"""Which *value* a name holds, as opposed to which type.

A registration is an identity fact -- "the thing you passed is the thing that
got stored" -- and types cannot express it: two unrelated parameters of the same
class are indistinguishable by type and completely different by identity. So
alongside the type lattice the collector tracks origins, and turns attribute
stores and receiver invocations into the escape and invoke evidence that
``registration`` later joins into rules.

Only ``TypeSummaryCollector`` records the resulting facts (see
``records_registry_facts`` on ``CollectorState``); the others compute origins and
discard them, which costs a few dictionary lookups and keeps one implementation
of the store shapes.

Requires from siblings: ``get_expr_types``, ``_attribute_owner_types``,
``_resolve_callees``, ``_infer_container_element_types``.
"""

from __future__ import annotations

import ast
from typing import Optional, Set
from ..ast_utils import unwrap_passthrough
from ..type_env import (
    Origin,
    attr_container_origin,
    attr_element_origin,
    param_element_origin,
)
from ..models import ClassAttr, FunctionEscapeSummary
from .constants import SELF_NAMES

from .state import CollectorState


class OriginsMixin(CollectorState):
    """Which *value* a name holds, as opposed to which type."""

    def _self_attr_keys(self, target: ast.AST) -> Set[ClassAttr]:
        """The ``(class, attribute)`` pairs an attribute expression names."""
        if not isinstance(target, ast.Attribute):
            return set()
        return {
            (owner, target.attr) for owner in self._attribute_owner_types(target)
        }

    def _value_origins(self, value: ast.AST) -> Set[Origin]:
        """Where the value of an expression came from.

        Only the forms that can carry a value through unchanged are followed. An
        arbitrary call or a computed expression produces a new value, so it has
        no origin and correctly stops the propagation.
        """
        value = unwrap_passthrough(value)
        if isinstance(value, ast.Name):
            return self.types.get_var_origins(value.id)
        if isinstance(value, ast.Starred):
            return self._value_origins(value.value)
        # ``x if flag else y`` yields one of its arms, so it carries both.
        if isinstance(value, ast.IfExp):
            return self._value_origins(value.body) | self._value_origins(value.orelse)
        # ``artists = self.get_children()`` -- an ordinary call, but one already
        # known to hand back the contents of an attribute. The value is the
        # collection, so it carries a container origin rather than an element
        # one; iterating it later is what recovers the elements.
        if isinstance(value, ast.Call):
            return {
                attr_container_origin(key)
                for key in self._returned_attrs_for_call(value)
            }
        return set()

    def _returned_attrs_for_call(self, value: ast.Call) -> Set[ClassAttr]:
        """Attributes whose contents a call is known to give back."""
        if not self.registry_facts.returned_attrs:
            return set()
        attrs: Set[ClassAttr] = set()
        for callee, _relation, is_resolved in self._resolve_callees(value.func):
            if is_resolved:
                attrs |= self.registry_facts.returned_attrs.get(callee, set())
        return attrs

    def _element_origins(self, value: ast.AST) -> Set[Origin]:
        """Where the *elements* of an expression came from.

        Three shapes matter. A literal collection holds whatever was written into
        it -- and for a dict that means the values, since the keys are labels;
        this is what makes ``self.children.update({name: child})`` a registration.
        Reading an attribute yields elements belonging to that attribute, which
        is how a value relayed from one registry into another stays traceable.
        A parameter yields elements *of* that parameter, which is the
        ``Pipeline(steps=[...])`` shape where the children arrive in bulk.
        """
        value = unwrap_passthrough(value)

        if isinstance(value, ast.Dict):
            origins: Set[Origin] = set()
            for item in value.values:
                origins |= self._value_origins(item)
            return origins
        if isinstance(value, (ast.List, ast.Set, ast.Tuple)):
            origins = set()
            for item in value.elts:
                # ``[*self._children, *self.spines.values()]`` splices two
                # collections in, so this literal's elements are *their*
                # elements. Taking the starred expression as one item would
                # lose every registry a framework merges on the way out.
                if isinstance(item, ast.Starred):
                    origins |= self._element_origins(item.value)
                else:
                    origins |= self._value_origins(item)
            return origins

        # ``self.registry``, ``self.registry[key]``, ``self.registry.values()``
        # and ``self.registry.items()`` all read out of the same attribute. The
        # subscript and the accessor call are peeled off because what matters is
        # which attribute the elements belong to, not the shape of the read.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr in {"values", "items", "keys"}:
                value = value.func.value
        if isinstance(value, ast.Subscript):
            value = value.value

        origins = {attr_element_origin(key) for key in self._self_attr_keys(value)}
        for kind, payload in self._value_origins(value):
            if kind == "param":
                origins.add(param_element_origin(payload))
            # Taking an item out of a collection that came from an attribute is
            # taking an item out of that attribute. This is the hop that lets a
            # registry survive being handed out of the class that owns it.
            elif kind == "attr_container":
                origins.add(attr_element_origin(payload))
        return origins

    def _note_attribute_store(
        self,
        container: ast.AST,
        origins: Set[Origin],
        *,
        is_container: bool,
        key: Optional[ast.AST] = None,
    ) -> None:
        """Record that values with these origins were stored on an attribute.

        A parameter landing here is an escape -- the caller's value is retained.
        An element of another attribute landing here is a relay between two
        registries, which is what lets the attribute registered into differ from
        the one invoked.

        ``key`` is the expression a keyed store filed the value under. When it is
        itself a parameter, the caller named this relationship and that name is
        worth keeping; see ``FunctionEscapeSummary.key_params``.
        """
        if not self.records_registry_facts or not origins:
            return
        targets = self._self_attr_keys(container)
        if not targets:
            return

        slots = {(attr, is_container) for _, attr in targets}
        for kind, payload in origins:
            if kind in {"param", "param_element"}:
                if self.current_callable is None:
                    continue
                summary = self.escape_summaries.setdefault(
                    self.current_callable, FunctionEscapeSummary()
                )
                if kind == "param":
                    summary.add(payload, slots)
                else:
                    summary.add_element(payload, slots)
                self._note_key_param(key)
            elif kind == "attr_element" and is_container:
                for target in targets:
                    self.registry_facts.add_element_flow(payload, target)

    def _note_key_param(self, key: Optional[ast.AST]) -> None:
        if key is None or self.current_callable is None:
            return
        for kind, payload in self._value_origins(key):
            if kind == "param":
                self.escape_summaries.setdefault(
                    self.current_callable, FunctionEscapeSummary()
                ).add_key_param(payload)

    def _note_receiver_invocation(self, func: ast.AST) -> None:
        """Record a method called on a value read out of a registry attribute.

        This is the evidence that a registry is *executed* rather than merely
        held, and it is what stops every ``self.config = config`` from looking
        like a registration.
        """
        if not self.records_registry_facts or not isinstance(func, ast.Attribute):
            return
        if self.current_callable is None:
            return

        if isinstance(func.value, ast.Name):
            origins = self.types.get_var_origins(func.value.id)
        else:
            origins = set()
        for kind, payload in origins:
            if kind == "attr_element":
                self.registry_facts.add_invocation(
                    payload, func.attr, self.current_callable
                )

        # ``self.hook()`` is what a template method does; the re-projection in
        # ``_registration_hook`` needs to know which hooks a base delegates to.
        if (
            isinstance(func.value, ast.Name)
            and func.value.id in SELF_NAMES
            and self.current_class
        ):
            self.registry_facts.add_self_delegation(self.current_callable, func.attr)
