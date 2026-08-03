"""The call-resolution passes.

``CallCollector`` is the heart of the analyzer. It walks a module tracking what
type each name currently holds, and turns every call -- and every implicit call
Python performs on your behalf, such as ``a + b`` reaching ``a.__add__`` -- into
a graph edge. Two subclasses reuse that machinery without emitting edges:

``ReturnSummaryCollector``
    Records what each callable gives back.
``TypeSummaryCollector``
    Records the types flowing into parameters and instance attributes.

All three read accumulated type facts through the ``return_summaries``,
``param_summaries``, and ``class_attr_types`` dictionaries handed to them. Those
are held by reference, not copied, so the pass driver in ``passes`` controls
when facts become visible.
"""


from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple

from .ast_utils import attribute_to_name, unwrap_passthrough
from .definitions import class_body_expressions, signature_expressions
from .discovery import module_callable_id
from .dunders import (
    AUGASSIGN_DUNDER_METHODS,
    BINOP_DUNDER_METHODS,
    COMPARE_DUNDER_METHODS,
    REVERSE_BINOP_DUNDER_METHODS,
    UNARY_DUNDER_METHODS,
)
from .project_index import ProjectIndex, unique
from .return_links import ReturnLink, ReturnLinkTable
from .type_env import (
    Origin,
    TypeEnv,
    TypeState,
    VarSource,
    attr_container_origin,
    attr_element_origin,
    param_element_origin,
    param_origin,
)
from .models import (
    CONFIDENCE_HIGH,
    CONFIDENCE_UNKNOWN,
    NO_SOURCE_SITE,
    CallGraphHealth,
    ClassAttr,
    ClassAttrTypes,
    Edge,
    FunctionEscapeSummary,
    FunctionParamSummary,
    FunctionReturnSummary,
    ModuleIndex,
    RegistryFacts,
    confidence_for_fanout,
    keyword_only_param_key,
    param_key,
)
from .registration import RegistrationRule, registration_child_exprs


# ``functools.partial(f, ...)`` evaluates to ``f`` with arguments pre-bound, so
# calling the result calls ``f``. Matched by source-level name, like the
# decorator names in ``definitions``, because importing to check would execute
# the analyzed project.
_PARTIAL_NAMES = frozenset({"partial", "functools.partial"})

# Element callables are keyed under the variable's name plus this suffix in the
# same scope map. A separate stack for one level of nesting would double the
# scope machinery for a dimension that is only ever read here, and the suffix
# cannot collide with a Python identifier.
_ELEMENT_SUFFIX = "[]"


def _callee_shape(func: ast.AST) -> str:
    """The syntactic form of a callee, for bucketing calls nothing resolved.

    Names the *shape* rather than the expression so the counts say which
    language feature is costing edges: ``call_result`` is a higher-order return,
    ``subscript`` a dispatch table, ``lambda`` an inline function.
    """
    shapes = {
        ast.Name: "name",
        ast.Attribute: "attribute",
        ast.Call: "call_result",
        ast.Subscript: "subscript",
        ast.Lambda: "lambda",
        ast.IfExp: "conditional",
        ast.BoolOp: "boolean",
    }
    return shapes.get(type(func), type(func).__name__.lower())

try:
    from microservice_pipeline.import_resolution import (
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from import_resolution import (  # type: ignore
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )


class CallCollector(ast.NodeVisitor):
    """Resolve calls and implicit Python operations into graph edges.

    The visitor also performs deliberately small-scale type inference, whose
    scoped state lives in ``self.types`` (:class:`~.type_env.TypeEnv`) and whose
    whole-project facts live in ``self.project_index``
    (:class:`~.project_index.ProjectIndex`).

    Resolution prefers known project definitions, but can retain an unresolved
    descriptive target when ``include_external`` is enabled.
    """
    def __init__(
        self,
        module: str,
        file: Path,
        module_index: ModuleIndex,
        callable_ids: Set[str],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        include_external: bool,
        package_prefix: Optional[str] | Sequence[str],
        return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
        param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
        class_attr_types: Optional[ClassAttrTypes] = None,
        return_links: Optional[ReturnLinkTable] = None,
        project_index: Optional[ProjectIndex] = None,
        escape_summaries: Optional[Dict[str, FunctionEscapeSummary]] = None,
        registry_facts: Optional[RegistryFacts] = None,
        registration_rules: Optional[Dict[str, RegistrationRule]] = None,
    ):
        self.module = module
        self.file = file
        self.module_index = module_index
        self.callable_ids = callable_ids
        self.module_map = module_map
        self.known_classes = known_classes
        self.include_external = include_external
        self.package_prefix = package_prefix
        self.return_summaries = (
            return_summaries if return_summaries is not None else {}
        )
        self.param_summaries = param_summaries if param_summaries is not None else {}
        self.class_attr_types = (
            class_attr_types if class_attr_types is not None else ClassAttrTypes()
        )
        # Registry evidence, held by reference like the summaries above so the
        # pass driver decides when a fact becomes visible. Only
        # ``TypeSummaryCollector`` writes to these; every collector reads them,
        # since re-projection needs the delegation facts during the edge pass.
        self.escape_summaries = (
            escape_summaries if escape_summaries is not None else {}
        )
        self.registry_facts = (
            registry_facts if registry_facts is not None else RegistryFacts()
        )
        # Populated only for the final edge pass, once escape and invoke facts
        # have been joined. Empty everywhere else, which is what keeps the
        # summary passes from emitting edges.
        self.registration_rules = registration_rules or {}
        # Whole-project class facts. Shared across every file and every pass of a
        # run when the caller supplies one; built here only for callers that have
        # no run to share (tests, one-off use).
        self.project_index = (
            project_index
            if project_index is not None
            else ProjectIndex(module_map, known_classes, callable_ids)
        )
        self.module_callable = module_callable_id(module)

        # Deferred "we return whatever that callee returns" facts. ``None``
        # switches recording off, which is what every collector except
        # ``ReturnSummaryCollector`` wants.
        self.return_links = return_links
        self._link_target: List[Tuple[str, Optional[int]]] = []

        self.current_callable: Optional[str] = None
        self.current_class: Optional[str] = None
        # Lexical *function* nesting, which is what decides a nested callable's
        # ID. Tracked apart from ``current_callable`` because a class body is a
        # callable that edges are attributed to but that never prefixes the name
        # of a method defined inside it -- ``C.method``, never ``C.<locals>.method``.
        self.enclosing_function: Optional[str] = None
        self.callable_stack: List[str] = []
        self.types = TypeEnv()
        self.edges: List[Edge] = []
        self.health = CallGraphHealth()

    @contextmanager
    def _recording_return_links(
        self, target_field: str, slot: Optional[int] = None
    ) -> Generator[None, None, None]:
        """Declare which part of the current callable's return summary is being filled.

        Type inference recurses through arbitrary expressions, and by the time it
        reaches a call it no longer knows what it is computing a value *for*.
        This marks that at the top, so ``_note_return_dependency`` further down
        can pair "which field we are filling" with "which field we read".
        """
        self._link_target.append((target_field, slot))
        try:
            yield
        finally:
            self._link_target.pop()

    @contextmanager
    def _resolving_receiver(self) -> Generator[None, None, None]:
        """Infer a receiver's type without charging it to the enclosing return.

        ``return build().submit()`` returns *submit's* value, not ``build``'s.
        Working out that the receiver is an ``Order`` runs the same inference
        that records return links, so without this the enclosing
        ``_recording_return_links`` block would still be in force and we would
        record "our return type includes build's" -- which is false.

        The link for ``submit`` itself is recorded by the caller of this block,
        outside the suspension, so only the receiver hop is dropped.
        """
        saved = self._link_target
        self._link_target = []
        try:
            yield
        finally:
            self._link_target = saved

    def _note_return_dependency(
        self, callee: str, source_field: str, source_slot: Optional[int] = None
    ) -> None:
        """Remember that our return type depends on ``callee``'s, known or not.

        Recorded even when the callee already has a summary: that summary can
        still grow in a later round, and the link is what carries the growth
        onwards.
        """
        if self.return_links is None or not self._link_target:
            return
        if not self.current_callable:
            return
        target_field, target_slot = self._link_target[-1]
        self.return_links.record(
            ReturnLink(
                caller=self.current_callable,
                callee=callee,
                source_field=source_field,
                target_field=target_field,
                source_slot=source_slot,
                target_slot=target_slot,
            )
        )

    def _is_internal_callee(self, callee: str) -> bool:
        """Apply optional package-prefix filtering to a resolved target."""
        if not self.package_prefix:
            return True
        if isinstance(self.package_prefix, str):
            prefixes = (self.package_prefix,)
        else:
            prefixes = tuple(self.package_prefix)
        return any(callee == prefix or callee.startswith(prefix + ".") for prefix in prefixes)

    # Delegating shims onto ``TypeEnv``. The visitor below uses these constantly,
    # so they stay on the collector rather than forcing ``self.types.`` at every
    # one of the several hundred call sites.
    def push_scope(self) -> None:
        self.types.push_scope()

    def pop_scope(self) -> None:
        self.types.pop_scope()

    def _copy_type_state(self) -> TypeState:
        return self.types.copy_state()

    def _restore_type_state(self, state: TypeState) -> None:
        self.types.restore_state(state)

    def _merge_type_states(self, states: Iterable[TypeState]) -> TypeState:
        return self.types.merge_states(states)

    def set_var_type(self, var: str, typ: str) -> None:
        self.types.set_var_type(var, typ)

    def set_var_types(self, var: str, types: Set[str]) -> None:
        self.types.set_var_types(var, types)

    def get_var_types(self, var: str) -> Set[str]:
        return self.types.get_var_types(var)

    def get_var_type(self, var: str) -> Optional[str]:
        return self.types.get_var_type(var)

    def set_container_types(self, var: str, types: Set[str]) -> None:
        self.types.set_container_types(var, types)

    def add_container_types(self, var: str, types: Set[str]) -> None:
        self.types.add_container_types(var, types)

    def get_container_types(self, var: str) -> Set[str]:
        return self.types.get_container_types(var)

    def set_var_source(self, var: str, value: ast.AST, slot: Optional[int]) -> None:
        self.types.set_var_source(var, value, slot)

    def get_var_sources(self, var: str) -> Set[VarSource]:
        return self.types.get_var_sources(var)

    def set_attr_expr_types(self, attr_expr: str, types: Set[str]) -> None:
        self.types.set_attr_expr_types(attr_expr, types)

    def get_attr_expr_types(self, attr_expr: str) -> Set[str]:
        return self.types.get_attr_expr_types(attr_expr)

    def get_expr_types(self, expr: ast.AST) -> Set[str]:
        """Infer possible class IDs for a simple name or attribute expression."""
        if isinstance(expr, ast.Name):
            return self.get_var_types(expr.id)
        if isinstance(expr, ast.Attribute):
            expr_name = attribute_to_name(expr)
            if expr_name:
                local_types = self.get_attr_expr_types(expr_name)
                if local_types:
                    return local_types

            base_types: Set[str] = set()
            if (
                isinstance(expr.value, ast.Name)
                and expr.value.id in {"self", "cls"}
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    base_types.add(class_id)
            base_types.update(self.get_expr_types(expr.value))

            attr_types: Set[str] = set()
            for base_type in base_types:
                for owner in self._class_and_ancestors(base_type):
                    attr_types.update(
                        self.class_attr_types.object_types.get((owner, expr.attr), set())
                    )
            attr_types.update(self._infer_property_return_class_types(expr))
            return attr_types
        return set()

    def _class_attr_element_types(self, expr: ast.AST) -> Set[str]:
        """Element types recorded for ``self.<attr>`` and friends across methods.

        This is what makes a registry attribute usable: ``add_child`` stores into
        ``self.children`` in one method and another method iterates it, so the
        fact has to survive the boundary between them.
        """
        if not isinstance(expr, ast.Attribute):
            return set()

        base_types: Set[str] = set()
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id in {"self", "cls"}
            and self.current_class
        ):
            class_id = self.current_class_id()
            if class_id:
                base_types.add(class_id)
        base_types.update(self.get_expr_types(expr.value))

        element_types: Set[str] = set()
        for base_type in base_types:
            for owner in self._class_and_ancestors(base_type):
                element_types.update(
                    self.class_attr_types.element_types.get((owner, expr.attr), set())
                )
        return element_types

    def _receiver_types_for_expr(self, expr: ast.AST) -> Set[str]:
        receiver_types: Set[str] = set()
        if (
            isinstance(expr, ast.Name)
            and expr.id in {"self", "cls"}
            and self.current_class
        ):
            class_id = self.current_class_id()
            if class_id:
                receiver_types.add(class_id)
        receiver_types.update(self.get_expr_types(expr))
        return receiver_types

    def _resolve_property_getter_targets(self, node: ast.Attribute) -> List[str]:
        targets: List[str] = []
        for receiver_type in sorted(self._receiver_types_for_expr(node.value)):
            for target in self._resolve_method_targets(receiver_type, node.attr):
                if target in self.project_index.property_ids:
                    targets.append(target)
        return self._unique(targets)

    def _infer_property_return_class_types(self, node: ast.Attribute) -> Set[str]:
        class_types: Set[str] = set()
        for getter in self._resolve_property_getter_targets(node):
            summary = self.return_summaries.get(getter)
            if summary:
                class_types.update(summary.class_types)
        return class_types

    def _infer_property_return_element_types(self, node: ast.Attribute) -> Set[str]:
        element_types: Set[str] = set()
        for getter in self._resolve_property_getter_targets(node):
            summary = self.return_summaries.get(getter)
            if summary:
                element_types.update(summary.element_types)
        return element_types

    def _attribute_owner_types(self, target: ast.Attribute) -> Set[str]:
        base_types: Set[str] = set()
        if (
            isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
            and self.current_class
        ):
            class_id = self.current_class_id()
            if class_id:
                base_types.add(class_id)
        base_types.update(self.get_expr_types(target.value))
        return base_types

    def set_attribute_types(self, target: ast.Attribute, types: Set[str]) -> None:
        if not types:
            return

        attr_name = attribute_to_name(target)
        if attr_name:
            self.set_attr_expr_types(attr_name, types)

        for base_type in self._attribute_owner_types(target):
            self.class_attr_types.add_object_types((base_type, target.attr), types)

    def set_attribute_element_types(self, target: ast.Attribute, types: Set[str]) -> None:
        """Record what an attribute *contains*, from a whole-value assignment."""
        if not types:
            return
        for base_type in self._attribute_owner_types(target):
            self.class_attr_types.add_element_types((base_type, target.attr), types)

    def record_attribute_container_store(self, container: ast.AST, values: Set[str]) -> None:
        """Note types added into a container attribute, however they got there.

        Covers both ``self.x[k] = v`` and ``self.x.append(v)`` styles. Only the
        element facts are touched: storing into a container says nothing about
        what the attribute itself is, and claiming otherwise would make the
        container look like the thing it holds.
        """
        if not values or not isinstance(container, ast.Attribute):
            return
        for base_type in self._attribute_owner_types(container):
            self.class_attr_types.add_element_types((base_type, container.attr), values)

    # -- value origins ---------------------------------------------------------
    #
    # Tracking which *value* a name currently holds, as opposed to which type.
    # A registration is an identity fact -- "the thing you passed is the thing
    # that got stored" -- and types cannot express it: two unrelated parameters
    # of the same class are indistinguishable by type and completely different
    # by identity. Only ``TypeSummaryCollector`` turns the resulting facts into
    # summaries; the others compute origins and discard them, which costs a few
    # dictionary lookups and keeps one implementation of the store shapes.

    records_registry_facts = False

    def set_var_origins(self, var: str, origins: Set[Origin]) -> None:
        self.types.set_var_origins(var, origins)

    def get_var_origins(self, var: str) -> Set[Origin]:
        return self.types.get_var_origins(var)

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
            return self.get_var_origins(value.id)
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
            origins = self.get_var_origins(func.value.id)
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
            and func.value.id in {"self", "cls"}
            and self.current_class
        ):
            self.registry_facts.add_self_delegation(self.current_callable, func.attr)

    def current_class_id(self) -> Optional[str]:
        if self.current_class:
            return f"{self.module}.{self.current_class}"
        return None

    # Delegating shims onto ``ProjectIndex``. The resolution below depends only
    # on whole-project definition facts, never on inference state, which is why
    # it can live outside the collector and be shared across every pass.
    def _is_known_class(self, class_id: str) -> bool:
        return self.project_index.is_known_class(class_id)

    def _known_class_id(self, class_id: str) -> str:
        return self.project_index.known_class_id(class_id)

    def _unique(self, values: Iterable[str]) -> List[str]:
        return unique(values)

    def _resolve_class_reference_name(self, name: str) -> List[str]:
        return self.project_index.resolve_class_reference_name(
            self.module, self.module_index, name
        )

    def _resolve_method_targets(
        self, class_id: str, method_name: str, seen: Optional[Set[str]] = None
    ) -> List[str]:
        return self.project_index.resolve_method_targets(class_id, method_name, seen)

    def _class_and_ancestors(self, class_id: str) -> List[str]:
        return self.project_index.class_and_ancestors(class_id)

    def _resolve_subclass_override_targets(
        self, class_id: str, method_name: str
    ) -> List[str]:
        return self.project_index.resolve_subclass_override_targets(
            class_id, method_name
        )

    def _resolve_constructor_targets(self, class_id: str) -> List[str]:
        return self.project_index.resolve_constructor_targets(class_id)

    def _resolve_super_method_targets(self, method_name: str) -> List[str]:
        return self.project_index.resolve_super_method_targets(
            self.current_class_id(), method_name
        )

    def _add_edge(
        self,
        callee: str,
        relation: str,
        is_resolved: bool,
        lineno: int,
        col_offset: int = NO_SOURCE_SITE,
        confidence: str = CONFIDENCE_HIGH,
    ) -> None:
        """Append an edge after applying unresolved/external filtering policy.

        ``col_offset`` completes the source *site*, which a line alone does not
        identify -- on climlab 30% of call expressions share a line with another
        call. It stays ``NO_SOURCE_SITE`` for edges whose position is not a call
        the interpreter makes at that spot, so they can be excluded from any
        site-level comparison instead of being matched against the wrong call.
        """
        if self.current_callable is None:
            self.health.dropped_no_caller += 1
            return
        # Tallied *before* the filter that drops it. Counting only what survives
        # is what made "unresolved: 0" a tautology rather than a measurement:
        # with include_external off, an unresolved edge cannot reach the list, so
        # the reported unresolved count was structurally always zero.
        if not is_resolved and not self.include_external:
            self.health.dropped_unresolved[relation] += 1
            return
        if is_resolved and not self._is_internal_callee(callee):
            self.health.dropped_external[relation] += 1
            return
        self.health.emitted[relation] += 1
        self.edges.append(
            Edge(
                caller=self.current_callable,
                callee=callee,
                file=str(self.file),
                lineno=lineno,
                col_offset=col_offset,
                resolved=is_resolved,
                relation=relation,
                confidence=confidence if is_resolved else CONFIDENCE_UNKNOWN,
            )
        )

    def _unique_callee_results(
        self, values: Iterable[Tuple[str, str, bool]]
    ) -> List[Tuple[str, str, bool]]:
        seen: Set[Tuple[str, str, bool]] = set()
        ordered: List[Tuple[str, str, bool]] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _resolve_method_callees_for_expr(
        self, receiver: ast.AST, method_name: str, relation: str
    ) -> List[Tuple[str, str, bool]]:
        targets: List[Tuple[str, str, bool]] = []
        for receiver_type in sorted(self._receiver_types_for_expr(receiver)):
            resolved_targets = self._resolve_method_targets(receiver_type, method_name)
            if resolved_targets:
                targets.extend(
                    (target, relation, True) for target in resolved_targets
                )
            else:
                targets.append((f"{receiver_type}.{method_name}", relation, False))
        return self._unique_callee_results(targets)

    def _callable_belongs_to_known_class(self, callable_id: str) -> bool:
        return self.project_index.callable_belongs_to_known_class(callable_id)

    def _implicit_receiver_arg_offset(self, callee: str, relation: str) -> int:
        """Account for implicit ``self``/``cls`` in positional parameter facts."""
        if callee in self.project_index.static_method_ids:
            return 0
        if relation == "constructor":
            return 1
        if relation == "class_method":
            return 1 if callee in self.project_index.class_method_ids else 0
        # ``virtual_override`` belongs here for the same reason ``self_method``
        # does: both arise from ``self.hook(...)`` and both name a bound method of
        # a known class, so argument 0 is the first *real* parameter. Omitting it
        # does not drop the fact, it files it one slot early -- argument 0's type
        # lands in the ``self`` slot of ``positional_types`` and is read back by
        # ``_seed_param_types``. Worse for the registry passes, which restrict
        # themselves to ``self``/``cls`` calls and so see little else.
        if relation in {
            "self_method",
            "super_method",
            "inferred_type",
            "virtual_override",
        }:
            if self._callable_belongs_to_known_class(callee):
                return 1
        return 0

    def _add_dunder_edges(
        self,
        receiver: ast.AST,
        method_name: str,
        relation: str,
        lineno: int,
        col_offset: int = NO_SOURCE_SITE,
        confidence: str = CONFIDENCE_HIGH,
    ) -> None:
        results = self._resolve_method_callees_for_expr(receiver, method_name, relation)
        confidence = confidence_for_fanout(sum(1 for result in results if result[2]))
        for callee, edge_relation, is_resolved in results:
            self._add_edge(
                callee, edge_relation, is_resolved, lineno, col_offset, confidence
            )

    def _add_membership_edges(
        self, container: ast.AST, lineno: int, col_offset: int = NO_SOURCE_SITE
    ) -> None:
        """Model ``x in container`` as ``__contains__`` or iteration fallback."""
        targets: List[Tuple[str, str, bool]] = []
        for receiver_type in sorted(self._receiver_types_for_expr(container)):
            contains_targets = self._resolve_method_targets(receiver_type, "__contains__")
            if contains_targets:
                targets.extend(
                    (target, "dunder_contains", True)
                    for target in contains_targets
                )
                continue

            iter_targets = self._resolve_method_targets(receiver_type, "__iter__")
            if iter_targets:
                targets.extend(
                    (target, "dunder_iter", True) for target in iter_targets
                )
            else:
                targets.append(
                    (f"{receiver_type}.__contains__", "dunder_contains", False)
                )

        results = self._unique_callee_results(targets)
        confidence = confidence_for_fanout(sum(1 for result in results if result[2]))
        for callee, relation, is_resolved in results:
            self._add_edge(
                callee, relation, is_resolved, lineno, col_offset, confidence
            )

    def _registration_parent_types(
        self, node: ast.Call, callee: str, relation: str
    ) -> Set[str]:
        """Which object this call registers a child *into*.

        Usually the receiver: ``parent.attach(child)``. For a constructor it is
        the class being built, since ``Pipeline(steps=[...])`` registers into the
        instance the call is producing and there is no receiver expression to
        read it from.
        """
        if relation == "constructor":
            owner = callee.rsplit(".", 1)[0]
            return {owner} if self._is_known_class(owner) else set()

        if not isinstance(node.func, ast.Attribute):
            return set()
        receiver = node.func.value
        if (
            isinstance(receiver, ast.Name)
            and receiver.id in {"self", "cls"}
            and self.current_class
        ):
            class_id = self.current_class_id()
            return {class_id} if class_id else set()
        return self._receiver_types_for_expr(receiver)

    def _registration_child_types(self, child_expr: ast.AST) -> Set[str]:
        if isinstance(child_expr, ast.Call):
            return self._infer_class_types_from_call(child_expr)
        if isinstance(child_expr, (ast.Name, ast.Attribute)):
            return self.get_expr_types(child_expr)
        return set()

    def _registration_hook(self, parent_type: str, child_type: str, method: str) -> str:
        """Pick the method to link, re-projecting through a template method.

        The registry is usually invoked through a method the framework's base
        class defines and nobody overrides -- climlab calls ``proc.compute()``,
        which resolves to ``TimeDependentProcess.compute`` for every process in
        the tree. An edge between two copies of that node says nothing, and hub
        policy discards it downstream anyway.

        So when the invoked method resolves to the same definition for parent and
        child, follow that definition's delegation one hop and look for the hook
        it hands off to that subclasses actually override -- the Template Method
        shape. ``compute`` delegates to ``_compute_type``, ``_compute`` and
        ``_build_process_type_list``; only ``_compute`` is overridden anywhere, so
        only ``_compute`` carries information about which process this is.

        Returns the method name to link, or ``""`` when re-projection is needed
        but does not resolve to exactly one hook.
        """
        parent_targets = self._resolve_method_targets(parent_type, method)
        child_targets = self._resolve_method_targets(child_type, method)
        if len(parent_targets) != 1 or len(child_targets) != 1:
            return ""
        if parent_targets[0] != child_targets[0]:
            # Parent and child give the invoked method different bodies, so it
            # already distinguishes them and needs no re-projection.
            return method

        shared = parent_targets[0]
        hooks = [
            hook
            for hook in sorted(self.registry_facts.self_delegations.get(shared, ()))
            if self._is_overridden_hook(shared, hook)
        ]
        # More than one candidate means the base delegates to several genuine
        # hooks and nothing here says which one carries the coupling. Guessing
        # would fabricate edges, so decline.
        return hooks[0] if len(hooks) == 1 else ""

    def _is_overridden_hook(self, base_callable: str, method: str) -> bool:
        owner = base_callable.rsplit(".", 1)[0]
        return bool(self._resolve_subclass_override_targets(owner, method))

    def _add_registration_edges(self, node: ast.Call) -> None:
        """Emit parent/child coupling for a call that registers one into the other.

        Registering a child establishes execution coupling that is not an
        ordinary source-level call from one hook to the other, so no amount of
        call resolution finds it. What makes this call a registration is derived
        rather than recognised by name: some parameter of the callee is known to
        escape into an attribute of ``self``, and elements of that attribute are
        known to be invoked. See ``registration``.

        The edge is emitted only when both sides resolve unambiguously. That
        precision is the whole value of doing this at the call site -- the
        attribute itself has lost which parent each child belongs to.
        """
        if not self.registration_rules:
            return

        for callee, relation, is_resolved in self._resolve_callees(node.func):
            if not is_resolved:
                continue
            rule = self.registration_rules.get(callee)
            if rule is None:
                continue

            offset = self._implicit_receiver_arg_offset(callee, relation)
            parent_types = sorted(
                self._registration_parent_types(node, callee, relation)
            )
            if len(parent_types) != 1:
                continue

            for child_expr in registration_child_exprs(node, rule, offset):
                child_types = sorted(self._registration_child_types(child_expr))
                if len(child_types) != 1:
                    continue

                hook = self._registration_hook(
                    parent_types[0], child_types[0], rule.invoked_method
                )
                if not hook:
                    continue

                parent_hook = self._resolve_method_targets(parent_types[0], hook)
                child_hook = self._resolve_method_targets(child_types[0], hook)
                if len(parent_hook) != 1 or len(child_hook) != 1:
                    continue
                if not self._is_internal_callee(parent_hook[0]):
                    continue

                # Emitted against the parent's hook rather than the enclosing
                # callable: the coupling belongs to the two objects, not to
                # whichever function happened to wire them together.
                caller = self.current_callable
                self.current_callable = parent_hook[0]
                try:
                    # No ``col_offset`` on purpose. Rebinding the caller above
                    # leaves the position pointing at the registration call,
                    # which sits in a different callable, so ``(caller, line,
                    # col)`` would not name a site inside ``parent_hook`` at all.
                    # NO_SOURCE_SITE keeps it out of any site-level comparison
                    # rather than matching it against an unrelated call.
                    self._add_edge(
                        child_hook[0], "registered_invoke", True, node.lineno
                    )
                finally:
                    self.current_callable = caller

    def _visit_call_children(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
        else:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def _call_is_empty_container(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"list", "set", "tuple"}
            and not node.args
        )

    def _is_container_literal(self, node: ast.AST) -> bool:
        return isinstance(
            node, (ast.List, ast.Set, ast.Tuple, ast.ListComp, ast.SetComp)
        ) or self._call_is_empty_container(node)

    def _assign_target_types(
        self,
        target: ast.AST,
        class_types: Set[str],
        container_types: Set[str],
        value: ast.AST,
        slot: Optional[int] = None,
    ) -> None:
        """Attach inferred value types to a name, attribute, or destructuring target.

        ``slot`` records which tuple position of ``value`` this target took, so
        provenance for ``a, b = make_pair()`` stays position-specific.
        """
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                self._assign_target_types(
                    item,
                    self._infer_sequence_slot_class_types(value, index),
                    self._infer_sequence_slot_container_types(value, index),
                    value,
                    slot=index,
                )
            return

        if isinstance(target, ast.Name) and self.types.var_types_stack:
            if class_types:
                self.set_var_types(target.id, class_types)
            if container_types or self._is_container_literal(value):
                self.set_container_types(target.id, container_types)
            # Unconditional: the point of provenance is the case where nothing
            # is known about ``value`` yet, so an empty inference above is
            # exactly when this matters most.
            self.set_var_source(target.id, value, slot)
            # A destructured position holds one element of the value, not the
            # value; ``for name, proc in registry.items()`` is the shape that
            # matters and it is handled where the loop is visited.
            if slot is None:
                self.set_var_origins(target.id, self._value_origins(value))
            return

        if isinstance(target, ast.Attribute):
            self.set_attribute_types(target, class_types)
            self.set_attribute_element_types(target, container_types)
            # ``self.parent = parent`` retains a single value rather than
            # collecting one, so it is recorded as a non-container escape and
            # will not on its own make the callable a registrar.
            self._note_attribute_store(
                target, self._value_origins(value), is_container=False
            )
            self._note_attribute_store(
                target, self._element_origins(value), is_container=True
            )
            return

        # ``self.registry[key] = value``: the assignment target is a subscript
        # of an attribute, so the value becomes an element of that attribute.
        if isinstance(target, ast.Subscript):
            self.record_attribute_container_store(target.value, class_types)
            self._note_attribute_store(
                target.value,
                self._value_origins(value),
                is_container=True,
                key=target.slice,
            )

    def visit_Module(self, node: ast.Module) -> None:
        """Treat top-level statements as the body of the synthetic module node."""
        prev_callable = self.current_callable
        self.current_callable = self.module_callable
        self.callable_stack.append(self.module_callable)
        self.push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable

    def _add_import_edge(
        self, module_name: str, lineno: int, col_offset: int = NO_SOURCE_SITE
    ) -> None:
        imported_module_callable = module_callable_id(module_name)
        if imported_module_callable in self.callable_ids:
            self._add_edge(
                imported_module_callable, "import", True, lineno, col_offset
            )
        else:
            self._add_edge(module_name, "import", False, lineno, col_offset)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_import_edge(alias.name, node.lineno, node.col_offset)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported_module = resolve_import_from_module(
            self.module,
            node.module,
            node.level,
            current_is_package=is_package_file(self.file),
        )
        if imported_module is None:
            return
        if node.module:
            self._add_import_edge(imported_module, node.lineno, node.col_offset)
            return
        for alias in node.names:
            target = resolve_import_from_target(
                self.module,
                node.module,
                node.level,
                alias.name,
                current_is_package=is_package_file(self.file),
            )
            if target:
                self._add_import_edge(target, node.lineno, node.col_offset)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class body, attributing what it evaluates to the class itself.

        The class body is a code object that runs once, when the ``class``
        statement executes, so a computed default argument belongs to the class
        and not to the method whose signature it is written in. The interpreter
        agrees: it reports the caller of ``make_slabatm_axis()`` as
        ``SlabAtmosphere``, not ``SlabAtmosphere.__init__``.
        """
        prev_class = self.current_class
        prev_callable = self.current_callable
        self.current_class = node.name

        # ``class_bodies``, not ``callable_ids`` -- a class body is a valid edge
        # caller but never a callee, so it is deliberately absent from the
        # resolution universe. See models.resolvable_callable_ids.
        class_id = f"{self.module}.{node.name}"
        if class_id in self.module_index.class_bodies:
            self.current_callable = class_id

        # Everything the class statement evaluates, attributed to the class...
        for expression in class_body_expressions(node):
            self.visit(expression)

        # ...and then each method, entered in its own scope with the signature
        # already accounted for above.
        self.current_callable = prev_callable
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._enter_callable(statement, statement.name)

        self.current_class = prev_class
        self.current_callable = prev_callable

    def _enter_callable(self, node: ast.AST, name: str) -> None:
        """Enter a function scope, seed parameter types, visit it, and restore state."""
        prev_callable = self.current_callable
        prev_enclosing = self.enclosing_function
        if self.enclosing_function is not None:
            self.current_callable = f"{self.enclosing_function}.<locals>.{name}"
        elif self.current_class:
            self.current_callable = f"{self.module}.{self.current_class}.{name}"
        else:
            self.current_callable = f"{self.module}.{name}"

        self.enclosing_function = self.current_callable
        self.callable_stack.append(self.current_callable)
        self.push_scope()

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if self.current_class and node.args.args:
                first_arg = node.args.args[0].arg
                if first_arg in {"self", "cls"}:
                    cls_id = f"{self.module}.{self.current_class}"
                    self.set_var_type(first_arg, cls_id)
            # An empty summary still matters: annotations are read here too, and
            # they are available whether or not any call site was observed.
            param_summary = self.param_summaries.get(self.current_callable) or FunctionParamSummary()
            positional_args = [*node.args.posonlyargs, *node.args.args]
            for index, arg in enumerate(positional_args):
                self._seed_param_types(param_summary, index, arg)
                self._seed_param_origin(index, arg, is_receiver=index == 0)
            for index, arg in enumerate(
                node.args.kwonlyargs, start=len(positional_args)
            ):
                self._seed_param_types(param_summary, index, arg)
                self._seed_param_origin(index, arg, keyword_only=True)

        # Only the body. Decorators, defaults and annotations were visited by the
        # enclosing scope, which is where the interpreter evaluates them.
        for statement in getattr(node, "body", []):
            self.visit(statement)
        self.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable
        self.enclosing_function = prev_enclosing

    # Typing constructs that wrap the type actually of interest. ``Optional[X]``
    # and ``Union[X, None]`` still denote an ``X``; the container forms denote a
    # collection *of* ``X``, which is a different fact and handled separately.
    _TRANSPARENT_ANNOTATION_NAMES = frozenset(
        {"Optional", "Union", "Final", "Annotated", "ClassVar", "typing"}
    )
    _CONTAINER_ANNOTATION_NAMES = frozenset(
        {
            "List", "list", "Set", "set", "FrozenSet", "frozenset",
            "Sequence", "Iterable", "Iterator", "Collection", "MutableSequence",
            "Tuple", "tuple", "Dict", "dict", "Mapping", "MutableMapping",
            "DefaultDict", "defaultdict", "OrderedDict", "Deque", "deque",
        }
    )

    @staticmethod
    def _annotation_head(node: ast.AST) -> str:
        name = attribute_to_name(node) or ""
        return name.rsplit(".", 1)[-1]

    def _annotation_types(self, node: Optional[ast.AST]) -> Tuple[Set[str], Set[str]]:
        """Resolve an annotation to ``(object types, element types)``.

        Annotations are free, reliable inference that the collectors previously
        ignored entirely. ``def solve(mesh: Mesh)`` states outright what a call
        site might never reveal.

        A container annotation contributes only element types: ``items: List[Order]``
        means ``items`` *holds* orders, so treating ``Order`` as the type of
        ``items`` itself would resolve ``items.submit()`` to a method that
        instances of ``list`` do not have. Mapping annotations follow the same
        rule as dict literals -- the values are the elements.
        """
        if node is None:
            return set(), set()

        # ``"Mesh"`` as a forward reference, and the string halves of
        # ``from __future__ import annotations`` output.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return set(self._resolve_class_reference_name(node.value)), set()

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left_objects, left_elements = self._annotation_types(node.left)
            right_objects, right_elements = self._annotation_types(node.right)
            return left_objects | right_objects, left_elements | right_elements

        if isinstance(node, ast.Subscript):
            head = self._annotation_head(node.value)
            inner = node.slice
            parts = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]

            if head in self._CONTAINER_ANNOTATION_NAMES:
                # ``Dict[K, V]`` -- the values are the elements; a one-argument
                # container annotates its elements directly.
                targets = parts[1:] if head.lower() in {"dict", "mapping", "mutablemapping", "defaultdict", "ordereddict"} and len(parts) > 1 else parts
                element_types: Set[str] = set()
                for part in targets:
                    part_objects, part_elements = self._annotation_types(part)
                    element_types |= part_objects | part_elements
                return set(), element_types

            if head in self._TRANSPARENT_ANNOTATION_NAMES:
                objects: Set[str] = set()
                elements: Set[str] = set()
                for part in parts:
                    part_objects, part_elements = self._annotation_types(part)
                    objects |= part_objects
                    elements |= part_elements
                return objects, elements
            return set(), set()

        name = attribute_to_name(node)
        if not name or name == "None":
            return set(), set()
        return set(self._resolve_class_reference_name(name)), set()

    def _seed_param_types(
        self, param_summary: FunctionParamSummary, index: int, arg: ast.arg
    ) -> None:
        """Bind one parameter to the types observed at its call sites.

        Object and element facts are seeded into different maps: a parameter that
        received a list of ``Order`` must answer ``Order`` when iterated, not
        when called on directly.
        """
        annotated_objects, annotated_elements = self._annotation_types(arg.annotation)

        param_types = set(param_summary.positional_types.get(index, set()))
        param_types.update(param_summary.named_types.get(arg.arg, set()))
        param_types |= annotated_objects
        if param_types:
            self.set_var_types(arg.arg, param_types)

        element_types = set(param_summary.positional_element_types.get(index, set()))
        element_types.update(param_summary.named_element_types.get(arg.arg, set()))
        element_types |= annotated_elements
        if element_types:
            self.set_container_types(arg.arg, element_types)

        # The callable dimension of the same fact. This is what makes
        # ``def apply(self, msg, func): return func(...)`` resolvable: the
        # callable arrived at a call site and has to be carried into the callee.
        param_callables = set(param_summary.positional_callables.get(index, set()))
        param_callables.update(param_summary.named_callables.get(arg.arg, set()))
        if param_callables:
            self.types.set_var_callables(arg.arg, param_callables)

    def _seed_param_origin(
        self,
        index: int,
        arg: ast.arg,
        *,
        is_receiver: bool = False,
        keyword_only: bool = False,
    ) -> None:
        """Mark a parameter name as holding the value its caller passed.

        The receiver is skipped. ``self`` is not something a caller registers --
        it is the thing being registered *into*, and treating it as a parameter
        would read every ``self.x = ...`` in a method as an escape of ``self``.
        """
        if is_receiver and self.current_class and arg.arg in {"self", "cls"}:
            return
        key = (
            keyword_only_param_key(arg.arg)
            if keyword_only
            else param_key(index, arg.arg)
        )
        origins = {param_origin(key)}
        # A parameter that callers fill with the contents of a registry keeps
        # that provenance inside the callee, so ``for a in artists: a.draw()``
        # in a free function still counts as invoking the registry.
        if self.current_callable is not None:
            origins.update(
                attr_container_origin(attr)
                for attr in self.registry_facts.attrs_for_param(
                    self.current_callable, index, arg.arg
                )
            )
        self.set_var_origins(arg.arg, origins)

    def _resolve_enclosing_local_callable(self, name: str) -> Optional[str]:
        for scope_callable in reversed(self.callable_stack):
            candidate = f"{scope_callable}.<locals>.{name}"
            if candidate in self.callable_ids:
                return candidate
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_signature(node)
        self._enter_callable(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_signature(node)
        self._enter_callable(node, node.name)

    def _visit_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Visit a ``def``'s decorators, defaults and annotations where they run.

        ``visit_ClassDef`` handles methods itself, so this only fires for module
        level and nested functions, whose enclosing scope is already current.
        """
        for expression in signature_expressions(node):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Enter a lambda body as its own callable scope.

        Without this a call inside a lambda body is attributed to the enclosing
        function, while the tracer reports it under ``<locals>.<lambda>`` -- the
        two graphs disagree about who made the call.
        """
        lambda_id = self._lambda_id(node)
        if lambda_id not in self.callable_ids:
            self.generic_visit(node)
            return

        prev_callable = self.current_callable
        prev_enclosing = self.enclosing_function
        self.current_callable = lambda_id
        self.enclosing_function = lambda_id
        self.callable_stack.append(lambda_id)
        self.push_scope()
        self.visit(node.body)
        self.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable
        self.enclosing_function = prev_enclosing

    def visit_If(self, node: ast.If) -> None:
        """Analyze both branches independently and keep the union of their facts.

        Static analysis does not know which branch will run. Mutating one shared
        map in source order would make the second branch incorrectly depend on
        the first, so both start from the same snapshot.
        """
        self.visit(node.test)
        base_state = self._copy_type_state()

        self._restore_type_state(base_state)
        for stmt in node.body:
            self.visit(stmt)
        body_state = self._copy_type_state()

        self._restore_type_state(base_state)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        orelse_state = self._copy_type_state()

        self._restore_type_state(self._merge_type_states([body_state, orelse_state]))

    def visit_Assign(self, node: ast.Assign) -> None:
        class_types = self._infer_class_types_from_value(node.value)
        container_types = self._infer_container_element_types(node.value)
        for target in node.targets:
            self._assign_target_types(target, class_types, container_types, node.value)
            self._assign_target_callables(target, node.value)
        self.generic_visit(node)

    def _assign_target_callables(self, target: ast.AST, value: ast.AST) -> None:
        """Record callables an assignment binds, for locals and ``self.<attr>``.

        Runs beside the class-type assignment rather than inside it: the two
        dimensions are independent, and ``handler = self.process`` produces a
        callable fact and no type fact at all.
        """
        callables = self._infer_callable_ids_from_value(value)
        elements = self._infer_container_callable_ids(value)
        if not callables and not elements:
            return

        if isinstance(target, ast.Name):
            self.types.add_var_callables(target.id, callables)
            self.types.add_var_callables(f"{target.id}{_ELEMENT_SUFFIX}", elements)
            return

        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
        ):
            class_id = self.current_class_id()
            if class_id:
                key = (class_id, target.attr)
                self.class_attr_types.add_callable_types(key, callables)
                self.class_attr_types.add_element_callable_types(key, elements)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # The annotation is a fact in its own right, so ``self.grid: Grid`` is
        # useful even with no assigned value -- which is exactly the form a
        # class-level attribute declaration takes.
        annotated_objects, annotated_elements = self._annotation_types(node.annotation)
        class_types = set(annotated_objects)
        container_types = set(annotated_elements)
        if node.value is not None:
            class_types |= self._infer_class_types_from_value(node.value)
            container_types |= self._infer_container_element_types(node.value)

        if class_types or container_types:
            # ``engine: Engine`` written directly in a class body declares an
            # instance attribute, not a local. Its target is a bare Name, so
            # without this it would be filed as a variable in the class scope and
            # never seen by ``self.engine`` lookups in the methods below it.
            if (
                isinstance(node.target, ast.Name)
                and isinstance(getattr(node, "parent", None), ast.ClassDef)
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    key = (class_id, node.target.id)
                    self.class_attr_types.add_object_types(key, class_types)
                    self.class_attr_types.add_element_types(key, container_types)
            else:
                self._assign_target_types(
                    node.target, class_types, container_types, node.value or node
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            self.add_container_types(
                node.target.id, self._infer_container_element_types(node.value)
            )
        method_name = AUGASSIGN_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.target, method_name, "dunder_operator", node.lineno, node.col_offset
            )
        self.visit(node.target)
        self.visit(node.value)

    def _visit_for_like(self, node: ast.For | ast.AsyncFor) -> None:
        """Model iteration and propagate known collection element types."""
        self._add_dunder_edges(node.iter, "__iter__", "dunder_iter", node.lineno, node.col_offset)
        self.visit(node.iter)
        element_origins = self._element_origins(node.iter)
        if isinstance(node.target, ast.Name):
            element_types = self._infer_container_element_types(node.iter)
            if element_types:
                self.set_var_types(node.target.id, element_types)
            self.set_var_origins(node.target.id, element_origins)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            # ``for name, proc in registry.items()``: each loop variable takes
            # one slot of the element, so bind them position by position.
            for index, item in enumerate(node.target.elts):
                if not isinstance(item, ast.Name):
                    continue
                slot_types = self._infer_sequence_slot_class_types(node.iter, index)
                if slot_types:
                    self.set_var_types(item.id, slot_types)
                # Destructuring a mapping's items gives keys at slot 0 and the
                # stored values at slot 1. Only the values are the registry's
                # elements; the keys are the labels they were filed under.
                self.set_var_origins(
                    item.id,
                    element_origins
                    if index == 1 and self._is_dict_items_call(node.iter)
                    else set(),
                )
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for_like(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for_like(node)

    def _container_mutation_element_types(self, node: ast.Call) -> Set[str]:
        """Element types a mutating container call introduces, if any.

        ``append``/``add`` contribute the argument's own type; ``extend``/
        ``update`` contribute the argument's *elements*, since they merge one
        collection into another. ``update`` is what makes dict-style registries
        work -- ``self.children.update({name: child})`` is the idiom frameworks
        use to register something under a key.
        """
        if not node.args:
            return set()
        if node.func.attr in {"append", "add"}:  # type: ignore[union-attr]
            return self._infer_class_types_from_value(node.args[0])
        if node.func.attr in {"extend", "update"}:  # type: ignore[union-attr]
            return self._infer_container_element_types(node.args[0])
        return set()

    def _container_mutation_value_origins(self, node: ast.Call) -> Set[Origin]:
        """Origins a mutating container call stores, mirroring the type rule.

        The same append/extend split applies: ``append`` stores the argument
        itself, ``update`` stores the argument's contents. Keeping the two rules
        in step is what makes ``self.children.update({name: child})`` register
        ``child`` rather than the dict wrapped around it.
        """
        if not node.args or not isinstance(node.func, ast.Attribute):
            return set()
        if node.func.attr in {"append", "add"}:
            return self._value_origins(node.args[0])
        if node.func.attr in {"extend", "update"}:
            return self._element_origins(node.args[0])
        return set()

    def _container_mutation_key(self, node: ast.Call) -> Optional[ast.AST]:
        """The key a mutating call filed its value under, when there is one.

        Only ``update`` with a single-entry dict literal names a key -- which is
        the registration idiom, ``self.children.update({name: child})``. Appending
        to a list files nothing under a name, and a multi-entry literal would make
        the pairing ambiguous.
        """
        if not node.args or not isinstance(node.func, ast.Attribute):
            return None
        if node.func.attr != "update":
            return None
        argument = unwrap_passthrough(node.args[0])
        if isinstance(argument, ast.Dict) and len(argument.keys) == 1:
            return argument.keys[0]
        return None

    def _record_container_mutation(self, node: ast.Call) -> None:
        """Learn element types introduced by mutating container calls.

        Handles both a local (``items.append(x)``) and an attribute container
        (``self.items.append(x)``). The attribute case is the one that crosses
        method boundaries, so it is what lets a registry populated in one method
        be resolved where another method iterates it.
        """
        if not isinstance(node.func, ast.Attribute):
            return

        container = node.func.value
        # ``self.buckets[key].append(x)`` -- a container nested one level inside
        # an attribute. The nesting level is deliberately flattened away: what
        # matters for resolving a later ``for item in self.buckets[k]`` is which
        # classes ended up somewhere under ``self.buckets``, not the shape of the
        # boxes they sit in. climlab's process tree needs exactly this, since it
        # bins subprocesses into ``self.process_types[time_type]`` lists.
        if isinstance(container, ast.Subscript):
            container = container.value

        # Recorded before the type facts and independently of them. Which value
        # was stored is knowable on the first pass; what type it has may take
        # several, and gating the escape on a type would lose the fact entirely
        # for a parameter whose callers are not yet resolved.
        if isinstance(container, ast.Attribute):
            self._note_attribute_store(
                container,
                self._container_mutation_value_origins(node),
                is_container=True,
                key=self._container_mutation_key(node),
            )

        element_types = self._container_mutation_element_types(node)
        if not element_types:
            return

        if isinstance(container, ast.Name):
            self.add_container_types(container.id, element_types)
        elif isinstance(container, ast.Attribute):
            self.record_attribute_container_store(container, element_types)

    def _replay_var_sources(self, name: str, *, elements: bool) -> Set[str]:
        """Re-infer what ``name`` was assigned from, under the current link target.

        ``x = build(); return x`` states the same fact as ``return build()``.
        But the call sits in an assignment, where no return-link target is in
        force, so its dependency went unrecorded and that chain fell back to
        advancing one hop per pass -- capped, and the very thing return links
        exist to avoid. Replaying the assigned expression here, inside the
        enclosing ``_recording_return_links`` block, records the link the
        assignment could not.

        Gated on an active link target: anywhere else this would be duplicated
        work, since whatever it can infer is already in the type environment.
        """
        if not self._link_target or name in self.types.replaying:
            return set()
        self.types.replaying.add(name)
        try:
            types: Set[str] = set()
            for source, slot in self.get_var_sources(name):
                if slot is not None:
                    if elements:
                        types.update(
                            self._infer_sequence_slot_container_types(source, slot)
                        )
                    else:
                        types.update(
                            self._infer_sequence_slot_class_types(source, slot)
                        )
                elif elements:
                    types.update(self._infer_container_element_types(source))
                else:
                    types.update(self._infer_class_types_from_value(source))
            return types
        finally:
            self.types.replaying.discard(name)

    def _infer_callable_ids_from_value(
        self, value: ast.AST, *, resolve_names: bool = True
    ) -> Set[str]:
        """Callables an expression may evaluate to, mirroring class inference.

        The counterpart of ``_infer_class_types_from_value`` for values that
        *are* code. Written as a separate walk rather than an extra branch there
        because the two answers must never mix: a class id and a callable id are
        both dotted strings, and every consumer of a type set assumes the former.

        ``resolve_names`` distinguishes the two contexts this is asked from. As a
        *value* -- ``handler = helper``, or ``f(helper)`` -- a bare name should
        resolve to the function it refers to. As a *callee* it must not: ordinary
        name resolution already handles ``helper()`` and gives it the more
        specific relation, so resolving it here too would relabel every plain
        call as ``inferred_callable``.
        """
        value = unwrap_passthrough(value)

        if isinstance(value, ast.Name):
            known = self.types.get_var_callables(value.id)
            if known:
                return set(known)
            if not resolve_names:
                return set()
            resolved = self._resolve_callee_definition(value)
            return {resolved} if resolved else set()

        if isinstance(value, ast.Attribute):
            callables = self._class_attr_callable_types(value)
            if callables:
                return callables
            if not resolve_names:
                return set()
            resolved = self._resolve_callee_definition(value)
            return {resolved} if resolved else set()

        if isinstance(value, ast.Lambda):
            lambda_id = self._lambda_id(value)
            return {lambda_id} if lambda_id in self.callable_ids else set()

        if isinstance(value, ast.Call):
            return self._infer_callable_ids_from_call(value)

        # Same recursion as the class walk: these evaluate to one of their
        # sub-expressions, and the answer is knowable for each.
        if isinstance(value, ast.IfExp):
            return self._infer_callable_ids_from_value(
                value.body
            ) | self._infer_callable_ids_from_value(value.orelse)
        if isinstance(value, ast.BoolOp):
            callables: Set[str] = set()
            for operand in value.values:
                callables.update(self._infer_callable_ids_from_value(operand))
            return callables

        # ``SOLVERS[name]()`` -- a dispatch table, which is the dominant
        # config-driven dispatch idiom in scientific code.
        if isinstance(value, ast.Subscript) and not isinstance(value.slice, ast.Slice):
            return self._infer_container_callable_ids(value.value)

        return set()

    def _infer_callable_ids_from_call(self, value: ast.Call) -> Set[str]:
        """Callables produced *by* a call: ``partial(f, x)`` and factories."""
        fn_name = attribute_to_name(value.func)
        if fn_name in _PARTIAL_NAMES and value.args:
            # ``partial(f, ...)`` is ``f`` with arguments pre-bound; calling the
            # result calls ``f``.
            return self._infer_callable_ids_from_value(value.args[0])

        callables: Set[str] = set()
        for callee, _relation, is_resolved in self._resolve_callees(value.func):
            if not is_resolved:
                continue
            summary = self.return_summaries.get(callee)
            if summary:
                callables.update(summary.callable_ids)
        return callables

    def _infer_container_callable_ids(self, value: ast.AST) -> Set[str]:
        """Callables held *inside* a collection, for dispatch tables."""
        value = unwrap_passthrough(value)
        if isinstance(value, ast.Name):
            return set(self.types.get_var_callables(f"{value.id}{_ELEMENT_SUFFIX}"))
        if isinstance(value, ast.Attribute):
            return self._class_attr_element_callable_types(value)
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            callables: Set[str] = set()
            for item in value.elts:
                callables.update(self._infer_callable_ids_from_value(item))
            return callables
        if isinstance(value, ast.Dict):
            callables = set()
            for item in value.values:
                callables.update(self._infer_callable_ids_from_value(item))
            return callables
        return set()

    def _resolve_callee_definition(self, value: ast.AST) -> Optional[str]:
        """Resolve a name/attribute to a project callable, or ``None``.

        A *reference* to a function, not a call of it. Only fully resolved
        project callables count -- an unresolved guess as a value would be a
        fabricated callable id rather than a fabricated callee, which is worse.
        """
        resolved = self._resolve_callee(value)
        if resolved and resolved[2] and resolved[0] in self.callable_ids:
            return resolved[0]
        return None

    def _lambda_id(self, node: ast.Lambda) -> str:
        """CPython's own name for a lambda's code object.

        Spelled ``<enclosing>.<locals>.<lambda>`` to match ``co_qualname``, so
        the runtime tracer's IDs keep agreeing with the static ones by
        construction. CPython does not disambiguate sibling lambdas in one
        scope and neither does this: inventing a disambiguator would buy
        precision at the cost of that agreement.
        """
        if self.enclosing_function:
            return f"{self.enclosing_function}.<locals>.<lambda>"
        if self.current_class:
            return f"{self.module}.{self.current_class}.<locals>.<lambda>"
        return f"{self.module}.<lambda>"

    def _class_attr_callable_types(self, value: ast.Attribute) -> Set[str]:
        """Callables stored on ``self.<attr>``, looked up across the hierarchy."""
        return self._attr_callables(value, self.class_attr_types.callable_types)

    def _class_attr_element_callable_types(self, value: ast.AST) -> Set[str]:
        if not isinstance(value, ast.Attribute):
            return set()
        return self._attr_callables(value, self.class_attr_types.element_callable_types)

    def _attr_callables(
        self, value: ast.Attribute, table: Dict[ClassAttr, Set[str]]
    ) -> Set[str]:
        if not (isinstance(value.value, ast.Name) and value.value.id in {"self", "cls"}):
            return set()
        class_id = self.current_class_id()
        if not class_id:
            return set()
        found: Set[str] = set()
        for ancestor in self.project_index.class_and_ancestors(class_id):
            found.update(table.get((ancestor, value.attr), set()))
        return found

    def _infer_class_types_from_value(self, value: ast.AST) -> Set[str]:
        value = unwrap_passthrough(value)
        if isinstance(value, ast.Call):
            return self._infer_class_types_from_call(value)
        if isinstance(value, ast.Name):
            return self.get_expr_types(value) | self._replay_var_sources(
                value.id, elements=False
            )
        if isinstance(value, ast.Attribute):
            return self.get_expr_types(value)
        # Expressions that evaluate to one of several sub-expressions. Recursing
        # rather than giving up matters twice over: the types are knowable, and
        # any call reached below still sees the enclosing return-link target, so
        # ``return a if flag else make_order()`` records its deferred link.
        if isinstance(value, ast.IfExp):
            return self._infer_class_types_from_value(
                value.body
            ) | self._infer_class_types_from_value(value.orelse)
        if isinstance(value, ast.BoolOp):
            class_types: Set[str] = set()
            for operand in value.values:
                class_types.update(self._infer_class_types_from_value(operand))
            return class_types
        # ``orders[0]`` produces one element of ``orders``; a slice does not, so
        # it is left to the element-type inference below.
        if isinstance(value, ast.Subscript) and not isinstance(value.slice, ast.Slice):
            return self._infer_container_element_types(value.value)
        return set()

    def _infer_class_types_from_call(self, value: ast.Call) -> Set[str]:
        """Infer objects constructed or returned by a call expression."""
        if self._is_copy_call(value) and value.args:
            return self._infer_class_types_from_value(value.args[0])

        fn_name = attribute_to_name(value.func)
        if fn_name is None:
            return set()

        class_matches = self._resolve_class_reference_name(fn_name)
        if class_matches:
            return set(class_matches)

        class_types: Set[str] = set()
        for callee, _relation, is_resolved in self._resolve_callees(value.func):
            if not is_resolved:
                continue
            self._note_return_dependency(callee, "class_types")
            summary = self.return_summaries.get(callee)
            if summary:
                class_types.update(summary.class_types)
        return class_types

    def _infer_container_element_types(self, value: ast.AST) -> Set[str]:
        """Infer possible element classes for common collection expressions."""
        value = unwrap_passthrough(value)
        if isinstance(value, ast.Name):
            return self.get_container_types(value.id) | self._replay_var_sources(
                value.id, elements=True
            )
        if isinstance(value, ast.Attribute):
            return (
                self._infer_property_return_element_types(value)
                | self._class_attr_element_types(value)
            )
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            element_types: Set[str] = set()
            for item in value.elts:
                element_types.update(self._infer_class_types_from_value(item))
            return element_types
        # A dict's elements are its values: ``{"albedo": P2Albedo()}`` is a
        # registry of albedo objects, and the keys are just labels. This is the
        # dominant config-driven dispatch idiom in scientific code.
        if isinstance(value, ast.Dict):
            element_types: Set[str] = set()
            for item in value.values:
                element_types.update(self._infer_class_types_from_value(item))
            return element_types
        if isinstance(value, ast.DictComp):
            return self._infer_class_types_from_value(value.value)
        if isinstance(value, (ast.ListComp, ast.SetComp)):
            return self._infer_class_types_from_value(value.elt)
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            return self._infer_container_element_types(
                value.left
            ) | self._infer_container_element_types(value.right)
        if isinstance(value, ast.IfExp):
            return self._infer_container_element_types(
                value.body
            ) | self._infer_container_element_types(value.orelse)
        if isinstance(value, ast.BoolOp):
            element_types: Set[str] = set()
            for operand in value.values:
                element_types.update(self._infer_container_element_types(operand))
            return element_types
        # Slicing a collection yields a collection holding the same elements.
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
            return self._infer_container_element_types(value.value)
        # Indexing an attribute that holds nested containers, as in
        # ``self.buckets[key]``, yields the inner container. Element facts are
        # recorded flattened by ``_record_container_mutation``, so they are read
        # back the same way rather than expecting a level of nesting that was
        # never stored.
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute):
            return self._class_attr_element_types(value.value)
        if isinstance(value, ast.Call):
            if self._is_copy_call(value) and value.args:
                return self._infer_container_element_types(value.args[0])
            if (
                isinstance(value.func, ast.Name)
                and value.func.id in {"list", "set", "tuple"}
            ):
                if value.args:
                    return self._infer_container_element_types(value.args[0])
                return set()
            # ``registry.values()`` yields what the registry holds. Dict-backed
            # registries are almost always walked this way rather than iterated
            # directly, so without this the element facts are never read back.
            if (
                isinstance(value.func, ast.Attribute)
                and value.func.attr == "values"
                and not value.args
            ):
                return self._infer_container_element_types(value.func.value)

            element_types: Set[str] = set()
            for callee, _relation, is_resolved in self._resolve_callees(value.func):
                if not is_resolved:
                    continue
                self._note_return_dependency(callee, "element_types")
                summary = self.return_summaries.get(callee)
                if summary:
                    element_types.update(summary.element_types)
            return element_types
        return set()

    def _is_dict_items_call(self, value: ast.AST) -> bool:
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "items"
            and not value.args
        )

    def _infer_sequence_slot_class_types(self, value: ast.AST, index: int) -> Set[str]:
        value = unwrap_passthrough(value)
        if isinstance(value, (ast.Tuple, ast.List)) and index < len(value.elts):
            return self._infer_class_types_from_value(value.elts[index])
        # ``for name, proc in self.subprocess.items()`` destructures a key/value
        # pair, so slot 1 is an element of the container. This is how frameworks
        # walk a registry when they need the key as well -- climlab's process
        # tree is traversed exactly this way.
        if self._is_dict_items_call(value) and index == 1:
            return self._infer_container_element_types(value.func.value)  # type: ignore[union-attr]
        if isinstance(value, ast.Call):
            slot_types: Set[str] = set()
            for callee, _relation, is_resolved in self._resolve_callees(value.func):
                if not is_resolved:
                    continue
                self._note_return_dependency(callee, "slot_types", source_slot=index)
                summary = self.return_summaries.get(callee)
                if summary:
                    slot_types.update(summary.slot_types.get(index, set()))
            return slot_types
        return set()

    def _infer_sequence_slot_container_types(
        self, value: ast.AST, index: int
    ) -> Set[str]:
        value = unwrap_passthrough(value)
        if isinstance(value, (ast.Tuple, ast.List)) and index < len(value.elts):
            return self._infer_container_element_types(value.elts[index])
        if isinstance(value, ast.Call):
            slot_types: Set[str] = set()
            for callee, _relation, is_resolved in self._resolve_callees(value.func):
                if not is_resolved:
                    continue
                self._note_return_dependency(
                    callee, "slot_element_types", source_slot=index
                )
                summary = self.return_summaries.get(callee)
                if summary:
                    slot_types.update(summary.slot_element_types.get(index, set()))
            return slot_types
        return set()

    def _is_copy_call(self, node: ast.Call) -> bool:
        fn_name = attribute_to_name(node.func)
        return fn_name in {"copy", "deepcopy", "copy.copy", "copy.deepcopy"}

    def _infer_class_from_call(self, value: ast.AST) -> Optional[str]:
        if not isinstance(value, ast.Call):
            return None
        inferred = sorted(self._infer_class_types_from_call(value))
        if inferred:
            return inferred[0]
        return None

    def _note_call_health(
        self, node: ast.Call, results: Sequence[Tuple[str, str, bool]]
    ) -> None:
        """Record what this call site cost the resolver, whatever the outcome."""
        if self.current_callable is None:
            return

        resolved = [result for result in results if result[2]]
        if resolved:
            self.health.site_fanout[len(resolved)] += 1
            return

        # Calls to a value the abstract domain cannot hold. Two shapes reach
        # here: a bound local whose contents are a function (``callable_value``),
        # and an expression the resolver produced no candidate for at all. Both
        # are the same underlying gap -- the lattice is a set of class ids, so a
        # value that *is* code is inexpressible rather than merely unknown.
        if any(relation == "callable_value" for _callee, relation, _ in results):
            self.health.unresolvable_calls[_callee_shape(node.func)] += 1
            return

        if results:
            # A named but unmatched callee. Already counted by ``_add_edge``.
            return

        self.health.unresolvable_calls[_callee_shape(node.func)] += 1

    def visit_Call(self, node: ast.Call) -> None:
        """Resolve an explicit call, record its edge, then visit child expressions."""
        self._record_container_mutation(node)
        self._note_receiver_invocation(node.func)

        dynamic_resolved = self._resolve_dynamic_getattr_callees(node.func)
        resolved_callees = dynamic_resolved or self._resolve_callees(node.func)
        self._note_call_health(node, resolved_callees)
        confidence = confidence_for_fanout(
            sum(1 for result in resolved_callees if result[2])
        )
        for callee, relation, is_resolved in resolved_callees:
            self._add_edge(
                callee,
                relation,
                is_resolved,
                node.lineno,
                node.col_offset,
                confidence,
            )
        self._add_registration_edges(node)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Turn a loaded property access into its implicit getter call."""
        if isinstance(node.ctx, ast.Load):
            targets = self._resolve_property_getter_targets(node)
            confidence = confidence_for_fanout(len(targets))
            for callee in targets:
                self._add_edge(
                    callee,
                    "property_getter",
                    True,
                    node.lineno,
                    node.col_offset,
                    confidence,
                )
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Model ``obj[key]`` load/store/delete through the relevant dunder method."""
        if isinstance(node.ctx, ast.Store):
            self._add_dunder_edges(
                node.value, "__setitem__", "dunder_setitem", node.lineno, node.col_offset
            )
        elif isinstance(node.ctx, ast.Del):
            self._add_dunder_edges(
                node.value, "__delitem__", "dunder_delitem", node.lineno, node.col_offset
            )
        else:
            self._add_dunder_edges(
                node.value, "__getitem__", "dunder_getitem", node.lineno, node.col_offset
            )
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        method_name = BINOP_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.left, method_name, "dunder_operator", node.lineno, node.col_offset
            )
        reverse_method_name = REVERSE_BINOP_DUNDER_METHODS.get(type(node.op))
        if reverse_method_name:
            self._add_dunder_edges(
                node.right, reverse_method_name, "dunder_operator", node.lineno, node.col_offset
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        self.visit(left)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                self._add_membership_edges(comparator, node.lineno, node.col_offset)
            else:
                method_name = COMPARE_DUNDER_METHODS.get(type(op))
                if method_name:
                    self._add_dunder_edges(
                        left, method_name, "dunder_operator", node.lineno, node.col_offset
                    )
            self.visit(comparator)
            left = comparator

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        method_name = UNARY_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.operand, method_name, "dunder_operator", node.lineno, node.col_offset
            )
        self.visit(node.operand)

    def _resolve_dynamic_getattr_callees(
        self, func: ast.AST
    ) -> List[Tuple[str, str, bool]]:
        """Resolve calls like getattr(module_alias, dynamic_name)(...) conservatively."""
        if not isinstance(func, ast.Call):
            return []
        if not isinstance(func.func, ast.Name) or func.func.id != "getattr":
            return []
        if len(func.args) < 2:
            return []

        target_obj, attr_arg = func.args[0], func.args[1]
        if not isinstance(target_obj, ast.Name):
            return []

        imported_base = self.module_index.imports.get(target_obj.id)
        if not imported_base:
            return []

        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
            target = f"{imported_base}.{attr_arg.value}"
            return [(target, "dynamic_getattr", target in self.callable_ids)]

        prefix = f"{imported_base}."
        matches = sorted(
            callable_id
            for callable_id in self.callable_ids
            if callable_id.startswith(prefix)
        )
        return [(callable_id, "dynamic_getattr", True) for callable_id in matches]

    def _resolve_name_from_module_exports(
        self, module_name: str, name: str, seen: Optional[Set[str]] = None
    ) -> List[str]:
        return self.project_index.resolve_name_from_module_exports(
            module_name, name, seen
        )

    def _resolve_star_import_targets(self, name: str) -> List[str]:
        return self.project_index.resolve_star_import_targets(self.module_index, name)

    def _resolve_callees(self, func: ast.AST) -> List[Tuple[str, str, bool]]:
        """Resolve a call target, allowing multiple method candidates.

        Attribute calls are checked from most informative to least informative:
        ``super()``, ``self``/``cls``, inferred receiver type, explicit class,
        then general name/import resolution.
        """
        if isinstance(func, ast.Attribute):
            full = attribute_to_name(func)
            if not full:
                return []

            if self._is_super_call(func.value):
                super_class_id = self._super_class_id(func.value)
                targets = self.project_index.resolve_super_method_targets(
                    super_class_id, func.attr
                )
                cooperative = self.project_index.resolve_cooperative_super_targets(
                    super_class_id, func.attr
                )
                if targets or cooperative:
                    return [
                        *((target, "super_method", True) for target in targets),
                        *(
                            (target, "cooperative_super", True)
                            for target in cooperative
                        ),
                    ]

            if (
                isinstance(func.value, ast.Name)
                and func.value.id in {"self", "cls"}
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    targets = self._resolve_method_targets(class_id, func.attr)
                    # ``self`` is not necessarily an instance of the class the
                    # code is written in, so a subclass override is an equally
                    # real target -- and usually the one that runs. This holds
                    # whether or not the base supplies its own definition, so the
                    # override edges are added alongside rather than as a
                    # fallback: on climlab the base defines the hook in most
                    # cases (``_Insolation._compute_fixed``,
                    # ``GreyGas._compute_emission``), which a fallback-only rule
                    # would never see past.
                    overrides = [
                        target
                        for target in self._resolve_subclass_override_targets(class_id, func.attr)
                        if target not in targets
                    ]
                    if targets or overrides:
                        return [
                            *((target, "self_method", True) for target in targets),
                            *((target, "virtual_override", True) for target in overrides),
                        ]
                    # No method of that name anywhere in the hierarchy. Before
                    # giving up, ask whether the attribute holds a *callable*:
                    # ``self.handler = on_event`` then ``self.handler()`` is a
                    # stored callback, not a method, and the class-id domain had
                    # no way to represent it.
                    stored = sorted(
                        target
                        for target in self._class_attr_callable_types(func)
                        if target in self.callable_ids
                    )
                    if stored:
                        return [
                            (target, "inferred_callable", True) for target in stored
                        ]
                    return [(f"{class_id}.{func.attr}", "self_method", False)]

            if isinstance(func.value, ast.Name):
                var_types = self.get_var_types(func.value.id)
                if var_types:
                    targets: List[str] = []
                    for var_type in sorted(var_types):
                        targets.extend(self._resolve_method_targets(var_type, func.attr))
                    if targets:
                        return [
                            (target, "inferred_type", True)
                            for target in self._unique(targets)
                        ]
                    first_type = sorted(var_types)[0]
                    return [(f"{first_type}.{func.attr}", "inferred_type", False)]

            # Not just names and attributes: a receiver can be any expression
            # whose type we can infer, including another call. ``get_expr_types``
            # alone left ``build().submit()`` unresolved.
            with self._resolving_receiver():
                receiver_types = self._infer_class_types_from_value(func.value)
            if receiver_types:
                targets = []
                for receiver_type in sorted(receiver_types):
                    targets.extend(
                        self._resolve_method_targets(receiver_type, func.attr)
                    )
                if targets:
                    return [
                        (target, "inferred_type", True)
                        for target in self._unique(targets)
                    ]
                first_type = sorted(receiver_types)[0]
                return [(f"{first_type}.{func.attr}", "inferred_type", False)]

            receiver_name = attribute_to_name(func.value)
            if receiver_name:
                class_ids = self._resolve_class_reference_name(receiver_name)
                if class_ids:
                    targets = []
                    for class_id in class_ids:
                        targets.extend(self._resolve_method_targets(class_id, func.attr))
                    if targets:
                        return [
                            (target, "class_method", True)
                            for target in self._unique(targets)
                        ]

        # ``cls(...)`` inside a classmethod constructs an instance of the class
        # the method belongs to. Resolved here rather than left to the bare-name
        # fallback, which cannot see that ``cls`` is bound at all and reported it
        # as a call to an unresolvable value.
        if isinstance(func, ast.Name) and func.id == "cls" and self.current_class:
            class_id = self.current_class_id()
            if class_id:
                targets = self._resolve_constructor_targets(class_id)
                if targets:
                    return [(target, "constructor", True) for target in targets]

        # The callable-value rung. Placed after every class-based rule, so a
        # receiver whose type is known still resolves as a method call, and
        # before the single-target fallback, which can only guess a bare name.
        callable_targets = sorted(
            target
            for target in self._infer_callable_ids_from_value(func, resolve_names=False)
            if target in self.callable_ids
        )
        if callable_targets:
            return [(target, "inferred_callable", True) for target in callable_targets]

        # ``model(x)`` where ``model`` is an instance of a class defining
        # ``__call__``. Cheap, and independent of everything above: the receiver
        # has an ordinary class type, it is the *call* that is implicit.
        dunder_call_targets: List[str] = []
        for receiver_type in sorted(self._callee_expression_types(func)):
            dunder_call_targets.extend(
                self._resolve_method_targets(receiver_type, "__call__")
            )
        if dunder_call_targets:
            return [
                (target, "dunder_call", True)
                for target in self._unique(dunder_call_targets)
            ]

        resolved = self._resolve_callee(func)
        if resolved:
            return [resolved]
        return []

    def _callee_expression_types(self, func: ast.AST) -> Set[str]:
        """Class types of the thing being called, for the ``__call__`` rung.

        ``cls`` is excluded. It is seeded with its class id like ``self``, but it
        denotes the *class object*, so ``cls(...)`` constructs an instance and
        does not invoke ``__call__`` on one. Treating the two alike turns every
        alternative constructor in a class that also defines ``__call__`` into an
        edge to the wrong method.
        """
        if isinstance(func, ast.Name):
            if func.id == "cls":
                return set()
            return self.get_var_types(func.id)
        if isinstance(func, ast.Attribute):
            with self._resolving_receiver():
                return self._infer_class_types_from_value(func)
        return set()

    def _super_class_id(self, node: ast.AST) -> Optional[str]:
        """Which class a ``super()`` call starts its MRO search after.

        Usually the enclosing class, but ``super(Other, self)`` names it
        explicitly and means something different. That distinction was harmless
        while resolution unioned over bases; under C3 the starting point decides
        the answer, so the argument has to be read.
        """
        if isinstance(node, ast.Call) and node.args:
            named = self._resolve_class_reference_name(attribute_to_name(node.args[0]) or "")
            if named:
                return named[0]
        return self.current_class_id()

    def _is_super_call(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "super"
        )

    def _resolve_callee(self, func: ast.AST) -> Optional[Tuple[str, str, bool]]:
        """Resolve a simple name or dotted attribute to one best target.

        The returned tuple is ``(callee_id, relation, resolved)``. Keeping a
        useful unresolved ID lets callers optionally expose external/dynamic
        dependencies instead of silently discarding them.
        """
        if isinstance(func, ast.Name):
            name = func.id
            enclosing_local = self._resolve_enclosing_local_callable(name)
            if enclosing_local:
                return (enclosing_local, "direct", True)

            local = f"{self.module}.{name}"
            if local in self.callable_ids:
                return (local, "direct", True)

            class_matches = self._resolve_class_reference_name(name)
            if class_matches:
                constructor_targets: List[str] = []
                for class_id in class_matches:
                    constructor_targets.extend(
                        self._resolve_constructor_targets(class_id)
                    )
                if len(constructor_targets) == 1:
                    return (constructor_targets[0], "constructor", True)
                if len(constructor_targets) > 1:
                    return (self._unique(constructor_targets)[0], "constructor", False)

            imported = self.module_index.imports.get(name)
            if imported:
                # Through the alias map first: a callable imported from a package
                # is recorded under the package path, not the defining module.
                canonical = self.project_index.canonical_callable_id(imported)
                is_known = canonical in self.callable_ids
                return (canonical if is_known else imported, "imported", is_known)

            star_import_targets = self._resolve_star_import_targets(name)
            star_import_matches = [
                target for target in star_import_targets if target in self.callable_ids
            ]
            if len(star_import_matches) == 1:
                return (star_import_matches[0], "imported", True)
            if len(star_import_matches) > 1:
                return (star_import_matches[0], "imported", False)

            star_import_class_matches = []
            for candidate_class in star_import_targets:
                if self._is_known_class(candidate_class):
                    class_id = self._known_class_id(candidate_class)
                    star_import_class_matches.extend(
                        self._resolve_constructor_targets(class_id)
                    )
            if len(star_import_class_matches) == 1:
                return (star_import_class_matches[0], "constructor", True)
            if len(star_import_class_matches) > 1:
                return (
                    self._unique(star_import_class_matches)[0],
                    "constructor",
                    False,
                )

            if self.types.is_bound_local(name):
                # The name resolved; what it holds is the problem. ``handler(x)``
                # where ``handler`` is a parameter or a local is a call to a
                # value, and the abstract domain is a set of *class* ids with no
                # way to say "this variable holds that function". Distinguished
                # from an ordinary unresolved name so the health report can
                # count the gap instead of burying it among third-party calls.
                return (name, "callable_value", False)
            return (name, "direct", False)

        if isinstance(func, ast.Attribute):
            full = attribute_to_name(func)
            if not full:
                return None

            # variable type inferred method call: obj.method()
            if isinstance(func.value, ast.Name):
                var_type = self.get_var_type(func.value.id)
                if var_type:
                    inferred_targets = self._resolve_method_targets(var_type, func.attr)
                    if inferred_targets:
                        return (inferred_targets[0], "inferred_type", True)
                    return (f"{var_type}.{func.attr}", "inferred_type", False)

                # imported module alias call: alias.fn()
                imported_base = self.module_index.imports.get(func.value.id)
                if imported_base:
                    target = f"{imported_base}.{func.attr}"
                    if self._is_known_class(target):
                        class_id = self._known_class_id(target)
                        init_targets = self._resolve_constructor_targets(class_id)
                        if init_targets:
                            return (init_targets[0], "constructor", True)
                    is_known = target in self.callable_ids
                    return (target, "imported", is_known)

            # nested imported alias: alias.sub.fn()
            if "." in full:
                base, attr = full.split(".", 1)
                imported_base = self.module_index.imports.get(base)
                if imported_base:
                    target = f"{imported_base}.{attr}"
                    if self._is_known_class(target):
                        class_id = self._known_class_id(target)
                        init_targets = self._resolve_constructor_targets(class_id)
                        if init_targets:
                            return (init_targets[0], "constructor", True)
                    is_known = target in self.callable_ids
                    return (target, "imported", is_known)

            # maybe fully qualified local reference
            is_known = full in self.callable_ids
            return (full, "attribute", is_known)

        return None
