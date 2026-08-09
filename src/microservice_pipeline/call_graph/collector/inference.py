"""Inferring class types, and the element types inside containers.

The deliberately small-scale type inference the resolver runs on: what class an
expression evaluates to, what a container holds, and what sits in a given tuple
slot. Container elements are tracked because ``for step in self.steps: step.run()``
is unresolvable without them, and that loop is how most pipelines are written.

This is one half of the cycle the package cannot break by moving code:
``_infer_class_types_from_call`` asks ``resolution`` what a call resolves to, and
``resolution`` asks here what the receiver's type is. Resolving
``build().submit()`` needs the inferred type of ``build()``, and inferring that
type needs to resolve ``build``. They are separate files but one object; see
``modularisation_plan.md``.

Requires from siblings: ``_resolve_callees``, ``get_expr_types``,
``_class_attr_element_types``, ``_attribute_owner_types``,
``_resolve_class_reference_name``, ``get_var_sources``.
"""

from __future__ import annotations

import ast
from typing import Set
from ..ast_utils import attribute_to_name, unwrap_passthrough
from .shapes import is_copy_call, is_dict_items_call

from .state import CollectorState


class InferenceMixin(CollectorState):
    """Inferring class types, and the element types inside containers."""

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
            for source, slot in self.types.get_var_sources(name):
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
        if is_copy_call(value) and value.args:
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
            return self.types.get_container_types(value.id) | self._replay_var_sources(
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
            if is_copy_call(value) and value.args:
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

    def _infer_sequence_slot_class_types(self, value: ast.AST, index: int) -> Set[str]:
        value = unwrap_passthrough(value)
        if isinstance(value, (ast.Tuple, ast.List)) and index < len(value.elts):
            return self._infer_class_types_from_value(value.elts[index])
        # ``for name, proc in self.subprocess.items()`` destructures a key/value
        # pair, so slot 1 is an element of the container. This is how frameworks
        # walk a registry when they need the key as well -- climlab's process
        # tree is traversed exactly this way.
        if is_dict_items_call(value) and index == 1:
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
