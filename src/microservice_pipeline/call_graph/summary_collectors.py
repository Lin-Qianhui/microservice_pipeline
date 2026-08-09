"""The two edge-free analysis passes.

Both subclass ``CallCollector`` to reuse its scope handling and expression
inference, and both override ``visit_Call`` so that visiting a call maintains
type facts without emitting a graph edge. What each one records instead is the
difference between them:

``ReturnSummaryCollector``
    What every callable gives back -- object types, element types, and the shape
    of a returned tuple.
``TypeSummaryCollector``
    What flows *into* parameters and instance attributes at every call site, and
    -- through ``records_registry_facts`` -- which parameters are retained on
    ``self`` and which retained values are later invoked. Those last two are what
    ``registration`` joins into registry coupling.

Both are driven as fixed points by ``passes``; see the module docstring there for
why a single pass over the files is not enough.
"""


from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Optional, Set

from .ast_utils import unwrap_passthrough
from .collector import CallCollector
from .models import (
    KEYWORD_ONLY_POSITION,
    AttrSlot,
    ClassAttr,
    ClassAttrTypes,
    FunctionEscapeSummary,
    FunctionParamSummary,
    FunctionReturnSummary,
    ModuleIndex,
    RegistryFacts,
    add_indexed_types,
    add_types,
)
from .type_env import Origin
from .project_index import ProjectIndex
from .return_links import ReturnLink, ReturnLinkTable


class ReturnSummaryCollector(CallCollector):
    """Analysis pass that records the types produced by each callable.

    It inherits scope and expression inference from ``CallCollector`` but does
    not emit call edges; calls are visited only to maintain local type facts.
    """
    def __init__(
        self,
        module: str,
        file: Path,
        module_index: ModuleIndex,
        callable_ids: Set[str],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
        return_links: Optional[ReturnLinkTable] = None,
        project_index: Optional[ProjectIndex] = None,
    ):
        super().__init__(
            module=module,
            file=file,
            module_index=module_index,
            callable_ids=callable_ids,
            module_map=module_map,
            known_classes=known_classes,
            include_external=False,
            package_prefix=None,
            return_summaries=return_summaries,
            return_links=return_links,
            project_index=project_index,
        )
        self.collected_summaries: Dict[str, FunctionReturnSummary] = {}

    def visit_Call(self, node: ast.Call) -> None:
        self._record_container_mutation(node)
        self._visit_call_children(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Merge one return statement's object, element, and slot type facts.

        Each inference runs inside ``_recording_return_links`` naming the field
        it fills. Whatever the inference reaches -- however deeply nested the
        expression -- then records a link from that field back to whichever
        field of the callee it consulted, so facts learned later still find
        their way here.
        """
        if self.current_callable and node.value is not None:
            summary = self.collected_summaries.setdefault(
                self.current_callable, FunctionReturnSummary()
            )
            # ``return await build()`` returns exactly what ``build()`` does, so
            # the shape tests below have to see through the wrapper.
            returned = unwrap_passthrough(node.value)
            with self._recording_return_links("class_types"):
                summary.class_types.update(
                    self._infer_class_types_from_value(returned)
                )
            # A callable handed back rather than an object: a factory returning a
            # function, or a decorator returning its wrapper.
            with self._recording_return_links("callable_ids"):
                summary.callable_ids.update(
                    self._infer_callable_ids_from_value(returned)
                )
            with self._recording_return_links("element_types"):
                summary.element_types.update(
                    self._infer_container_element_types(returned)
                )
            if isinstance(returned, (ast.Tuple, ast.List)):
                for index, item in enumerate(returned.elts):
                    with self._recording_return_links("slot_types", index):
                        add_indexed_types(
                            summary.slot_types,
                            index,
                            self._infer_class_types_from_value(item),
                        )
                    with self._recording_return_links("slot_element_types", index):
                        add_indexed_types(
                            summary.slot_element_types,
                            index,
                            self._infer_container_element_types(item),
                        )
            elif isinstance(returned, ast.Call):
                self._note_returned_call_slots(returned)
            elif isinstance(returned, ast.Name):
                self._note_returned_var_slots(returned.id)
        self.generic_visit(node)

    def _note_returned_var_slots(self, name: str) -> None:
        """Carry a tuple shape across ``pair = make_pair(); return pair``.

        The bare-call branch above cannot see through the local, and the
        class/element replay only reproduces flat fields, so without this the
        positions are lost exactly as they were before return links existed.
        """
        if name in self.types.replaying:
            return
        self.types.replaying.add(name)
        try:
            for source, slot in self.types.get_var_sources(name):
                # A destructured binding is one position, not a whole shape.
                if slot is not None:
                    continue
                source = unwrap_passthrough(source)
                if isinstance(source, ast.Call):
                    self._note_returned_call_slots(source)
                elif isinstance(source, ast.Name):
                    self._note_returned_var_slots(source.id)
        finally:
            self.types.replaying.discard(name)

    def _note_returned_call_slots(self, value: ast.Call) -> None:
        """Carry a callee's whole tuple shape across ``return make_pair()``.

        The branch above only learns slot types from a literal tuple or list.
        For a bare call there is no literal to walk, so link the slot-keyed
        fields wholesale: whatever positions the callee is found to return, this
        callable returns at the same positions.
        """
        if self.return_links is None or not self.current_callable:
            return
        for callee, _relation, is_resolved in self._resolve_callees(value.func):
            if not is_resolved:
                continue
            for field_name in ("slot_types", "slot_element_types"):
                self.return_links.record(
                    ReturnLink(
                        caller=self.current_callable,
                        callee=callee,
                        source_field=field_name,
                        target_field=field_name,
                    )
                )


class TypeSummaryCollector(CallCollector):
    """Analysis pass that propagates argument and instance-attribute types."""

    # This is the one pass that writes registry evidence. It already visits every
    # store and every call with the full type environment in scope, so gathering
    # the escape and invoke facts here costs one extra walk of nothing.
    records_registry_facts = True

    def __init__(
        self,
        module: str,
        file: Path,
        module_index: ModuleIndex,
        callable_ids: Set[str],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        return_summaries: Dict[str, FunctionReturnSummary],
        param_summaries: Dict[str, FunctionParamSummary],
        class_attr_types: ClassAttrTypes,
        project_index: Optional[ProjectIndex] = None,
        escape_summaries: Optional[Dict[str, FunctionEscapeSummary]] = None,
        registry_facts: Optional[RegistryFacts] = None,
    ):
        super().__init__(
            module=module,
            file=file,
            module_index=module_index,
            callable_ids=callable_ids,
            module_map=module_map,
            known_classes=known_classes,
            include_external=False,
            package_prefix=None,
            return_summaries=return_summaries,
            param_summaries=param_summaries,
            class_attr_types=class_attr_types,
            project_index=project_index,
            escape_summaries=escape_summaries,
            registry_facts=registry_facts,
        )

    def visit_Return(self, node: ast.Return) -> None:
        """Record that this callable hands back the contents of an attribute.

        ``get_children`` returning ``[*self._children, ...]`` is how a registry
        leaves the class that owns it. Without this the ``.draw()`` its caller
        performs on the result cannot be attributed to ``self._children``, and
        the registry is invisible.
        """
        if self.current_callable is not None and node.value is not None:
            self.registry_facts.add_returned_attrs(
                self.current_callable, self._attr_origins(self._element_origins(node.value))
            )
        self.generic_visit(node)

    @staticmethod
    def _attr_origins(origins: Set[Origin]) -> Set[ClassAttr]:
        return {payload for kind, payload in origins if kind == "attr_element"}

    def visit_Call(self, node: ast.Call) -> None:
        """Feed inferred argument types into the resolved callee's parameters."""
        self._record_container_mutation(node)
        self._note_receiver_invocation(node.func)
        self._note_argument_escapes(node)
        self._note_argument_registries(node)

        dynamic_resolved = self._resolve_dynamic_getattr_callees(node.func)
        resolved_callees = dynamic_resolved or self._resolve_callees(node.func)
        for callee, relation, is_resolved in resolved_callees:
            if not is_resolved:
                continue
            summary = self.param_summaries.setdefault(callee, FunctionParamSummary())
            positional_offset = self._implicit_receiver_arg_offset(callee, relation)
            for index, arg in enumerate(node.args):
                if isinstance(arg, ast.Starred):
                    continue
                add_indexed_types(
                    summary.positional_types,
                    index + positional_offset,
                    self._infer_class_types_from_value(arg),
                )
                add_indexed_types(
                    summary.positional_element_types,
                    index + positional_offset,
                    self._infer_container_element_types(arg),
                )
                # A function passed as an argument. Recorded here so the callee
                # can resolve a call to its own parameter -- the fact has to
                # cross the call boundary, and nothing else carries it.
                add_indexed_types(
                    summary.positional_callables,
                    index + positional_offset,
                    self._infer_callable_ids_from_value(arg),
                )
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                add_types(
                    summary.named_types,
                    keyword.arg,
                    self._infer_class_types_from_value(keyword.value),
                )
                add_types(
                    summary.named_element_types,
                    keyword.arg,
                    self._infer_container_element_types(keyword.value),
                )
                add_types(
                    summary.named_callables,
                    keyword.arg,
                    self._infer_callable_ids_from_value(keyword.value),
                )

        self.generic_visit(node)

    def _note_argument_escapes(self, node: ast.Call) -> None:
        """Carry escapes across a delegating registration method.

        ``def add_child(self, name, child): self._store(name, child)`` retains
        ``child`` just as surely as storing it directly, but the store is one
        call away. Composing the callee's escapes onto our own parameters closes
        that gap, and the surrounding fixed point closes chains of any length.

        Restricted to calls on ``self``. ``other._store(child)`` also retains the
        child, but on ``other`` -- attributing it to our own class would name the
        wrong parent, which is the one thing this analysis must get right.
        """
        if not isinstance(node.func, ast.Attribute) or self.current_callable is None:
            return
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"self", "cls"}
        ):
            return

        for callee, relation, is_resolved in self._resolve_callees(node.func):
            if not is_resolved:
                continue
            callee_escapes = self.escape_summaries.get(callee)
            if callee_escapes is None:
                continue
            offset = self._implicit_receiver_arg_offset(callee, relation)

            for escapes, is_element in (
                (callee_escapes.escapes, False),
                (callee_escapes.element_escapes, True),
            ):
                for (position, name), slots in escapes.items():
                    argument = self._argument_at(node, position, name, offset)
                    if argument is None:
                        continue
                    # Which origin we read decides what the callee retained: the
                    # value we handed over, or what was inside it.
                    origins = (
                        self._element_origins(argument)
                        if is_element
                        else self._value_origins(argument)
                    )
                    self._compose_escape(origins, slots)

    def _note_argument_registries(self, node: ast.Call) -> None:
        """Record which parameters callers fill with a registry's contents.

        The other half of carrying provenance across a call. ``_draw_all(artists)``
        hands a registry to a free function, and only this lets the ``.draw()``
        inside that function count as invoking the registry.
        """
        if self.current_callable is None:
            return
        for callee, relation, is_resolved in self._resolve_callees(node.func):
            if not is_resolved:
                continue
            offset = self._implicit_receiver_arg_offset(callee, relation)
            for index, argument in enumerate(node.args):
                if isinstance(argument, ast.Starred):
                    continue
                attrs = self._attr_origins(self._element_origins(argument))
                if attrs:
                    self.registry_facts.add_param_attrs(
                        callee, (index + offset, ""), attrs
                    )
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                attrs = self._attr_origins(self._element_origins(keyword.value))
                if attrs:
                    self.registry_facts.add_param_attrs(
                        callee, (KEYWORD_ONLY_POSITION, keyword.arg), attrs
                    )

    @staticmethod
    def _argument_at(
        node: ast.Call, position: int, name: str, offset: int
    ) -> Optional[ast.AST]:
        for keyword in node.keywords:
            if keyword.arg == name:
                return keyword.value
        index = position - offset
        if position >= 0 and 0 <= index < len(node.args):
            candidate = node.args[index]
            if not isinstance(candidate, ast.Starred):
                return candidate
        return None

    def _compose_escape(self, origins: Set[Origin], slots: Set[AttrSlot]) -> None:
        """Attribute a callee's escape to whichever of our parameters supplied it."""
        if self.current_callable is None:
            return
        for kind, payload in origins:
            if kind not in {"param", "param_element"}:
                continue
            summary = self.escape_summaries.setdefault(
                self.current_callable, FunctionEscapeSummary()
            )
            if kind == "param":
                summary.add(payload, set(slots))
            else:
                summary.add_element(payload, set(slots))
