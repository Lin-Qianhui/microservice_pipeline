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
from .discovery import module_callable_id
from .return_links import ReturnLink, ReturnLinkTable
from .models import (
    ClassAttrTypes,
    Edge,
    FunctionParamSummary,
    FunctionReturnSummary,
    ModuleIndex,
    add_indexed_types,
    add_types,
    copy_type_map,
)

try:
    from microservice_pipeline.import_resolution import (
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )
    from microservice_pipeline.subprocess_coupling import (
        add_subprocess_child_expr,
        is_add_subprocess_call,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from import_resolution import (  # type: ignore
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )
    from subprocess_coupling import (  # type: ignore
        add_subprocess_child_expr,
        is_add_subprocess_call,
    )


# Python syntax can invoke methods implicitly. These tables translate AST
# operator node types into the corresponding data-model ("dunder") methods so,
# for example, ``left + right`` can produce an edge to ``left.__add__``.
BINOP_DUNDER_METHODS = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.MatMult: "__matmul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
    ast.BitAnd: "__and__",
}

REVERSE_BINOP_DUNDER_METHODS = {
    ast.Add: "__radd__",
    ast.Sub: "__rsub__",
    ast.Mult: "__rmul__",
    ast.MatMult: "__rmatmul__",
    ast.Div: "__rtruediv__",
    ast.FloorDiv: "__rfloordiv__",
    ast.Mod: "__rmod__",
    ast.Pow: "__rpow__",
    ast.LShift: "__rlshift__",
    ast.RShift: "__rrshift__",
    ast.BitOr: "__ror__",
    ast.BitXor: "__rxor__",
    ast.BitAnd: "__rand__",
}

AUGASSIGN_DUNDER_METHODS = {
    ast.Add: "__iadd__",
    ast.Sub: "__isub__",
    ast.Mult: "__imul__",
    ast.MatMult: "__imatmul__",
    ast.Div: "__itruediv__",
    ast.FloorDiv: "__ifloordiv__",
    ast.Mod: "__imod__",
    ast.Pow: "__ipow__",
    ast.LShift: "__ilshift__",
    ast.RShift: "__irshift__",
    ast.BitOr: "__ior__",
    ast.BitXor: "__ixor__",
    ast.BitAnd: "__iand__",
}

COMPARE_DUNDER_METHODS = {
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
}

UNARY_DUNDER_METHODS = {
    ast.UAdd: "__pos__",
    ast.USub: "__neg__",
    ast.Invert: "__invert__",
}


# What a local variable was assigned from: the value expression, plus the tuple
# position taken from it when the assignment destructured (``a, b = make_pair()``
# stores slot 0 for ``a``). Kept so ``return a`` can re-run inference on the
# expression -- see ``CallCollector._replay_var_sources``.
VarSource = Tuple[ast.AST, Optional[int]]

TypeStack = List[Dict[str, Set[str]]]
SourceStack = List[Dict[str, Set[VarSource]]]
TypeState = Tuple[TypeStack, TypeStack, TypeStack, SourceStack]


class CallCollector(ast.NodeVisitor):
    """Resolve calls and implicit Python operations into graph edges.

    The visitor also performs deliberately small-scale type inference. Each
    active lexical scope has three maps:

    * ``var_types_stack``: variable -> possible object class IDs
    * ``container_types_stack``: variable -> possible element class IDs
    * ``attr_types_stack``: dotted attribute expression -> possible class IDs

    Values are sets because control flow can make more than one runtime type
    possible. Resolution prefers known project definitions, but can retain an
    unresolved descriptive target when ``include_external`` is enabled.
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
        self.class_bases = {
            class_id: bases
            for module_info in module_map.values()
            for class_id, bases in module_info.class_bases.items()
        }
        # Reverse of ``class_bases``, for resolving a ``self.hook()`` call in a
        # base class down to the subclass overrides that really run. Base IDs are
        # canonicalized on the way in because a base recorded through a package
        # re-export is an alias of the defining path.
        self.subclasses: Dict[str, Set[str]] = {}
        for class_id, bases in self.class_bases.items():
            for base_id in bases:
                canonical_base = known_classes.get(base_id, base_id)
                self.subclasses.setdefault(canonical_base, set()).add(class_id)
        self.module_callable = module_callable_id(module)
        self.property_ids = {
            property_id
            for module_info in module_map.values()
            for property_id in module_info.properties
        }
        self.static_method_ids = {
            method_id
            for module_info in module_map.values()
            for method_id in module_info.static_methods
        }
        self.class_method_ids = {
            method_id
            for module_info in module_map.values()
            for method_id in module_info.class_methods
        }

        # Deferred "we return whatever that callee returns" facts. ``None``
        # switches recording off, which is what every collector except
        # ``ReturnSummaryCollector`` wants.
        self.return_links = return_links
        self._link_target: List[Tuple[str, Optional[int]]] = []

        self.current_callable: Optional[str] = None
        self.current_class: Optional[str] = None
        self.callable_stack: List[str] = []
        self.var_types_stack: TypeStack = []
        self.container_types_stack: TypeStack = []
        self.attr_types_stack: TypeStack = []
        self.var_sources_stack: SourceStack = []
        # Variable names whose assigned expression is being replayed, so a
        # self-referential binding such as ``x = x.next()`` cannot loop.
        self._replaying: Set[str] = set()
        self.edges: List[Edge] = []

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

    def push_scope(self) -> None:
        """Start empty type-inference maps for a module/function scope."""
        self.var_types_stack.append({})
        self.container_types_stack.append({})
        self.attr_types_stack.append({})
        self.var_sources_stack.append({})

    def pop_scope(self) -> None:
        self.var_types_stack.pop()
        self.container_types_stack.pop()
        self.attr_types_stack.pop()
        self.var_sources_stack.pop()

    def _copy_type_state(self) -> TypeState:
        """Deep-copy mutable inference state before exploring a branch."""
        return (
            [copy_type_map(scope) for scope in self.var_types_stack],
            [copy_type_map(scope) for scope in self.container_types_stack],
            [copy_type_map(scope) for scope in self.attr_types_stack],
            [self._copy_source_map(scope) for scope in self.var_sources_stack],
        )

    def _restore_type_state(self, state: TypeState) -> None:
        self.var_types_stack = [copy_type_map(scope) for scope in state[0]]
        self.container_types_stack = [copy_type_map(scope) for scope in state[1]]
        self.attr_types_stack = [copy_type_map(scope) for scope in state[2]]
        self.var_sources_stack = [self._copy_source_map(scope) for scope in state[3]]

    @staticmethod
    def _copy_source_map(
        source_map: Dict[str, Set[VarSource]],
    ) -> Dict[str, Set[VarSource]]:
        return {key: set(values) for key, values in source_map.items()}

    def _merge_type_maps(
        self, maps: Iterable[Dict[str, Set[str]]]
    ) -> Dict[str, Set[str]]:
        merged: Dict[str, Set[str]] = {}
        for type_map in maps:
            for key, values in type_map.items():
                add_types(merged, key, values)
        return merged

    def _merge_type_stack_states(
        self, states: Iterable[List[Dict[str, Set[str]]]]
    ) -> List[Dict[str, Set[str]]]:
        state_list = list(states)
        if not state_list:
            return []
        depth = len(state_list[0])
        return [
            self._merge_type_maps(state[index] for state in state_list)
            for index in range(depth)
        ]

    def _merge_source_stack_states(
        self, states: Iterable[SourceStack]
    ) -> SourceStack:
        """Union provenance across branches.

        A variable assigned in both arms of an ``if`` was really assigned by
        either, so ``return x`` afterwards depends on both expressions. Merging
        here is what stops the second arm's assignment from simply overwriting
        the first's.
        """
        state_list = list(states)
        if not state_list:
            return []
        depth = len(state_list[0])
        merged: SourceStack = []
        for index in range(depth):
            scope: Dict[str, Set[VarSource]] = {}
            for state in state_list:
                for key, values in state[index].items():
                    if values:
                        scope.setdefault(key, set()).update(values)
            merged.append(scope)
        return merged

    def _merge_type_states(self, states: Iterable[TypeState]) -> TypeState:
        """Union facts from control-flow alternatives into one safe state."""
        state_list = list(states)
        return (
            self._merge_type_stack_states(state[0] for state in state_list),
            self._merge_type_stack_states(state[1] for state in state_list),
            self._merge_type_stack_states(state[2] for state in state_list),
            self._merge_source_stack_states(state[3] for state in state_list),
        )

    def set_var_type(self, var: str, typ: str) -> None:
        self.set_var_types(var, {typ})

    def set_var_types(self, var: str, types: Set[str]) -> None:
        if self.var_types_stack:
            self.var_types_stack[-1][var] = set(types)

    def get_var_types(self, var: str) -> Set[str]:
        for scope in reversed(self.var_types_stack):
            if var in scope:
                return set(scope[var])
        return set()

    def set_container_types(self, var: str, types: Set[str]) -> None:
        if self.container_types_stack:
            self.container_types_stack[-1][var] = set(types)

    def add_container_types(self, var: str, types: Set[str]) -> None:
        if not self.container_types_stack or not types:
            return
        existing = self.container_types_stack[-1].setdefault(var, set())
        existing.update(types)

    def get_container_types(self, var: str) -> Set[str]:
        for scope in reversed(self.container_types_stack):
            if var in scope:
                return set(scope[var])
        return set()

    def set_var_source(self, var: str, value: ast.AST, slot: Optional[int]) -> None:
        """Remember the expression ``var`` was just assigned from.

        Replaces rather than accumulates, matching ``set_var_types``: a rebound
        name no longer depends on what it used to hold.
        """
        if self.var_sources_stack:
            self.var_sources_stack[-1][var] = {(value, slot)}

    def get_var_sources(self, var: str) -> Set[VarSource]:
        for scope in reversed(self.var_sources_stack):
            if var in scope:
                return set(scope[var])
        return set()

    def get_var_type(self, var: str) -> Optional[str]:
        types = sorted(self.get_var_types(var))
        if types:
            return types[0]
        return None

    def set_attr_expr_types(self, attr_expr: str, types: Set[str]) -> None:
        if self.attr_types_stack and types:
            add_types(self.attr_types_stack[-1], attr_expr, types)

    def get_attr_expr_types(self, attr_expr: str) -> Set[str]:
        for scope in reversed(self.attr_types_stack):
            if attr_expr in scope:
                return set(scope[attr_expr])
        return set()

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
                if target in self.property_ids:
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

    def current_class_id(self) -> Optional[str]:
        if self.current_class:
            return f"{self.module}.{self.current_class}"
        return None

    def _is_known_class(self, class_id: str) -> bool:
        return (
            class_id in self.known_classes
            or class_id in self.known_classes.values()
        )

    def _known_class_id(self, class_id: str) -> str:
        return self.known_classes.get(class_id, class_id)

    def _unique(self, values: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _resolve_class_reference_name(self, name: str) -> List[str]:
        matches: Set[str] = set()

        if "." not in name:
            local_class = self.module_index.classes.get(name)
            if local_class:
                matches.add(local_class)

            imported = self.module_index.imports.get(name)
            if imported and self._is_known_class(imported):
                matches.add(self._known_class_id(imported))

            for imported in self._resolve_star_import_targets(name):
                if self._is_known_class(imported):
                    matches.add(self._known_class_id(imported))

            local_name = f"{self.module}.{name}"
            if self._is_known_class(local_name):
                matches.add(self._known_class_id(local_name))

            return sorted(matches)

        base, rest = name.split(".", 1)
        imported_base = self.module_index.imports.get(base)
        if imported_base:
            candidate = f"{imported_base}.{rest}"
            if self._is_known_class(candidate):
                matches.add(self._known_class_id(candidate))

        if self._is_known_class(name):
            matches.add(self._known_class_id(name))

        return sorted(matches)

    def _resolve_method_targets(
        self, class_id: str, method_name: str, seen: Optional[Set[str]] = None
    ) -> List[str]:
        """Find a method on a class or, recursively, its indexed base classes."""
        if seen is None:
            seen = set()
        if class_id in seen:
            return []
        seen.add(class_id)

        direct = f"{class_id}.{method_name}"
        if direct in self.callable_ids:
            return [direct]

        targets: List[str] = []
        for base_id in self.class_bases.get(class_id, []):
            if not self._is_known_class(base_id):
                continue
            targets.extend(
                self._resolve_method_targets(
                    self._known_class_id(base_id), method_name, seen=seen.copy()
                )
            )
        return self._unique(targets)

    def _class_and_ancestors(self, class_id: str) -> List[str]:
        """``class_id`` followed by every base class reachable from it.

        Attribute facts are recorded against whichever class the assigning method
        belongs to, which is often a base. ``self.subprocess`` is populated by
        ``Process.add_subprocess`` but read by ``TimeDependentProcess``, so a
        lookup that checks only the reading class finds nothing.
        """
        chain: List[str] = []
        queue: List[str] = [class_id]
        seen: Set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            chain.append(current)
            for base_id in self.class_bases.get(current, []):
                canonical = self._known_class_id(base_id)
                if canonical not in seen:
                    queue.append(canonical)
        return chain

    def _resolve_subclass_override_targets(self, class_id: str, method_name: str) -> List[str]:
        """Find overrides of ``method_name`` below ``class_id`` in the hierarchy.

        This is the Template Method pattern, which scientific frameworks lean on
        heavily. ``_SurfaceFlux._compute_heating_rates`` calls
        ``self._compute_flux()``, but ``_SurfaceFlux`` never defines
        ``_compute_flux`` -- its two subclasses do, and one of those is what
        actually runs. Resolving ``self`` only to the enclosing class, as normal
        method resolution does, finds nothing at all here. On climlab this shape
        was 49 of the 55 edges still missing after alias canonicalization.

        Every override found is a genuine runtime possibility, so all of them are
        returned. That deliberately over-approximates -- a base with ten
        subclasses yields ten edges -- which is why callers label these
        ``virtual_override`` and the structural graph weights them below literal
        calls rather than treating them as ordinary resolved edges.
        """
        targets: List[str] = []
        queue: List[str] = [class_id]
        seen: Set[str] = {class_id}
        while queue:
            current = queue.pop()
            for subclass_id in sorted(self.subclasses.get(current, ())):
                if subclass_id in seen:
                    continue
                seen.add(subclass_id)
                queue.append(subclass_id)
                # A subclass that overrides the method is one possible target;
                # keep descending regardless, because a deeper subclass can
                # override it again and instances of that class dispatch there.
                candidate = f"{subclass_id}.{method_name}"
                if candidate in self.callable_ids:
                    targets.append(candidate)
        return self._unique(targets)

    def _resolve_constructor_targets(self, class_id: str) -> List[str]:
        return self._resolve_method_targets(class_id, "__init__")

    def _resolve_super_method_targets(self, method_name: str) -> List[str]:
        class_id = self.current_class_id()
        if not class_id:
            return []

        targets: List[str] = []
        for base_id in self.class_bases.get(class_id, []):
            if not self._is_known_class(base_id):
                continue
            targets.extend(
                self._resolve_method_targets(
                    self._known_class_id(base_id), method_name
                )
            )
        return self._unique(targets)

    def _add_edge(
        self, callee: str, relation: str, is_resolved: bool, lineno: int
    ) -> None:
        """Append an edge after applying unresolved/external filtering policy."""
        if self.current_callable is None:
            return
        if not is_resolved and not self.include_external:
            return
        if is_resolved and not self._is_internal_callee(callee):
            return
        self.edges.append(
            Edge(
                caller=self.current_callable,
                callee=callee,
                file=str(self.file),
                lineno=lineno,
                resolved=is_resolved,
                relation=relation,
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
        return any(
            callable_id.startswith(f"{class_id}.")
            for class_id in self.known_classes.values()
        )

    def _implicit_receiver_arg_offset(self, callee: str, relation: str) -> int:
        """Account for implicit ``self``/``cls`` in positional parameter facts."""
        if callee in self.static_method_ids:
            return 0
        if relation == "constructor":
            return 1
        if relation == "class_method":
            return 1 if callee in self.class_method_ids else 0
        if relation in {"self_method", "super_method", "inferred_type"}:
            if self._callable_belongs_to_known_class(callee):
                return 1
        return 0

    def _add_dunder_edges(
        self, receiver: ast.AST, method_name: str, relation: str, lineno: int
    ) -> None:
        for callee, edge_relation, is_resolved in self._resolve_method_callees_for_expr(
            receiver, method_name, relation
        ):
            self._add_edge(callee, edge_relation, is_resolved, lineno)

    def _add_membership_edges(self, container: ast.AST, lineno: int) -> None:
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

        for callee, relation, is_resolved in self._unique_callee_results(targets):
            self._add_edge(callee, relation, is_resolved, lineno)

    def _subprocess_parent_types(self, receiver: ast.AST) -> Set[str]:
        if (
            isinstance(receiver, ast.Name)
            and receiver.id in {"self", "cls"}
            and self.current_class
        ):
            class_id = self.current_class_id()
            return {class_id} if class_id else set()
        return self._receiver_types_for_expr(receiver)

    def _subprocess_child_types(self, child_expr: ast.AST) -> Set[str]:
        if isinstance(child_expr, ast.Call):
            return self._infer_class_types_from_call(child_expr)
        if isinstance(child_expr, (ast.Name, ast.Attribute)):
            return self.get_expr_types(child_expr)
        return set()

    def _add_subprocess_compute_edge(self, node: ast.Call) -> None:
        """Add the framework-specific parent/child ``_compute`` dependency.

        ``add_subprocess(child)`` establishes execution coupling that is not an
        ordinary source-level call from one ``_compute`` method to the other.
        The edge is emitted only when both sides resolve unambiguously.
        """
        if not is_add_subprocess_call(node):
            return
        child_expr = add_subprocess_child_expr(node)
        if child_expr is None or not isinstance(node.func, ast.Attribute):
            return

        parent_types = sorted(self._subprocess_parent_types(node.func.value))
        child_types = sorted(self._subprocess_child_types(child_expr))
        if len(parent_types) != 1 or len(child_types) != 1:
            return

        parent_compute = self._resolve_method_targets(parent_types[0], "_compute")
        child_compute = self._resolve_method_targets(child_types[0], "_compute")
        if len(parent_compute) != 1 or len(child_compute) != 1:
            return

        if not self._is_internal_callee(parent_compute[0]) or not self._is_internal_callee(child_compute[0]):
            return
        self.edges.append(
            Edge(
                caller=parent_compute[0],
                callee=child_compute[0],
                file=str(self.file),
                lineno=node.lineno,
                resolved=True,
                relation="subprocess_compute",
            )
        )

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

        if isinstance(target, ast.Name) and self.var_types_stack:
            if class_types:
                self.set_var_types(target.id, class_types)
            if container_types or self._is_container_literal(value):
                self.set_container_types(target.id, container_types)
            # Unconditional: the point of provenance is the case where nothing
            # is known about ``value`` yet, so an empty inference above is
            # exactly when this matters most.
            self.set_var_source(target.id, value, slot)
            return

        if isinstance(target, ast.Attribute):
            self.set_attribute_types(target, class_types)
            self.set_attribute_element_types(target, container_types)
            return

        # ``self.registry[key] = value``: the assignment target is a subscript
        # of an attribute, so the value becomes an element of that attribute.
        if isinstance(target, ast.Subscript):
            self.record_attribute_container_store(target.value, class_types)

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

    def _add_import_edge(self, module_name: str, lineno: int) -> None:
        imported_module_callable = module_callable_id(module_name)
        if imported_module_callable in self.callable_ids:
            self._add_edge(imported_module_callable, "import", True, lineno)
        else:
            self._add_edge(module_name, "import", False, lineno)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_import_edge(alias.name, node.lineno)

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
            self._add_import_edge(imported_module, node.lineno)
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
                self._add_import_edge(target, node.lineno)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev

    def _enter_callable(self, node: ast.AST, name: str) -> None:
        """Enter a function scope, seed parameter types, visit it, and restore state."""
        prev_callable = self.current_callable
        if prev_callable is not None and prev_callable != self.module_callable:
            self.current_callable = f"{prev_callable}.<locals>.{name}"
        elif self.current_class:
            self.current_callable = f"{self.module}.{self.current_class}.{name}"
        else:
            self.current_callable = f"{self.module}.{name}"

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
            for index, arg in enumerate(
                node.args.kwonlyargs, start=len(positional_args)
            ):
                self._seed_param_types(param_summary, index, arg)

        self.generic_visit(node)
        self.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable

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

    def _resolve_enclosing_local_callable(self, name: str) -> Optional[str]:
        for scope_callable in reversed(self.callable_stack):
            candidate = f"{scope_callable}.<locals>.{name}"
            if candidate in self.callable_ids:
                return candidate
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_callable(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_callable(node, node.name)

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
        self.generic_visit(node)

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
                node.target, method_name, "dunder_operator", node.lineno
            )
        self.visit(node.target)
        self.visit(node.value)

    def _visit_for_like(self, node: ast.For | ast.AsyncFor) -> None:
        """Model iteration and propagate known collection element types."""
        self._add_dunder_edges(node.iter, "__iter__", "dunder_iter", node.lineno)
        self.visit(node.iter)
        if isinstance(node.target, ast.Name):
            element_types = self._infer_container_element_types(node.iter)
            if element_types:
                self.set_var_types(node.target.id, element_types)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            # ``for name, proc in registry.items()``: each loop variable takes
            # one slot of the element, so bind them position by position.
            for index, item in enumerate(node.target.elts):
                if not isinstance(item, ast.Name):
                    continue
                slot_types = self._infer_sequence_slot_class_types(node.iter, index)
                if slot_types:
                    self.set_var_types(item.id, slot_types)
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

    def _record_container_mutation(self, node: ast.Call) -> None:
        """Learn element types introduced by mutating container calls.

        Handles both a local (``items.append(x)``) and an attribute container
        (``self.items.append(x)``). The attribute case is the one that crosses
        method boundaries, so it is what lets a registry populated in one method
        be resolved where another method iterates it.
        """
        if not isinstance(node.func, ast.Attribute):
            return

        element_types = self._container_mutation_element_types(node)
        if not element_types:
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
        work, since whatever it can infer is already in ``var_types_stack``.
        """
        if not self._link_target or name in self._replaying:
            return set()
        self._replaying.add(name)
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
            self._replaying.discard(name)

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

    def visit_Call(self, node: ast.Call) -> None:
        """Resolve an explicit call, record its edge, then visit child expressions."""
        self._record_container_mutation(node)

        dynamic_resolved = self._resolve_dynamic_getattr_callees(node.func)
        resolved_callees = dynamic_resolved or self._resolve_callees(node.func)
        for callee, relation, is_resolved in resolved_callees:
            self._add_edge(callee, relation, is_resolved, node.lineno)
        self._add_subprocess_compute_edge(node)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Turn a loaded property access into its implicit getter call."""
        if isinstance(node.ctx, ast.Load):
            for callee in self._resolve_property_getter_targets(node):
                self._add_edge(callee, "property_getter", True, node.lineno)
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Model ``obj[key]`` load/store/delete through the relevant dunder method."""
        if isinstance(node.ctx, ast.Store):
            self._add_dunder_edges(
                node.value, "__setitem__", "dunder_setitem", node.lineno
            )
        elif isinstance(node.ctx, ast.Del):
            self._add_dunder_edges(
                node.value, "__delitem__", "dunder_delitem", node.lineno
            )
        else:
            self._add_dunder_edges(
                node.value, "__getitem__", "dunder_getitem", node.lineno
            )
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        method_name = BINOP_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.left, method_name, "dunder_operator", node.lineno
            )
        reverse_method_name = REVERSE_BINOP_DUNDER_METHODS.get(type(node.op))
        if reverse_method_name:
            self._add_dunder_edges(
                node.right, reverse_method_name, "dunder_operator", node.lineno
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        self.visit(left)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                self._add_membership_edges(comparator, node.lineno)
            else:
                method_name = COMPARE_DUNDER_METHODS.get(type(op))
                if method_name:
                    self._add_dunder_edges(
                        left, method_name, "dunder_operator", node.lineno
                    )
            self.visit(comparator)
            left = comparator

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        method_name = UNARY_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.operand, method_name, "dunder_operator", node.lineno
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
        if seen is None:
            seen = set()
        if module_name in seen:
            return []
        seen.add(module_name)

        matches: Set[str] = set()
        module_index = self.module_map.get(module_name)
        local_name = f"{module_name}.{name}"

        if local_name in self.callable_ids:
            matches.add(local_name)
        if module_index and name in module_index.classes:
            matches.add(module_index.classes[name])
        elif local_name in self.known_classes.values() or local_name in self.known_classes:
            matches.add(self.known_classes.get(local_name, local_name))

        if not module_index:
            return sorted(matches)

        imported = module_index.imports.get(name)
        if imported:
            matches.add(self.known_classes.get(imported, imported))

        for imported_module in module_index.star_imports:
            matches.update(
                self._resolve_name_from_module_exports(
                    imported_module, name, seen=seen.copy()
                )
            )

        return sorted(matches)

    def _resolve_star_import_targets(self, name: str) -> List[str]:
        matches: Set[str] = set()
        for imported_module in self.module_index.star_imports:
            matches.update(self._resolve_name_from_module_exports(imported_module, name))
        return sorted(matches)

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
                targets = self._resolve_super_method_targets(func.attr)
                if targets:
                    return [(target, "super_method", True) for target in targets]

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

        resolved = self._resolve_callee(func)
        if resolved:
            return [resolved]
        return []

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
                is_known = imported in self.callable_ids
                return (imported, "imported", is_known)

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
        if name in self._replaying:
            return
        self._replaying.add(name)
        try:
            for source, slot in self.get_var_sources(name):
                # A destructured binding is one position, not a whole shape.
                if slot is not None:
                    continue
                source = unwrap_passthrough(source)
                if isinstance(source, ast.Call):
                    self._note_returned_call_slots(source)
                elif isinstance(source, ast.Name):
                    self._note_returned_var_slots(source.id)
        finally:
            self._replaying.discard(name)

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
        )

    def visit_Call(self, node: ast.Call) -> None:
        """Feed inferred argument types into the resolved callee's parameters."""
        self._record_container_mutation(node)

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

        self.generic_visit(node)
