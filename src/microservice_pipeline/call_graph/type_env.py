"""The lexically scoped type environment used during a collector's walk.

``TypeEnv`` is the collector's memory of what names currently hold. Each active
scope -- module, function, comprehension -- contributes one frame to four
parallel stacks:

``var_types``
    variable -> possible object class IDs
``container_types``
    variable -> possible *element* class IDs, so a list of ``Order`` is
    distinguishable from an ``Order``
``attr_types``
    dotted attribute expression -> possible class IDs, scoped to this walk
``var_sources``
    variable -> the expression it was assigned from, so ``return x`` can re-run
    inference on whatever produced ``x``

Values are sets throughout because control flow can make more than one runtime
type possible; ``merge_states`` unions the alternatives at a join point rather
than letting one branch overwrite the other.

This module owns state and nothing else. It performs no name resolution and
knows nothing about the project -- inference that needs to consult class
hierarchies or return summaries stays in the collector.
"""


from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .models import add_types, copy_type_map

# What a local variable was assigned from: the value expression, plus the tuple
# position taken from it when the assignment destructured (``a, b = make_pair()``
# stores slot 0 for ``a``). Kept so ``return a`` can re-run inference on the
# expression -- see ``CallCollector._replay_var_sources``.
VarSource = Tuple[ast.AST, Optional[int]]

TypeStack = List[Dict[str, Set[str]]]
SourceStack = List[Dict[str, Set[VarSource]]]
TypeState = Tuple[TypeStack, TypeStack, TypeStack, SourceStack]


class TypeEnv:
    """Scoped variable, container, and attribute type facts for one walk."""

    def __init__(self) -> None:
        self.var_types_stack: TypeStack = []
        self.container_types_stack: TypeStack = []
        self.attr_types_stack: TypeStack = []
        self.var_sources_stack: SourceStack = []
        # Variable names whose assigned expression is being replayed, so a
        # self-referential binding such as ``x = x.next()`` cannot loop.
        self.replaying: Set[str] = set()

    # -- scopes ---------------------------------------------------------------

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

    # -- branch state ---------------------------------------------------------

    def copy_state(self) -> TypeState:
        """Deep-copy mutable inference state before exploring a branch."""
        return (
            [copy_type_map(scope) for scope in self.var_types_stack],
            [copy_type_map(scope) for scope in self.container_types_stack],
            [copy_type_map(scope) for scope in self.attr_types_stack],
            [self._copy_source_map(scope) for scope in self.var_sources_stack],
        )

    def restore_state(self, state: TypeState) -> None:
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

    def merge_states(self, states: Iterable[TypeState]) -> TypeState:
        """Union facts from control-flow alternatives into one safe state."""
        state_list = list(states)
        return (
            self._merge_type_stack_states(state[0] for state in state_list),
            self._merge_type_stack_states(state[1] for state in state_list),
            self._merge_type_stack_states(state[2] for state in state_list),
            self._merge_source_stack_states(state[3] for state in state_list),
        )

    # -- variable types -------------------------------------------------------

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

    def get_var_type(self, var: str) -> Optional[str]:
        types = sorted(self.get_var_types(var))
        if types:
            return types[0]
        return None

    # -- container element types ----------------------------------------------

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

    # -- assignment provenance ------------------------------------------------

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

    # -- attribute-expression types -------------------------------------------

    def set_attr_expr_types(self, attr_expr: str, types: Set[str]) -> None:
        if self.attr_types_stack and types:
            add_types(self.attr_types_stack[-1], attr_expr, types)

    def get_attr_expr_types(self, attr_expr: str) -> Set[str]:
        for scope in reversed(self.attr_types_stack):
            if attr_expr in scope:
                return set(scope[attr_expr])
        return set()