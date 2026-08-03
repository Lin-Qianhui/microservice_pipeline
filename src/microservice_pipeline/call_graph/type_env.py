"""The lexically scoped type environment used during a collector's walk.

``TypeEnv`` is the collector's memory of what names currently hold. Each active
scope -- module, function, comprehension -- contributes one frame to five
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
``var_origins``
    variable -> where the *value itself* came from, as opposed to what type it
    has. A parameter and an element read out of ``self.children`` are the two
    origins tracked, and both are needed to tell a registration call site from
    an ordinary one; see ``registration``.

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

from .models import ClassAttr, ParamKey, add_types, copy_type_map

# What a local variable was assigned from: the value expression, plus the tuple
# position taken from it when the assignment destructured (``a, b = make_pair()``
# stores slot 0 for ``a``). Kept so ``return a`` can re-run inference on the
# expression -- see ``CallCollector._replay_var_sources``.
VarSource = Tuple[ast.AST, Optional[int]]

# Where a value came from, tracked by identity rather than by type: one of the
# enclosing callable's parameters, an element of one of them, or an element read
# out of some class attribute. Each is tagged so they can share one stack without
# a consumer having to guess which it is looking at.
#
# The element forms matter because a registry is often filled wholesale --
# ``self.steps.extend(steps)`` retains what the parameter contained rather than
# the parameter itself, and the two name different children at the call site.
ParamOrigin = Tuple[str, ParamKey]  # ("param", key)
ParamElementOrigin = Tuple[str, ParamKey]  # ("param_element", key)
AttrOrigin = Tuple[str, ClassAttr]  # ("attr_element", (class id, attribute))
# A whole collection whose *contents* came out of an attribute -- what
# ``artists = self.get_children()`` binds. Distinct from ``attr_element``, which
# is a single item taken from the attribute; iterating a container origin is
# what turns it back into element origins.
AttrContainerOrigin = Tuple[str, ClassAttr]  # ("attr_container", (class id, attr))
Origin = Tuple[str, object]

TypeStack = List[Dict[str, Set[str]]]
SourceStack = List[Dict[str, Set[VarSource]]]
OriginStack = List[Dict[str, Set[Origin]]]
# Six parallel stacks, not one map of richer values. ``var_callables`` is the
# same idea as ``container_types``: a second, independent thing worth knowing
# about a name. Keeping callable ids out of ``var_types`` is deliberate --
# every consumer of a type set assumes its members are *class* ids it may call
# ``resolve_method_targets`` on, and a callable id leaking into one of those
# sets fabricates callees rather than failing loudly.
TypeState = Tuple[
    TypeStack, TypeStack, TypeStack, TypeStack, SourceStack, OriginStack
]


def param_origin(key: ParamKey) -> Origin:
    return ("param", key)


def param_element_origin(key: ParamKey) -> Origin:
    return ("param_element", key)


def attr_element_origin(key: ClassAttr) -> Origin:
    return ("attr_element", key)


def attr_container_origin(key: ClassAttr) -> Origin:
    return ("attr_container", key)


class TypeEnv:
    """Scoped variable, container, and attribute type facts for one walk."""

    def __init__(self) -> None:
        self.var_types_stack: TypeStack = []
        # What a name *is* vs. what code it *denotes*. A value that is itself
        # callable has no class to record, which is why the class-id domain
        # cannot express ``handler = self.process`` at all.
        self.var_callables_stack: TypeStack = []
        self.container_types_stack: TypeStack = []
        self.attr_types_stack: TypeStack = []
        self.var_sources_stack: SourceStack = []
        self.var_origins_stack: OriginStack = []
        # Variable names whose assigned expression is being replayed, so a
        # self-referential binding such as ``x = x.next()`` cannot loop.
        self.replaying: Set[str] = set()

    # -- scopes ---------------------------------------------------------------

    def push_scope(self) -> None:
        """Start empty type-inference maps for a module/function scope."""
        self.var_types_stack.append({})
        self.var_callables_stack.append({})
        self.container_types_stack.append({})
        self.attr_types_stack.append({})
        self.var_sources_stack.append({})
        self.var_origins_stack.append({})

    def pop_scope(self) -> None:
        self.var_types_stack.pop()
        self.var_callables_stack.pop()
        self.container_types_stack.pop()
        self.attr_types_stack.pop()
        self.var_sources_stack.pop()
        self.var_origins_stack.pop()

    # -- branch state ---------------------------------------------------------

    def copy_state(self) -> TypeState:
        """Deep-copy mutable inference state before exploring a branch."""
        return (
            [copy_type_map(scope) for scope in self.var_types_stack],
            [copy_type_map(scope) for scope in self.var_callables_stack],
            [copy_type_map(scope) for scope in self.container_types_stack],
            [copy_type_map(scope) for scope in self.attr_types_stack],
            [self._copy_source_map(scope) for scope in self.var_sources_stack],
            [self._copy_origin_map(scope) for scope in self.var_origins_stack],
        )

    def restore_state(self, state: TypeState) -> None:
        self.var_types_stack = [copy_type_map(scope) for scope in state[0]]
        self.var_callables_stack = [copy_type_map(scope) for scope in state[1]]
        self.container_types_stack = [copy_type_map(scope) for scope in state[2]]
        self.attr_types_stack = [copy_type_map(scope) for scope in state[3]]
        self.var_sources_stack = [self._copy_source_map(scope) for scope in state[4]]
        self.var_origins_stack = [self._copy_origin_map(scope) for scope in state[5]]

    @staticmethod
    def _copy_source_map(
        source_map: Dict[str, Set[VarSource]],
    ) -> Dict[str, Set[VarSource]]:
        return {key: set(values) for key, values in source_map.items()}

    @staticmethod
    def _copy_origin_map(
        origin_map: Dict[str, Set[Origin]],
    ) -> Dict[str, Set[Origin]]:
        return {key: set(values) for key, values in origin_map.items()}

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

    def _merge_provenance_stack_states(self, states: Iterable[List[Dict[str, Set]]]):
        """Union provenance across branches.

        A variable assigned in both arms of an ``if`` was really assigned by
        either, so ``return x`` afterwards depends on both expressions. Merging
        here is what stops the second arm's assignment from simply overwriting
        the first's. Shared by the source and origin stacks, which differ only in
        what they record about the assignment.
        """
        state_list = list(states)
        if not state_list:
            return []
        depth = len(state_list[0])
        merged: List[Dict[str, Set]] = []
        for index in range(depth):
            scope: Dict[str, Set] = {}
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
            self._merge_type_stack_states(state[3] for state in state_list),
            self._merge_provenance_stack_states(state[4] for state in state_list),
            self._merge_provenance_stack_states(state[5] for state in state_list),
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

    # -- value origins --------------------------------------------------------

    def set_var_origins(self, var: str, origins: Set[Origin]) -> None:
        """Record where ``var``'s current value came from.

        Replaces rather than accumulates, for the same reason ``set_var_types``
        does: a rebound name no longer holds what it used to. An empty set is
        still written, so rebinding a parameter to an unrelated value clears the
        parameter origin instead of leaving a stale one behind.
        """
        if self.var_origins_stack:
            self.var_origins_stack[-1][var] = set(origins)

    def set_var_callables(self, var: str, callables: Set[str]) -> None:
        if self.var_callables_stack and callables:
            self.var_callables_stack[-1][var] = set(callables)

    def add_var_callables(self, var: str, callables: Set[str]) -> None:
        if self.var_callables_stack and callables:
            self.var_callables_stack[-1].setdefault(var, set()).update(callables)

    def get_var_callables(self, var: str) -> Set[str]:
        """Callable ids a name may hold, innermost scope first."""
        for scope in reversed(self.var_callables_stack):
            if var in scope:
                return scope[var]
        return set()

    def is_bound_local(self, var: str) -> bool:
        """Whether ``var`` names something bound in an enclosing local scope.

        A call to a bound local is a call to a *value*, which is a different
        failure from a call to a name that was never looked up successfully.
        Only the first kind is a limitation of the abstract domain: the name
        resolved fine, and what it holds simply cannot be written down.

        Any of the three stacks counts. A parameter with no inferred type still
        has an origin, and a variable assigned from an unanalyzable expression
        still has a source, so checking types alone would miss both.
        """
        for stack in (
            self.var_types_stack,
            self.var_callables_stack,
            self.var_sources_stack,
            self.var_origins_stack,
        ):
            for scope in stack:
                if var in scope:
                    return True
        return False

    def get_var_origins(self, var: str) -> Set[Origin]:
        for scope in reversed(self.var_origins_stack):
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