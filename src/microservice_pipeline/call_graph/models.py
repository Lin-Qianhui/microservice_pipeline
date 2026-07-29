"""Data structures shared by every call-graph analysis pass.

This module is the bottom of the ``call_graph`` package: it holds the graph
node/edge records, the per-module index, the inferred-type summaries, and the
small helpers that union type facts together. It imports nothing else from the
package, so every other module here is free to depend on it.

The type-fact helpers all *merge* rather than replace. Inference runs repeatedly
over the same code and each visit can only ever learn more, so a later pass must
never be able to erase what an earlier one established.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Imports and top-level statements execute in module scope, so each module gets
# a synthetic callable node that can act as their caller.
MODULE_CALLABLE_QUALNAME = "<module>"


@dataclass
class CallableDef:
    """A graph node representing executable code discovered in one source file.

    ``id`` is the globally unique dotted identifier used by edges. ``qualname``
    is its module-relative form, while ``kind`` distinguishes module scope,
    functions, and methods. ``class_name`` is populated only for class members.
    """
    id: str
    module: str
    qualname: str
    file: str
    lineno: int
    kind: str  # module | function | method
    class_name: Optional[str] = None


@dataclass
class ModuleIndex:
    """Names and class metadata learned during the definition-indexing pass.

    This is the analyzer's lightweight substitute for importing a module and
    inspecting its runtime namespace.
    """
    module: str
    path: Path
    imports: Dict[str, str] = field(default_factory=dict)  # alias -> fully qualified target
    classes: Dict[str, str] = field(default_factory=dict)  # class name -> callable id namespace
    class_bases: Dict[str, List[str]] = field(default_factory=dict)  # class id -> base class ids
    properties: Set[str] = field(default_factory=set)  # property getter callable ids
    static_methods: Set[str] = field(default_factory=set)
    class_methods: Set[str] = field(default_factory=set)
    star_imports: List[str] = field(default_factory=list)  # modules imported via from x import *


@dataclass
class Edge:
    """A directed relationship from one callable to another.

    ``resolved`` means ``callee`` was matched to an indexed project callable;
    ``relation`` records the rule that produced the match. ``file`` and
    ``lineno`` point to the source expression responsible for the edge.
    """
    caller: str
    callee: str
    file: str
    lineno: int
    resolved: bool
    # direct | self_method | super_method | imported | inferred_type
    # attribute | dynamic_getattr | constructor | import
    # property_getter | dunder_getitem | dunder_setitem | dunder_delitem
    # dunder_contains | dunder_iter | dunder_operator
    relation: str


@dataclass(frozen=True)
class AnalysisFile:
    """A physical Python file paired with its import-style module name."""
    path: Path
    module: str


@dataclass
class FunctionReturnSummary:
    """Types inferred from all return statements in one callable.

    Besides direct returned object types, the analyzer tracks collection element
    types and individual tuple/list slots. Slot facts make destructuring such as
    ``order, items = build()`` useful to later inference.
    """
    class_types: Set[str] = field(default_factory=set)
    element_types: Set[str] = field(default_factory=set)
    slot_types: Dict[int, Set[str]] = field(default_factory=dict)
    slot_element_types: Dict[int, Set[str]] = field(default_factory=dict)


@dataclass
class FunctionParamSummary:
    """Object types observed at a callable's positional and named parameters."""
    positional_types: Dict[int, Set[str]] = field(default_factory=dict)
    named_types: Dict[str, Set[str]] = field(default_factory=dict)


def add_types(target: Dict[str, Set[str]], key: str, types: Set[str]) -> None:
    """Union non-empty type facts into a string-keyed inference map."""
    if not types:
        return
    target.setdefault(key, set()).update(types)


def add_indexed_types(target: Dict[int, Set[str]], key: int, types: Set[str]) -> None:
    """Union non-empty type facts into an integer-keyed inference map."""
    if not types:
        return
    target.setdefault(key, set()).update(types)


def copy_type_map(source: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    return {key: set(values) for key, values in source.items()}


def copy_indexed_type_map(source: Dict[int, Set[str]]) -> Dict[int, Set[str]]:
    return {key: set(values) for key, values in source.items()}


def copy_return_summaries(
    summaries: Dict[str, FunctionReturnSummary],
) -> Dict[str, FunctionReturnSummary]:
    return {
        key: FunctionReturnSummary(
            class_types=set(value.class_types),
            element_types=set(value.element_types),
            slot_types=copy_indexed_type_map(value.slot_types),
            slot_element_types=copy_indexed_type_map(value.slot_element_types),
        )
        for key, value in summaries.items()
    }


def copy_param_summaries(
    summaries: Dict[str, FunctionParamSummary],
) -> Dict[str, FunctionParamSummary]:
    return {
        key: FunctionParamSummary(
            positional_types=copy_indexed_type_map(value.positional_types),
            named_types=copy_type_map(value.named_types),
        )
        for key, value in summaries.items()
    }


def copy_class_attr_types(
    class_attr_types: Dict[Tuple[str, str], Set[str]],
) -> Dict[Tuple[str, str], Set[str]]:
    return {key: set(values) for key, values in class_attr_types.items()}
