"""Syntactic predicates over AST nodes, and one de-duplication helper.

Everything here answers a question about the *shape* of an expression, using
nothing but the node itself. No collector state is consulted, so these are free
functions rather than methods -- they are the parts of the visitor that are
genuinely testable in isolation.
"""

from __future__ import annotations

import ast
from typing import Iterable, List, Optional, Set, Tuple

from ..ast_utils import attribute_to_name, unwrap_passthrough


def unique_callee_results(
    values: Iterable[Tuple[str, str, bool]]
) -> List[Tuple[str, str, bool]]:
    """De-duplicate resolution results, preserving first-seen order.

    Order is preserved rather than sorted because callers have already sorted
    the inputs they care about, and edge output must stay deterministic.
    """
    seen: Set[Tuple[str, str, bool]] = set()
    ordered: List[Tuple[str, str, bool]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def call_is_empty_container(node: ast.AST) -> bool:
    """``list()``, ``set()``, ``tuple()`` with no arguments."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "set", "tuple"}
        and not node.args
    )


def is_container_literal(node: ast.AST) -> bool:
    """An expression that plainly constructs a container."""
    return isinstance(
        node, (ast.List, ast.Set, ast.Tuple, ast.ListComp, ast.SetComp)
    ) or call_is_empty_container(node)


def is_dict_items_call(value: ast.AST) -> bool:
    """``d.items()`` -- the source of the ``(key, value)`` tuple shape."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "items"
        and not value.args
    )


def is_copy_call(node: ast.Call) -> bool:
    """A copy that preserves the argument's type, so types flow through it."""
    fn_name = attribute_to_name(node.func)
    return fn_name in {"copy", "deepcopy", "copy.copy", "copy.deepcopy"}


def is_super_call(node: ast.AST) -> bool:
    """``super()`` -- the receiver whose type is the *lexical* class, not the value's."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )


def annotation_head(node: ast.AST) -> str:
    """The rightmost name in an annotation, e.g. ``t.Optional`` -> ``Optional``."""
    name = attribute_to_name(node) or ""
    return name.rsplit(".", 1)[-1]


def container_mutation_key(node: ast.Call) -> Optional[ast.AST]:
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
