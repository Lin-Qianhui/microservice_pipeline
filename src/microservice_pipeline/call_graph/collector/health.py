"""What each call site cost the resolver.

``callee_shape`` buckets an unresolved callee by syntactic form and
``note_call_health`` charges the outcome to a ``CallGraphHealth``. They live
together because the bucket is only meaningful alongside the accounting that
uses it: the counts are how you tell which language feature is costing edges.

Health counters do not affect the edge list, so a regression here is invisible
in ``edges.csv`` -- ``call_graph_health.json`` is the only artifact that catches
it.
"""

from __future__ import annotations

import ast
from typing import Optional, Sequence, Tuple

from ..models import CallGraphHealth


def callee_shape(func: ast.AST) -> str:
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


def note_call_health(
    health: CallGraphHealth,
    current_callable: Optional[str],
    node: ast.Call,
    results: Sequence[Tuple[str, str, bool]],
) -> None:
    """Record what this call site cost the resolver, whatever the outcome."""
    if current_callable is None:
        return

    resolved = [result for result in results if result[2]]
    if resolved:
        health.site_fanout[len(resolved)] += 1
        return

    # Calls to a value the abstract domain cannot hold. Two shapes reach
    # here: a bound local whose contents are a function (``callable_value``),
    # and an expression the resolver produced no candidate for at all. Both
    # are the same underlying gap -- the lattice is a set of class ids, so a
    # value that *is* code is inexpressible rather than merely unknown.
    if any(relation == "callable_value" for _callee, relation, _ in results):
        health.unresolvable_calls[callee_shape(node.func)] += 1
        return

    if results:
        # A named but unmatched callee. Already counted by ``_add_edge``.
        return

    health.unresolvable_calls[callee_shape(node.func)] += 1
