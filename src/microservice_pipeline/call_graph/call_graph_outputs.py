"""Convert an in-memory call graph into stable, human-readable artifacts.

The AST analyzer builds two groups of Python objects:

* ``CallableDef`` objects are the graph's *nodes*. Each one represents a module,
  function, or method that was found in the analyzed source code.
* ``Edge`` objects are the graph's directed *edges*. Each one says that a caller
  invokes (or otherwise depends on) a callee at a particular source line.

This module deliberately contains no analysis logic. Its only job is to turn
those objects into CSV rows and JSON, then write the following files:

``nodes.csv``
    One row per discovered callable. This is convenient for spreadsheets and
    graph/database imports.
``edges.csv``
    One row per caller-to-callee relationship.
``call_graph.json``
    Both collections in one document, preserving native JSON booleans.
``call_graph_health.json``
    What the resolver could not do: calls dropped as unresolved or external,
    calls it produced no candidate for at all, and the fan-out of the sites it
    did resolve. Written only when the caller asks for it. The other files can
    describe only the edges that exist, which is why a count taken from them
    can never report a failure.

Rows are sorted before writing. Deterministic output makes generated artifacts
easy to diff in code review and prevents filesystem traversal order from
causing unrelated changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

try:
    from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import ensure_dir, write_csv_rows, write_json  # type: ignore

if TYPE_CHECKING:
    # These imports are needed only by static type checkers. Importing them at
    # runtime would create a cycle: the analyzer imports ``write_outputs`` from
    # this module, while this module would import the analyzer's data classes.
    from microservice_pipeline.call_graph.generate_call_graph_ast import (
        CallableDef,
        CallGraphHealth,
        Edge,
    )


# Column order is declared explicitly instead of relying on dictionary order.
# The shared CSV writer uses these lists as its header/schema.
#
# Node fields:
# - ``id``: Project-wide callable identifier used as the endpoint of an edge,
#   for example ``shop.orders.Order.submit`` or ``shop.orders.<module>``.
# - ``module``: Import-style name of the module containing the callable, such
#   as ``shop.orders``.
# - ``qualname``: Callable name relative to its module. Methods include their
#   class (``Order.submit``), and nested functions include ``<locals>``.
# - ``file``: Path of the Python source file in which the callable is defined.
# - ``lineno``: One-based source line on which its definition begins.
# - ``kind``: Definition category: ``module``, ``function``, ``method``, or
#   ``class_body`` for a class whose body itself runs code.
# - ``class_name``: Name of the containing class for a method; empty for
#   module-level code and functions that do not belong to a class.
NODE_FIELDS = ["id", "module", "qualname", "file", "lineno", "kind", "class_name"]

# Edge fields:
# - ``caller``: Callable ID of the code containing the call or dependency.
# - ``callee``: Callable ID (or best inferred external name) being invoked.
# - ``file``: Path of the source file containing the call expression.
# - ``lineno``: One-based line on which the call or implicit operation occurs.
# - ``col_offset``: Zero-based UTF-8 byte column of the same expression, or
#   ``-1`` when the edge has no single call expression behind it. Line and
#   column together identify the call *site*; a line alone does not, because
#   distinct calls routinely share one.
# - ``resolved``: Whether ``callee`` was matched to a callable indexed from the
#   analyzed project. CSV encodes this boolean as ``1`` (yes) or ``0`` (no).
# - ``confidence``: How sure the analyzer is the call really happens --
#   ``high``, ``medium``, ``low``, or ``unknown``. Graded by how many targets the
#   call site resolved to, and deliberately separate from ``relation``: two edges
#   can share a relation while one is the only possibility and the other is one
#   guess among five.
# - ``relation``: How the analyzer found the relationship, such as ``direct``,
#   ``imported``, ``constructor``, ``self_method``, ``property_getter``, or a
#   ``dunder_*`` relation for an implicit Python operation.
EDGE_FIELDS = [
    "caller",
    "callee",
    "file",
    "lineno",
    "col_offset",
    "resolved",
    "relation",
    "confidence",
]


def node_rows(nodes: Dict[str, CallableDef]) -> list[dict]:
    """Return node objects as CSV-ready dictionaries, sorted by stable ID.

    ``nodes`` is keyed by the same globally unique ID stored on each node (for
    example ``shop.orders.Order.submit``). A missing ``class_name`` is written
    as an empty string because that is clearer than the text ``None`` in CSV.
    """
    return [
        {
            "id": node.id,
            "module": node.module,
            "qualname": node.qualname,
            "file": node.file,
            "lineno": node.lineno,
            "kind": node.kind,
            "class_name": node.class_name or "",
        }
        for node in sorted(nodes.values(), key=lambda item: item.id)
    ]


def edge_rows(edges: List[Edge]) -> list[dict]:
    """Return edges as deterministic, CSV-ready dictionaries.

    CSV has no native boolean type, so ``resolved`` is encoded as ``1`` or
    ``0``. The JSON writer below keeps it as a true JSON boolean.
    """
    return [
        {
            "caller": edge.caller,
            "callee": edge.callee,
            "file": edge.file,
            "lineno": edge.lineno,
            "col_offset": edge.col_offset,
            "resolved": int(edge.resolved),
            "relation": edge.relation,
            "confidence": edge.confidence,
        }
        for edge in sorted(
            edges,
            key=lambda item: (item.caller, item.callee, item.lineno, item.col_offset),
        )
    ]


def write_outputs(
    outdir: Path,
    nodes: Dict[str, CallableDef],
    edges: List[Edge],
    health: Optional[CallGraphHealth] = None,
) -> None:
    """Write all call-graph artifacts beneath ``outdir``.

    The directory is created when needed. Existing files with the known output
    names are replaced by the shared artifact-writing helpers.

    ``health`` writes ``call_graph_health.json``, which is the only artifact
    reporting what the resolver *failed* to do. The other files can only
    describe edges that exist.
    """
    ensure_dir(outdir)

    # The CSV projections are intentionally flat and use the schemas above.
    write_csv_rows(outdir / "nodes.csv", NODE_FIELDS, node_rows(nodes))
    write_csv_rows(outdir / "edges.csv", EDGE_FIELDS, edge_rows(edges))

    # JSON can represent the dataclasses directly as dictionaries. Sorting here
    # mirrors the CSV order so all output formats are reproducible.
    write_json(
        outdir / "call_graph.json",
        {
            "nodes": [node.__dict__ for node in sorted(nodes.values(), key=lambda item: item.id)],
            "edges": [edge.__dict__ for edge in sorted(edges, key=lambda item: (item.caller, item.callee, item.lineno))],
        },
    )

    if health is not None:
        write_json(outdir / "call_graph_health.json", health.as_dict())
