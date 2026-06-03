"""Artifact writers for the AST call graph generator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

try:
    from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import ensure_dir, write_csv_rows, write_json  # type: ignore

if TYPE_CHECKING:
    from microservice_pipeline.call_graph.generate_call_graph_ast import CallableDef, Edge


NODE_FIELDS = ["id", "module", "qualname", "file", "lineno", "kind", "class_name"]
EDGE_FIELDS = ["caller", "callee", "file", "lineno", "resolved", "relation"]


def node_rows(nodes: Dict[str, CallableDef]) -> list[dict]:
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
    return [
        {
            "caller": edge.caller,
            "callee": edge.callee,
            "file": edge.file,
            "lineno": edge.lineno,
            "resolved": int(edge.resolved),
            "relation": edge.relation,
        }
        for edge in sorted(edges, key=lambda item: (item.caller, item.callee, item.lineno))
    ]


def write_outputs(outdir: Path, nodes: Dict[str, CallableDef], edges: List[Edge]) -> None:
    ensure_dir(outdir)
    write_csv_rows(outdir / "nodes.csv", NODE_FIELDS, node_rows(nodes))
    write_csv_rows(outdir / "edges.csv", EDGE_FIELDS, edge_rows(edges))
    write_json(
        outdir / "call_graph.json",
        {
            "nodes": [node.__dict__ for node in sorted(nodes.values(), key=lambda item: item.id)],
            "edges": [edge.__dict__ for edge in sorted(edges, key=lambda item: (item.caller, item.callee, item.lineno))],
        },
    )
