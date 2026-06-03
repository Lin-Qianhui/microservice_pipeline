"""Notebook-scoped pruning for refined structural cluster assignments."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from microservice_pipeline.structural_dependency_graph.clustering.exclusions import (
        enclosing_callable_ids,
        int_value,
        node_label,
        node_type,
        orphaned_data_without_remaining_callables,
        text,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from structural_dependency_graph.clustering.exclusions import (  # type: ignore
        enclosing_callable_ids,
        int_value,
        node_label,
        node_type,
        orphaned_data_without_remaining_callables,
        text,
    )


NOTEBOOK_UNOBSERVED_EXCLUDED_FIELDS = [
    "node",
    "node_type",
    "reason",
    "label",
    "kind",
    "cluster_id",
    "refined_cluster_id",
    "in_degree",
    "out_degree",
    "total_degree",
    "file",
    "lineno",
]

CALLABLE_PRUNE_REASON = "notebook_unobserved_zero_in_degree_callable"
ORPHAN_DATA_PRUNE_REASON = "orphaned_by_notebook_unobserved_callable"


@dataclass(frozen=True)
class NotebookPruningResult:
    refined_rows: list[dict[str, Any]]
    excluded_rows: list[dict[str, Any]]
    pruned_callable_count: int
    pruned_data_count: int


def _notebook_used_callables(usage_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        text(row.get("callable_node"))
        for row in usage_rows
        if text(row.get("resolved")) == "1" and text(row.get("callable_node"))
    }


def _is_dunder_callable(row: Mapping[str, Any]) -> bool:
    values = [
        text(row.get("label")),
        text(row.get("display_name")),
        text(row.get("qualname")),
        text(row.get("node")).removeprefix("callable:"),
    ]
    for value in values:
        leaf = value.rsplit(".", 1)[-1]
        if re.fullmatch(r"__[A-Za-z0-9_]+__", leaf):
            return True
    return False


def _has_notebook_used_enclosing_callable(
    node_id: str,
    available_nodes: set[str],
    notebook_used: set[str],
) -> bool:
    return any(parent_id in notebook_used for parent_id in enclosing_callable_ids(node_id, available_nodes))


def _excluded_row(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    node_id = text(row.get("node"))
    return {
        "node": node_id,
        "node_type": node_type(node_id, row),
        "reason": reason,
        "label": node_label(node_id, row),
        "kind": text(row.get("kind")),
        "cluster_id": text(row.get("cluster_id")),
        "refined_cluster_id": text(row.get("refined_cluster_id")),
        "in_degree": int_value(row.get("in_degree")),
        "out_degree": int_value(row.get("out_degree")),
        "total_degree": int_value(row.get("total_degree")),
        "file": text(row.get("file")),
        "lineno": text(row.get("lineno")),
    }


def _with_recomputed_cluster_sizes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sizes = Counter(text(row.get("cluster_id")) for row in rows)
    updated: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        cluster_id = text(next_row.get("cluster_id"))
        next_row["cluster_size"] = sizes[cluster_id]
        if "refined_cluster_size" in next_row:
            next_row["refined_cluster_size"] = sizes[cluster_id]
        updated.append(next_row)
    return updated


def prune_notebook_unobserved_assignments(
    *,
    refined_rows: Sequence[Mapping[str, Any]],
    structural_edges: Sequence[Mapping[str, Any]],
    usage_rows: Sequence[Mapping[str, Any]],
) -> NotebookPruningResult:
    """Remove notebook-unobserved zero-in-degree callables and orphan data."""
    rows_by_node = {
        text(row.get("node")): dict(row)
        for row in refined_rows
        if text(row.get("node"))
    }
    available_nodes = set(rows_by_node)
    notebook_used = _notebook_used_callables(usage_rows)

    pruned_callables = {
        node_id
        for node_id, row in rows_by_node.items()
        if node_type(node_id, row) == "callable"
        and int_value(row.get("in_degree")) == 0
        and node_id not in notebook_used
        and not _has_notebook_used_enclosing_callable(node_id, available_nodes, notebook_used)
        and not _is_dunder_callable(row)
    }
    pruned_data = orphaned_data_without_remaining_callables(
        rows_by_node,
        structural_edges,
        pruned_callables,
    )
    pruned_nodes = pruned_callables | pruned_data

    kept_rows = [
        dict(row)
        for row in refined_rows
        if text(row.get("node")) and text(row.get("node")) not in pruned_nodes
    ]
    excluded_rows = [
        _excluded_row(rows_by_node[node_id], CALLABLE_PRUNE_REASON)
        for node_id in sorted(pruned_callables)
    ]
    excluded_rows.extend(
        _excluded_row(rows_by_node[node_id], ORPHAN_DATA_PRUNE_REASON)
        for node_id in sorted(pruned_data)
    )
    excluded_rows.sort(key=lambda row: (row["reason"], row["node"]))

    return NotebookPruningResult(
        refined_rows=_with_recomputed_cluster_sizes(kept_rows),
        excluded_rows=excluded_rows,
        pruned_callable_count=len(pruned_callables),
        pruned_data_count=len(pruned_data),
    )
