"""Shared exclusion helpers for structural clustering and notebook overlays."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Tuple


CALLABLE_PREFIX = "callable:"
DATA_PREFIX = "data:"
LOCAL_CALLABLE_MARKER = ".<locals>."


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def edge_value(edge: Any, key: str, default: Any = "") -> Any:
    if isinstance(edge, Mapping):
        return edge.get(key, default)
    return getattr(edge, key, default)


def node_type(node_id: str, row: Mapping[str, Any] | None = None) -> str:
    if row:
        current = text(row.get("node_type"))
        if current:
            return current
    if node_id.startswith(CALLABLE_PREFIX):
        return "callable"
    if node_id.startswith(DATA_PREFIX):
        return "data"
    return ""


def node_label(node_id: str, row: Mapping[str, Any] | None = None) -> str:
    if row:
        for key in ("display_name", "label", "qualname"):
            value = text(row.get(key))
            if value:
                return value
    return node_id


def is_data_node(
    node_id: str,
    nodes: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    row = nodes.get(node_id, {}) if nodes is not None else None
    return node_type(node_id, row) == "data"


def is_callable_node(
    node_id: str,
    nodes: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    row = nodes.get(node_id, {}) if nodes is not None else None
    return node_type(node_id, row) == "callable"


def build_exclusion_row(
    node_id: str,
    row: Mapping[str, Any],
    reason: str,
    degree_map: Mapping[str, Mapping[str, Any]],
) -> dict:
    return {
        "node": node_id,
        "node_type": node_type(node_id, row),
        "reason": reason,
        "label": node_label(node_id, row),
        "kind": text(row.get("kind")),
        "total_degree": int_value(degree_map.get(node_id, {}).get("total_degree")),
        "weighted_degree": f"{float_value(degree_map.get(node_id, {}).get('weighted_degree')):.6f}",
        "file": text(row.get("file")),
        "lineno": text(row.get("lineno")),
    }


def build_orphaned_data_exclusions(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Any],
    degree_map: Mapping[str, Mapping[str, Any]],
    excluded: Set[str],
    dropped_callables: Set[str],
    *,
    reason: str = "orphaned_by_callable_hub",
) -> Tuple[Set[str], list[dict]]:
    """Return data nodes with no active incident edges after callable removal."""
    if not dropped_callables:
        return set(), []

    active_nodes = set(nodes) - excluded
    active_degree: Counter[str] = Counter()
    data_touched_by_dropped_callable: Set[str] = set()

    for edge in edges:
        src = text(edge_value(edge, "src"))
        dst = text(edge_value(edge, "dst"))
        if src in dropped_callables and dst in active_nodes and is_data_node(dst, nodes):
            data_touched_by_dropped_callable.add(dst)
        if dst in dropped_callables and src in active_nodes and is_data_node(src, nodes):
            data_touched_by_dropped_callable.add(src)

        if src in active_nodes and dst in active_nodes:
            active_degree[src] += 1
            active_degree[dst] += 1

    orphaned = {
        node_id
        for node_id in data_touched_by_dropped_callable
        if node_id in active_nodes and active_degree[node_id] == 0
    }
    rows = [
        build_exclusion_row(
            node_id,
            nodes.get(node_id, {}),
            reason,
            degree_map,
        )
        for node_id in sorted(orphaned)
    ]
    return orphaned, rows


def orphaned_data_without_remaining_callables(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Any],
    excluded_callables: Set[str],
) -> Set[str]:
    """Return data nodes touched by excluded callables and no kept callable."""
    if not excluded_callables:
        return set()

    active_nodes = set(nodes) - excluded_callables
    remaining_callables = {
        node_id for node_id in active_nodes if is_callable_node(node_id, nodes)
    }
    data_touched_by_excluded: Set[str] = set()
    data_touched_by_remaining_callable: Set[str] = set()

    for edge in edges:
        src = text(edge_value(edge, "src"))
        dst = text(edge_value(edge, "dst"))

        if src in excluded_callables and dst in active_nodes and is_data_node(dst, nodes):
            data_touched_by_excluded.add(dst)
        if dst in excluded_callables and src in active_nodes and is_data_node(src, nodes):
            data_touched_by_excluded.add(src)

        if src in remaining_callables and dst in active_nodes and is_data_node(dst, nodes):
            data_touched_by_remaining_callable.add(dst)
        if dst in remaining_callables and src in active_nodes and is_data_node(src, nodes):
            data_touched_by_remaining_callable.add(src)

    return {
        node_id
        for node_id in data_touched_by_excluded
        if node_id in active_nodes and node_id not in data_touched_by_remaining_callable
    }


def local_callable_parent_id(node_id: str, available_nodes: Set[str]) -> Optional[str]:
    if not node_id.startswith(CALLABLE_PREFIX) or LOCAL_CALLABLE_MARKER not in node_id:
        return None

    callable_body = node_id[len(CALLABLE_PREFIX) :]
    parent_body = callable_body.rsplit(LOCAL_CALLABLE_MARKER, 1)[0]
    while parent_body:
        parent_id = f"{CALLABLE_PREFIX}{parent_body}"
        if parent_id in available_nodes and parent_id != node_id:
            return parent_id
        if LOCAL_CALLABLE_MARKER not in parent_body:
            break
        parent_body = parent_body.rsplit(LOCAL_CALLABLE_MARKER, 1)[0]
    return None


def enclosing_callable_ids(node_id: str, available_nodes: Set[str]) -> Tuple[str, ...]:
    parents: list[str] = []
    current = node_id
    while True:
        parent_id = local_callable_parent_id(current, available_nodes)
        if not parent_id or parent_id in parents:
            break
        parents.append(parent_id)
        current = parent_id
    return tuple(parents)
