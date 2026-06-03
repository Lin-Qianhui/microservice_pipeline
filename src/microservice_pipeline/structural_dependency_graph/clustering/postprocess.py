"""Post-processing helpers for structural clustering assignments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CALLABLE_PREFIX = "callable:"
DATA_PREFIX = "data:"
MUTATING_DATA_ACCESSES = frozenset({"create", "write", "read_write"})
ACCESS_PRIORITY = {"create": 0, "write": 1, "read_write": 2}


@dataclass(frozen=True)
class DataOnlyClusterPostprocessResult:
    cluster_of: Dict[str, str]
    reassigned_nodes: Dict[str, str]
    removed_nodes: Tuple[str, ...]


def _edge_value(edge: Any, key: str) -> Any:
    if isinstance(edge, Mapping):
        return edge.get(key)
    return getattr(edge, key, None)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_callable_node(node_id: str) -> bool:
    return node_id.startswith(CALLABLE_PREFIX)


def _is_data_node(node_id: str) -> bool:
    return node_id.startswith(DATA_PREFIX)


def _data_access_endpoint(edge: Any) -> Tuple[str, str]:
    src = _text(_edge_value(edge, "src"))
    dst = _text(_edge_value(edge, "dst"))
    if _is_callable_node(src) and _is_data_node(dst):
        return src, dst
    if _is_callable_node(dst) and _is_data_node(src):
        return dst, src
    return "", ""


def _groups_by_cluster(cluster_of: Mapping[str, str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for node_id, cluster_id in cluster_of.items():
        groups[cluster_id].append(node_id)
    return groups


def _mutating_edges_by_data(
    edges: Iterable[Any],
    clustered_nodes: Mapping[str, str],
) -> Dict[str, List[Tuple[str, str, float]]]:
    by_data: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    for edge in edges:
        if _text(_edge_value(edge, "edge_type")) != "data_access":
            continue
        access = _text(_edge_value(edge, "access"))
        if access not in MUTATING_DATA_ACCESSES:
            continue
        callable_id, data_id = _data_access_endpoint(edge)
        if not callable_id or not data_id:
            continue
        if callable_id not in clustered_nodes or data_id not in clustered_nodes:
            continue
        by_data[data_id].append((callable_id, access, _float(_edge_value(edge, "weight"), 1.0)))
    return by_data


def _best_mutating_callable(candidates: Sequence[Tuple[str, str, float]]) -> str:
    callable_id, _access, _weight = sorted(
        candidates,
        key=lambda item: (
            -item[2],
            ACCESS_PRIORITY.get(item[1], 99),
            item[0],
        ),
    )[0]
    return callable_id


def postprocess_data_only_clusters(
    cluster_of: Mapping[str, str],
    edges: Iterable[Any],
) -> DataOnlyClusterPostprocessResult:
    """Attach or remove data-only clusters after graph clustering.

    Data-only clusters are not useful service candidates by themselves. If a
    data node has an active callable that creates, writes, or read-writes it,
    assign that data node to the strongest such callable's cluster. If it only
    has read evidence, remove it from the cluster result.
    """
    updated = dict(cluster_of)
    groups = _groups_by_cluster(updated)
    data_only_nodes = {
        node_id
        for members in groups.values()
        if members and all(_is_data_node(node_id) for node_id in members)
        for node_id in members
    }
    if not data_only_nodes:
        return DataOnlyClusterPostprocessResult(updated, {}, tuple())

    mutating_by_data = _mutating_edges_by_data(edges, updated)
    reassigned: Dict[str, str] = {}
    removed: List[str] = []

    for data_id in sorted(data_only_nodes):
        candidates = mutating_by_data.get(data_id, [])
        if candidates:
            callable_id = _best_mutating_callable(candidates)
            updated[data_id] = updated[callable_id]
            reassigned[data_id] = callable_id
        else:
            updated.pop(data_id, None)
            removed.append(data_id)

    return DataOnlyClusterPostprocessResult(updated, reassigned, tuple(removed))
