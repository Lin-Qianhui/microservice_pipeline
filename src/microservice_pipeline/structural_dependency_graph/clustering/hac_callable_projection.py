"""Callable-projection HAC clustering for structural dependency graphs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Set, Tuple

from .common import StructuralClusteringInput, ordered_cluster_map


ALGORITHM = "hac_callable_projection"
CALLABLE_PREFIX = "callable:"
DATA_PREFIX = "data:"


def cluster_hac_callable_projection(
    nodes: Set[str],
    typed_undirected_edges: Mapping[str, Mapping[Tuple[str, str], float]],
    members_of: Mapping[str, List[str]],
    edge_type_layer_weights: Mapping[str, float],
    n_clusters: int,
) -> Dict[str, str]:
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HAC callable projection requires optional dependency 'scikit-learn'. "
            "Install project dependencies before using --algorithm hac_callable_projection."
        ) from exc

    callable_nodes = _callable_supernodes(nodes, members_of)
    if not callable_nodes:
        return {}

    if len(callable_nodes) <= max(1, int(n_clusters)):
        return ordered_cluster_map(([node] for node in callable_nodes), members_of)

    distance_matrix = callable_distance_matrix(
        callable_nodes=callable_nodes,
        typed_undirected_edges=typed_undirected_edges,
        members_of=members_of,
        edge_type_layer_weights=edge_type_layer_weights,
    )
    model = _agglomerative_model(
        n_clusters=max(1, int(n_clusters)),
        agglomerative_cls=AgglomerativeClustering,
    )
    labels = list(model.fit_predict(distance_matrix))

    groups: Dict[int, List[str]] = defaultdict(list)
    for node, label in zip(callable_nodes, labels):
        groups[int(label)].append(node)
    return ordered_cluster_map(groups.values(), members_of)


def callable_distance_matrix(
    callable_nodes: List[str],
    typed_undirected_edges: Mapping[str, Mapping[Tuple[str, str], float]],
    members_of: Mapping[str, List[str]],
    edge_type_layer_weights: Mapping[str, float],
) -> List[List[float]]:
    node_set = set(callable_nodes)
    call_features = _call_features(
        node_set,
        typed_undirected_edges.get("call", {}),
        members_of,
    )
    data_features = _data_access_features(
        node_set,
        typed_undirected_edges.get("data_access", {}),
        members_of,
    )
    call_weight = max(0.0, float(edge_type_layer_weights.get("call", 1.0)))
    data_weight = max(0.0, float(edge_type_layer_weights.get("data_access", 1.0)))
    denominator = call_weight + data_weight
    if denominator <= 0:
        call_weight = data_weight = 1.0
        denominator = 2.0

    matrix: List[List[float]] = []
    for left in callable_nodes:
        row: List[float] = []
        for right in callable_nodes:
            if left == right:
                row.append(0.0)
                continue
            call_similarity = _cosine(call_features.get(left, {}), call_features.get(right, {}))
            data_similarity = _cosine(data_features.get(left, {}), data_features.get(right, {}))
            similarity = (
                call_weight * call_similarity + data_weight * data_similarity
            ) / denominator
            row.append(1.0 - min(1.0, max(0.0, similarity)))
        matrix.append(row)
    return matrix


def _agglomerative_model(
    n_clusters: int,
    agglomerative_cls: object,
) -> object:
    try:
        return agglomerative_cls(  # type: ignore[misc]
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        return agglomerative_cls(  # type: ignore[misc]
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )


def _callable_supernodes(
    nodes: Set[str],
    members_of: Mapping[str, List[str]],
) -> List[str]:
    return sorted(
        node
        for node in nodes
        if any(member.startswith(CALLABLE_PREFIX) for member in members_of.get(node, [node]))
    )


def _call_features(
    callable_nodes: Set[str],
    edges: Mapping[Tuple[str, str], float],
    members_of: Mapping[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    features: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (left, right), weight in sorted(edges.items()):
        if left not in callable_nodes or right not in callable_nodes:
            continue
        value = max(0.0, float(weight))
        if value <= 0:
            continue
        edge_feature = f"call_edge:{left}|{right}"
        features[left][edge_feature] += value
        features[right][edge_feature] += value
        features[left][f"call_neighbor:{right}"] += value
        features[right][f"call_neighbor:{left}"] += value
        for member in members_of.get(right, [right]):
            features[left][f"call_target_member:{member}"] += value
        for member in members_of.get(left, [left]):
            features[right][f"call_target_member:{member}"] += value
    return {node: dict(values) for node, values in features.items()}


def _data_access_features(
    callable_nodes: Set[str],
    edges: Mapping[Tuple[str, str], float],
    members_of: Mapping[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    features: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (left, right), weight in sorted(edges.items()):
        callable_node = ""
        data_node = ""
        if left in callable_nodes and _is_data_supernode(right, members_of):
            callable_node = left
            data_node = right
        elif right in callable_nodes and _is_data_supernode(left, members_of):
            callable_node = right
            data_node = left
        if not callable_node or not data_node:
            continue
        value = max(0.0, float(weight))
        if value <= 0:
            continue
        features[callable_node][f"data:{data_node}"] += value
    return {node: dict(values) for node, values in features.items()}


def _is_data_supernode(node: str, members_of: Mapping[str, List[str]]) -> bool:
    members = members_of.get(node, [node])
    return bool(members) and all(member.startswith(DATA_PREFIX) for member in members)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def cluster(input_data: StructuralClusteringInput) -> Dict[str, str]:
    return cluster_hac_callable_projection(
        nodes=input_data.supernodes,
        typed_undirected_edges=input_data.typed_undirected_edges,
        members_of=input_data.members_of,
        edge_type_layer_weights=input_data.edge_type_layer_weights,
        n_clusters=input_data.hac_n_clusters,
    )
