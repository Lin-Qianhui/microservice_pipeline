"""Leiden clustering for structural dependency graph supernodes."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Set, Tuple

from .common import StructuralClusteringInput, ordered_cluster_map


def cluster_leiden(
    nodes: Set[str],
    undirected_edges: Mapping[Tuple[str, str], float],
    leiden_quality: str,
    resolution: float,
    seed: int,
    members_of: Mapping[str, List[str]],
) -> Dict[str, str]:
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Leiden clustering requires optional dependencies 'igraph' and 'leidenalg'. "
            "Run with the project .venv or install them before using --algorithm leiden."
        ) from exc

    node_list = sorted(nodes)
    if not node_list:
        return {}
    if not undirected_edges:
        return ordered_cluster_map(([node] for node in node_list), members_of)

    node_index = {node: idx for idx, node in enumerate(node_list)}
    graph_edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for (src, dst), weight in sorted(undirected_edges.items()):
        if src not in node_index or dst not in node_index:
            continue
        graph_edges.append((node_index[src], node_index[dst]))
        weights.append(weight)

    graph = ig.Graph(n=len(node_list), edges=graph_edges, directed=False)
    if weights:
        graph.es["weight"] = weights

    if leiden_quality == "rb_configuration":
        partition_type = leidenalg.RBConfigurationVertexPartition
    elif leiden_quality == "cpm":
        partition_type = leidenalg.CPMVertexPartition
    else:
        raise ValueError(f"Unsupported Leiden quality function: {leiden_quality}")

    partition = leidenalg.find_partition(
        graph,
        partition_type,
        weights=weights if weights else None,
        seed=seed,
        resolution_parameter=resolution,
    )

    groups: Dict[int, List[str]] = defaultdict(list)
    for node, membership in zip(node_list, partition.membership):
        groups[int(membership)].append(node)
    return ordered_cluster_map(groups.values(), members_of)


def cluster(input_data: StructuralClusteringInput) -> Dict[str, str]:
    return cluster_leiden(
        nodes=input_data.supernodes,
        undirected_edges=input_data.undirected_edges,
        leiden_quality=input_data.leiden_quality,
        resolution=input_data.resolution,
        seed=input_data.seed,
        members_of=input_data.members_of,
    )
