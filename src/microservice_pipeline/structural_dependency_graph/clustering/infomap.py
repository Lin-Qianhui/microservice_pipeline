"""Infomap clustering for structural dependency graph supernodes."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Set, Tuple

from .common import StructuralClusteringInput, ordered_cluster_map


def cluster_infomap(
    nodes: Set[str],
    directed_edges: Mapping[Tuple[str, str], float],
    seed: int,
    markov_time: float,
    members_of: Mapping[str, List[str]],
) -> Dict[str, str]:
    try:
        from infomap import Infomap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Infomap clustering requires optional dependency 'infomap'. "
            "Run with the project .venv or install it before using --algorithm infomap."
        ) from exc

    node_list = sorted(nodes)
    if not node_list:
        return {}
    if not directed_edges:
        return ordered_cluster_map(([node] for node in node_list), members_of)

    node_index = {node: idx for idx, node in enumerate(node_list, start=1)}
    reverse_index = {idx: node for node, idx in node_index.items()}
    infomap = Infomap(
        two_level=True,
        directed=True,
        silent=True,
        seed=seed,
        markov_time=markov_time,
    )
    for (src, dst), weight in sorted(directed_edges.items()):
        if src in node_index and dst in node_index:
            infomap.add_link(node_index[src], node_index[dst], weight)
    infomap.run()

    groups: Dict[int, List[str]] = defaultdict(list)
    for tree_node in infomap.tree:
        if not getattr(tree_node, "is_leaf", False):
            continue
        node_id = getattr(tree_node, "physical_id", None)
        if node_id is None:
            node_id = getattr(tree_node, "node_id", None)
        if node_id in reverse_index:
            groups[int(tree_node.module_id)].append(reverse_index[node_id])

    assigned = {node for group in groups.values() for node in group}
    next_group = (max(groups) + 1) if groups else 1
    for node in node_list:
        if node not in assigned:
            groups[next_group] = [node]
            next_group += 1

    return ordered_cluster_map(groups.values(), members_of)


def cluster(input_data: StructuralClusteringInput) -> Dict[str, str]:
    return cluster_infomap(
        nodes=input_data.supernodes,
        directed_edges=input_data.directed_edges,
        seed=input_data.seed,
        markov_time=input_data.markov_time,
        members_of=input_data.members_of,
    )
