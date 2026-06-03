"""Label propagation clustering for structural dependency graph supernodes."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Mapping, Set, Tuple

from .common import StructuralClusteringInput, ordered_cluster_map


def cluster_label_propagation(
    nodes: Set[str],
    undirected_edges: Mapping[Tuple[str, str], float],
    max_iter: int,
    seed: int,
    members_of: Mapping[str, List[str]],
) -> Dict[str, str]:
    adjacency: Dict[str, Dict[str, float]] = {node: {} for node in nodes}
    for (src, dst), weight in undirected_edges.items():
        adjacency.setdefault(src, {})[dst] = adjacency.setdefault(src, {}).get(dst, 0.0) + weight
        adjacency.setdefault(dst, {})[src] = adjacency.setdefault(dst, {}).get(src, 0.0) + weight

    rng = random.Random(seed)
    labels = {node: node for node in nodes}
    node_list = list(nodes)
    for _ in range(max_iter):
        changed = False
        rng.shuffle(node_list)
        for node in node_list:
            scores: Dict[str, float] = defaultdict(float)
            for neighbor, weight in adjacency.get(node, {}).items():
                scores[labels[neighbor]] += weight
            if not scores:
                continue
            best_score = max(scores.values())
            best_labels = sorted(label for label, score in scores.items() if score == best_score)
            chosen = best_labels[0]
            if labels[node] != chosen:
                labels[node] = chosen
                changed = True
        if not changed:
            break

    groups: Dict[str, List[str]] = defaultdict(list)
    for node, label in labels.items():
        groups[label].append(node)
    return ordered_cluster_map(groups.values(), members_of)


def cluster(input_data: StructuralClusteringInput) -> Dict[str, str]:
    return cluster_label_propagation(
        nodes=input_data.supernodes,
        undirected_edges=input_data.undirected_edges,
        max_iter=input_data.max_iter,
        seed=input_data.seed,
        members_of=input_data.members_of,
    )
