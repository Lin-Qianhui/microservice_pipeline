"""Shared helpers for structural dependency graph clustering algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Set, Tuple


@dataclass(frozen=True)
class StructuralClusteringInput:
    supernodes: Set[str]
    directed_edges: Mapping[Tuple[str, str], float]
    undirected_edges: Mapping[Tuple[str, str], float]
    members_of: Mapping[str, List[str]]
    seed: int
    resolution: float
    markov_time: float
    max_iter: int
    leiden_quality: str
    hac_n_clusters: int = 13
    typed_undirected_edges: Mapping[str, Mapping[Tuple[str, str], float]] = field(
        default_factory=dict
    )
    edge_type_layer_weights: Mapping[str, float] = field(default_factory=dict)
    edge_type_layer_resolutions: Mapping[str, float] = field(default_factory=dict)


def ordered_cluster_map(
    groups: Iterable[Iterable[str]],
    members_of: Mapping[str, List[str]] | None = None,
) -> Dict[str, str]:
    materialized: List[List[str]] = []
    for group in groups:
        members = sorted(group)
        if members:
            materialized.append(members)

    def expanded_members(group: List[str]) -> List[str]:
        if members_of is None:
            return group
        expanded: List[str] = []
        for node in group:
            expanded.extend(members_of.get(node, [node]))
        return sorted(expanded)

    materialized.sort(key=lambda group: (-len(expanded_members(group)), min(expanded_members(group))))
    cluster_of: Dict[str, str] = {}
    for idx, group in enumerate(materialized, start=1):
        cluster_id = f"C{idx:03d}"
        for node in group:
            cluster_of[node] = cluster_id
    return cluster_of
