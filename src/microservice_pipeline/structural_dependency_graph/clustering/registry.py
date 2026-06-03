"""Registry for structural dependency graph clustering algorithms."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from .common import StructuralClusteringInput
from .hac_callable_projection import cluster as cluster_hac_callable_projection
from .infomap import cluster as cluster_infomap
from .label_propagation import cluster as cluster_label_propagation
from .leiden import cluster as cluster_leiden
from .leiden_multiplex import cluster as cluster_leiden_multiplex
from .leiden_reweighted import cluster as cluster_leiden_reweighted


Algorithm = Callable[[StructuralClusteringInput], Dict[str, str]]

_ALGORITHMS: Dict[str, Algorithm] = {
    "leiden": cluster_leiden,
    "leiden_reweighted": cluster_leiden_reweighted,
    "leiden_multiplex": cluster_leiden_multiplex,
    "infomap": cluster_infomap,
    "label_propagation": cluster_label_propagation,
    "hac_callable_projection": cluster_hac_callable_projection,
}


def algorithm_choices() -> Tuple[str, ...]:
    return tuple(_ALGORITHMS)


def cluster_with_algorithm(
    algorithm: str,
    input_data: StructuralClusteringInput,
) -> Dict[str, str]:
    try:
        cluster = _ALGORITHMS[algorithm]
    except KeyError as exc:
        raise ValueError(f"Unsupported algorithm: {algorithm}") from exc
    return cluster(input_data)
