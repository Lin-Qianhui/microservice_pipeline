"""Type-aware multiplex Leiden clustering for structural dependency graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from .common import StructuralClusteringInput, ordered_cluster_map


ALGORITHM = "leiden_multiplex"
EDGE_TYPE_LAYER_ORDER = ("call", "data_access", "data_lineage")


def cluster_leiden_multiplex(
    nodes: Set[str],
    typed_undirected_edges: Mapping[str, Mapping[Tuple[str, str], float]],
    edge_type_layer_weights: Mapping[str, float],
    edge_type_layer_resolutions: Mapping[str, float],
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
            "Multiplex Leiden clustering requires optional dependencies 'igraph' and 'leidenalg'. "
            "Run with the project .venv or install them before using --algorithm leiden_multiplex."
        ) from exc

    node_list = sorted(nodes)
    if not node_list:
        return {}

    edge_types = _ordered_edge_types(typed_undirected_edges)
    if not edge_types:
        return ordered_cluster_map(([node] for node in node_list), members_of)

    if leiden_quality == "rb_configuration":
        partition_type = leidenalg.RBConfigurationVertexPartition
    elif leiden_quality == "cpm":
        partition_type = leidenalg.CPMVertexPartition
    else:
        raise ValueError(f"Unsupported Leiden quality function: {leiden_quality}")

    node_index = {node: idx for idx, node in enumerate(node_list)}
    partitions: List[Any] = []
    layer_weights: List[float] = []
    for edge_type in edge_types:
        graph_edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        for (src, dst), weight in sorted(typed_undirected_edges[edge_type].items()):
            if src not in node_index or dst not in node_index:
                continue
            graph_edges.append((node_index[src], node_index[dst]))
            weights.append(weight)
        if not graph_edges:
            continue

        graph = ig.Graph(n=len(node_list), edges=graph_edges, directed=False)
        graph.es["weight"] = weights
        partitions.append(
            partition_type(
                graph,
                weights="weight",
                resolution_parameter=float(
                    edge_type_layer_resolutions.get(edge_type, resolution)
                ),
            )
        )
        layer_weights.append(float(edge_type_layer_weights.get(edge_type, 1.0)))

    if not partitions:
        return ordered_cluster_map(([node] for node in node_list), members_of)

    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(seed)
    optimiser.optimise_partition_multiplex(
        partitions,
        layer_weights=layer_weights,
        n_iterations=2,
    )
    membership = partitions[0].membership

    groups: Dict[int, List[str]] = defaultdict(list)
    for node, cluster_id in zip(node_list, membership):
        groups[int(cluster_id)].append(node)
    return ordered_cluster_map(groups.values(), members_of)


def multiplex_must_link_relations(
    weight_config: Mapping[str, Any],
    default_relations: Set[str],
) -> Set[str]:
    configured = _multiplex_setting(weight_config, "must_link_relations")
    if isinstance(configured, str):
        relations = [part.strip() for part in configured.split(",")]
    elif isinstance(configured, Sequence):
        relations = [_text(relation).strip() for relation in configured]
    else:
        relations = list(default_relations)
    cleaned = {relation for relation in relations if relation}
    return cleaned or set(default_relations)


def _ordered_edge_types(
    typed_undirected_edges: Mapping[str, Mapping[Tuple[str, str], float]],
) -> List[str]:
    known = [
        edge_type
        for edge_type in EDGE_TYPE_LAYER_ORDER
        if typed_undirected_edges.get(edge_type)
    ]
    extra = sorted(
        edge_type
        for edge_type, edges in typed_undirected_edges.items()
        if edge_type not in EDGE_TYPE_LAYER_ORDER and edges
    )
    return known + extra


def _multiplex_setting(weight_config: Mapping[str, Any], key: str) -> Any:
    clustering = _mapping(weight_config.get("clustering"))
    multiplex = _mapping(clustering.get("multiplex"))
    return multiplex.get(key)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def cluster(input_data: StructuralClusteringInput) -> Dict[str, str]:
    typed_undirected_edges = input_data.typed_undirected_edges
    if not typed_undirected_edges and input_data.undirected_edges:
        typed_undirected_edges = {"aggregate": input_data.undirected_edges}
    return cluster_leiden_multiplex(
        nodes=input_data.supernodes,
        typed_undirected_edges=typed_undirected_edges,
        edge_type_layer_weights=input_data.edge_type_layer_weights,
        edge_type_layer_resolutions=input_data.edge_type_layer_resolutions,
        leiden_quality=input_data.leiden_quality,
        resolution=input_data.resolution,
        seed=input_data.seed,
        members_of=input_data.members_of,
    )
