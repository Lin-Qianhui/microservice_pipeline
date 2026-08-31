#!/usr/bin/env python3
"""Cluster the heterogeneous structural dependency graph.

This script turns the structural graph produced by
``microservice-pipeline structural-graph`` into microservice candidate
clusters containing both callables and service-relevant data objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown
    from microservice_pipeline.jsonc_config import load_jsonc
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown  # type: ignore
    from jsonc_config import load_jsonc  # type: ignore

try:
    from microservice_pipeline.structural_dependency_graph.clustering.common import (
        StructuralClusteringInput,
        ordered_cluster_map as _ordered_cluster_map,
    )
    from microservice_pipeline.structural_dependency_graph.clustering.hac_callable_projection import (
        ALGORITHM as HAC_CALLABLE_PROJECTION_ALGORITHM,
    )
    from microservice_pipeline.structural_dependency_graph.clustering.infomap import cluster_infomap
    from microservice_pipeline.structural_dependency_graph.clustering.label_propagation import cluster_label_propagation
    from microservice_pipeline.structural_dependency_graph.clustering.leiden import cluster_leiden
    from microservice_pipeline.structural_dependency_graph.clustering.leiden_multiplex import (
        ALGORITHM as LEIDEN_MULTIPLEX_ALGORITHM,
        multiplex_must_link_relations,
    )
    from microservice_pipeline.structural_dependency_graph.clustering.leiden_reweighted import (
        ALGORITHM as LEIDEN_REWEIGHTED_ALGORITHM,
        reweighted_edge_weights,
        reweighted_must_link_relations,
        single_writer_must_link_pairs,
    )
    from microservice_pipeline.structural_dependency_graph.clustering.postprocess import (
        postprocess_data_only_clusters,
    )
    from microservice_pipeline.structural_dependency_graph.clustering.exclusions import (
        build_exclusion_row as _shared_exclusion_row,
        build_orphaned_data_exclusions as _shared_build_orphaned_data_exclusions,
        is_callable_node as _shared_is_callable_node,
        is_data_node as _shared_is_data_node,
        local_callable_parent_id as _shared_local_callable_parent_id,
        node_label as _shared_node_label,
        node_type as _shared_node_type,
    )
    from microservice_pipeline.structural_dependency_graph.clustering.registry import (
        algorithm_choices,
        cluster_with_algorithm,
    )
    from microservice_pipeline.structural_dependency_graph.hub_nodes import (
        HubDetectionOptions,
        compute_degrees as compute_structural_degrees,
        identify_hub_nodes as identify_structural_hub_nodes,
        load_hub_node_rows,
    )
    from microservice_pipeline.structural_dependency_graph.weight_config import (
        StructuralWeightConfig,
        load_weight_config,
        resolve_weight_config_reference,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from structural_dependency_graph.clustering.common import (  # type: ignore
        StructuralClusteringInput,
        ordered_cluster_map as _ordered_cluster_map,
    )
    from structural_dependency_graph.clustering.hac_callable_projection import (  # type: ignore
        ALGORITHM as HAC_CALLABLE_PROJECTION_ALGORITHM,
    )
    from structural_dependency_graph.clustering.infomap import cluster_infomap  # type: ignore
    from structural_dependency_graph.clustering.label_propagation import cluster_label_propagation  # type: ignore
    from structural_dependency_graph.clustering.leiden import cluster_leiden  # type: ignore
    from structural_dependency_graph.clustering.leiden_multiplex import (  # type: ignore
        ALGORITHM as LEIDEN_MULTIPLEX_ALGORITHM,
        multiplex_must_link_relations,
    )
    from structural_dependency_graph.clustering.leiden_reweighted import (  # type: ignore
        ALGORITHM as LEIDEN_REWEIGHTED_ALGORITHM,
        reweighted_edge_weights,
        reweighted_must_link_relations,
        single_writer_must_link_pairs,
    )
    from structural_dependency_graph.clustering.postprocess import (  # type: ignore
        postprocess_data_only_clusters,
    )
    from structural_dependency_graph.clustering.exclusions import (  # type: ignore
        build_exclusion_row as _shared_exclusion_row,
        build_orphaned_data_exclusions as _shared_build_orphaned_data_exclusions,
        is_callable_node as _shared_is_callable_node,
        is_data_node as _shared_is_data_node,
        local_callable_parent_id as _shared_local_callable_parent_id,
        node_label as _shared_node_label,
        node_type as _shared_node_type,
    )
    from structural_dependency_graph.clustering.registry import (  # type: ignore
        algorithm_choices,
        cluster_with_algorithm,
    )
    from structural_dependency_graph.hub_nodes import (  # type: ignore
        HubDetectionOptions,
        compute_degrees as compute_structural_degrees,
        identify_hub_nodes as identify_structural_hub_nodes,
        load_hub_node_rows,
    )
    from structural_dependency_graph.weight_config import (  # type: ignore
        StructuralWeightConfig,
        load_weight_config,
        resolve_weight_config_reference,
    )

try:
    from microservice_pipeline.evaluation.evaluate_microservice_clustering import (
        DEFAULT_EVALUATION_KIND_TOKENS,
        DEFAULT_EVALUATION_NODE_TYPES,
        DEFAULT_MANUAL,
        NA_DEFAULTS,
        build_evaluation_input_from_rows,
        build_evaluation_payload,
        evaluate_assignment_rows,
        evaluation_summary_row,
        parse_evaluation_tokens,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from microservice_pipeline.evaluation.evaluate_microservice_clustering import (  # type: ignore
        DEFAULT_EVALUATION_KIND_TOKENS,
        DEFAULT_EVALUATION_NODE_TYPES,
        DEFAULT_MANUAL,
        NA_DEFAULTS,
        build_evaluation_input_from_rows,
        build_evaluation_payload,
        evaluate_assignment_rows,
        evaluation_summary_row,
        parse_evaluation_tokens,
    )


CALLABLE_PREFIX = "callable:"
DATA_PREFIX = "data:"
HAC_MUTATING_DATA_ACCESSES = frozenset({"create", "write", "read_write"})
LOCAL_CALLABLE_MARKER = ".<locals>."
LOCAL_CALLABLE_MUST_LINK_RELATION = "local_callable"

# A must-link is a hard constraint: the two nodes end up in one cluster whatever
# the rest of the graph says. So only relations that claim the two names hold the
# *same object* belong here. ``derived_from`` -- "this value was made from that
# one" -- is deliberately absent, and so is ``arg_to_param``.
STRONG_MUST_LINK_RELATIONS = {
    "state_assign",
    "tuple_unpack",
    "return_value",
    "return_slot",
    "local_assign",
}

RB_CONFIGURATION_SWEEP_DEFAULTS = (0.6, 0.8, 1.0, 1.2, 1.5)
CPM_SWEEP_DEFAULTS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
INFOMAP_MARKOV_TIME_SWEEP_DEFAULTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
SWEEP_DEFAULTS = RB_CONFIGURATION_SWEEP_DEFAULTS
DEFAULT_WEIGHT_CONFIG = load_weight_config()
LEIDEN_ALGORITHMS = frozenset(
    {"leiden", LEIDEN_REWEIGHTED_ALGORITHM, LEIDEN_MULTIPLEX_ALGORITHM}
)
SWEEP_EVALUATION_FIELDNAMES = [
    "evaluation_joined_rows",
    "evaluation_known_joined_rows",
    "evaluation_known_coverage",
    "evaluation_unmatched_manual_rows",
    "evaluation_unmatched_cluster_rows",
    "evaluation_n",
    "evaluation_manual_cluster_count",
    "evaluation_predicted_cluster_count",
    "evaluation_adjusted_rand_index",
    "evaluation_v_measure",
    "evaluation_homogeneity",
    "evaluation_completeness",
    "evaluation_nmi",
    "evaluation_pairwise_precision",
    "evaluation_pairwise_recall",
    "evaluation_pairwise_f1",
    "evaluation_bcubed_precision",
    "evaluation_bcubed_recall",
    "evaluation_bcubed_f1",
    "evaluation_purity",
    "evaluation_inverse_purity",
    "evaluation_purity_f1",
    "evaluation_macro_purity_precision",
    "evaluation_macro_purity_recall",
    "evaluation_macro_purity_f1",
    "evaluation_predicted_match_precision",
    "evaluation_predicted_match_recall",
    "evaluation_predicted_match_f1",
    "evaluation_predicted_match_pair_macro_f1",
    "evaluation_hungarian_accuracy",
    "evaluation_best_match_macro_f1",
    "evaluation_best_match_weighted_f1",
    "evaluation_best_match_macro_jaccard",
    "evaluation_best_match_weighted_jaccard",
]
DEFAULT_SWEEP_BEST_METRIC = "evaluation_predicted_match_f1"
SWEEP_BEST_DATA_HUB_POLICIES = ("auto", "drop_data_hubs", "keep_data_hubs")


@dataclass(frozen=True)
class SweepBestSelectionOptions:
    """User-facing controls for materializing one sweep row as full cluster output."""

    enabled: bool = True
    metric: str = DEFAULT_SWEEP_BEST_METRIC
    metric_direction: str = "max"
    resolution: Optional[float] = None
    markov_time: Optional[float] = None
    hac_n_clusters: Optional[int] = None
    call_resolution: Optional[float] = None
    data_access_resolution: Optional[float] = None
    data_lineage_resolution: Optional[float] = None
    data_hub_policy: str = "auto"
    min_metric: Optional[str] = None
    min_value: Optional[float] = None


@dataclass(frozen=True)
class SweepBestSelection:
    """Result of choosing one row from the parameter sweep table."""

    selected_index: Optional[int]
    selected_row: Optional[dict]
    metric: str
    metric_direction: str
    candidate_count: int
    filtered_count: int
    reason: str

    @property
    def selected(self) -> bool:
        return self.selected_index is not None and self.selected_row is not None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    edge_type: str
    relation: str = ""
    access: str = ""
    operation: str = ""
    weight: float = 1.0
    evidence_count: int = 1
    confidence: str = ""
    files_preview: str = ""
    linenos_preview: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Edge":
        return cls(
            src=_text(row.get("src")),
            dst=_text(row.get("dst")),
            edge_type=_text(row.get("edge_type")),
            relation=_text(row.get("relation")),
            access=_text(row.get("access")),
            operation=_text(row.get("operation")),
            weight=_float(row.get("weight"), 1.0),
            evidence_count=_int(row.get("evidence_count"), 1),
            confidence=_text(row.get("confidence")),
            files_preview=_text(row.get("files_preview")),
            linenos_preview=_text(row.get("linenos_preview")),
        )


@dataclass(frozen=True)
class ClusterOptions:
    algorithm: str = "leiden"
    leiden_quality: str = "rb_configuration"
    multiplex_layer_mode: str = "edge_type"
    resolution: float = 1.0
    markov_time: float = 1.0
    call_resolution: Optional[float] = None
    data_access_resolution: Optional[float] = None
    data_lineage_resolution: Optional[float] = None
    hac_n_clusters: int = 13
    seed: int = 42
    max_iter: int = 100
    sweep_resolutions: Tuple[float, ...] = tuple()
    sweep_markov_times: Tuple[float, ...] = tuple()
    sweep_call_resolutions: Tuple[float, ...] = tuple()
    sweep_data_access_resolutions: Tuple[float, ...] = tuple()
    sweep_data_lineage_resolutions: Tuple[float, ...] = tuple()
    sweep_hac_n_clusters: Tuple[int, ...] = tuple()
    run_sweep: bool = False
    exclude_module_callables: bool = True
    callable_hub_policy: Optional[str] = None
    callable_hub_drop: Tuple[str, ...] = tuple()
    callable_hub_keep: Tuple[str, ...] = tuple()
    callable_hub_nodes_path: Optional[str] = None
    data_hub_nodes_path: Optional[str] = None
    drop_callable_hubs: bool = True
    drop_data_hubs: bool = False
    call_weight_scale: float = DEFAULT_WEIGHT_CONFIG.clustering_scale("call")
    data_access_weight_scale: float = DEFAULT_WEIGHT_CONFIG.clustering_scale("data_access")
    data_lineage_weight_scale: float = DEFAULT_WEIGHT_CONFIG.clustering_scale("data_lineage")
    weight_config: Dict[str, Any] = field(default_factory=lambda: load_weight_config().to_dict())
    hub_callable_degree_percentile: float = 95.0
    hub_callable_min_degree: int = 25
    hub_callable_min_in_degree: int = 2
    hub_callable_min_out_degree: int = 2
    hub_entrypoint_min_out_degree: int = 12
    hub_orchestrator_max_in_degree: int = 1
    hub_orchestrator_min_out_degree: int = 12
    hub_orchestrator_min_out_call_degree: int = 4
    hub_orchestrator_min_target_modules: int = 3
    hub_orchestrator_min_target_callables: int = 4
    hub_orchestrator_min_target_data: int = 4
    hub_orchestrator_min_data_to_call_ratio: float = 1.0
    hub_data_min_degree: int = 20
    hub_data_min_callable_count: int = 10
    hub_data_min_access_count: int = 100


@dataclass
class ClusterResult:
    cluster_of: Dict[str, str]
    cluster_summary: List[dict]
    cluster_edges: List[dict]
    excluded_nodes: List[dict]
    hub_nodes: List[dict]
    hub_cluster_links: List[dict]
    must_link_groups: List[dict]
    cycle_findings: List[dict]
    degree_map: Dict[str, dict]
    options: ClusterOptions


@dataclass(frozen=True)
class HacDataAssignmentResult:
    cluster_of: Dict[str, str]
    excluded_reasons: Dict[str, str]


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _unique_text_tuple(values: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(text for value in values if (text := _text(value))))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _node_type(node_id: str, row: Mapping[str, Any] | None = None) -> str:
    return _shared_node_type(node_id, row)


def _node_label(node_id: str, row: Mapping[str, Any] | None = None) -> str:
    return _shared_node_label(node_id, row)


def _preview(values: Iterable[Any], limit: int = 8) -> str:
    cleaned = [_text(value) for value in values if _text(value)]
    if not cleaned:
        return ""
    unique = sorted(set(cleaned))
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";... ({len(unique)} total)"
    return ";".join(unique)


def _counter_preview(counter: Mapping[str, int], limit: int = 5) -> str:
    if not counter:
        return ""
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ";".join(f"{key}({value})" for key, value in items)


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if percentile <= 0:
        return float(ordered[0])
    if percentile >= 100:
        return float(ordered[-1])
    rank = math.ceil((percentile / 100.0) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return float(ordered[rank])


def load_node_rows(nodes_csv: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with nodes_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            node_id = _text(row.get("id")).strip()
            if node_id:
                rows[node_id] = dict(row)
    return rows


def load_csv_rows(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def load_edges(edges_csv: Path) -> List[Edge]:
    edges: List[Edge] = []
    with edges_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            edge = Edge.from_row(row)
            if edge.src and edge.dst:
                edges.append(edge)
    return edges


def scale_edges_for_clustering(edges: Sequence[Edge], options: ClusterOptions) -> List[Edge]:
    if options.algorithm == LEIDEN_REWEIGHTED_ALGORITHM:
        weights = reweighted_edge_weights(edges, options.weight_config)
        return [replace(edge, weight=weight) for edge, weight in zip(edges, weights)]
    if options.algorithm == LEIDEN_MULTIPLEX_ALGORITHM:
        return [replace(edge) for edge in edges]

    scaled: List[Edge] = []
    for edge in edges:
        weight = edge.weight
        if edge.edge_type == "call":
            weight *= options.call_weight_scale
        elif edge.edge_type == "data_access":
            weight *= options.data_access_weight_scale
        elif edge.edge_type == "data_lineage":
            weight *= options.data_lineage_weight_scale
        scaled.append(replace(edge, weight=weight))
    return scaled


def ensure_edge_nodes(nodes: Dict[str, dict], edges: Iterable[Edge]) -> Dict[str, dict]:
    rows = dict(nodes)
    for edge in edges:
        for node_id in (edge.src, edge.dst):
            if node_id not in rows:
                rows[node_id] = {
                    "id": node_id,
                    "node_type": _node_type(node_id),
                    "label": node_id,
                    "kind": "",
                    "module": "",
                    "qualname": "",
                    "class_name": "",
                    "display_name": node_id,
                    "scope": "",
                    "owner": "",
                    "file": "",
                    "lineno": "",
                    "raw_object_count": "",
                    "callable_count": "",
                    "access_count": "",
                }
    return rows


def compute_degrees(nodes: Mapping[str, Mapping[str, Any]], edges: Iterable[Edge]) -> Dict[str, dict]:
    return compute_structural_degrees(nodes, edges)


def callable_hub_decisions_from_json(path: Path) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Callable hub decisions must be a JSON object: {path}")

    container = payload.get("callable_hubs", payload)
    if not isinstance(container, Mapping):
        raise ValueError(f"Callable hub decisions must contain an object: {path}")

    def _node_tuple(key: str) -> Tuple[str, ...]:
        value = container.get(key, ())
        if value is None:
            return tuple()
        if not isinstance(value, list):
            raise ValueError(f"Callable hub decisions field '{key}' must be a list: {path}")
        return tuple(_text(item) for item in value if _text(item))

    return _node_tuple("drop"), _node_tuple("keep")


def _effective_callable_hub_policy(options: ClusterOptions) -> str:
    if options.callable_hub_policy:
        return options.callable_hub_policy
    return "drop-all" if options.drop_callable_hubs else "keep"


def _callable_fanout_features(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Edge],
) -> Dict[str, dict]:
    features: Dict[str, dict] = {
        node_id: {
            "out_call_degree": 0,
            "out_data_degree": 0,
            "target_callable_count": 0,
            "target_data_count": 0,
            "target_callables": set(),
            "target_data": set(),
            "target_modules": Counter(),
            "target_edge_types": Counter(),
        }
        for node_id, row in nodes.items()
        if _node_type(node_id, row) == "callable"
    }

    for edge in edges:
        row = nodes.get(edge.src)
        if row is None or _node_type(edge.src, row) != "callable":
            continue
        feature = features.setdefault(
            edge.src,
            {
                "out_call_degree": 0,
                "out_data_degree": 0,
                "target_callable_count": 0,
                "target_data_count": 0,
                "target_callables": set(),
                "target_data": set(),
                "target_modules": Counter(),
                "target_edge_types": Counter(),
            },
        )
        feature["target_edge_types"][edge.edge_type] += 1
        target_row = nodes.get(edge.dst, {})
        target_type = _node_type(edge.dst, target_row)
        if edge.edge_type == "call":
            feature["out_call_degree"] += 1
        else:
            feature["out_data_degree"] += 1
        if target_type == "callable":
            feature["target_callables"].add(edge.dst)
            module = _text(target_row.get("module")) or "(unknown)"
            feature["target_modules"][module] += 1
        elif target_type == "data":
            feature["target_data"].add(edge.dst)

    for feature in features.values():
        feature["target_callable_count"] = len(feature.get("target_callables", ()))
        feature["target_data_count"] = len(feature.get("target_data", ()))

    return features


def _is_orchestrator_candidate(
    degrees: Mapping[str, Any],
    features: Mapping[str, Any],
    options: ClusterOptions,
) -> bool:
    in_degree = _int(degrees.get("in_degree"))
    out_degree = _int(degrees.get("out_degree"))
    out_call_degree = _int(features.get("out_call_degree"))
    target_modules = features.get("target_modules", Counter())
    target_module_count = len(target_modules)
    target_callable_count = _int(features.get("target_callable_count"))
    target_data_count = _int(features.get("target_data_count"))
    enough_data_fanout = (
        target_data_count >= options.hub_orchestrator_min_target_data
        and target_data_count
        >= target_callable_count * options.hub_orchestrator_min_data_to_call_ratio
    )
    return (
        in_degree <= options.hub_orchestrator_max_in_degree
        and out_degree >= options.hub_orchestrator_min_out_degree
        and out_call_degree >= options.hub_orchestrator_min_out_call_degree
        and enough_data_fanout
        and (
            target_module_count >= options.hub_orchestrator_min_target_modules
            or target_callable_count >= options.hub_orchestrator_min_target_callables
        )
    )


def _should_drop_callable_hub(node_id: str, options: ClusterOptions) -> bool:
    keep = set(options.callable_hub_keep)
    drop = set(options.callable_hub_drop)
    if node_id in keep:
        return False
    policy = _effective_callable_hub_policy(options)
    if policy == "drop-all":
        return True
    if policy == "drop-configured":
        return node_id in drop
    return False


def _hub_detection_options(options: ClusterOptions) -> HubDetectionOptions:
    return HubDetectionOptions(
        hub_callable_degree_percentile=options.hub_callable_degree_percentile,
        hub_callable_min_degree=options.hub_callable_min_degree,
        hub_callable_min_in_degree=options.hub_callable_min_in_degree,
        hub_callable_min_out_degree=options.hub_callable_min_out_degree,
        hub_entrypoint_min_out_degree=options.hub_entrypoint_min_out_degree,
        hub_orchestrator_max_in_degree=options.hub_orchestrator_max_in_degree,
        hub_orchestrator_min_out_degree=options.hub_orchestrator_min_out_degree,
        hub_orchestrator_min_out_call_degree=options.hub_orchestrator_min_out_call_degree,
        hub_orchestrator_min_target_modules=options.hub_orchestrator_min_target_modules,
        hub_orchestrator_min_target_callables=options.hub_orchestrator_min_target_callables,
        hub_orchestrator_min_target_data=options.hub_orchestrator_min_target_data,
        hub_orchestrator_min_data_to_call_ratio=options.hub_orchestrator_min_data_to_call_ratio,
        hub_data_min_degree=options.hub_data_min_degree,
        hub_data_min_callable_count=options.hub_data_min_callable_count,
        hub_data_min_access_count=options.hub_data_min_access_count,
    )


def _hub_node_file_rows(
    nodes: Mapping[str, Mapping[str, Any]],
    options: ClusterOptions,
) -> Tuple[List[dict], List[dict]] | None:
    if not options.callable_hub_nodes_path or not options.data_hub_nodes_path:
        return None
    callable_path = Path(options.callable_hub_nodes_path)
    data_path = Path(options.data_hub_nodes_path)
    if not callable_path.is_file() or not data_path.is_file():
        return None
    callable_rows = [
        row for row in load_hub_node_rows(callable_path) if _text(row.get("node")) in nodes
    ]
    data_rows = [
        row for row in load_hub_node_rows(data_path) if _text(row.get("node")) in nodes
    ]
    return callable_rows, data_rows


def _cluster_hub_row(
    node_id: str,
    source_row: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    degree_map: Mapping[str, Mapping[str, Any]],
    action: str,
) -> dict:
    node_row = nodes.get(node_id, {})
    degrees = degree_map.get(node_id, {})

    def value(key: str, fallback: Any = "") -> Any:
        current = source_row.get(key)
        if current not in (None, ""):
            return current
        return fallback

    return {
        "node": node_id,
        "node_type": value("node_type", _node_type(node_id, node_row)),
        "label": value("label", _node_label(node_id, node_row)),
        "kind": value("kind", _text(node_row.get("kind"))),
        "action": action,
        "candidate_types": value("candidate_types"),
        "reasons": value("reasons"),
        "in_degree": _int(value("in_degree", degrees.get("in_degree"))),
        "out_degree": _int(value("out_degree", degrees.get("out_degree"))),
        "total_degree": _int(value("total_degree", degrees.get("total_degree"))),
        "weighted_out_degree": f"{_float(value('weighted_out_degree', degrees.get('weighted_out_degree'))):.6f}",
        "weighted_degree": f"{_float(value('weighted_degree', degrees.get('weighted_degree'))):.6f}",
        "callable_count": value("callable_count", _text(node_row.get("callable_count"))),
        "access_count": value("access_count", _text(node_row.get("access_count"))),
        "out_call_degree": value("out_call_degree"),
        "out_data_degree": value("out_data_degree"),
        "target_callable_count": value("target_callable_count"),
        "target_data_count": value("target_data_count"),
        "target_module_count": value("target_module_count"),
        "target_modules": value("target_modules"),
    }


def identify_hubs(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Edge],
    degree_map: Mapping[str, Mapping[str, Any]],
    options: ClusterOptions,
) -> Tuple[Set[str], Set[str], List[dict]]:
    file_rows = _hub_node_file_rows(nodes, options)
    if file_rows is None:
        callable_rows, data_rows, _structural_degree_map = identify_structural_hub_nodes(
            nodes,
            edges,
            _hub_detection_options(options),
        )
    else:
        callable_rows, data_rows = file_rows

    callable_hubs: Set[str] = set()
    data_hubs: Set[str] = {_text(row.get("node")) for row in data_rows if _text(row.get("node"))}
    hub_rows: List[dict] = []
    for row in callable_rows:
        node_id = _text(row.get("node"))
        if not node_id:
            continue
        action = "dropped" if _should_drop_callable_hub(node_id, options) else "kept"
        if action == "dropped":
            callable_hubs.add(node_id)
        hub_rows.append(_cluster_hub_row(node_id, row, nodes, degree_map, action))

    for row in data_rows:
        node_id = _text(row.get("node"))
        if not node_id:
            continue
        action = "dropped" if options.drop_data_hubs else "kept"
        hub_rows.append(_cluster_hub_row(node_id, row, nodes, degree_map, action))

    hub_rows.sort(key=lambda row: (row["node_type"], row["action"], -_int(row["total_degree"]), row["node"]))
    return callable_hubs, data_hubs, hub_rows


def build_exclusions(
    nodes: Mapping[str, Mapping[str, Any]],
    degree_map: Mapping[str, Mapping[str, Any]],
    callable_hubs: Set[str],
    data_hubs: Set[str],
    options: ClusterOptions,
) -> Tuple[Set[str], List[dict]]:
    excluded: Set[str] = set()
    rows: List[dict] = []

    for node_id, row in sorted(nodes.items()):
        node_type = _node_type(node_id, row)
        total_degree = _int(degree_map.get(node_id, {}).get("total_degree"))
        reason = ""
        if (
            options.exclude_module_callables
            and node_type == "callable"
            and _text(row.get("kind")) == "module"
        ):
            reason = "module_callable"
        elif node_type == "callable" and total_degree == 0:
            reason = "isolated_callable"
        elif node_type == "data" and total_degree == 0:
            reason = "isolated_data"
        elif node_id in callable_hubs:
            reason = "callable_hub"
        elif node_id in data_hubs and options.drop_data_hubs:
            reason = "data_hub"

        if not reason:
            continue
        excluded.add(node_id)
        rows.append(
            {
                "node": node_id,
                "node_type": node_type,
                "reason": reason,
                "label": _node_label(node_id, row),
                "kind": _text(row.get("kind")),
                "total_degree": total_degree,
                "weighted_degree": f"{_float(degree_map.get(node_id, {}).get('weighted_degree')):.6f}",
                "file": _text(row.get("file")),
                "lineno": _text(row.get("lineno")),
            }
        )

    return excluded, rows


def _exclusion_row(
    node_id: str,
    row: Mapping[str, Any],
    reason: str,
    degree_map: Mapping[str, Mapping[str, Any]],
) -> dict:
    return _shared_exclusion_row(node_id, row, reason, degree_map)


def build_orphaned_data_exclusions(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Edge],
    degree_map: Mapping[str, Mapping[str, Any]],
    excluded: Set[str],
    callable_hubs: Set[str],
) -> Tuple[Set[str], List[dict]]:
    return _shared_build_orphaned_data_exclusions(
        nodes,
        edges,
        degree_map,
        excluded,
        callable_hubs,
        reason="orphaned_by_callable_hub",
    )


def _is_data_node(node_id: str, nodes: Mapping[str, Mapping[str, Any]]) -> bool:
    return _shared_is_data_node(node_id, nodes)


def _is_callable_node(node_id: str, nodes: Mapping[str, Mapping[str, Any]]) -> bool:
    return _shared_is_callable_node(node_id, nodes)


def local_callable_parent_id(node_id: str, available_nodes: Set[str]) -> Optional[str]:
    return _shared_local_callable_parent_id(node_id, available_nodes)


def local_callable_must_link_pairs(
    node_ids: Iterable[str],
) -> Tuple[Tuple[str, str, str], ...]:
    available_nodes = set(node_ids)
    pairs: List[Tuple[str, str, str]] = []
    for node_id in sorted(available_nodes):
        parent_id = local_callable_parent_id(node_id, available_nodes)
        if parent_id:
            pairs.append((parent_id, node_id, LOCAL_CALLABLE_MUST_LINK_RELATION))
    return tuple(pairs)


def _is_producer_data_node(node_id: str, row: Mapping[str, Any] | None = None) -> bool:
    kind = _text(row.get("kind") if row else "")
    return node_id.startswith(f"{DATA_PREFIX}local_exposed:") or "local_exposed" in kind


def build_must_link_groups(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Edge],
    active_nodes: Set[str],
    strong_relations: Set[str] | None = None,
    extra_must_link_pairs: Iterable[Tuple[str, str, str]] | None = None,
) -> Tuple[Dict[str, str], Dict[str, List[str]], List[dict]]:
    if strong_relations is None:
        strong_relations = STRONG_MUST_LINK_RELATIONS
    active_data_nodes = {
        node_id for node_id in active_nodes if _node_type(node_id, nodes.get(node_id, {})) == "data"
    }
    uf = UnionFind(active_nodes)
    strong_links: List[Tuple[str, str, str, str, str]] = []

    for edge in edges:
        if edge.edge_type != "data_lineage":
            continue
        if edge.relation not in strong_relations:
            continue
        if edge.src not in active_data_nodes or edge.dst not in active_data_nodes:
            continue
        uf.union(edge.src, edge.dst)
        strong_links.append(
            (
                edge.src,
                edge.dst,
                edge.relation,
                edge.files_preview,
                edge.linenos_preview,
            )
        )

    for src, dst, relation in extra_must_link_pairs or ():
        if src not in active_nodes or dst not in active_nodes:
            continue
        uf.union(src, dst)
        strong_links.append((src, dst, relation, "", ""))

    grouped: Dict[str, List[str]] = defaultdict(list)
    for node_id in sorted(active_nodes):
        grouped[uf.find(node_id)].append(node_id)

    multi_groups = [members for members in grouped.values() if len(members) > 1]
    multi_groups.sort(key=lambda members: min(members))

    supernode_of: Dict[str, str] = {node_id: node_id for node_id in active_nodes}
    members_of: Dict[str, List[str]] = {node_id: [node_id] for node_id in active_nodes}
    rows: List[dict] = []

    for idx, members in enumerate(multi_groups, start=1):
        member_set = set(members)
        group_links = [
            link for link in strong_links if link[0] in member_set and link[1] in member_set
        ]
        callable_members = [
            node_id
            for node_id in members
            if _node_type(node_id, nodes.get(node_id, {})) == "callable"
        ]
        if callable_members:
            anchor = sorted(callable_members)[0]
        else:
            producer_candidates = [
                src
                for src, _dst, _relation, _files, _linenos in group_links
                if _is_producer_data_node(src, nodes.get(src, {}))
            ]
            if not producer_candidates:
                producer_candidates = [src for src, _dst, _relation, _files, _linenos in group_links]
            producer_counts = Counter(producer_candidates)
            if producer_counts:
                anchor = sorted(producer_counts, key=lambda node: (-producer_counts[node], node))[0]
            else:
                anchor = sorted(members)[0]

        group_id = f"ML{idx:03d}"
        supernode_id = f"must_link:{group_id}"
        for node_id in members:
            supernode_of[node_id] = supernode_id
        members_of[supernode_id] = sorted(members)
        for node_id in members:
            if node_id != supernode_id:
                members_of.pop(node_id, None)

        rows.append(
            {
                "must_link_group": group_id,
                "anchor_node": anchor,
                "anchor_label": _node_label(anchor, nodes.get(anchor, {})),
                "size": len(members),
                "relations": _preview(relation for _src, _dst, relation, _files, _linenos in group_links),
                "members": ";".join(sorted(members)),
                "evidence": _preview(
                    (
                        f"{relation}@{files}:{linenos}"
                        if files or linenos
                        else relation
                    )
                    for _src, _dst, relation, files, linenos in group_links
                ),
            }
        )

    return supernode_of, members_of, rows


def build_contracted_edge_views(
    edges: Iterable[Edge],
    active_nodes: Set[str],
    supernode_of: Mapping[str, str],
) -> Tuple[
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], float],
    Dict[str, Dict[Tuple[str, str], float]],
]:
    directed: Dict[Tuple[str, str], float] = defaultdict(float)
    undirected: Dict[Tuple[str, str], float] = defaultdict(float)
    typed_undirected: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for edge in edges:
        if edge.src not in active_nodes or edge.dst not in active_nodes:
            continue
        src = supernode_of[edge.src]
        dst = supernode_of[edge.dst]
        if src == dst:
            continue
        directed[(src, dst)] += edge.weight
        left, right = sorted((src, dst))
        key = (left, right)
        undirected[key] += edge.weight
        typed_undirected[edge.edge_type][key] += edge.weight
    return (
        dict(directed),
        dict(undirected),
        {edge_type: dict(weights) for edge_type, weights in typed_undirected.items()},
    )


def build_contracted_edges(
    edges: Iterable[Edge],
    active_nodes: Set[str],
    supernode_of: Mapping[str, str],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    directed, undirected, _typed_undirected = build_contracted_edge_views(
        edges,
        active_nodes,
        supernode_of,
    )
    return directed, undirected


def _edge_type_layer_weights(options: ClusterOptions) -> Dict[str, float]:
    if options.multiplex_layer_mode == "call_data":
        return {
            "call": options.call_weight_scale,
            "data": 1.0,
        }
    return {
        "call": options.call_weight_scale,
        "data_access": options.data_access_weight_scale,
        "data_lineage": options.data_lineage_weight_scale,
    }


def _edge_type_layer_resolutions(options: ClusterOptions) -> Dict[str, float]:
    call_resolution = (
        options.call_resolution
        if options.call_resolution is not None
        else options.resolution
    )
    data_access_resolution = (
        options.data_access_resolution
        if options.data_access_resolution is not None
        else options.resolution
    )
    if options.multiplex_layer_mode == "call_data":
        return {
            "call": call_resolution,
            "data": data_access_resolution,
        }
    return {
        "call": call_resolution,
        "data_access": data_access_resolution,
        "data_lineage": (
            options.data_lineage_resolution
            if options.data_lineage_resolution is not None
            else options.resolution
        ),
    }


def _multiplex_layer_edges(
    typed_undirected_edges: Mapping[str, Mapping[Tuple[str, str], float]],
    options: ClusterOptions,
) -> Mapping[str, Mapping[Tuple[str, str], float]]:
    if options.multiplex_layer_mode != "call_data":
        return typed_undirected_edges

    data_edges: Dict[Tuple[str, str], float] = defaultdict(float)
    for key, weight in typed_undirected_edges.get("data_access", {}).items():
        data_edges[key] += weight * options.data_access_weight_scale
    for key, weight in typed_undirected_edges.get("data_lineage", {}).items():
        data_edges[key] += weight * options.data_lineage_weight_scale
    for edge_type, edges in typed_undirected_edges.items():
        if edge_type in {"call", "data_access", "data_lineage"}:
            continue
        for key, weight in edges.items():
            data_edges[key] += weight

    layers: Dict[str, Mapping[Tuple[str, str], float]] = {}
    call_edges = typed_undirected_edges.get("call")
    if call_edges:
        layers["call"] = call_edges
    if data_edges:
        layers["data"] = dict(data_edges)
    return layers


def cluster_supernodes(
    supernodes: Set[str],
    directed_edges: Mapping[Tuple[str, str], float],
    undirected_edges: Mapping[Tuple[str, str], float],
    typed_undirected_edges: Mapping[str, Mapping[Tuple[str, str], float]],
    members_of: Mapping[str, List[str]],
    options: ClusterOptions,
) -> Dict[str, str]:
    input_data = StructuralClusteringInput(
        supernodes=supernodes,
        directed_edges=directed_edges,
        undirected_edges=undirected_edges,
        members_of=members_of,
        seed=options.seed,
        resolution=options.resolution,
        markov_time=options.markov_time,
        max_iter=options.max_iter,
        leiden_quality=options.leiden_quality,
        hac_n_clusters=options.hac_n_clusters,
        typed_undirected_edges=_multiplex_layer_edges(typed_undirected_edges, options),
        edge_type_layer_weights=_edge_type_layer_weights(options),
        edge_type_layer_resolutions=_edge_type_layer_resolutions(options),
    )
    return cluster_with_algorithm(options.algorithm, input_data)


def expand_cluster_assignments(
    super_cluster_of: Mapping[str, str],
    members_of: Mapping[str, List[str]],
) -> Dict[str, str]:
    cluster_of: Dict[str, str] = {}
    for supernode, cluster_id in super_cluster_of.items():
        for member in members_of.get(supernode, [supernode]):
            cluster_of[member] = cluster_id
    return reindex_clusters(cluster_of)


def assign_hac_mutating_data(
    cluster_of: Mapping[str, str],
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Edge],
    active_nodes: Set[str],
    supernode_of: Mapping[str, str],
    members_of: Mapping[str, List[str]],
) -> HacDataAssignmentResult:
    updated = dict(cluster_of)
    data_supernodes = {
        supernode
        for supernode, members in members_of.items()
        if _is_data_only_members(members, nodes)
    }
    mutating_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for edge in edges:
        if edge.edge_type != "data_access":
            continue
        if edge.src not in active_nodes or edge.dst not in active_nodes:
            continue
        callable_id, data_id = _callable_data_access_endpoint(edge, nodes)
        if not callable_id or not data_id:
            continue
        callable_cluster = updated.get(callable_id)
        if not callable_cluster:
            continue
        data_supernode = supernode_of.get(data_id, data_id)
        if data_supernode not in data_supernodes:
            continue
        if edge.access in HAC_MUTATING_DATA_ACCESSES:
            mutating_scores[data_supernode][callable_cluster] += edge.weight

    excluded_reasons: Dict[str, str] = {}
    for data_supernode in sorted(data_supernodes):
        data_members = [
            member
            for member in members_of.get(data_supernode, [data_supernode])
            if member in active_nodes and _is_data_node(member, nodes)
        ]
        if not data_members:
            continue
        scores = mutating_scores.get(data_supernode, {})
        if not scores:
            for member in data_members:
                excluded_reasons[member] = "read_only_data"
            continue

        best_score = max(scores.values())
        best_clusters = sorted(
            cluster_id
            for cluster_id, score in scores.items()
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
        )
        if len(best_clusters) != 1:
            for member in data_members:
                excluded_reasons[member] = "ambiguous_mutating_data"
            continue

        for member in data_members:
            updated[member] = best_clusters[0]

    return HacDataAssignmentResult(updated, excluded_reasons)


def _callable_data_access_endpoint(
    edge: Edge,
    nodes: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, str]:
    if _is_callable_node(edge.src, nodes) and _is_data_node(edge.dst, nodes):
        return edge.src, edge.dst
    if _is_callable_node(edge.dst, nodes) and _is_data_node(edge.src, nodes):
        return edge.dst, edge.src
    return "", ""


def _is_data_only_members(
    members: Sequence[str],
    nodes: Mapping[str, Mapping[str, Any]],
) -> bool:
    return bool(members) and all(_is_data_node(member, nodes) for member in members)


def reindex_clusters(cluster_of: Mapping[str, str]) -> Dict[str, str]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for node_id, cluster_id in cluster_of.items():
        groups[cluster_id].append(node_id)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    remapped: Dict[str, str] = {}
    for idx, members in enumerate(ordered, start=1):
        cluster_id = f"C{idx:03d}"
        for node_id in members:
            remapped[node_id] = cluster_id
    return remapped


def build_semantic_edges(edges: Iterable[Edge]) -> List[Edge]:
    semantic: List[Edge] = []
    for edge in edges:
        if edge.edge_type == "call":
            semantic.append(edge)
        elif edge.edge_type == "data_lineage":
            semantic.append(edge)
        elif edge.edge_type == "data_access":
            if edge.access == "read":
                semantic.append(replace(edge, src=edge.dst, dst=edge.src))
            elif edge.access == "read_write":
                semantic.append(edge)
                semantic.append(replace(edge, src=edge.dst, dst=edge.src))
            else:
                semantic.append(edge)
    return semantic


def strongly_connected_components(edges: Iterable[Edge], nodes: Set[str] | None = None) -> List[List[str]]:
    adjacency: Dict[str, List[str]] = defaultdict(list)
    known_nodes: Set[str] = set(nodes or set())
    self_loops: Set[str] = set()
    for edge in edges:
        known_nodes.add(edge.src)
        known_nodes.add(edge.dst)
        adjacency[edge.src].append(edge.dst)
        if edge.src == edge.dst:
            self_loops.add(edge.src)

    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    components: List[List[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(adjacency.get(node, [])):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or component[0] in self_loops:
                components.append(sorted(component))

    for node in sorted(known_nodes):
        if node not in indices:
            strongconnect(node)

    components.sort(key=lambda members: (-len(members), members[0]))
    return components


def build_cycle_findings(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Edge],
    cluster_of: Mapping[str, str],
) -> List[dict]:
    semantic_edges = build_semantic_edges(edges)
    findings: List[dict] = []

    node_sccs = strongly_connected_components(semantic_edges)
    for idx, members in enumerate(node_sccs, start=1):
        member_set = set(members)
        internal = [
            edge
            for edge in semantic_edges
            if edge.src in member_set and edge.dst in member_set
        ]
        findings.append(
            {
                "finding_id": f"N{idx:03d}",
                "level": "node",
                "size": len(members),
                "members": ";".join(members),
                "members_preview": _preview(members, limit=12),
                "node_types": _counter_preview(
                    Counter(_node_type(node, nodes.get(node, {})) for node in members)
                ),
                "clusters": _preview(cluster_of.get(node, "(unclustered)") for node in members),
                "internal_edge_count": len(internal),
                "edge_types": _preview(edge.edge_type for edge in internal),
                "relations": _preview(edge.relation or edge.access for edge in internal),
            }
        )

    cluster_edge_map: Dict[Tuple[str, str], Edge] = {}
    aggregate: Dict[Tuple[str, str], dict] = defaultdict(lambda: {"weight": 0.0, "edge_count": 0, "types": set(), "relations": set()})
    for edge in semantic_edges:
        src_cluster = cluster_of.get(edge.src)
        dst_cluster = cluster_of.get(edge.dst)
        if not src_cluster or not dst_cluster or src_cluster == dst_cluster:
            continue
        key = (src_cluster, dst_cluster)
        aggregate[key]["weight"] += edge.weight
        aggregate[key]["edge_count"] += 1
        aggregate[key]["types"].add(edge.edge_type)
        aggregate[key]["relations"].add(edge.relation or edge.access)
    for (src, dst), stats in aggregate.items():
        cluster_edge_map[(src, dst)] = Edge(
            src=src,
            dst=dst,
            edge_type=_preview(stats["types"]),
            relation=_preview(stats["relations"]),
            weight=stats["weight"],
            evidence_count=stats["edge_count"],
        )

    cluster_sccs = strongly_connected_components(cluster_edge_map.values(), set(cluster_of.values()))
    for idx, members in enumerate(cluster_sccs, start=1):
        member_set = set(members)
        internal = [
            edge
            for edge in cluster_edge_map.values()
            if edge.src in member_set and edge.dst in member_set
        ]
        findings.append(
            {
                "finding_id": f"C{idx:03d}",
                "level": "cluster",
                "size": len(members),
                "members": ";".join(members),
                "members_preview": _preview(members, limit=12),
                "node_types": "cluster",
                "clusters": ";".join(members),
                "internal_edge_count": len(internal),
                "edge_types": _preview(edge.edge_type for edge in internal),
                "relations": _preview(edge.relation for edge in internal),
            }
        )

    return findings


def compute_cluster_edges(
    edges: Iterable[Edge],
    cluster_of: Mapping[str, str],
) -> List[dict]:
    semantic_edges = build_semantic_edges(edges)
    aggregate: Dict[Tuple[str, str], dict] = defaultdict(
        lambda: {
            "weight": 0.0,
            "edge_count": 0,
            "edge_types": Counter(),
            "relations": Counter(),
        }
    )
    for edge in semantic_edges:
        src_cluster = cluster_of.get(edge.src)
        dst_cluster = cluster_of.get(edge.dst)
        if not src_cluster or not dst_cluster or src_cluster == dst_cluster:
            continue
        row = aggregate[(src_cluster, dst_cluster)]
        row["weight"] += edge.weight
        row["edge_count"] += 1
        row["edge_types"][edge.edge_type] += 1
        row["relations"][edge.relation or edge.access or "(none)"] += 1

    rows: List[dict] = []
    for (src_cluster, dst_cluster), stats in aggregate.items():
        rows.append(
            {
                "src_cluster": src_cluster,
                "dst_cluster": dst_cluster,
                "weight": f"{stats['weight']:.6f}",
                "edge_count": stats["edge_count"],
                "edge_types": _counter_preview(stats["edge_types"]),
                "relations": _counter_preview(stats["relations"]),
            }
        )
    rows.sort(key=lambda row: (-_float(row["weight"]), row["src_cluster"], row["dst_cluster"]))
    return rows


def compute_cluster_summary(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Edge],
    cluster_of: Mapping[str, str],
    data_hubs: Set[str],
    cycle_findings: Sequence[Mapping[str, Any]],
    degree_map: Mapping[str, Mapping[str, Any]],
) -> List[dict]:
    members: Dict[str, List[str]] = defaultdict(list)
    for node_id, cluster_id in cluster_of.items():
        members[cluster_id].append(node_id)

    internal_weight: Dict[str, float] = defaultdict(float)
    incoming_weight: Dict[str, float] = defaultdict(float)
    outgoing_weight: Dict[str, float] = defaultdict(float)
    for edge in edges:
        src_cluster = cluster_of.get(edge.src)
        dst_cluster = cluster_of.get(edge.dst)
        if not src_cluster or not dst_cluster:
            continue
        if src_cluster == dst_cluster:
            internal_weight[src_cluster] += edge.weight
        else:
            outgoing_weight[src_cluster] += edge.weight
            incoming_weight[dst_cluster] += edge.weight

    cyclic_clusters: Set[str] = set()
    for finding in cycle_findings:
        if finding.get("level") != "cluster":
            continue
        cyclic_clusters.update(_text(finding.get("members")).split(";"))

    rows: List[dict] = []
    for cluster_id, cluster_members in members.items():
        callable_count = 0
        data_count = 0
        module_counts: Counter[str] = Counter()
        data_kind_counts: Counter[str] = Counter()
        weighted_degrees: List[Tuple[float, str]] = []
        warnings: List[str] = []
        for node_id in cluster_members:
            row = nodes.get(node_id, {})
            if _node_type(node_id, row) == "callable":
                callable_count += 1
                module = _text(row.get("module")) or "(unknown)"
                module_counts[module] += 1
            elif _node_type(node_id, row) == "data":
                data_count += 1
                for kind in _text(row.get("kind")).split(";"):
                    kind = kind.strip()
                    if kind:
                        data_kind_counts[kind] += 1
            weighted_degrees.append((_float(degree_map.get(node_id, {}).get("weighted_degree")), node_id))

        if any(node_id in data_hubs for node_id in cluster_members):
            warnings.append("contains_data_hub")
        if callable_count == 0:
            warnings.append("data_only")
        if data_count == 0:
            warnings.append("callable_only")
        if cluster_id in cyclic_clusters:
            warnings.append("cluster_dependency_cycle")

        internal = internal_weight[cluster_id]
        incoming = incoming_weight[cluster_id]
        outgoing = outgoing_weight[cluster_id]
        total = internal + incoming + outgoing
        cohesion = internal / total if total else 0.0
        anchor = sorted(weighted_degrees, key=lambda item: (-item[0], item[1]))[0][1]
        rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster_members),
                "callable_count": callable_count,
                "data_count": data_count,
                "anchor_node": anchor,
                "anchor_label": _node_label(anchor, nodes.get(anchor, {})),
                "internal_weight": f"{internal:.6f}",
                "outgoing_weight": f"{outgoing:.6f}",
                "incoming_weight": f"{incoming:.6f}",
                "cohesion": f"{cohesion:.6f}",
                "top_modules": _counter_preview(module_counts),
                "data_kinds": _counter_preview(data_kind_counts),
                "warnings": ";".join(warnings),
                "members_preview": _preview(sorted(cluster_members), limit=10),
            }
        )

    rows.sort(key=lambda row: (-_int(row["size"]), row["cluster_id"]))
    return rows


def build_hub_cluster_links(
    edges: Iterable[Edge],
    cluster_of: Mapping[str, str],
    hub_nodes: Sequence[Mapping[str, Any]],
) -> List[dict]:
    hub_ids = {_text(row.get("node")) for row in hub_nodes}
    aggregate: Dict[Tuple[str, str, str, str], dict] = defaultdict(
        lambda: {"weight": 0.0, "edge_count": 0, "edge_types": Counter(), "relations": Counter()}
    )
    for edge in edges:
        for hub_id, other_id, direction in (
            (edge.src, edge.dst, "outgoing"),
            (edge.dst, edge.src, "incoming"),
        ):
            if hub_id not in hub_ids:
                continue
            other_cluster = cluster_of.get(other_id)
            if not other_cluster:
                continue
            own_cluster = cluster_of.get(hub_id)
            scope = "own_cluster" if own_cluster == other_cluster else "external_cluster"
            key = (hub_id, other_cluster, direction, scope)
            row = aggregate[key]
            row["weight"] += edge.weight
            row["edge_count"] += 1
            row["edge_types"][edge.edge_type] += 1
            row["relations"][edge.relation or edge.access or "(none)"] += 1

    rows: List[dict] = []
    for (hub_id, cluster_id, direction, scope), stats in aggregate.items():
        rows.append(
            {
                "hub_node": hub_id,
                "linked_cluster": cluster_id,
                "direction": direction,
                "scope": scope,
                "weight": f"{stats['weight']:.6f}",
                "edge_count": stats["edge_count"],
                "edge_types": _counter_preview(stats["edge_types"]),
                "relations": _counter_preview(stats["relations"]),
            }
        )
    rows.sort(key=lambda row: (row["hub_node"], row["scope"], row["linked_cluster"], row["direction"]))
    return rows


def must_link_group_by_node(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    group_of: Dict[str, str] = {}
    for row in rows:
        group_id = _text(row.get("must_link_group"))
        for node_id in _text(row.get("members")).split(";"):
            if node_id:
                group_of[node_id] = group_id
    return group_of


def assignment_rows(
    nodes: Mapping[str, Mapping[str, Any]],
    cluster_of: Mapping[str, str],
    degree_map: Mapping[str, Mapping[str, Any]],
    data_hubs: Set[str],
    must_link_rows: Sequence[Mapping[str, Any]],
) -> List[dict]:
    size_map = Counter(cluster_of.values())
    group_of = must_link_group_by_node(must_link_rows)
    rows: List[dict] = []
    for node_id in sorted(
        cluster_of,
        key=lambda node: (cluster_of[node], _node_type(node, nodes.get(node, {})), node),
    ):
        row = nodes.get(node_id, {})
        degrees = degree_map.get(node_id, {})
        warnings: List[str] = []
        if node_id in data_hubs:
            warnings.append("data_hub")
        if node_id in group_of:
            warnings.append("must_link")
        rows.append(
            {
                "cluster_id": cluster_of[node_id],
                "cluster_size": size_map[cluster_of[node_id]],
                "node": node_id,
                "node_type": _node_type(node_id, row),
                "label": _node_label(node_id, row),
                "kind": _text(row.get("kind")),
                "module": _text(row.get("module")),
                "qualname": _text(row.get("qualname")),
                "owner": _text(row.get("owner")),
                "in_degree": _int(degrees.get("in_degree")),
                "out_degree": _int(degrees.get("out_degree")),
                "total_degree": _int(degrees.get("total_degree")),
                "weighted_degree": f"{_float(degrees.get('weighted_degree')):.6f}",
                "must_link_group": group_of.get(node_id, ""),
                "warnings": ";".join(warnings),
                "file": _text(row.get("file")),
                "lineno": _text(row.get("lineno")),
            }
        )
    return rows


def _cluster_stats_for_sweep(
    cluster_of: Mapping[str, str],
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Edge],
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict:
    groups: Dict[str, List[str]] = defaultdict(list)
    for node_id, cluster_id in cluster_of.items():
        groups[cluster_id].append(node_id)
    sizes = sorted((len(members) for members in groups.values()), reverse=True)
    mixed = 0
    for members in groups.values():
        types = {_node_type(node_id, nodes.get(node_id, {})) for node_id in members}
        if "callable" in types and "data" in types:
            mixed += 1
    cohesions = [_float(row.get("cohesion")) for row in summary_rows]
    mean_cohesion = sum(cohesions) / len(cohesions) if cohesions else 0.0
    partition_stats = _partition_quality_stats(cluster_of, edges)
    true_sm = mean_cohesion - _float(partition_stats.get("mean_cluster_coupling"))
    return {
        "num_clusters": len(groups),
        "mixed_clusters": mixed,
        "max_cluster_size": sizes[0] if sizes else 0,
        "median_cluster_size": sizes[len(sizes) // 2] if sizes else 0,
        "size_distribution": ";".join(str(size) for size in sizes[:20]),
        "mean_cohesion": f"{mean_cohesion:.6f}",
        "true_sm": f"{true_sm:.6f}",
        **partition_stats,
    }


def _partition_quality_stats(cluster_of: Mapping[str, str], edges: Sequence[Edge]) -> dict:
    """Return sweep-level weighted cut and modularity metrics.

    The structural clustering graph is undirected for Leiden, so these metrics
    use the same view: every clustered structural edge contributes once to total
    edge weight, regardless of its source orientation in the artifact.
    """
    internal_weight = 0.0
    external_weight = 0.0
    degree_by_node: Dict[str, float] = defaultdict(float)
    internal_by_cluster: Dict[str, float] = defaultdict(float)
    external_incident_by_cluster: Dict[str, float] = defaultdict(float)
    clusters = set(cluster_of.values())

    for edge in edges:
        src_cluster = cluster_of.get(edge.src)
        dst_cluster = cluster_of.get(edge.dst)
        if not src_cluster or not dst_cluster:
            continue

        weight = edge.weight
        if edge.src == edge.dst:
            degree_by_node[edge.src] += 2.0 * weight
            internal_weight += weight
            internal_by_cluster[src_cluster] += weight
            continue

        degree_by_node[edge.src] += weight
        degree_by_node[edge.dst] += weight
        if src_cluster == dst_cluster:
            internal_weight += weight
            internal_by_cluster[src_cluster] += weight
        else:
            external_weight += weight
            external_incident_by_cluster[src_cluster] += weight
            external_incident_by_cluster[dst_cluster] += weight

    total_weight = internal_weight + external_weight
    cluster_degree: Dict[str, float] = defaultdict(float)
    for node_id, degree in degree_by_node.items():
        cluster_id = cluster_of.get(node_id)
        if cluster_id:
            cluster_degree[cluster_id] += degree

    if total_weight > 0:
        coupling = external_weight / total_weight
        newman_modularity_q = sum(
            (internal_by_cluster[cluster_id] / total_weight)
            - (cluster_degree[cluster_id] / (2.0 * total_weight)) ** 2
            for cluster_id in clusters
        )
    else:
        coupling = 0.0
        newman_modularity_q = 0.0

    cluster_couplings: List[float] = []
    for cluster_id in clusters:
        degree = cluster_degree.get(cluster_id, 0.0)
        if degree <= 0:
            cluster_couplings.append(0.0)
        else:
            cluster_couplings.append(external_incident_by_cluster[cluster_id] / degree)
    mean_cluster_coupling = (
        sum(cluster_couplings) / len(cluster_couplings) if cluster_couplings else 0.0
    )

    return {
        "internal_weight": f"{internal_weight:.6f}",
        "external_weight": f"{external_weight:.6f}",
        "coupling": f"{coupling:.6f}",
        "mean_cluster_coupling": f"{mean_cluster_coupling:.6f}",
        "newman_modularity_Q": f"{newman_modularity_q:.6f}",
    }


def _format_sweep_metric(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _evaluation_metrics_for_sweep(
    nodes: Mapping[str, Mapping[str, Any]],
    result: ClusterResult,
    manual_rows: Optional[Sequence[Mapping[str, Any]]],
    manual_fields: Sequence[str],
    manual_label_column: Optional[str],
    node_mode: str,
    na_labels: Sequence[str],
    evaluation_node_types: Sequence[str],
    evaluation_kind_tokens: Sequence[str],
    all_evaluation_nodes: bool,
) -> Dict[str, Any]:
    if manual_rows is None:
        return {}

    data_hubs = {row["node"] for row in result.hub_nodes if row["node_type"] == "data"}
    cluster_rows = assignment_rows(
        nodes=nodes,
        cluster_of=result.cluster_of,
        degree_map=result.degree_map,
        data_hubs=data_hubs,
        must_link_rows=result.must_link_groups,
    )
    cluster_fields = list(cluster_rows[0].keys()) if cluster_rows else []
    payload = evaluate_assignment_rows(
        manual_rows=manual_rows,
        manual_fields=manual_fields,
        cluster_rows=cluster_rows,
        cluster_fields=cluster_fields,
        manual_label_column=manual_label_column,
        node_mode=node_mode,
        na_labels=set(na_labels),
        evaluation_node_types=evaluation_node_types,
        evaluation_kind_tokens=evaluation_kind_tokens,
        all_evaluation_nodes=all_evaluation_nodes,
    )
    return {
        key: _format_sweep_metric(value)
        for key, value in evaluation_summary_row(payload).items()
    }


def _optional_float(value: Any) -> Optional[float]:
    text = _text(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def _numbers_match(value: Any, expected: Optional[float]) -> bool:
    if expected is None:
        return True
    actual = _optional_float(value)
    if actual is None:
        return False
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def _normalize_sweep_best_metric(metric: str) -> str:
    name = metric.strip()
    if not name:
        return DEFAULT_SWEEP_BEST_METRIC
    # Evaluation metrics in parameter_sweep.csv carry the evaluation_ prefix,
    # but accepting the shorter CLI form keeps commands readable.
    if f"evaluation_{name}" in SWEEP_EVALUATION_FIELDNAMES:
        return f"evaluation_{name}"
    return name


def _sweep_best_filters_config(options: SweepBestSelectionOptions) -> dict:
    return {
        "resolution": options.resolution,
        "markov_time": options.markov_time,
        "hac_n_clusters": options.hac_n_clusters,
        "call_resolution": options.call_resolution,
        "data_access_resolution": options.data_access_resolution,
        "data_lineage_resolution": options.data_lineage_resolution,
        "data_hub_policy": options.data_hub_policy,
        "min_metric": options.min_metric,
        "min_value": options.min_value,
    }


def _has_explicit_sweep_best_filter(options: SweepBestSelectionOptions) -> bool:
    filters = _sweep_best_filters_config(options)
    return any(
        value is not None and value != "auto"
        for value in filters.values()
    )


def _row_matches_sweep_best_filters(
    row: Mapping[str, Any],
    options: SweepBestSelectionOptions,
) -> bool:
    if options.data_hub_policy != "auto" and row.get("data_hub_policy") != options.data_hub_policy:
        return False
    if not _numbers_match(row.get("resolution"), options.resolution):
        return False
    if not _numbers_match(row.get("markov_time"), options.markov_time):
        return False
    if options.hac_n_clusters is not None:
        hac_value = _optional_float(row.get("hac_n_clusters"))
        if hac_value is None or int(hac_value) != options.hac_n_clusters:
            return False
    if not _numbers_match(row.get("call_resolution"), options.call_resolution):
        return False
    if not _numbers_match(row.get("data_access_resolution"), options.data_access_resolution):
        return False
    if not _numbers_match(row.get("data_lineage_resolution"), options.data_lineage_resolution):
        return False
    if options.min_value is not None:
        min_metric = _normalize_sweep_best_metric(options.min_metric or options.metric)
        min_metric_value = _optional_float(row.get(min_metric))
        if min_metric_value is None or min_metric_value <= options.min_value:
            return False
    return True


def select_sweep_best_row(
    rows: Sequence[Mapping[str, Any]],
    options: SweepBestSelectionOptions,
) -> SweepBestSelection:
    """Choose the sweep row that should be rerun and written as full artifacts.

    The default selector is evaluation-driven: maximize predicted_match_f1.
    Explicit row filters, such as a fixed resolution or data-hub policy, narrow
    the candidate rows first. Ties are stable: the first matching sweep row wins.
    """

    metric = _normalize_sweep_best_metric(options.metric)
    candidates = [
        (index, row)
        for index, row in enumerate(rows)
        if _row_matches_sweep_best_filters(row, options)
    ]
    if not candidates:
        return SweepBestSelection(
            selected_index=None,
            selected_row=None,
            metric=metric,
            metric_direction=options.metric_direction,
            candidate_count=len(rows),
            filtered_count=0,
            reason="no_sweep_rows_matched_selection_filters",
        )

    best_index: Optional[int] = None
    best_row: Optional[Mapping[str, Any]] = None
    best_score: Optional[float] = None
    for index, row in candidates:
        score = _optional_float(row.get(metric))
        if score is None:
            continue
        if best_score is None:
            best_index, best_row, best_score = index, row, score
            continue
        if options.metric_direction == "min":
            is_better = score < best_score
        else:
            is_better = score > best_score
        if is_better:
            best_index, best_row, best_score = index, row, score

    if best_index is not None and best_row is not None:
        return SweepBestSelection(
            selected_index=best_index,
            selected_row=dict(best_row),
            metric=metric,
            metric_direction=options.metric_direction,
            candidate_count=len(rows),
            filtered_count=len(candidates),
            reason="selected_by_metric",
        )

    if len(candidates) == 1 and _has_explicit_sweep_best_filter(options):
        index, row = candidates[0]
        return SweepBestSelection(
            selected_index=index,
            selected_row=dict(row),
            metric=metric,
            metric_direction=options.metric_direction,
            candidate_count=len(rows),
            filtered_count=len(candidates),
            reason="selected_by_unique_filter_without_metric",
        )

    return SweepBestSelection(
        selected_index=None,
        selected_row=None,
        metric=metric,
        metric_direction=options.metric_direction,
        candidate_count=len(rows),
        filtered_count=len(candidates),
        reason="selection_metric_missing_or_non_numeric",
    )


def mark_selected_sweep_row(
    rows: Sequence[dict],
    selection: SweepBestSelection,
) -> None:
    for index, row in enumerate(rows):
        row["selected_best"] = "yes" if index == selection.selected_index else ""


def sweep_options_from_row(
    base_options: ClusterOptions,
    row: Mapping[str, Any],
) -> ClusterOptions:
    """Recreate the clustering options represented by one parameter-sweep row."""

    updates: Dict[str, Any] = {
        "run_sweep": False,
        "sweep_resolutions": tuple(),
        "sweep_markov_times": tuple(),
        "sweep_hac_n_clusters": tuple(),
        "sweep_call_resolutions": tuple(),
        "sweep_data_access_resolutions": tuple(),
        "sweep_data_lineage_resolutions": tuple(),
        "drop_data_hubs": row.get("data_hub_policy") == "drop_data_hubs",
    }
    if _text(row.get("resolution")).strip():
        updates["resolution"] = float(_text(row.get("resolution")))
    if _text(row.get("markov_time")).strip():
        updates["markov_time"] = float(_text(row.get("markov_time")))
    if _text(row.get("hac_n_clusters")).strip():
        updates["hac_n_clusters"] = int(float(_text(row.get("hac_n_clusters"))))
    if _text(row.get("call_resolution")).strip():
        updates["call_resolution"] = float(_text(row.get("call_resolution")))
    if _text(row.get("data_access_resolution")).strip():
        updates["data_access_resolution"] = float(_text(row.get("data_access_resolution")))
    if _text(row.get("data_lineage_resolution")).strip():
        updates["data_lineage_resolution"] = float(_text(row.get("data_lineage_resolution")))
    return replace(base_options, **updates)


def _has_layer_resolution_sweep(options: ClusterOptions) -> bool:
    return bool(
        options.sweep_call_resolutions
        or options.sweep_data_access_resolutions
        or options.sweep_data_lineage_resolutions
    )


def _layer_resolution_sweep_values(
    options: ClusterOptions,
) -> Tuple[Tuple[float, float, float], ...]:
    resolved = _edge_type_layer_resolutions(options)
    call_values = options.sweep_call_resolutions or (resolved["call"],)
    data_default = resolved.get("data_access", resolved.get("data", options.resolution))
    data_access_values = options.sweep_data_access_resolutions or (
        data_default,
    )
    lineage_default = resolved.get("data_lineage", options.resolution)
    data_lineage_values = options.sweep_data_lineage_resolutions or (
        lineage_default,
    )
    return tuple(product(call_values, data_access_values, data_lineage_values))


def run_parameter_sweep(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Edge],
    options: ClusterOptions,
    manual_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    manual_fields: Sequence[str] = tuple(),
    manual_label_column: Optional[str] = None,
    node_mode: str = "auto",
    na_labels: Sequence[str] = NA_DEFAULTS,
    evaluation_node_types: Sequence[str] = DEFAULT_EVALUATION_NODE_TYPES,
    evaluation_kind_tokens: Sequence[str] = DEFAULT_EVALUATION_KIND_TOKENS,
    all_evaluation_nodes: bool = False,
) -> List[dict]:
    sweep_parameter: str
    sweep_values: Sequence[Any]
    if options.algorithm == LEIDEN_MULTIPLEX_ALGORITHM and _has_layer_resolution_sweep(options):
        sweep_parameter = "layer_resolution"
        sweep_values = _layer_resolution_sweep_values(options)
    elif options.algorithm in LEIDEN_ALGORITHMS:
        sweep_parameter = "resolution"
        sweep_values = options.sweep_resolutions
    elif options.algorithm == "infomap":
        sweep_parameter = "markov_time"
        sweep_values = options.sweep_markov_times
    elif options.algorithm == HAC_CALLABLE_PROJECTION_ALGORITHM:
        sweep_parameter = "hac_n_clusters"
        sweep_values = options.sweep_hac_n_clusters
    else:
        return []

    if not sweep_values:
        return []

    rows: List[dict] = []
    for hub_policy, drop_data_hubs in (
        ("drop_data_hubs", True),
        ("keep_data_hubs", False),
    ):
        for sweep_value in sweep_values:
            updates: Dict[str, Any] = {
                "run_sweep": False,
                "sweep_resolutions": tuple(),
                "sweep_markov_times": tuple(),
                "sweep_hac_n_clusters": tuple(),
                "drop_data_hubs": drop_data_hubs,
            }
            if sweep_parameter == "resolution":
                if isinstance(sweep_value, tuple):
                    raise ValueError(f"Invalid scalar resolution sweep value: {sweep_value}")
                updates["resolution"] = float(sweep_value)
            elif sweep_parameter == "markov_time":
                if isinstance(sweep_value, tuple):
                    raise ValueError(f"Invalid scalar markov-time sweep value: {sweep_value}")
                updates["markov_time"] = float(sweep_value)
            elif sweep_parameter == "hac_n_clusters":
                if isinstance(sweep_value, tuple):
                    raise ValueError(f"Invalid scalar HAC sweep value: {sweep_value}")
                updates["hac_n_clusters"] = int(sweep_value)
            else:
                if not isinstance(sweep_value, tuple) or len(sweep_value) != 3:
                    raise ValueError(f"Invalid layer-resolution sweep value: {sweep_value}")
                call_resolution, data_access_resolution, data_lineage_resolution = (
                    float(sweep_value[0]),
                    float(sweep_value[1]),
                    float(sweep_value[2]),
                )
                updates["call_resolution"] = call_resolution
                updates["data_access_resolution"] = data_access_resolution
                updates["data_lineage_resolution"] = data_lineage_resolution
            sweep_options = replace(options, **updates)
            scaled_edges = scale_edges_for_clustering(edges, sweep_options)
            result = cluster_structural_graph(nodes, edges, sweep_options)
            stats = _cluster_stats_for_sweep(
                result.cluster_of, nodes, scaled_edges, result.cluster_summary
            )
            layer_resolutions = _edge_type_layer_resolutions(sweep_options)
            call_layer_resolution = layer_resolutions.get("call")
            data_layer_resolution = layer_resolutions.get(
                "data_access",
                layer_resolutions.get("data"),
            )
            data_lineage_layer_resolution = layer_resolutions.get("data_lineage")
            evaluation_metrics = _evaluation_metrics_for_sweep(
                nodes=nodes,
                result=result,
                manual_rows=manual_rows,
                manual_fields=manual_fields,
                manual_label_column=manual_label_column,
                node_mode=node_mode,
                na_labels=na_labels,
                evaluation_node_types=evaluation_node_types,
                evaluation_kind_tokens=evaluation_kind_tokens,
                all_evaluation_nodes=all_evaluation_nodes,
            )
            rows.append(
                {
                    "algorithm": options.algorithm,
                    "sweep_parameter": sweep_parameter,
                    "leiden_quality": (
                        options.leiden_quality if options.algorithm in LEIDEN_ALGORITHMS else ""
                    ),
                    "resolution": (
                        f"{sweep_options.resolution:g}"
                        if sweep_parameter == "resolution"
                        else ""
                    ),
                    "call_resolution": (
                        f"{call_layer_resolution:g}"
                        if options.algorithm == LEIDEN_MULTIPLEX_ALGORITHM
                        and call_layer_resolution is not None
                        else ""
                    ),
                    "data_access_resolution": (
                        f"{data_layer_resolution:g}"
                        if options.algorithm == LEIDEN_MULTIPLEX_ALGORITHM
                        and data_layer_resolution is not None
                        else ""
                    ),
                    "data_lineage_resolution": (
                        f"{data_lineage_layer_resolution:g}"
                        if options.algorithm == LEIDEN_MULTIPLEX_ALGORITHM
                        and data_lineage_layer_resolution is not None
                        else ""
                    ),
                    "markov_time": (
                        f"{sweep_options.markov_time:g}"
                        if sweep_parameter == "markov_time"
                        else ""
                    ),
                    "hac_n_clusters": (
                        str(sweep_options.hac_n_clusters)
                        if sweep_parameter == "hac_n_clusters"
                        else ""
                    ),
                    "data_hub_policy": hub_policy,
                    "nodes_clustered": len(result.cluster_of),
                    "hubs_dropped": sum(1 for row in result.excluded_nodes if row["reason"] == "data_hub"),
                    **stats,
                    **evaluation_metrics,
                }
            )
    return rows


def cluster_structural_graph(
    node_rows: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Edge],
    options: ClusterOptions,
) -> ClusterResult:
    edges = scale_edges_for_clustering(edges, options)
    nodes = ensure_edge_nodes({node_id: dict(row) for node_id, row in node_rows.items()}, edges)
    degree_map = compute_degrees(nodes, edges)
    callable_hubs, data_hubs, hub_rows = identify_hubs(nodes, edges, degree_map, options)
    excluded, excluded_rows = build_exclusions(nodes, degree_map, callable_hubs, data_hubs, options)
    orphaned_data, orphaned_rows = build_orphaned_data_exclusions(
        nodes,
        edges,
        degree_map,
        excluded,
        callable_hubs,
    )
    if orphaned_data:
        excluded.update(orphaned_data)
        excluded_rows.extend(orphaned_rows)
        excluded_rows.sort(key=lambda row: row["node"])
    active_nodes = set(nodes) - excluded
    local_callable_links = local_callable_must_link_pairs(active_nodes)

    if options.algorithm == LEIDEN_REWEIGHTED_ALGORITHM:
        must_link_relations = reweighted_must_link_relations(options.weight_config)
        extra_must_link_pairs = local_callable_links + tuple(
            (callable_id, data_id, "single_writer")
            for callable_id, data_id in sorted(single_writer_must_link_pairs(edges))
        )
    elif options.algorithm == LEIDEN_MULTIPLEX_ALGORITHM:
        must_link_relations = multiplex_must_link_relations(
            options.weight_config,
            STRONG_MUST_LINK_RELATIONS,
        )
        extra_must_link_pairs = local_callable_links
    else:
        must_link_relations = STRONG_MUST_LINK_RELATIONS
        extra_must_link_pairs = local_callable_links
    supernode_of, members_of, must_link_rows = build_must_link_groups(
        nodes,
        edges,
        active_nodes,
        strong_relations=must_link_relations,
        extra_must_link_pairs=extra_must_link_pairs,
    )
    supernodes = set(members_of)
    directed_edges, undirected_edges, typed_undirected_edges = build_contracted_edge_views(
        edges, active_nodes, supernode_of
    )
    super_cluster_of = cluster_supernodes(
        supernodes,
        directed_edges,
        undirected_edges,
        typed_undirected_edges,
        members_of,
        options,
    )
    if options.algorithm == HAC_CALLABLE_PROJECTION_ALGORITHM:
        cluster_of = expand_cluster_assignments(super_cluster_of, members_of)
        hac_assignment = assign_hac_mutating_data(
            cluster_of=cluster_of,
            nodes=nodes,
            edges=edges,
            active_nodes=active_nodes,
            supernode_of=supernode_of,
            members_of=members_of,
        )
        cluster_of = reindex_clusters(hac_assignment.cluster_of)
        excluded_rows.extend(
            _exclusion_row(node_id, nodes.get(node_id, {}), reason, degree_map)
            for node_id, reason in sorted(hac_assignment.excluded_reasons.items())
        )
        excluded_rows.sort(key=lambda row: row["node"])
    else:
        cluster_of = expand_cluster_assignments(super_cluster_of, members_of)
        postprocess_result = postprocess_data_only_clusters(cluster_of, edges)
        if postprocess_result.removed_nodes or postprocess_result.reassigned_nodes:
            cluster_of = reindex_clusters(postprocess_result.cluster_of)
            excluded_rows.extend(
                _exclusion_row(
                    node_id,
                    nodes.get(node_id, {}),
                    "data_only_read_only",
                    degree_map,
                )
                for node_id in postprocess_result.removed_nodes
            )
            excluded_rows.sort(key=lambda row: row["node"])

    cycle_findings = build_cycle_findings(nodes, edges, cluster_of)
    cluster_edges = compute_cluster_edges(edges, cluster_of)
    cluster_summary = compute_cluster_summary(
        nodes=nodes,
        edges=edges,
        cluster_of=cluster_of,
        data_hubs=data_hubs,
        cycle_findings=cycle_findings,
        degree_map=degree_map,
    )
    hub_cluster_links = build_hub_cluster_links(edges, cluster_of, hub_rows)

    return ClusterResult(
        cluster_of=cluster_of,
        cluster_summary=cluster_summary,
        cluster_edges=cluster_edges,
        excluded_nodes=excluded_rows,
        hub_nodes=hub_rows,
        hub_cluster_links=hub_cluster_links,
        must_link_groups=must_link_rows,
        cycle_findings=cycle_findings,
        degree_map=degree_map,
        options=options,
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv_rows(path, fieldnames, rows)


def write_cycle_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    node_cycles = [row for row in rows if row.get("level") == "node"]
    cluster_cycles = [row for row in rows if row.get("level") == "cluster"]
    lines = [
        "# Cycle Findings",
        "",
        f"- Node-level cycles: `{len(node_cycles)}`",
        f"- Cluster-level cycles: `{len(cluster_cycles)}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('finding_id')} ({row.get('level')})",
                "",
                f"- Size: `{row.get('size')}`",
                f"- Members: `{row.get('members_preview')}`",
                f"- Internal edges: `{row.get('internal_edge_count')}`",
                f"- Edge types: `{row.get('edge_types')}`",
                f"- Relations: `{row.get('relations')}`",
                "",
            ]
        )
    write_markdown(path, lines)


def write_sweep_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Structural Parameter Sweep", ""]
    if not rows:
        lines.append("No sweep was run.")
        write_markdown(path, lines, trailing_newline=True)
        return
    has_evaluation = any(
        _text(row.get("evaluation_adjusted_rand_index")) for row in rows
    )
    metric_header = ""
    metric_alignment = ""
    if has_evaluation:
        metric_header = (
            " | Eval ARI | Eval V | Eval Pairwise F1 | Eval BCubed F1"
            " | Eval PredMatch P | Eval PredMatch R | Eval PredMatch F1"
        )
        metric_alignment = " | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    lines.extend(
        [
            "| Best | Algorithm | Quality | Resolution | Markov Time | HAC k | Data Hub Policy | Nodes | Clusters | Mixed | Max Size | Median Size | Mean Cohesion | Mean Cluster Coupling | True SM | Coupling | Newman Modularity Q{metric_header} |".format(
                metric_header=metric_header
            ),
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:{metric_alignment} |".format(
                metric_alignment=metric_alignment
            ),
        ]
    )
    for row in rows:
        display_row = {
            "selected_best": "",
            "algorithm": "",
            "leiden_quality": "",
            "resolution": "",
            "call_resolution": "",
            "data_access_resolution": "",
            "data_lineage_resolution": "",
            "markov_time": "",
            "hac_n_clusters": "",
            **row,
        }
        resolution_text = _text(display_row.get("resolution"))
        if not resolution_text and _text(display_row.get("call_resolution")):
            resolution_text = "/".join(
                [
                    _text(display_row.get("call_resolution")),
                    _text(display_row.get("data_access_resolution")),
                    _text(display_row.get("data_lineage_resolution")),
                ]
            )
        display_row["resolution"] = resolution_text
        metric_cells = ""
        if has_evaluation:
            metric_cells = (
                f" | {_text(row.get('evaluation_adjusted_rand_index'))}"
                f" | {_text(row.get('evaluation_v_measure'))}"
                f" | {_text(row.get('evaluation_pairwise_f1'))}"
                f" | {_text(row.get('evaluation_bcubed_f1'))}"
                f" | {_text(row.get('evaluation_predicted_match_precision'))}"
                f" | {_text(row.get('evaluation_predicted_match_recall'))}"
                f" | {_text(row.get('evaluation_predicted_match_f1'))}"
            )
        lines.append(
            "| {selected_best} | {algorithm} | {leiden_quality} | {resolution} | {markov_time} | {hac_n_clusters} | {data_hub_policy} | {nodes_clustered} | {num_clusters} | "
            "{mixed_clusters} | {max_cluster_size} | {median_cluster_size} | {mean_cohesion} | "
            "{mean_cluster_coupling} | {true_sm} | {coupling} | {newman_modularity_Q}{metric_cells} |".format(
                metric_cells=metric_cells,
                **display_row,
            )
        )
    write_markdown(path, lines, trailing_newline=True)


def remove_legacy_sweep_outputs(outdir: Path) -> None:
    for name in ("parameter_sweep.csv", "parameter_sweep.md", "parameter_sweep.json"):
        path = outdir / name
        if path.is_file():
            path.unlink()


def write_outputs(outdir: Path, nodes: Mapping[str, Mapping[str, Any]], result: ClusterResult) -> None:
    ensure_dir(outdir)
    remove_legacy_sweep_outputs(outdir)
    assignments = assignment_rows(
        nodes=nodes,
        cluster_of=result.cluster_of,
        degree_map=result.degree_map,
        data_hubs={row["node"] for row in result.hub_nodes if row["node_type"] == "data"},
        must_link_rows=result.must_link_groups,
    )

    _write_csv(
        outdir / "cluster_assignments.csv",
        [
            "cluster_id",
            "cluster_size",
            "node",
            "node_type",
            "label",
            "kind",
            "module",
            "qualname",
            "owner",
            "in_degree",
            "out_degree",
            "total_degree",
            "weighted_degree",
            "must_link_group",
            "warnings",
            "file",
            "lineno",
        ],
        assignments,
    )
    _write_csv(
        outdir / "cluster_summary.csv",
        [
            "cluster_id",
            "size",
            "callable_count",
            "data_count",
            "anchor_node",
            "anchor_label",
            "internal_weight",
            "outgoing_weight",
            "incoming_weight",
            "cohesion",
            "top_modules",
            "data_kinds",
            "warnings",
            "members_preview",
        ],
        result.cluster_summary,
    )
    _write_csv(
        outdir / "cluster_edges.csv",
        ["src_cluster", "dst_cluster", "weight", "edge_count", "edge_types", "relations"],
        result.cluster_edges,
    )
    _write_csv(
        outdir / "excluded_nodes.csv",
        ["node", "node_type", "reason", "label", "kind", "total_degree", "weighted_degree", "file", "lineno"],
        result.excluded_nodes,
    )
    _write_csv(
        outdir / "hub_nodes.csv",
        [
            "node",
            "node_type",
            "label",
            "kind",
            "action",
            "candidate_types",
            "reasons",
            "in_degree",
            "out_degree",
            "total_degree",
            "weighted_out_degree",
            "weighted_degree",
            "callable_count",
            "access_count",
            "out_call_degree",
            "out_data_degree",
            "target_callable_count",
            "target_data_count",
            "target_module_count",
            "target_modules",
        ],
        result.hub_nodes,
    )
    _write_csv(
        outdir / "hub_cluster_links.csv",
        ["hub_node", "linked_cluster", "direction", "scope", "weight", "edge_count", "edge_types", "relations"],
        result.hub_cluster_links,
    )
    _write_csv(
        outdir / "must_link_groups.csv",
        ["must_link_group", "anchor_node", "anchor_label", "size", "relations", "members", "evidence"],
        result.must_link_groups,
    )
    _write_csv(
        outdir / "cycle_findings.csv",
        [
            "finding_id",
            "level",
            "size",
            "members",
            "members_preview",
            "node_types",
            "clusters",
            "internal_edge_count",
            "edge_types",
            "relations",
        ],
        result.cycle_findings,
    )
    write_cycle_report(outdir / "cycle_findings.md", result.cycle_findings)

    payload = {
        "schema": "structural_microservice_candidates.v1",
        "algorithm": result.options.algorithm,
        "leiden_quality": result.options.leiden_quality,
        "resolution": result.options.resolution,
        "markov_time": result.options.markov_time,
        "seed": result.options.seed,
        "num_nodes_clustered": len(result.cluster_of),
        "num_clusters": len(set(result.cluster_of.values())),
        "options": {
            "leiden_quality": result.options.leiden_quality,
            "multiplex_layer_mode": result.options.multiplex_layer_mode,
            "markov_time": result.options.markov_time,
            "call_resolution": result.options.call_resolution,
            "data_access_resolution": result.options.data_access_resolution,
            "data_lineage_resolution": result.options.data_lineage_resolution,
            "hac_n_clusters": result.options.hac_n_clusters,
            "callable_hub_policy": _effective_callable_hub_policy(result.options),
            "callable_hub_drop": list(result.options.callable_hub_drop),
            "callable_hub_keep": list(result.options.callable_hub_keep),
            "callable_hub_nodes_path": result.options.callable_hub_nodes_path,
            "data_hub_nodes_path": result.options.data_hub_nodes_path,
            "drop_callable_hubs": result.options.drop_callable_hubs,
            "drop_data_hubs": result.options.drop_data_hubs,
            "exclude_module_callables": result.options.exclude_module_callables,
            "call_weight_scale": result.options.call_weight_scale,
            "data_access_weight_scale": result.options.data_access_weight_scale,
            "data_lineage_weight_scale": result.options.data_lineage_weight_scale,
            "weight_config": result.options.weight_config,
            "hub_callable_degree_percentile": result.options.hub_callable_degree_percentile,
            "hub_callable_min_degree": result.options.hub_callable_min_degree,
            "hub_callable_min_in_degree": result.options.hub_callable_min_in_degree,
            "hub_callable_min_out_degree": result.options.hub_callable_min_out_degree,
            "hub_entrypoint_min_out_degree": result.options.hub_entrypoint_min_out_degree,
            "hub_orchestrator_max_in_degree": result.options.hub_orchestrator_max_in_degree,
            "hub_orchestrator_min_out_degree": result.options.hub_orchestrator_min_out_degree,
            "hub_orchestrator_min_out_call_degree": result.options.hub_orchestrator_min_out_call_degree,
            "hub_orchestrator_min_target_modules": result.options.hub_orchestrator_min_target_modules,
            "hub_orchestrator_min_target_callables": result.options.hub_orchestrator_min_target_callables,
            "hub_orchestrator_min_target_data": result.options.hub_orchestrator_min_target_data,
            "hub_orchestrator_min_data_to_call_ratio": result.options.hub_orchestrator_min_data_to_call_ratio,
            "hub_data_min_degree": result.options.hub_data_min_degree,
            "hub_data_min_callable_count": result.options.hub_data_min_callable_count,
            "hub_data_min_access_count": result.options.hub_data_min_access_count,
        },
        "summary": result.cluster_summary,
        "cluster_edges": result.cluster_edges,
        "excluded_nodes": result.excluded_nodes,
        "hub_nodes": result.hub_nodes,
        "must_link_groups": result.must_link_groups,
        "cycle_findings": result.cycle_findings,
    }
    write_json(outdir / "clusters.json", payload)


def write_sweep_outputs(
    outdir: Path, rows: Sequence[Mapping[str, Any]], options: ClusterOptions
) -> None:
    ensure_dir(outdir)
    fieldnames = [
        "selected_best",
        "algorithm",
        "sweep_parameter",
        "leiden_quality",
        "resolution",
        "call_resolution",
        "data_access_resolution",
        "data_lineage_resolution",
        "markov_time",
        "hac_n_clusters",
        "data_hub_policy",
        "nodes_clustered",
        "hubs_dropped",
        "num_clusters",
        "mixed_clusters",
        "max_cluster_size",
        "median_cluster_size",
        "size_distribution",
        "mean_cohesion",
        "mean_cluster_coupling",
        "true_sm",
        "internal_weight",
        "external_weight",
        "coupling",
        "newman_modularity_Q",
        *SWEEP_EVALUATION_FIELDNAMES,
    ]
    _write_csv(outdir / "parameter_sweep.csv", fieldnames, rows)
    write_sweep_report(outdir / "parameter_sweep.md", rows)
    write_json(
        outdir / "parameter_sweep.json",
        {
            "schema": "structural_microservice_parameter_sweep.v1",
            "algorithm": options.algorithm,
            "leiden_quality": options.leiden_quality,
            "multiplex_layer_mode": options.multiplex_layer_mode,
            "resolution": options.resolution,
            "call_resolution": options.call_resolution,
            "data_access_resolution": options.data_access_resolution,
            "data_lineage_resolution": options.data_lineage_resolution,
            "hac_n_clusters": options.hac_n_clusters,
            "markov_time": options.markov_time,
            "seed": options.seed,
            "hub_policy_axis": "data_hubs",
            "callable_hub_policy": _effective_callable_hub_policy(options),
            "callable_hub_drop": list(options.callable_hub_drop),
            "callable_hub_keep": list(options.callable_hub_keep),
            "callable_hub_nodes_path": options.callable_hub_nodes_path,
            "data_hub_nodes_path": options.data_hub_nodes_path,
            "drop_callable_hubs": options.drop_callable_hubs,
            "base_drop_data_hubs": options.drop_data_hubs,
            "exclude_module_callables": options.exclude_module_callables,
            "call_weight_scale": options.call_weight_scale,
            "data_access_weight_scale": options.data_access_weight_scale,
            "data_lineage_weight_scale": options.data_lineage_weight_scale,
            "weight_config": options.weight_config,
            "hub_callable_degree_percentile": options.hub_callable_degree_percentile,
            "hub_callable_min_degree": options.hub_callable_min_degree,
            "hub_callable_min_in_degree": options.hub_callable_min_in_degree,
            "hub_callable_min_out_degree": options.hub_callable_min_out_degree,
            "hub_entrypoint_min_out_degree": options.hub_entrypoint_min_out_degree,
            "hub_orchestrator_max_in_degree": options.hub_orchestrator_max_in_degree,
            "hub_orchestrator_min_out_degree": options.hub_orchestrator_min_out_degree,
            "hub_orchestrator_min_out_call_degree": options.hub_orchestrator_min_out_call_degree,
            "hub_orchestrator_min_target_modules": options.hub_orchestrator_min_target_modules,
            "hub_orchestrator_min_target_callables": options.hub_orchestrator_min_target_callables,
            "hub_orchestrator_min_target_data": options.hub_orchestrator_min_target_data,
            "hub_orchestrator_min_data_to_call_ratio": options.hub_orchestrator_min_data_to_call_ratio,
            "hub_data_min_degree": options.hub_data_min_degree,
            "hub_data_min_callable_count": options.hub_data_min_callable_count,
            "hub_data_min_access_count": options.hub_data_min_access_count,
            "sweep_resolutions": list(options.sweep_resolutions),
            "sweep_markov_times": list(options.sweep_markov_times),
            "sweep_hac_n_clusters": list(options.sweep_hac_n_clusters),
            "sweep_call_resolutions": list(options.sweep_call_resolutions),
            "sweep_data_access_resolutions": list(options.sweep_data_access_resolutions),
            "sweep_data_lineage_resolutions": list(options.sweep_data_lineage_resolutions),
            "rows": list(rows),
        },
    )


def _cluster_options_payload(options: ClusterOptions) -> dict:
    return {
        "algorithm": options.algorithm,
        "leiden_quality": options.leiden_quality,
        "multiplex_layer_mode": options.multiplex_layer_mode,
        "resolution": options.resolution,
        "call_resolution": options.call_resolution,
        "data_access_resolution": options.data_access_resolution,
        "data_lineage_resolution": options.data_lineage_resolution,
        "markov_time": options.markov_time,
        "hac_n_clusters": options.hac_n_clusters,
        "seed": options.seed,
        "drop_data_hubs": options.drop_data_hubs,
        "callable_hub_policy": _effective_callable_hub_policy(options),
    }


def write_sweep_best_selection(
    outdir: Path,
    selection: SweepBestSelection,
    selection_options: SweepBestSelectionOptions,
    best_outdir: Optional[Path],
    selected_options: Optional[ClusterOptions],
) -> None:
    ensure_dir(outdir)
    payload = {
        "schema": "structural_microservice_sweep_best_selection.v1",
        "selected": selection.selected,
        "reason": selection.reason,
        "metric": selection.metric,
        "metric_direction": selection.metric_direction,
        "candidate_count": selection.candidate_count,
        "filtered_count": selection.filtered_count,
        "selection_options": {
            "enabled": selection_options.enabled,
            "metric": selection_options.metric,
            "metric_direction": selection_options.metric_direction,
            "filters": _sweep_best_filters_config(selection_options),
        },
        "best_outdir": str(best_outdir) if best_outdir else "",
        "selected_row": selection.selected_row or {},
        "selected_cluster_options": (
            _cluster_options_payload(selected_options) if selected_options else {}
        ),
    }
    write_json(outdir / "sweep_best_selection.json", payload)

    lines = [
        "# Sweep Best Selection",
        "",
        f"- Selected: `{selection.selected}`",
        f"- Reason: `{selection.reason}`",
        f"- Metric: `{selection.metric}`",
        f"- Direction: `{selection.metric_direction}`",
        f"- Candidate rows: `{selection.candidate_count}`",
        f"- Rows after filters: `{selection.filtered_count}`",
        f"- Best output: `{best_outdir if best_outdir else ''}`",
        "",
    ]
    if selection.selected_row:
        lines.extend(
            [
                "## Selected Row",
                "",
                f"- Resolution: `{selection.selected_row.get('resolution', '')}`",
                f"- Markov time: `{selection.selected_row.get('markov_time', '')}`",
                f"- HAC k: `{selection.selected_row.get('hac_n_clusters', '')}`",
                f"- Call resolution: `{selection.selected_row.get('call_resolution', '')}`",
                f"- Data access resolution: `{selection.selected_row.get('data_access_resolution', '')}`",
                f"- Data lineage resolution: `{selection.selected_row.get('data_lineage_resolution', '')}`",
                f"- Data hub policy: `{selection.selected_row.get('data_hub_policy', '')}`",
                f"- Metric value: `{selection.selected_row.get(selection.metric, '')}`",
                "",
            ]
        )
    write_markdown(outdir / "sweep_best_selection.md", lines, trailing_newline=True)


def materialize_sweep_best_cluster(
    best_outdir: Path,
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Edge],
    base_options: ClusterOptions,
    selected_row: Mapping[str, Any],
) -> ClusterOptions:
    selected_options = sweep_options_from_row(base_options, selected_row)
    selected_result = cluster_structural_graph(nodes, edges, selected_options)
    write_outputs(best_outdir, nodes, selected_result)
    return selected_options


def _parse_sweep_resolutions(value: str) -> Tuple[float, ...]:
    stripped = value.strip()
    if not stripped:
        return tuple()
    if ":" in stripped and "," not in stripped:
        return _parse_sweep_range(stripped)
    return tuple(float(part.strip()) for part in stripped.split(",") if part.strip())


def _parse_sweep_ints(value: str) -> Tuple[int, ...]:
    if not value.strip():
        return tuple()
    values: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        number = float(part)
        if not number.is_integer():
            raise ValueError(f"Sweep value must be an integer: {part}")
        values.append(int(number))
    return tuple(values)


def _parse_sweep_range(value: str) -> Tuple[float, ...]:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 3 or not all(parts):
        raise ValueError("Sweep range must use START:END:STEP, for example 0.01:5:0.01")
    try:
        start, end, step = (Decimal(part) for part in parts)
    except InvalidOperation as exc:
        raise ValueError("Sweep range values must be numeric") from exc
    if step <= 0:
        raise ValueError("Sweep range step must be positive")
    if end < start:
        raise ValueError("Sweep range end must be greater than or equal to start")

    values: List[float] = []
    current = start
    # Include the end value when it lands exactly on the decimal step.
    while current <= end:
        values.append(float(current))
        current += step
    return tuple(values)


def _parse_sweep_int_range(value: str) -> Tuple[int, ...]:
    values = _parse_sweep_range(value)
    int_values: List[int] = []
    for number in values:
        if not float(number).is_integer():
            raise ValueError(f"Sweep range value must be an integer: {number:g}")
        int_values.append(int(number))
    return tuple(int_values)


def _default_sweep_resolutions(leiden_quality: str) -> Tuple[float, ...]:
    if leiden_quality == "cpm":
        return CPM_SWEEP_DEFAULTS
    return RB_CONFIGURATION_SWEEP_DEFAULTS


def _default_sweep_markov_times() -> Tuple[float, ...]:
    return INFOMAP_MARKOV_TIME_SWEEP_DEFAULTS


def _default_sweep_hac_n_clusters() -> Tuple[int, ...]:
    return tuple(range(10, 26))


def _resolve_clustering_weight_scales(
    args: argparse.Namespace,
    weight_config: StructuralWeightConfig,
) -> Tuple[float, float, float]:
    return (
        args.call_weight_scale
        if args.call_weight_scale is not None
        else weight_config.clustering_scale("call"),
        args.data_access_weight_scale
        if args.data_access_weight_scale is not None
        else weight_config.clustering_scale("data_access"),
        args.data_lineage_weight_scale
        if args.data_lineage_weight_scale is not None
        else weight_config.clustering_scale("data_lineage"),
    )


def _resolve_project_path(project_root: Path, value: Any) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _resolve_optional_project_path(project_root: Path, value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    return _resolve_project_path(project_root, value)


def _expand_algorithm_path_template(value: Any, algorithm: str) -> Any:
    if value in (None, ""):
        return value
    text = str(value)
    return text.replace("${algorithm}", algorithm).replace("{algorithm}", algorithm)


def _expand_algorithm_output_path_templates(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("outdir", "sweep_outdir", "sweep_best_outdir"):
        if hasattr(args, name):
            setattr(args, name, _expand_algorithm_path_template(getattr(args, name), args.algorithm))
    return args


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Structural clustering config section must be an object: {name}")
    return value


def _config_csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ",".join(_text(item) for item in value if _text(item))
    return value


def _config_string_list(value: Any, key: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_text(value)] if _text(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_text(item) for item in value if _text(item)]
    raise ValueError(f"Structural clustering config value must be a string or array: {key}")


def _set_config_default(
    defaults: Dict[str, Any],
    section: Mapping[str, Any],
    key: str,
    dest: Optional[str] = None,
    *,
    path: bool = False,
    csv_value: bool = False,
    string_list: bool = False,
    project_root: Optional[Path] = None,
) -> None:
    if key not in section:
        return
    value = section[key]
    if path:
        if project_root is None:
            raise ValueError("project_root is required for path config values")
        path_value = resolve_weight_config_reference(value, project_root=project_root) if key == "weight_config" else _resolve_optional_project_path(project_root, value)
        value = str(path_value) if path_value is not None else None
    elif csv_value:
        value = _config_csv_value(value)
    elif string_list:
        value = _config_string_list(value, key)
    defaults[dest or key] = value


def structural_config_defaults(
    config: Mapping[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    if not config:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("Structural clustering config must be a JSON object")

    defaults: Dict[str, Any] = {}
    paths = _config_section(config, "paths")
    algorithm = _config_section(config, "algorithm")
    evaluation = _config_section(config, "evaluation")
    sweep = _config_section(config, "sweep")
    sweep_best = _config_section(config, "sweep_best")
    hub_policy = _config_section(config, "hub_policy")
    weighting = _config_section(config, "weighting")

    for key in (
        "nodes",
        "edges",
        "callable_hub_nodes",
        "data_hub_nodes",
        "outdir",
        "sweep_outdir",
        "sweep_best_outdir",
        "sweep_manual",
    ):
        _set_config_default(defaults, paths, key, path=True, project_root=project_root)
    _set_config_default(defaults, paths, "manual_mapping", "sweep_manual", path=True, project_root=project_root)

    for key in (
        "algorithm",
        "leiden_quality",
        "multiplex_layer_mode",
        "resolution",
        "call_resolution",
        "data_access_resolution",
        "data_lineage_resolution",
        "markov_time",
        "hac_n_clusters",
        "seed",
        "max_iter",
    ):
        _set_config_default(defaults, algorithm, key)

    for key in ("run_sweep", "sweep_range", "sweep_markov_range"):
        _set_config_default(defaults, sweep, key)
    for key in (
        "sweep_resolutions",
        "sweep_markov_times",
        "sweep_hac_n_clusters",
        "sweep_call_resolutions",
        "sweep_data_access_resolutions",
        "sweep_data_lineage_resolutions",
        "sweep_evaluation_node_types",
        "sweep_evaluation_kind_tokens",
    ):
        _set_config_default(defaults, sweep, key, csv_value=True)
    _set_config_default(defaults, sweep, "resolutions", "sweep_resolutions", csv_value=True)
    _set_config_default(defaults, sweep, "markov_times", "sweep_markov_times", csv_value=True)
    _set_config_default(defaults, sweep, "hac_n_clusters", "sweep_hac_n_clusters", csv_value=True)
    _set_config_default(defaults, sweep, "call_resolutions", "sweep_call_resolutions", csv_value=True)
    _set_config_default(defaults, sweep, "data_access_resolutions", "sweep_data_access_resolutions", csv_value=True)
    _set_config_default(defaults, sweep, "data_lineage_resolutions", "sweep_data_lineage_resolutions", csv_value=True)
    _set_config_default(defaults, sweep, "range", "sweep_range")
    _set_config_default(defaults, sweep, "markov_range", "sweep_markov_range")
    _set_config_default(defaults, sweep, "outdir", "sweep_outdir", path=True, project_root=project_root)
    _set_config_default(defaults, sweep, "manual", "sweep_manual", path=True, project_root=project_root)
    _set_config_default(defaults, sweep, "manual_mapping", "sweep_manual", path=True, project_root=project_root)
    _set_config_default(defaults, sweep, "manual_label_column", "sweep_manual_label_column")
    _set_config_default(defaults, sweep, "node_mode", "sweep_node_mode")
    _set_config_default(defaults, sweep, "na_labels", "sweep_na_label", string_list=True)
    _set_config_default(defaults, sweep, "evaluation_node_types", "sweep_evaluation_node_types", csv_value=True)
    _set_config_default(defaults, sweep, "evaluation_kind_tokens", "sweep_evaluation_kind_tokens", csv_value=True)
    _set_config_default(defaults, sweep, "all_evaluation_nodes", "sweep_all_evaluation_nodes")
    if "evaluation_enabled" in sweep:
        defaults["no_sweep_evaluation"] = not bool(sweep["evaluation_enabled"])
    _set_config_default(defaults, sweep, "no_sweep_evaluation")

    _set_config_default(defaults, sweep_best, "select_sweep_best")
    _set_config_default(defaults, sweep_best, "enabled", "select_sweep_best")
    _set_config_default(defaults, sweep_best, "outdir", "sweep_best_outdir", path=True, project_root=project_root)
    _set_config_default(defaults, sweep_best, "metric", "sweep_best_metric")
    _set_config_default(defaults, sweep_best, "metric_direction", "sweep_best_metric_direction")
    _set_config_default(defaults, sweep_best, "resolution", "sweep_best_resolution")
    _set_config_default(defaults, sweep_best, "markov_time", "sweep_best_markov_time")
    _set_config_default(defaults, sweep_best, "hac_n_clusters", "sweep_best_hac_n_clusters")
    _set_config_default(defaults, sweep_best, "call_resolution", "sweep_best_call_resolution")
    _set_config_default(defaults, sweep_best, "data_access_resolution", "sweep_best_data_access_resolution")
    _set_config_default(defaults, sweep_best, "data_lineage_resolution", "sweep_best_data_lineage_resolution")
    _set_config_default(defaults, sweep_best, "data_hub_policy", "sweep_best_data_hub_policy")
    _set_config_default(defaults, sweep_best, "min_metric", "sweep_best_min_metric")
    _set_config_default(defaults, sweep_best, "minimum_metric", "sweep_best_min_metric")
    _set_config_default(defaults, sweep_best, "min_value", "sweep_best_min_value")
    _set_config_default(defaults, sweep_best, "minimum_value", "sweep_best_min_value")

    _set_config_default(defaults, hub_policy, "drop_callable_hubs")
    _set_config_default(defaults, hub_policy, "callable_hub_policy")
    _set_config_default(defaults, hub_policy, "callable_hub_decisions", path=True, project_root=project_root)
    _set_config_default(defaults, hub_policy, "drop_callable_hub", string_list=True)
    _set_config_default(defaults, hub_policy, "keep_callable_hub", string_list=True)
    _set_config_default(defaults, hub_policy, "drop_data_hubs")
    _set_config_default(defaults, hub_policy, "exclude_module_callables")
    for key in (
        "hub_callable_degree_percentile",
        "hub_callable_min_degree",
        "hub_callable_min_in_degree",
        "hub_callable_min_out_degree",
        "hub_entrypoint_min_out_degree",
        "hub_orchestrator_max_in_degree",
        "hub_orchestrator_min_out_degree",
        "hub_orchestrator_min_out_call_degree",
        "hub_orchestrator_min_target_modules",
        "hub_orchestrator_min_target_callables",
        "hub_orchestrator_min_target_data",
        "hub_orchestrator_min_data_to_call_ratio",
        "hub_data_min_degree",
        "hub_data_min_callable_count",
        "hub_data_min_access_count",
    ):
        _set_config_default(defaults, hub_policy, key)

    _set_config_default(defaults, weighting, "weight_config", path=True, project_root=project_root)
    _set_config_default(defaults, weighting, "call_weight_scale")
    _set_config_default(defaults, weighting, "data_access_weight_scale")
    _set_config_default(defaults, weighting, "data_lineage_weight_scale")
    _set_config_default(defaults, weighting, "call_scale", "call_weight_scale")
    _set_config_default(defaults, weighting, "data_access_scale", "data_access_weight_scale")
    _set_config_default(defaults, weighting, "data_lineage_scale", "data_lineage_weight_scale")

    return defaults


def load_structural_config_defaults(
    config_path: Optional[Path],
    project_root: Path,
) -> Dict[str, Any]:
    if config_path is None:
        return {}
    payload = load_jsonc(config_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Structural clustering config must be a JSON object: {config_path}")
    return structural_config_defaults(payload, project_root)


def _preparse_project_and_config(argv: Optional[Sequence[str]]) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=None)
    args, _unknown = parser.parse_known_args(argv)
    project_root = Path(args.project_root).resolve()
    config_path = _resolve_optional_project_path(project_root, args.config)
    defaults = load_structural_config_defaults(config_path, project_root)
    defaults["project_root"] = str(project_root)
    defaults["config"] = str(config_path) if config_path else None
    return project_root, config_path, defaults


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    _project_root, _config_path, config_defaults = _preparse_project_and_config(argv)
    parser = argparse.ArgumentParser(description="Cluster a heterogeneous structural dependency graph")
    parser.add_argument("--project-root", default=".", help="Project root used to resolve config-relative paths")
    parser.add_argument("--config", default=None, help="Optional structural clustering JSON/JSONC config")
    parser.add_argument("--nodes", default="artifacts/structural_dependency_graph/nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="artifacts/structural_dependency_graph/edges.csv", help="Path to edges.csv")
    parser.add_argument(
        "--callable-hub-nodes",
        type=Path,
        default=None,
        help=(
            "Path to callable_hub_nodes.csv from the structural graph stage. "
            "Defaults to the nodes.csv directory when present."
        ),
    )
    parser.add_argument(
        "--data-hub-nodes",
        type=Path,
        default=None,
        help=(
            "Path to data_hub_nodes.csv from the structural graph stage. "
            "Defaults to the nodes.csv directory when present."
        ),
    )
    parser.add_argument("--outdir", default="artifacts/structural_microservice_candidates", help="Output directory")
    parser.add_argument(
        "--run-sweep",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Also run an algorithm-specific parameter sweep and write it to a separate sweep output directory",
    )
    parser.add_argument(
        "--sweep-outdir",
        default=None,
        help="Output directory for sweep artifacts; defaults to OUTDIR/sweep_results",
    )
    parser.add_argument(
        "--sweep-manual",
        type=Path,
        default=DEFAULT_MANUAL,
        help="Manual mapping CSV used to add evaluation metrics to sweep rows",
    )
    parser.add_argument(
        "--sweep-manual-label-column",
        default=None,
        help="Manual microservice label column for sweep evaluation",
    )
    parser.add_argument(
        "--sweep-node-mode",
        choices=("auto", "exact", "callable"),
        default="auto",
        help="Node matching mode for sweep evaluation",
    )
    parser.add_argument(
        "--sweep-na-label",
        action="append",
        default=None,
        help="Manual label to treat as unknown/unadjudicated in sweep evaluation. Can be repeated.",
    )
    parser.add_argument(
        "--sweep-evaluation-node-types",
        default=",".join(DEFAULT_EVALUATION_NODE_TYPES),
        help="Comma-separated node_type values to include in sweep evaluation.",
    )
    parser.add_argument(
        "--sweep-evaluation-kind-tokens",
        default=",".join(DEFAULT_EVALUATION_KIND_TOKENS),
        help="Comma-separated semicolon-tokenized kind values to include in sweep evaluation.",
    )
    parser.add_argument(
        "--sweep-all-evaluation-nodes",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Evaluate all joined manual and cluster rows during sweeps.",
    )
    parser.add_argument(
        "--no-sweep-evaluation",
        action="store_true",
        help="Do not add manual clustering evaluation metrics to sweep rows",
    )
    parser.add_argument(
        "--select-sweep-best",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "When a sweep is run, materialize one selected sweep row as full cluster "
            "artifacts under the sweep output directory. Enabled by default."
        ),
    )
    parser.add_argument(
        "--sweep-best-outdir",
        default=None,
        help="Output directory for the selected sweep cluster; defaults to SWEEP_OUTDIR/best",
    )
    parser.add_argument(
        "--sweep-best-metric",
        default=DEFAULT_SWEEP_BEST_METRIC,
        help=(
            "Metric used for automatic sweep best selection. Evaluation metric names may "
            "be passed with or without the evaluation_ prefix. Defaults to predicted_match_f1."
        ),
    )
    parser.add_argument(
        "--sweep-best-metric-direction",
        choices=("max", "min"),
        default="max",
        help="Whether the selected sweep metric should be maximized or minimized.",
    )
    parser.add_argument(
        "--sweep-best-resolution",
        type=float,
        default=None,
        help="Select only sweep rows with this scalar Leiden resolution before metric ranking.",
    )
    parser.add_argument(
        "--sweep-best-markov-time",
        type=float,
        default=None,
        help="Select only sweep rows with this Infomap Markov time before metric ranking.",
    )
    parser.add_argument(
        "--sweep-best-hac-n-clusters",
        type=int,
        default=None,
        help="Select only sweep rows with this HAC target cluster count before metric ranking.",
    )
    parser.add_argument(
        "--sweep-best-call-resolution",
        type=float,
        default=None,
        help="Select only multiplex sweep rows with this call-layer resolution.",
    )
    parser.add_argument(
        "--sweep-best-data-access-resolution",
        type=float,
        default=None,
        help="Select only multiplex sweep rows with this data-access-layer resolution.",
    )
    parser.add_argument(
        "--sweep-best-data-lineage-resolution",
        type=float,
        default=None,
        help="Select only multiplex sweep rows with this data-lineage-layer resolution.",
    )
    parser.add_argument(
        "--sweep-best-data-hub-policy",
        choices=SWEEP_BEST_DATA_HUB_POLICIES,
        default="auto",
        help=(
            "Filter best selection by the sweep's data-hub policy. "
            "Use keep_data_hubs to keep high-degree data nodes, or drop_data_hubs to exclude them."
        ),
    )
    parser.add_argument(
        "--sweep-best-min-metric",
        default=None,
        help=(
            "Only consider sweep rows whose metric is greater than --sweep-best-min-value. "
            "Evaluation metric names may be passed with or without the evaluation_ prefix."
        ),
    )
    parser.add_argument(
        "--sweep-best-min-value",
        type=float,
        default=None,
        help="Minimum exclusive threshold for --sweep-best-min-metric.",
    )
    parser.add_argument(
        "--algorithm",
        default="leiden",
        choices=algorithm_choices(),
        help="Clustering algorithm",
    )
    parser.add_argument(
        "--weight-config",
        default=None,
        help="Optional structural weight profile JSON or builtin:NAME alias. Clustering scales are read from it unless scale flags override them.",
    )
    parser.add_argument(
        "--leiden-quality",
        default="rb_configuration",
        choices=["rb_configuration", "cpm"],
        help="Leiden quality function: rb_configuration is modularity-family; cpm is Constant Potts Model",
    )
    parser.add_argument(
        "--multiplex-layer-mode",
        choices=("edge_type", "call_data"),
        default="edge_type",
        help=(
            "Layer layout for --algorithm leiden_multiplex: edge_type uses call, "
            "data_access, and data_lineage layers; call_data uses call and combined data layers."
        ),
    )
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter")
    parser.add_argument(
        "--call-resolution",
        type=float,
        default=None,
        help="Leiden multiplex resolution for the call layer; defaults to --resolution",
    )
    parser.add_argument(
        "--data-access-resolution",
        type=float,
        default=None,
        help="Leiden multiplex resolution for the data_access layer; defaults to --resolution",
    )
    parser.add_argument(
        "--data-lineage-resolution",
        type=float,
        default=None,
        help="Leiden multiplex resolution for the data_lineage layer; defaults to --resolution",
    )
    parser.add_argument("--markov-time", type=float, default=1.0, help="Infomap Markov time parameter")
    parser.add_argument(
        "--hac-n-clusters",
        type=int,
        default=13,
        help="Target callable cluster count for --algorithm hac_callable_projection",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations for label propagation")
    parser.add_argument(
        "--call-weight-scale",
        type=float,
        default=None,
        help="Override the callable-callable edge multiplier from --weight-config during clustering",
    )
    parser.add_argument(
        "--data-access-weight-scale",
        type=float,
        default=None,
        help="Override the callable-data access edge multiplier from --weight-config during clustering",
    )
    parser.add_argument(
        "--data-lineage-weight-scale",
        type=float,
        default=None,
        help="Override the data-lineage edge multiplier from --weight-config during clustering",
    )
    parser.add_argument(
        "--sweep-resolutions",
        default=None,
        help=(
            "Comma-separated Leiden resolutions for the optional sweep. "
            "Specifying this option implies --run-sweep. When omitted for a sweep, "
            "defaults depend on --leiden-quality."
        ),
    )
    parser.add_argument(
        "--sweep-markov-times",
        default=None,
        help=(
            "Comma-separated Infomap Markov times for the optional sweep. "
            "Specifying this option implies --run-sweep. When omitted for an "
            "Infomap sweep, built-in Markov-time defaults are used."
        ),
    )
    parser.add_argument(
        "--sweep-hac-n-clusters",
        default=None,
        help=(
            "Comma-separated target cluster counts for --algorithm hac_callable_projection. "
            "Specifying this option implies --run-sweep. Defaults to 10..25 for a HAC sweep."
        ),
    )
    parser.add_argument(
        "--sweep-call-resolutions",
        default=None,
        help=(
            "Comma-separated Leiden multiplex resolutions, or START:END:STEP range, "
            "for the call layer. "
            "Specifying any layer-resolution sweep option implies --run-sweep."
        ),
    )
    parser.add_argument(
        "--sweep-data-access-resolutions",
        default=None,
        help=(
            "Comma-separated Leiden multiplex resolutions, or START:END:STEP range, "
            "for the data_access layer. "
            "Specifying any layer-resolution sweep option implies --run-sweep."
        ),
    )
    parser.add_argument(
        "--sweep-data-lineage-resolutions",
        default=None,
        help=(
            "Comma-separated Leiden multiplex resolutions, or START:END:STEP range, "
            "for the data_lineage layer. "
            "Specifying any layer-resolution sweep option implies --run-sweep."
        ),
    )
    parser.add_argument(
        "--sweep-range",
        default=None,
        help=(
            "Compact Leiden resolution sweep range as START:END:STEP, for example 0.01:5:0.01. "
            "For --algorithm infomap, this is accepted as a Markov-time range unless "
            "--sweep-markov-range is supplied. For --algorithm hac_callable_projection, "
            "this is accepted as a cluster-count range, for example 10:25:1. "
            "Specifying this option implies --run-sweep."
        ),
    )
    parser.add_argument(
        "--sweep-markov-range",
        default=None,
        help=(
            "Compact Infomap Markov-time sweep range as START:END:STEP, for example 0.25:5:0.25. "
            "Specifying this option implies --run-sweep."
        ),
    )
    parser.add_argument(
        "--drop-callable-hubs",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Legacy shortcut for --callable-hub-policy drop-all when no explicit "
            "callable hub policy is supplied"
        ),
    )
    parser.add_argument(
        "--callable-hub-policy",
        choices=("keep", "drop-all", "drop-configured"),
        default=None,
        help=(
            "Callable hub exclusion policy. keep reports candidates only; drop-all "
            "drops every detected candidate; drop-configured drops only nodes listed "
            "by --drop-callable-hub or --callable-hub-decisions. Explicit keeps win."
        ),
    )
    parser.add_argument(
        "--callable-hub-decisions",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with callable hub decisions, either "
            "{\"drop\": [...], \"keep\": [...]} or nested under callable_hubs."
        ),
    )
    parser.add_argument(
        "--drop-callable-hub",
        action="append",
        default=None,
        metavar="NODE",
        help="Callable hub node id to drop when --callable-hub-policy drop-configured is active",
    )
    parser.add_argument(
        "--keep-callable-hub",
        action="append",
        default=None,
        metavar="NODE",
        help="Callable hub node id to keep even when another policy would drop it",
    )
    parser.add_argument(
        "--drop-data-hubs",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Drop high-degree data hubs from clustering; default keeps but flags them",
    )
    parser.add_argument(
        "--exclude-module-callables",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Exclude synthetic module-level callable nodes from clustering; use --no-exclude-module-callables to keep them",
    )
    parser.add_argument(
        "--hub-callable-degree-percentile",
        type=float,
        default=95.0,
        help="Percentile used with --hub-callable-min-degree to detect callable hubs",
    )
    parser.add_argument(
        "--hub-callable-min-degree",
        type=int,
        default=25,
        help="Minimum structural degree for callable hub detection",
    )
    parser.add_argument(
        "--hub-callable-min-in-degree",
        type=int,
        default=2,
        help="Minimum incoming structural degree for degree-based callable hub detection",
    )
    parser.add_argument(
        "--hub-callable-min-out-degree",
        type=int,
        default=2,
        help="Minimum outgoing structural degree for degree-based callable hub detection",
    )
    parser.add_argument(
        "--hub-entrypoint-min-out-degree",
        type=int,
        default=12,
        help="Minimum outgoing structural degree for zero-incoming callable entrypoint hub detection",
    )
    parser.add_argument(
        "--hub-orchestrator-max-in-degree",
        type=int,
        default=1,
        help="Maximum incoming structural degree for low-in/high-out callable orchestrator hub detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-out-degree",
        type=int,
        default=12,
        help="Minimum outgoing structural degree for low-in/high-out callable orchestrator hub detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-out-call-degree",
        type=int,
        default=4,
        help="Minimum outgoing call edges for callable orchestrator hub detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-target-modules",
        type=int,
        default=3,
        help="Minimum distinct target callable modules for callable orchestrator hub detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-target-callables",
        type=int,
        default=4,
        help="Minimum distinct target callables for mixed callable/data orchestrator hub detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-target-data",
        type=int,
        default=4,
        help="Minimum distinct target data objects for mixed callable/data orchestrator hub detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-data-to-call-ratio",
        type=float,
        default=1.0,
        help="Minimum distinct target data/callable ratio for callable orchestrator hub detection",
    )
    parser.add_argument("--hub-data-min-degree", type=int, default=20, help="Data hub structural degree threshold")
    parser.add_argument(
        "--hub-data-min-callable-count",
        type=int,
        default=10,
        help="Data hub callable_count threshold",
    )
    parser.add_argument(
        "--hub-data-min-access-count",
        type=int,
        default=100,
        help="Data hub access_count threshold",
    )
    parser.set_defaults(**config_defaults)
    return _expand_algorithm_output_path_templates(parser.parse_args(argv))

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    nodes_path = _resolve_project_path(project_root, args.nodes)
    edges_path = _resolve_project_path(project_root, args.edges)
    outdir = _resolve_project_path(project_root, args.outdir)
    default_callable_hub_nodes = nodes_path.parent / "callable_hub_nodes.csv"
    default_data_hub_nodes = nodes_path.parent / "data_hub_nodes.csv"
    callable_hub_nodes_path = (
        _resolve_project_path(project_root, args.callable_hub_nodes)
        if args.callable_hub_nodes is not None
        else default_callable_hub_nodes if default_callable_hub_nodes.is_file() else None
    )
    data_hub_nodes_path = (
        _resolve_project_path(project_root, args.data_hub_nodes)
        if args.data_hub_nodes is not None
        else default_data_hub_nodes if default_data_hub_nodes.is_file() else None
    )
    weight_config_reference = resolve_weight_config_reference(args.weight_config, project_root=project_root)
    weight_config = load_weight_config(weight_config_reference)
    call_weight_scale, data_access_weight_scale, data_lineage_weight_scale = (
        _resolve_clustering_weight_scales(args, weight_config)
    )
    configured_callable_hub_drop: Tuple[str, ...] = tuple()
    configured_callable_hub_keep: Tuple[str, ...] = tuple()
    callable_hub_decisions_path = _resolve_optional_project_path(
        project_root,
        args.callable_hub_decisions,
    )
    if callable_hub_decisions_path is not None:
        configured_callable_hub_drop, configured_callable_hub_keep = (
            callable_hub_decisions_from_json(callable_hub_decisions_path)
        )
    callable_hub_drop = _unique_text_tuple(
        [
            *configured_callable_hub_drop,
            *(args.drop_callable_hub or []),
        ]
    )
    callable_hub_keep = _unique_text_tuple(
        [
            *configured_callable_hub_keep,
            *(args.keep_callable_hub or []),
        ]
    )
    run_sweep = (
        args.run_sweep
        or args.sweep_range is not None
        or args.sweep_resolutions is not None
        or args.sweep_markov_range is not None
        or args.sweep_markov_times is not None
        or args.sweep_hac_n_clusters is not None
        or args.sweep_call_resolutions is not None
        or args.sweep_data_access_resolutions is not None
        or args.sweep_data_lineage_resolutions is not None
    )
    sweep_markov_times = tuple()
    sweep_hac_n_clusters = tuple()
    sweep_call_resolutions = tuple()
    sweep_data_access_resolutions = tuple()
    sweep_data_lineage_resolutions = tuple()
    if run_sweep:
        if args.algorithm == "infomap":
            if args.sweep_markov_range:
                sweep_markov_times = _parse_sweep_range(args.sweep_markov_range)
            elif args.sweep_markov_times is not None:
                sweep_markov_times = _parse_sweep_resolutions(args.sweep_markov_times)
            elif args.sweep_range:
                sweep_markov_times = _parse_sweep_range(args.sweep_range)
            elif args.sweep_resolutions is not None:
                sweep_markov_times = _parse_sweep_resolutions(args.sweep_resolutions)
            else:
                sweep_markov_times = _default_sweep_markov_times()
            sweep_resolutions = tuple()
        elif args.algorithm == HAC_CALLABLE_PROJECTION_ALGORITHM:
            if args.sweep_hac_n_clusters is not None:
                sweep_hac_n_clusters = _parse_sweep_ints(args.sweep_hac_n_clusters)
            elif args.sweep_range:
                sweep_hac_n_clusters = _parse_sweep_int_range(args.sweep_range)
            elif args.sweep_resolutions is not None:
                sweep_hac_n_clusters = _parse_sweep_ints(args.sweep_resolutions)
            else:
                sweep_hac_n_clusters = _default_sweep_hac_n_clusters()
            sweep_resolutions = tuple()
        elif args.algorithm == LEIDEN_MULTIPLEX_ALGORITHM and (
            args.sweep_call_resolutions is not None
            or args.sweep_data_access_resolutions is not None
            or args.sweep_data_lineage_resolutions is not None
        ):
            sweep_call_resolutions = (
                _parse_sweep_resolutions(args.sweep_call_resolutions)
                if args.sweep_call_resolutions is not None
                else tuple()
            )
            sweep_data_access_resolutions = (
                _parse_sweep_resolutions(args.sweep_data_access_resolutions)
                if args.sweep_data_access_resolutions is not None
                else tuple()
            )
            sweep_data_lineage_resolutions = (
                _parse_sweep_resolutions(args.sweep_data_lineage_resolutions)
                if args.sweep_data_lineage_resolutions is not None
                else tuple()
            )
            sweep_resolutions = tuple()
        else:
            if args.sweep_range:
                sweep_resolutions = _parse_sweep_range(args.sweep_range)
            elif args.sweep_resolutions is None:
                sweep_resolutions = _default_sweep_resolutions(args.leiden_quality)
            else:
                sweep_resolutions = _parse_sweep_resolutions(args.sweep_resolutions)
    else:
        sweep_resolutions = tuple()

    options = ClusterOptions(
        algorithm=args.algorithm,
        leiden_quality=args.leiden_quality,
        multiplex_layer_mode=args.multiplex_layer_mode,
        resolution=args.resolution,
        call_resolution=args.call_resolution,
        data_access_resolution=args.data_access_resolution,
        data_lineage_resolution=args.data_lineage_resolution,
        markov_time=args.markov_time,
        hac_n_clusters=args.hac_n_clusters,
        seed=args.seed,
        max_iter=args.max_iter,
        sweep_resolutions=sweep_resolutions,
        sweep_markov_times=sweep_markov_times,
        sweep_hac_n_clusters=sweep_hac_n_clusters,
        sweep_call_resolutions=sweep_call_resolutions,
        sweep_data_access_resolutions=sweep_data_access_resolutions,
        sweep_data_lineage_resolutions=sweep_data_lineage_resolutions,
        run_sweep=run_sweep
        and bool(
            sweep_resolutions
            or sweep_markov_times
            or sweep_hac_n_clusters
            or sweep_call_resolutions
            or sweep_data_access_resolutions
            or sweep_data_lineage_resolutions
        ),
        exclude_module_callables=args.exclude_module_callables,
        callable_hub_policy=args.callable_hub_policy,
        callable_hub_drop=callable_hub_drop,
        callable_hub_keep=callable_hub_keep,
        callable_hub_nodes_path=str(callable_hub_nodes_path) if callable_hub_nodes_path else None,
        data_hub_nodes_path=str(data_hub_nodes_path) if data_hub_nodes_path else None,
        drop_callable_hubs=args.drop_callable_hubs,
        drop_data_hubs=args.drop_data_hubs,
        call_weight_scale=call_weight_scale,
        data_access_weight_scale=data_access_weight_scale,
        data_lineage_weight_scale=data_lineage_weight_scale,
        weight_config=weight_config.to_dict(),
        hub_callable_degree_percentile=args.hub_callable_degree_percentile,
        hub_callable_min_degree=args.hub_callable_min_degree,
        hub_callable_min_in_degree=args.hub_callable_min_in_degree,
        hub_callable_min_out_degree=args.hub_callable_min_out_degree,
        hub_entrypoint_min_out_degree=args.hub_entrypoint_min_out_degree,
        hub_orchestrator_max_in_degree=args.hub_orchestrator_max_in_degree,
        hub_orchestrator_min_out_degree=args.hub_orchestrator_min_out_degree,
        hub_orchestrator_min_out_call_degree=args.hub_orchestrator_min_out_call_degree,
        hub_orchestrator_min_target_modules=args.hub_orchestrator_min_target_modules,
        hub_orchestrator_min_target_callables=args.hub_orchestrator_min_target_callables,
        hub_orchestrator_min_target_data=args.hub_orchestrator_min_target_data,
        hub_orchestrator_min_data_to_call_ratio=args.hub_orchestrator_min_data_to_call_ratio,
        hub_data_min_degree=args.hub_data_min_degree,
        hub_data_min_callable_count=args.hub_data_min_callable_count,
        hub_data_min_access_count=args.hub_data_min_access_count,
    )

    nodes = load_node_rows(nodes_path)
    edges = load_edges(edges_path)
    result = cluster_structural_graph(nodes, edges, options)
    nodes_with_edge_refs = ensure_edge_nodes(nodes, edges)
    write_outputs(outdir, nodes_with_edge_refs, result)

    sweep_rows: List[dict] = []
    sweep_outdir: Optional[Path] = None
    sweep_evaluation_enabled = False
    sweep_best_selection: Optional[SweepBestSelection] = None
    sweep_best_outdir: Optional[Path] = None
    if options.run_sweep:
        sweep_outdir = (
            _resolve_project_path(project_root, args.sweep_outdir)
            if args.sweep_outdir
            else outdir / "sweep_results"
        )
        manual_rows: Optional[List[dict]] = None
        manual_fields: List[str] = []
        if not args.no_sweep_evaluation:
            manual_path = _resolve_project_path(project_root, args.sweep_manual)
            if manual_path.exists():
                manual_rows, manual_fields = load_csv_rows(manual_path)
                sweep_evaluation_enabled = True
            else:
                print(f"Sweep evaluation skipped; manual mapping not found: {manual_path}")
        sweep_rows = run_parameter_sweep(
            nodes_with_edge_refs,
            edges,
            options,
            manual_rows=manual_rows,
            manual_fields=manual_fields,
            manual_label_column=args.sweep_manual_label_column,
            node_mode=args.sweep_node_mode,
            na_labels=tuple(args.sweep_na_label or NA_DEFAULTS),
            evaluation_node_types=parse_evaluation_tokens(args.sweep_evaluation_node_types),
            evaluation_kind_tokens=parse_evaluation_tokens(args.sweep_evaluation_kind_tokens),
            all_evaluation_nodes=args.sweep_all_evaluation_nodes,
        )
        sweep_best_selection_options = SweepBestSelectionOptions(
            enabled=args.select_sweep_best,
            metric=args.sweep_best_metric,
            metric_direction=args.sweep_best_metric_direction,
            resolution=args.sweep_best_resolution,
            markov_time=args.sweep_best_markov_time,
            hac_n_clusters=args.sweep_best_hac_n_clusters,
            call_resolution=args.sweep_best_call_resolution,
            data_access_resolution=args.sweep_best_data_access_resolution,
            data_lineage_resolution=args.sweep_best_data_lineage_resolution,
            data_hub_policy=args.sweep_best_data_hub_policy,
            min_metric=args.sweep_best_min_metric,
            min_value=args.sweep_best_min_value,
        )
        selected_options: Optional[ClusterOptions] = None
        sweep_best_outdir = (
            _resolve_project_path(project_root, args.sweep_best_outdir)
            if args.sweep_best_outdir
            else sweep_outdir / "best"
        )
        if sweep_best_selection_options.enabled:
            sweep_best_selection = select_sweep_best_row(
                sweep_rows,
                sweep_best_selection_options,
            )
            mark_selected_sweep_row(sweep_rows, sweep_best_selection)
            if sweep_best_selection.selected and sweep_best_selection.selected_row:
                selected_options = materialize_sweep_best_cluster(
                    sweep_best_outdir,
                    nodes_with_edge_refs,
                    edges,
                    options,
                    sweep_best_selection.selected_row,
                )
        else:
            sweep_best_selection = SweepBestSelection(
                selected_index=None,
                selected_row=None,
                metric=_normalize_sweep_best_metric(sweep_best_selection_options.metric),
                metric_direction=sweep_best_selection_options.metric_direction,
                candidate_count=len(sweep_rows),
                filtered_count=0,
                reason="disabled",
            )
            mark_selected_sweep_row(sweep_rows, sweep_best_selection)
        write_sweep_best_selection(
            sweep_outdir,
            sweep_best_selection,
            sweep_best_selection_options,
            sweep_best_outdir,
            selected_options,
        )
        write_sweep_outputs(sweep_outdir, sweep_rows, options)

    print(f"Structural clustering output written to: {outdir}")
    print(f"Algorithm: {options.algorithm}")
    print(f"Weight profile: {weight_config.name or '(unnamed)'}")
    if options.algorithm in LEIDEN_ALGORITHMS:
        print(f"Leiden quality: {options.leiden_quality}")
    if options.algorithm == "infomap":
        print(f"Infomap Markov time: {options.markov_time:g}")
    if options.algorithm == HAC_CALLABLE_PROJECTION_ALGORITHM:
        print(f"HAC target callable clusters: {options.hac_n_clusters}")
    print(
        "Weight scales: "
        f"call={options.call_weight_scale:g}, "
        f"data_access={options.data_access_weight_scale:g}, "
        f"data_lineage={options.data_lineage_weight_scale:g}"
    )
    print(f"Callable hub policy: {_effective_callable_hub_policy(options)}")
    if options.callable_hub_nodes_path and options.data_hub_nodes_path:
        print("Hub node source: structural_dependency_graph CSVs")
    if options.algorithm == LEIDEN_MULTIPLEX_ALGORITHM:
        layer_resolutions = _edge_type_layer_resolutions(options)
        if options.multiplex_layer_mode == "call_data":
            print(
                "Layer resolutions: "
                f"call={layer_resolutions['call']:g}, "
                f"data={layer_resolutions['data']:g}"
            )
        else:
            print(
                "Layer resolutions: "
                f"call={layer_resolutions['call']:g}, "
                f"data_access={layer_resolutions['data_access']:g}, "
                f"data_lineage={layer_resolutions['data_lineage']:g}"
            )
    print(f"Nodes clustered: {len(result.cluster_of)}")
    print(f"Clusters: {len(set(result.cluster_of.values()))}")
    print(f"Excluded nodes: {len(result.excluded_nodes)}")
    print(f"Hub nodes flagged: {len(result.hub_nodes)}")
    print(f"Cycle findings: {len(result.cycle_findings)}")
    if sweep_outdir is not None:
        print(f"Parameter sweep output written to: {sweep_outdir}")
        print(f"Sweep rows: {len(sweep_rows)}")
        print(f"Sweep evaluation metrics: {'included' if sweep_evaluation_enabled else 'not included'}")
        if sweep_best_selection is not None:
            print(f"Sweep best selection: {sweep_best_selection.reason}")
            if sweep_best_selection.selected and sweep_best_outdir is not None:
                print(f"Sweep best cluster output written to: {sweep_best_outdir}")


if __name__ == "__main__":
    main()
