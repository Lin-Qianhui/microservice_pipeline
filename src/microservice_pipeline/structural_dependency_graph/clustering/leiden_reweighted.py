"""Ownership-biased Leiden clustering for structural dependency graphs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .common import StructuralClusteringInput
from .leiden import cluster as cluster_leiden
from ..weight_config import load_weight_config


CALLABLE_PREFIX = "callable:"
DATA_PREFIX = "data:"

ALGORITHM = "leiden_reweighted"
DEFAULT_WEIGHT_CONFIG = load_weight_config().to_dict()
DEFAULT_SUBTYPE_RATIO_MIN = 0.5
DEFAULT_SUBTYPE_RATIO_MAX = 1.5
DEFAULT_SINGLE_WRITER_BOOST = 1.5
DEFAULT_SINGLE_WRITER_MAX_WEIGHT = 8.0
WRITER_ACCESSES = frozenset({"write", "create", "read_write"})
DEFAULT_MUST_LINK_RELATIONS = frozenset({"state_assign", "local_assign", "tuple_unpack"})


def cluster(input_data: StructuralClusteringInput) -> Dict[str, str]:
    return cluster_leiden(input_data)


def reweighted_edge_weights(
    edges: Iterable[Any],
    weight_config: Mapping[str, Any],
) -> List[float]:
    materialized = list(edges)
    single_writer_pairs = single_writer_must_link_pairs(materialized)
    return [
        reweighted_edge_weight(
            edge=edge,
            weight_config=weight_config,
            single_writer_pairs=single_writer_pairs,
        )
        for edge in materialized
    ]


def reweighted_edge_weight(
    edge: Any,
    weight_config: Mapping[str, Any],
    single_writer_pairs: Set[Tuple[str, str]] | None = None,
) -> float:
    edge_type = _text(getattr(edge, "edge_type", ""))
    base_weight = _float(getattr(edge, "weight", 1.0), 1.0)
    weight = (
        base_weight
        * _edge_type_scale(weight_config, edge_type)
        * _bounded_subtype_ratio(edge, weight_config, edge_type)
    )
    if _is_single_writer_edge(edge, single_writer_pairs or set()):
        weight = min(
            weight * _reweighted_float(
                weight_config,
                "single_writer_boost",
                DEFAULT_SINGLE_WRITER_BOOST,
            ),
            _reweighted_float(
                weight_config,
                "single_writer_max_weight",
                DEFAULT_SINGLE_WRITER_MAX_WEIGHT,
            ),
        )
    return weight


def reweighted_must_link_relations(weight_config: Mapping[str, Any]) -> Set[str]:
    configured = _reweighted_setting(weight_config, "must_link_relations")
    if isinstance(configured, str):
        relations = [part.strip() for part in configured.split(",")]
    elif isinstance(configured, Sequence):
        relations = [_text(relation).strip() for relation in configured]
    else:
        relations = list(DEFAULT_MUST_LINK_RELATIONS)
    cleaned = {relation for relation in relations if relation}
    return cleaned or set(DEFAULT_MUST_LINK_RELATIONS)


def single_writer_must_link_pairs(edges: Iterable[Any]) -> Set[Tuple[str, str]]:
    """Return callable-data pairs where one callable owns all mutating touches."""

    writers_by_data: Dict[str, Set[str]] = {}
    touchers_by_data: Dict[str, Set[str]] = {}
    for edge in edges:
        pair = _callable_data_pair(edge)
        if pair is None:
            continue
        callable_id, data_id = pair
        touchers_by_data.setdefault(data_id, set()).add(callable_id)
        if _text(getattr(edge, "access", "")) in WRITER_ACCESSES:
            writers_by_data.setdefault(data_id, set()).add(callable_id)

    pairs: Set[Tuple[str, str]] = set()
    for data_id, writers in writers_by_data.items():
        if len(writers) == 1 and touchers_by_data.get(data_id, set()) == writers:
            pairs.add((next(iter(writers)), data_id))
    return pairs


def _is_single_writer_edge(edge: Any, single_writer_pairs: Set[Tuple[str, str]]) -> bool:
    if _text(getattr(edge, "access", "")) not in WRITER_ACCESSES:
        return False
    pair = _callable_data_pair(edge)
    return pair in single_writer_pairs


def _callable_data_pair(edge: Any) -> Tuple[str, str] | None:
    src = _text(getattr(edge, "src", ""))
    dst = _text(getattr(edge, "dst", ""))
    if src.startswith(CALLABLE_PREFIX) and dst.startswith(DATA_PREFIX):
        return src, dst
    if dst.startswith(CALLABLE_PREFIX) and src.startswith(DATA_PREFIX):
        return dst, src
    return None


def _edge_type_scale(weight_config: Mapping[str, Any], edge_type: str) -> float:
    clustering = _mapping(weight_config.get("clustering"))
    scales = _mapping(clustering.get("edge_type_scales"))
    return _float(scales.get(edge_type), 1.0)


def _bounded_subtype_ratio(
    edge: Any,
    weight_config: Mapping[str, Any],
    edge_type: str,
) -> float:
    configured = _subtype_weight(edge, weight_config, edge_type)
    default = _subtype_weight(edge, DEFAULT_WEIGHT_CONFIG, edge_type)
    if default <= 0:
        return 1.0
    ratio = configured / default
    lower = _reweighted_float(weight_config, "subtype_ratio_min", DEFAULT_SUBTYPE_RATIO_MIN)
    upper = _reweighted_float(weight_config, "subtype_ratio_max", DEFAULT_SUBTYPE_RATIO_MAX)
    if lower > upper:
        lower, upper = upper, lower
    return min(max(ratio, lower), upper)


def _subtype_weight(edge: Any, weight_config: Mapping[str, Any], edge_type: str) -> float:
    generation = _mapping(weight_config.get("generation"))
    if edge_type == "call":
        call = _mapping(generation.get("call"))
        return _float(call.get("base_weight"), 1.0)
    if edge_type == "data_access":
        data_access = _mapping(generation.get("data_access"))
        return _weighted_subtype(
            weights=_mapping(data_access.get("weights")),
            subtype=_text(getattr(edge, "access", "")),
            fallback=_float(data_access.get("fallback_weight"), 1.0),
        )
    if edge_type == "data_lineage":
        data_lineage = _mapping(generation.get("data_lineage"))
        return _weighted_subtype(
            weights=_mapping(data_lineage.get("weights")),
            subtype=_text(getattr(edge, "relation", "")),
            fallback=_float(data_lineage.get("fallback_weight"), 1.0),
        )
    return _float(getattr(edge, "weight", 1.0), 1.0)


def _weighted_subtype(
    weights: Mapping[str, Any],
    subtype: str,
    fallback: float,
) -> float:
    return _float(weights.get(subtype), fallback)


def _reweighted_float(
    weight_config: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    return _float(_reweighted_setting(weight_config, key), default)


def _reweighted_setting(weight_config: Mapping[str, Any], key: str) -> Any:
    clustering = _mapping(weight_config.get("clustering"))
    reweighted = _mapping(clustering.get("reweighted"))
    return reweighted.get(key)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
