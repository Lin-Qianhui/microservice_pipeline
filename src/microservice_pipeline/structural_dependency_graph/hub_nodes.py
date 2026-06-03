"""Structural hub-node detection for dependency graph artifacts."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


CALLABLE_PREFIX = "callable:"
DATA_PREFIX = "data:"

CALLABLE_HUB_NODE_FIELDNAMES = [
    "node",
    "node_type",
    "label",
    "kind",
    "candidate_types",
    "reasons",
    "in_degree",
    "out_degree",
    "total_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "weighted_degree",
    "out_call_degree",
    "out_data_degree",
    "target_callable_count",
    "target_data_count",
    "target_module_count",
    "target_modules",
    "file",
    "lineno",
]

DATA_HUB_NODE_FIELDNAMES = [
    "node",
    "node_type",
    "label",
    "kind",
    "reasons",
    "in_degree",
    "out_degree",
    "total_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "weighted_degree",
    "callable_count",
    "access_count",
    "file",
    "lineno",
]


@dataclass(frozen=True)
class HubDetectionOptions:
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


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _edge_value(edge: Any, key: str, default: Any = "") -> Any:
    if isinstance(edge, Mapping):
        return edge.get(key, default)
    return getattr(edge, key, default)


def _node_type(node_id: str, row: Mapping[str, Any] | None = None) -> str:
    if row:
        node_type = _text(row.get("node_type"))
        if node_type:
            return node_type
    if node_id.startswith(CALLABLE_PREFIX):
        return "callable"
    if node_id.startswith(DATA_PREFIX):
        return "data"
    return ""


def _node_label(node_id: str, row: Mapping[str, Any] | None = None) -> str:
    if row:
        for key in ("display_name", "label", "qualname"):
            value = _text(row.get(key))
            if value:
                return value
    return node_id


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


def node_rows_by_id(nodes: Iterable[Mapping[str, Any]]) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for row in nodes:
        node_id = _text(row.get("id") or row.get("node"))
        if node_id:
            rows[node_id] = dict(row)
    return rows


def compute_degrees(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Any],
) -> Dict[str, dict]:
    incoming = Counter()
    outgoing = Counter()
    incoming_weight: Dict[str, float] = defaultdict(float)
    outgoing_weight: Dict[str, float] = defaultdict(float)

    for edge in edges:
        src = _text(_edge_value(edge, "src"))
        dst = _text(_edge_value(edge, "dst"))
        if not src or not dst:
            continue
        weight = _float(_edge_value(edge, "weight"), 1.0)
        outgoing[src] += 1
        incoming[dst] += 1
        outgoing_weight[src] += weight
        incoming_weight[dst] += weight

    degree_map: Dict[str, dict] = {}
    for node_id in nodes:
        in_degree = incoming[node_id]
        out_degree = outgoing[node_id]
        in_weight = incoming_weight[node_id]
        out_weight = outgoing_weight[node_id]
        degree_map[node_id] = {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": in_degree + out_degree,
            "weighted_in_degree": round(in_weight, 6),
            "weighted_out_degree": round(out_weight, 6),
            "weighted_degree": round(in_weight + out_weight, 6),
        }
    return degree_map


def _callable_fanout_features(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Iterable[Any],
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
        }
        for node_id, row in nodes.items()
        if _node_type(node_id, row) == "callable"
    }

    for edge in edges:
        src = _text(_edge_value(edge, "src"))
        dst = _text(_edge_value(edge, "dst"))
        edge_type = _text(_edge_value(edge, "edge_type"))
        row = nodes.get(src)
        if row is None or _node_type(src, row) != "callable":
            continue
        feature = features.setdefault(
            src,
            {
                "out_call_degree": 0,
                "out_data_degree": 0,
                "target_callable_count": 0,
                "target_data_count": 0,
                "target_callables": set(),
                "target_data": set(),
                "target_modules": Counter(),
            },
        )
        target_row = nodes.get(dst, {})
        target_type = _node_type(dst, target_row)
        if edge_type == "call":
            feature["out_call_degree"] += 1
        else:
            feature["out_data_degree"] += 1
        if target_type == "callable":
            feature["target_callables"].add(dst)
            module = _text(target_row.get("module")) or "(unknown)"
            feature["target_modules"][module] += 1
        elif target_type == "data":
            feature["target_data"].add(dst)

    for feature in features.values():
        feature["target_callable_count"] = len(feature.get("target_callables", ()))
        feature["target_data_count"] = len(feature.get("target_data", ()))

    return features


def _is_orchestrator_candidate(
    degrees: Mapping[str, Any],
    features: Mapping[str, Any],
    options: HubDetectionOptions,
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


def _is_entrypoint_candidate(
    degrees: Mapping[str, Any],
    options: HubDetectionOptions,
) -> bool:
    return (
        _int(degrees.get("in_degree")) == 0
        and _int(degrees.get("out_degree")) >= options.hub_entrypoint_min_out_degree
    )


def identify_hub_nodes(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Any],
    options: HubDetectionOptions | None = None,
) -> Tuple[list[dict], list[dict], Dict[str, dict]]:
    options = options or HubDetectionOptions()
    degree_map = compute_degrees(nodes, edges)
    callable_degrees = [
        _int(degree_map.get(node_id, {}).get("total_degree"))
        for node_id, row in nodes.items()
        if _node_type(node_id, row) == "callable"
    ]
    callable_threshold = max(
        _percentile(callable_degrees, options.hub_callable_degree_percentile),
        float(options.hub_callable_min_degree),
    )
    callable_features = _callable_fanout_features(nodes, edges)

    callable_rows: list[dict] = []
    data_rows: list[dict] = []
    for node_id, row in nodes.items():
        node_type = _node_type(node_id, row)
        degrees = degree_map.get(node_id, {})
        in_degree = _int(degrees.get("in_degree"))
        out_degree = _int(degrees.get("out_degree"))
        total_degree = _int(degrees.get("total_degree"))
        reasons: list[str] = []
        candidate_types: list[str] = []

        if node_type == "callable":
            if (
                total_degree > 0
                and total_degree >= callable_threshold
                and in_degree >= options.hub_callable_min_in_degree
                and out_degree >= options.hub_callable_min_out_degree
            ):
                candidate_types.append("degree")
                reasons.append(f"callable_degree>={callable_threshold:g}")
            if _is_entrypoint_candidate(degrees, options):
                candidate_types.append("entrypoint")
                reasons.append(
                    f"entrypoint_fanout:in=0;out>={options.hub_entrypoint_min_out_degree}"
                )
            features = callable_features.get(node_id, {})
            if _is_orchestrator_candidate(degrees, features, options):
                candidate_types.append("orchestrator")
                reasons.append(
                    "orchestrator_fanout"
                    f":in<={options.hub_orchestrator_max_in_degree}"
                    f";out>={options.hub_orchestrator_min_out_degree}"
                    f";out_call>={options.hub_orchestrator_min_out_call_degree}"
                    f";data/call>={options.hub_orchestrator_min_data_to_call_ratio:g}"
                )
            if not reasons:
                continue
            target_modules = features.get("target_modules", Counter())
            callable_rows.append(
                {
                    "node": node_id,
                    "node_type": node_type,
                    "label": _node_label(node_id, row),
                    "kind": _text(row.get("kind")),
                    "candidate_types": ";".join(candidate_types),
                    "reasons": ";".join(reasons),
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                    "total_degree": total_degree,
                    "weighted_in_degree": f"{_float(degrees.get('weighted_in_degree')):.6f}",
                    "weighted_out_degree": f"{_float(degrees.get('weighted_out_degree')):.6f}",
                    "weighted_degree": f"{_float(degrees.get('weighted_degree')):.6f}",
                    "out_call_degree": _int(features.get("out_call_degree")),
                    "out_data_degree": _int(features.get("out_data_degree")),
                    "target_callable_count": _int(features.get("target_callable_count")),
                    "target_data_count": _int(features.get("target_data_count")),
                    "target_module_count": len(target_modules),
                    "target_modules": _counter_preview(target_modules, limit=8),
                    "file": _text(row.get("file")),
                    "lineno": _text(row.get("lineno")),
                }
            )
        elif node_type == "data":
            callable_count = _int(row.get("callable_count"))
            access_count = _int(row.get("access_count"))
            if total_degree >= options.hub_data_min_degree:
                reasons.append(f"data_degree>={options.hub_data_min_degree}")
            if callable_count >= options.hub_data_min_callable_count:
                reasons.append(f"callable_count>={options.hub_data_min_callable_count}")
            if access_count >= options.hub_data_min_access_count:
                reasons.append(f"access_count>={options.hub_data_min_access_count}")
            if not reasons:
                continue
            data_rows.append(
                {
                    "node": node_id,
                    "node_type": node_type,
                    "label": _node_label(node_id, row),
                    "kind": _text(row.get("kind")),
                    "reasons": ";".join(reasons),
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                    "total_degree": total_degree,
                    "weighted_in_degree": f"{_float(degrees.get('weighted_in_degree')):.6f}",
                    "weighted_out_degree": f"{_float(degrees.get('weighted_out_degree')):.6f}",
                    "weighted_degree": f"{_float(degrees.get('weighted_degree')):.6f}",
                    "callable_count": _text(row.get("callable_count")),
                    "access_count": _text(row.get("access_count")),
                    "file": _text(row.get("file")),
                    "lineno": _text(row.get("lineno")),
                }
            )

    callable_rows.sort(key=lambda item: (-_int(item["total_degree"]), item["node"]))
    data_rows.sort(key=lambda item: (-_int(item["total_degree"]), item["node"]))
    return callable_rows, data_rows, degree_map


def load_hub_node_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
