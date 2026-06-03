"""Shared structural graph generation weights and clustering scales."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from microservice_pipeline.package_resources import builtin_weight_profile_resource


BUILTIN_WEIGHT_CONFIG_PREFIX = "builtin:"


@dataclass(frozen=True)
class StructuralWeightConfig:
    schema: str
    name: str
    call_weight: float
    data_access_weights: Dict[str, float]
    data_access_fallback_weight: float
    data_lineage_weights: Dict[str, float]
    data_lineage_fallback_weight: float
    confidence_weights: Dict[str, float]
    confidence_fallback_weight: float
    clustering_scales: Dict[str, float]
    reweighted_settings: Dict[str, Any]
    multiplex_settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StructuralWeightConfig":
        generation = _required_mapping(payload, "generation")
        call = _required_mapping(generation, "call")
        data_access = _required_mapping(generation, "data_access")
        data_lineage = _required_mapping(generation, "data_lineage")
        confidence = _required_mapping(generation, "confidence")
        clustering = _required_mapping(payload, "clustering")

        return cls(
            schema=_required_text(payload, "schema"),
            name=str(payload.get("name") or ""),
            call_weight=_required_float(call, "base_weight"),
            data_access_weights=_float_mapping(_required_mapping(data_access, "weights")),
            data_access_fallback_weight=_required_float(data_access, "fallback_weight"),
            data_lineage_weights=_float_mapping(_required_mapping(data_lineage, "weights")),
            data_lineage_fallback_weight=_required_float(data_lineage, "fallback_weight"),
            confidence_weights=_float_mapping(_required_mapping(confidence, "weights")),
            confidence_fallback_weight=_required_float(confidence, "fallback_weight"),
            clustering_scales=_float_mapping(_required_mapping(clustering, "edge_type_scales")),
            reweighted_settings=_plain_mapping(clustering.get("reweighted", {})),
            multiplex_settings=_plain_mapping(clustering.get("multiplex", {})),
        )

    def data_access_weight(self, access: str) -> float:
        return self.data_access_weights.get(access, self.data_access_fallback_weight)

    def data_lineage_weight(self, relation: str) -> float:
        return self.data_lineage_weights.get(relation, self.data_lineage_fallback_weight)

    def confidence_weight(self, confidence: str) -> float:
        return self.confidence_weights.get(confidence, self.confidence_fallback_weight)

    def clustering_scale(self, edge_type: str) -> float:
        return self.clustering_scales.get(edge_type, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "generation": {
                "call": {
                    "base_weight": self.call_weight,
                },
                "data_access": {
                    "weights": dict(self.data_access_weights),
                    "fallback_weight": self.data_access_fallback_weight,
                },
                "data_lineage": {
                    "weights": dict(self.data_lineage_weights),
                    "fallback_weight": self.data_lineage_fallback_weight,
                },
                "confidence": {
                    "weights": dict(self.confidence_weights),
                    "fallback_weight": self.confidence_fallback_weight,
                },
            },
            "clustering": _clustering_to_dict(
                self.clustering_scales,
                self.reweighted_settings,
                self.multiplex_settings,
            ),
        }


def load_weight_config(path: Path | str | None = None) -> StructuralWeightConfig:
    default_payload = _load_json(builtin_weight_profile_resource("default"))
    if path is not None:
        payload = _deep_merge(default_payload, _load_json(_weight_config_resource(path)))
    else:
        payload = default_payload
    return StructuralWeightConfig.from_mapping(payload)


def is_builtin_weight_config_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(BUILTIN_WEIGHT_CONFIG_PREFIX)


def resolve_weight_config_reference(value: Any, *, project_root: Path | None = None) -> Path | str | None:
    if value in (None, ""):
        return None
    if is_builtin_weight_config_reference(value):
        return str(value)
    path = Path(value)
    if project_root is not None and not path.is_absolute():
        return (project_root / path).resolve()
    return path


def _weight_config_resource(path: Path | str):
    if is_builtin_weight_config_reference(path):
        name = str(path).removeprefix(BUILTIN_WEIGHT_CONFIG_PREFIX)
        return builtin_weight_profile_resource(name)
    return Path(path)


def _load_json(path: Any) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Weight config must be a JSON object: {path}")
    return payload


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Weight config field must be an object: {key}")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Weight config field must be a non-empty string: {key}")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"Weight config field is required: {key}")
    value = payload[key]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Weight config field must be numeric: {key}") from exc


def _float_mapping(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight config mapping value must be numeric: {key}") from exc
    return values


def _plain_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_ready(value) for key, value in value.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _clustering_to_dict(
    clustering_scales: Mapping[str, float],
    reweighted_settings: Mapping[str, Any],
    multiplex_settings: Mapping[str, Any],
) -> Dict[str, Any]:
    clustering: Dict[str, Any] = {
        "edge_type_scales": dict(clustering_scales),
    }
    if reweighted_settings:
        clustering["reweighted"] = _plain_mapping(reweighted_settings)
    if multiplex_settings:
        clustering["multiplex"] = _plain_mapping(multiplex_settings)
    return clustering
