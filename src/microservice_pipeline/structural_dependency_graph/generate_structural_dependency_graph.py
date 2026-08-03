#!/usr/bin/env python3
"""Build a heterogeneous structural dependency graph.

The graph combines callable-callable call edges with callable-data access edges
and data-data lineage edges. It is intended as the input for later vertical
microservice-candidate clustering, where clusters can contain both behavior and
the data objects that behavior owns or depends on.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown
    from microservice_pipeline.jsonc_config import load_jsonc
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown  # type: ignore
    from jsonc_config import load_jsonc  # type: ignore

try:
    from microservice_pipeline.structural_dependency_graph.hub_nodes import (
        CALLABLE_HUB_NODE_FIELDNAMES,
        DATA_HUB_NODE_FIELDNAMES,
        HubDetectionOptions,
        identify_hub_nodes,
        node_rows_by_id,
    )
    from microservice_pipeline.structural_dependency_graph.weight_config import (
        StructuralWeightConfig,
        load_weight_config,
        resolve_weight_config_reference,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from structural_dependency_graph.hub_nodes import (  # type: ignore
        CALLABLE_HUB_NODE_FIELDNAMES,
        DATA_HUB_NODE_FIELDNAMES,
        HubDetectionOptions,
        identify_hub_nodes,
        node_rows_by_id,
    )
    from structural_dependency_graph.weight_config import (  # type: ignore
        StructuralWeightConfig,
        load_weight_config,
        resolve_weight_config_reference,
    )


CALLABLE_PREFIX = "callable:"
DATA_PREFIX = "data:"

INCLUDED_DATA_KINDS = {
    "class_attr_state",
    "object_state",
    "class_state",
    "module_global",
    "file",
    "df_col",
    "dict_key",
    "container_field",
    "local_exposed",
    "param",
    "unknown",
}

NODE_FIELDNAMES = [
    "id",
    "node_type",
    "label",
    "kind",
    "inferred_type",
    "module",
    "qualname",
    "class_name",
    "display_name",
    "scope",
    "owner",
    "access_path",
    "structural_role",
    "file",
    "lineno",
    "raw_object_count",
    "callable_count",
    "access_count",
]

EDGE_FIELDNAMES = [
    "src",
    "dst",
    "edge_type",
    "relation",
    "access",
    "operation",
    "base_weight",
    "weight",
    "evidence_count",
    "confidence",
    "confidence_weight",
    "files_preview",
    "linenos_preview",
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence_weight(row: Mapping[str, Any], weight_config: StructuralWeightConfig) -> float:
    if "confidence_weight" in row:
        return _as_float(row.get("confidence_weight"), weight_config.confidence_fallback_weight)
    return weight_config.confidence_weight(_text(row.get("confidence")))


def _preview(values: Iterable[Any], limit: int = 5) -> str:
    cleaned = {_text(value) for value in values if _text(value)}
    if not cleaned:
        return ""
    return ";".join(sorted(cleaned)[:limit])


def _line_preview(values: Iterable[Any], limit: int = 5) -> str:
    cleaned = {_text(value) for value in values if _text(value)}

    def key(value: str) -> Tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return ";".join(sorted(cleaned, key=key)[:limit])


def _access_root(access_path: str) -> str:
    access_path = access_path.strip()
    if not access_path:
        return ""
    stop = len(access_path)
    for marker in (".", "["):
        idx = access_path.find(marker)
        if idx >= 0:
            stop = min(stop, idx)
    return access_path[:stop]


def _prefixed_callable(callable_id: str) -> str:
    return f"{CALLABLE_PREFIX}{callable_id}"


def _prefixed_data(object_id: str) -> str:
    return f"{DATA_PREFIX}{object_id}"


def _best_data_object(objects: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not objects:
        return {}
    included = [
        obj
        for obj in objects
        if _text(obj.get("kind")) in INCLUDED_DATA_KINDS
        and _text(obj.get("structural_role") or "primary") != "coarse"
    ]
    candidates = included or list(objects)
    return sorted(
        candidates,
        key=lambda obj: (
            _text(obj.get("display_name")),
            _text(obj.get("id")),
        ),
    )[0]


def _object_groups(
    data_access_payload: Mapping[str, Any],
) -> Tuple[Dict[str, List[Mapping[str, Any]]], Dict[str, str], Dict[str, Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    object_to_data_id: Dict[str, str] = {}
    objects_by_id: Dict[str, Mapping[str, Any]] = {}

    for obj in data_access_payload.get("objects", []):
        object_id = _text(obj.get("id"))
        if not object_id:
            continue
        objects_by_id[object_id] = obj
        object_to_data_id[object_id] = object_id
        groups[object_id].append(obj)

    return groups, object_to_data_id, objects_by_id


def _included_data_ids(
    data_access_payload: Mapping[str, Any],
    objects_by_id: Mapping[str, Mapping[str, Any]],
    precise_roots: Mapping[str, set[str]],
) -> set[str]:
    included: set[str] = set()

    def include(object_id: str) -> None:
        obj = objects_by_id.get(object_id)
        if not obj:
            return
        if _text(obj.get("kind")) not in INCLUDED_DATA_KINDS:
            return
        if _text(obj.get("structural_role") or "primary") == "coarse":
            return
        included.add(object_id)

    for edge in data_access_payload.get("edges", []):
        if _skip_coarse_data_access_edge(edge, objects_by_id, precise_roots):
            continue
        include(_text(edge.get("object_id")))

    for edge in data_access_payload.get("lineage_edges", []):
        include(_text(edge.get("src_object_id")))
        include(_text(edge.get("dst_object_id")))

    return included


def _touching_callables_by_object(
    data_access_payload: Mapping[str, Any],
    objects_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    precise_roots: Mapping[str, set[str]] | None = None,
) -> Dict[str, set[str]]:
    touching: Dict[str, set[str]] = defaultdict(set)
    for edge in data_access_payload.get("edges", []):
        if objects_by_id is not None and precise_roots is not None:
            if _skip_coarse_data_access_edge(edge, objects_by_id, precise_roots):
                continue
        callable_id = _text(edge.get("callable"))
        object_id = _text(edge.get("object_id"))
        if callable_id and object_id:
            touching[object_id].add(callable_id)
    return touching


def _access_counts_by_object(
    data_access_payload: Mapping[str, Any],
    objects_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    precise_roots: Mapping[str, set[str]] | None = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for edge in data_access_payload.get("edges", []):
        if objects_by_id is not None and precise_roots is not None:
            if _skip_coarse_data_access_edge(edge, objects_by_id, precise_roots):
                continue
        object_id = _text(edge.get("object_id"))
        if object_id:
            counts[object_id] += 1
    return counts


def _precise_roots_by_callable(
    data_access_payload: Mapping[str, Any],
    objects_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, set[str]]:
    roots: Dict[str, set[str]] = defaultdict(set)
    for edge in data_access_payload.get("edges", []):
        callable_id = _text(edge.get("callable"))
        object_id = _text(edge.get("object_id"))
        obj = objects_by_id.get(object_id)
        if not callable_id or not obj:
            continue
        if _text(obj.get("structural_role") or "primary") != "precise":
            continue
        root = _access_root(_text(obj.get("access_path") or obj.get("display_name")))
        if root:
            roots[callable_id].add(root)
    return roots


def _skip_coarse_data_access_edge(
    edge: Mapping[str, Any],
    objects_by_id: Mapping[str, Mapping[str, Any]],
    precise_roots: Mapping[str, set[str]],
) -> bool:
    callable_id = _text(edge.get("callable"))
    object_id = _text(edge.get("object_id"))
    obj = objects_by_id.get(object_id)
    if not obj:
        return False
    if _text(obj.get("structural_role") or "primary") == "coarse":
        return True
    if _text(obj.get("kind")) != "param":
        return False
    root = _text(obj.get("field") or obj.get("display_name"))
    return bool(root and root in precise_roots.get(callable_id, set()))


def _callable_rows(
    call_graph_payload: Mapping[str, Any],
    data_access_payload: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    rows: Dict[str, Mapping[str, Any]] = {}
    for row in call_graph_payload.get("nodes", []):
        callable_id = _text(row.get("id"))
        if callable_id:
            rows[callable_id] = row

    for row in data_access_payload.get("callables", []):
        callable_id = _text(row.get("id"))
        if callable_id and callable_id not in rows:
            rows[callable_id] = row

    for edge in call_graph_payload.get("edges", []):
        for key in ("caller", "callee"):
            callable_id = _text(edge.get(key))
            if callable_id and callable_id not in rows:
                rows[callable_id] = {"id": callable_id}

    for edge in data_access_payload.get("edges", []):
        callable_id = _text(edge.get("callable"))
        if callable_id and callable_id not in rows:
            rows[callable_id] = {"id": callable_id}

    return rows


def _callable_node(callable_id: str, row: Mapping[str, Any]) -> dict:
    label = _text(row.get("qualname") or row.get("id") or callable_id)
    return {
        "id": _prefixed_callable(callable_id),
        "node_type": "callable",
        "label": label,
        "kind": _text(row.get("kind")),
        "inferred_type": "",
        "module": _text(row.get("module")),
        "qualname": _text(row.get("qualname")),
        "class_name": _text(row.get("class_name")),
        "display_name": label,
        "scope": "",
        "owner": "",
        "access_path": "",
        "structural_role": "",
        "file": _text(row.get("file")),
        "lineno": _text(row.get("lineno")),
        "raw_object_count": 0,
        "callable_count": 0,
        "access_count": 0,
    }


def _data_node(
    object_id: str,
    objects: Sequence[Mapping[str, Any]],
    touching_callables: Iterable[str],
    access_count: int,
) -> dict:
    representative = _best_data_object(objects)
    display_name = _text(representative.get("display_name") or object_id)
    kind = _preview(obj.get("kind") for obj in objects)
    inferred_type = _preview(obj.get("inferred_type") for obj in objects)
    structural_role = _preview(obj.get("structural_role") or "primary" for obj in objects)
    return {
        "id": _prefixed_data(object_id),
        "node_type": "data",
        "label": display_name,
        "kind": kind,
        "inferred_type": inferred_type,
        "module": "",
        "qualname": "",
        "class_name": "",
        "display_name": display_name,
        "scope": _text(representative.get("scope")),
        "owner": _text(representative.get("owner")),
        "access_path": _text(representative.get("access_path")),
        "structural_role": structural_role,
        "file": _text(representative.get("file")),
        "lineno": _text(representative.get("lineno")),
        "raw_object_count": len(objects),
        "callable_count": len(set(touching_callables)),
        "access_count": access_count,
    }


def _add_edge(
    edge_map: MutableMapping[Tuple[str, str, str, str, str, str], dict],
    src: str,
    dst: str,
    edge_type: str,
    relation: str = "",
    access: str = "",
    operation: str = "",
    base_weight: float = 1.0,
    weight: float = 1.0,
    confidence: str = "high",
    confidence_weight: float = 1.0,
    file: str = "",
    lineno: Any = "",
) -> None:
    key = (src, dst, edge_type, relation, access, operation)
    row = edge_map.get(key)
    if row is None:
        row = {
            "src": src,
            "dst": dst,
            "edge_type": edge_type,
            "relation": relation,
            "access": access,
            "operation": operation,
            "base_weight": 0.0,
            "weight": 0.0,
            "evidence_count": 0,
            "_confidences": [],
            "_confidence_weights": [],
            "_files": set(),
            "_linenos": set(),
        }
        edge_map[key] = row

    row["base_weight"] += base_weight
    row["weight"] += weight
    row["evidence_count"] += 1
    row["_confidences"].append(confidence)
    row["_confidence_weights"].append(confidence_weight)
    if file:
        row["_files"].add(file)
    if _text(lineno):
        row["_linenos"].add(_text(lineno))


def _evidence_count_factor(edge_type: str, evidence_count: int) -> float:
    if evidence_count <= 1:
        return 1.0
    if edge_type == "data_lineage":
        return 1.0
    return 1.0 + math.log2(evidence_count)


def _finalize_edges(edge_map: Mapping[Tuple[str, str, str, str, str, str], dict]) -> List[dict]:
    rows: List[dict] = []
    for row in edge_map.values():
        evidence_count = row["evidence_count"]
        count_factor = _evidence_count_factor(row["edge_type"], evidence_count)
        base_weight = (row["base_weight"] / evidence_count) * count_factor
        weight = (row["weight"] / evidence_count) * count_factor
        confidence_weights = row["_confidence_weights"] or [1.0]
        confidence_weight = sum(confidence_weights) / len(confidence_weights)
        rows.append(
            {
                "src": row["src"],
                "dst": row["dst"],
                "edge_type": row["edge_type"],
                "relation": row["relation"],
                "access": row["access"],
                "operation": row["operation"],
                "base_weight": round(base_weight, 6),
                "weight": round(weight, 6),
                "evidence_count": evidence_count,
                "confidence": _preview(row["_confidences"]),
                "confidence_weight": round(confidence_weight, 6),
                "files_preview": _preview(row["_files"]),
                "linenos_preview": _line_preview(row["_linenos"]),
            }
        )

    return sorted(
        rows,
        key=lambda edge: (
            edge["src"],
            edge["dst"],
            edge["edge_type"],
            edge["relation"],
            edge["access"],
            edge["operation"],
        ),
    )


def build_structural_graph(
    call_graph_payload: Mapping[str, Any],
    data_access_payload: Mapping[str, Any],
    weight_config: StructuralWeightConfig | None = None,
    hub_options: HubDetectionOptions | None = None,
) -> dict:
    """Return nodes and weighted edges for the heterogeneous structural graph."""
    weight_config = weight_config or load_weight_config()
    hub_options = hub_options or HubDetectionOptions()
    object_groups, object_to_data_id, objects_by_id = _object_groups(data_access_payload)
    callables = _callable_rows(call_graph_payload, data_access_payload)
    precise_roots = _precise_roots_by_callable(data_access_payload, objects_by_id)
    included_data_ids = _included_data_ids(data_access_payload, objects_by_id, precise_roots)
    touching_by_object = _touching_callables_by_object(
        data_access_payload,
        objects_by_id,
        precise_roots,
    )
    access_counts = _access_counts_by_object(
        data_access_payload,
        objects_by_id,
        precise_roots,
    )

    nodes: List[dict] = []
    for callable_id, row in sorted(callables.items()):
        nodes.append(_callable_node(callable_id, row))

    for object_id in sorted(included_data_ids):
        nodes.append(
            _data_node(
                object_id=object_id,
                objects=object_groups[object_id],
                touching_callables=touching_by_object.get(object_id, set()),
                access_count=access_counts.get(object_id, 0),
            )
        )

    edge_map: Dict[Tuple[str, str, str, str, str, str], dict] = {}

    for edge in call_graph_payload.get("edges", []):
        caller = _text(edge.get("caller"))
        callee = _text(edge.get("callee"))
        if not caller or not callee:
            continue
        relation = _text(edge.get("relation"))
        # Two independent discounts, because they answer different questions.
        #
        # ``relation`` is the *kind* of dependency: a registered child is coupled
        # to its parent differently from a literal call, whether or not either is
        # certain.
        #
        # ``confidence`` is how sure the analyzer is that this call happens at
        # all, graded by how many targets its call site resolved to. Without it a
        # receiver with one possible type and a receiver with five both arrived
        # here as ``inferred_type`` at full weight, and a guess pulled two
        # unrelated classes into one cluster exactly as hard as a certainty. On
        # climlab the graded buckets differ by 70x in measured falsification rate.
        base_weight = weight_config.call_relation_weight(relation)
        confidence = _text(edge.get("confidence")) or "high"
        confidence_weight = weight_config.confidence_weight(confidence)
        _add_edge(
            edge_map=edge_map,
            src=_prefixed_callable(caller),
            dst=_prefixed_callable(callee),
            edge_type="call",
            relation=relation,
            base_weight=base_weight,
            weight=base_weight * confidence_weight,
            confidence=confidence,
            confidence_weight=confidence_weight,
            file=_text(edge.get("file")),
            lineno=edge.get("lineno"),
        )

    for edge in data_access_payload.get("edges", []):
        callable_id = _text(edge.get("callable"))
        object_id = _text(edge.get("object_id"))
        if _skip_coarse_data_access_edge(edge, objects_by_id, precise_roots):
            continue
        if not callable_id or object_id not in included_data_ids:
            continue

        access = _text(edge.get("access"))
        base_weight = weight_config.data_access_weight(access)
        confidence_weight = _confidence_weight(edge, weight_config)
        touching_count = len(touching_by_object.get(object_id, set())) or 1
        final_weight = base_weight * confidence_weight / math.log2(2 + touching_count)

        _add_edge(
            edge_map=edge_map,
            src=_prefixed_callable(callable_id),
            dst=_prefixed_data(object_id),
            edge_type="data_access",
            access=access,
            operation=_text(edge.get("operation")),
            base_weight=base_weight,
            weight=final_weight,
            confidence=_text(edge.get("confidence")),
            confidence_weight=confidence_weight,
            file=_text(edge.get("file")),
            lineno=edge.get("lineno"),
        )

    for edge in data_access_payload.get("lineage_edges", []):
        src_data_id = object_to_data_id.get(_text(edge.get("src_object_id")))
        dst_data_id = object_to_data_id.get(_text(edge.get("dst_object_id")))
        if not src_data_id or not dst_data_id:
            continue
        if src_data_id == dst_data_id:
            continue
        if src_data_id not in included_data_ids or dst_data_id not in included_data_ids:
            continue

        relation = _text(edge.get("relation"))
        base_weight = weight_config.data_lineage_weight(relation)
        _add_edge(
            edge_map=edge_map,
            src=_prefixed_data(src_data_id),
            dst=_prefixed_data(dst_data_id),
            edge_type="data_lineage",
            relation=relation,
            base_weight=base_weight,
            weight=base_weight,
            confidence="high",
            confidence_weight=1.0,
            file=_text(edge.get("file")),
            lineno=edge.get("lineno"),
        )

    edges = _finalize_edges(edge_map)
    nodes = sorted(nodes, key=lambda node: (node["node_type"], node["id"]))
    callable_hub_nodes, data_hub_nodes, _degree_map = identify_hub_nodes(
        node_rows_by_id(nodes),
        edges,
        hub_options,
    )

    return {
        "schema": "heterogeneous_structural_dependency_graph.v1",
        "nodes": nodes,
        "edges": edges,
        "callable_hub_nodes": callable_hub_nodes,
        "data_hub_nodes": data_hub_nodes,
        "included_data_kinds": sorted(INCLUDED_DATA_KINDS),
        "weights": {
            "call": {"call": weight_config.call_weight},
            "data_access": dict(weight_config.data_access_weights),
            "data_access_fallback": weight_config.data_access_fallback_weight,
            "data_lineage": dict(weight_config.data_lineage_weights),
            "data_lineage_fallback": weight_config.data_lineage_fallback_weight,
            "confidence": dict(weight_config.confidence_weights),
            "confidence_fallback": weight_config.confidence_fallback_weight,
        },
        "weight_config": weight_config.to_dict(),
        "hub_detection_options": asdict(hub_options),
        "stats": {
            "node_count": len(nodes),
            "callable_node_count": sum(1 for node in nodes if node["node_type"] == "callable"),
            "data_node_count": sum(1 for node in nodes if node["node_type"] == "data"),
            "edge_count": len(edges),
            "callable_hub_node_count": len(callable_hub_nodes),
            "data_hub_node_count": len(data_hub_nodes),
        },
    }


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_artifact_guide(path: Path) -> None:
    lines = [
        "# Structural Dependency Graph",
        "",
        "This folder contains a heterogeneous structural dependency graph generated",
        "by `microservice-pipeline structural-graph`.",
        "",
        "The graph combines callable-callable call edges, callable-data access",
        "edges, and data-data lineage edges. It is intended for later vertical",
        "microservice-candidate clustering where a candidate can contain both",
        "functions/methods and their service-relevant data objects.",
        "",
        "## Files",
        "",
        "- `nodes.csv`: callable and data nodes with metadata.",
        "- `edges.csv`: typed weighted graph edges with aggregated evidence.",
        "- `callable_hub_nodes.csv`: structural callable hub nodes with degree/fanout reasons.",
        "- `data_hub_nodes.csv`: structural data hub nodes with degree/access reasons.",
        "- `structural_graph.json`: complete machine-readable graph payload.",
        "- `README.md`: this guide.",
        "",
        "## Node IDs",
        "",
        "- Callable nodes are prefixed with `callable:`.",
        "- Data nodes are prefixed with `data:` and use data-access `object_id` identity.",
        "",
        "## Edge Types",
        "",
        "- `call`: callable-to-callable control dependency.",
        "- `data_access`: callable-to-data read/write/create dependency.",
        "- `data_lineage`: data-to-data alias, return, assignment, or state lineage.",
        "",
        "## Weights",
        "",
        "Repeated evidence is aggregated into one edge and recorded as",
        "`evidence_count`. For `call` and `data_access` edges, repeated evidence",
        "uses saturated weighting with `1 + log2(evidence_count)` instead of a",
        "linear multiplier. Repeated `data_lineage` evidence is treated as",
        "identity/flow evidence and does not increase edge weight.",
        "",
        "Data-access weights are downweighted by the number of callables touching",
        "the same data object so common objects do not dominate future clustering.",
        "Generation weights are resolved from the default structural weight profile",
        "or from the profile supplied with `--weight-config`; the resolved profile",
        "is recorded in `structural_graph.json` as `weight_config`.",
        "",
        "## Data Node Metadata",
        "",
        "Data node `kind` is the extractor object category used by",
        "`included_data_kinds`, such as `class_attr_state`, `df_col`, or",
        "`local_exposed`. Data node `inferred_type` is the coarse type family",
        "reported by data access, such as `object`, `dataframe`, `dict`, `list`,",
        "`file`, or `unknown`; it is metadata and is not the inclusion filter.",
        "`access_path` preserves the source-like nested object path selected",
        "by the data-access extractor when available. `structural_role` records",
        "whether a data object is a primary object, a precise nested path, or a",
        "coarse audit object. Coarse objects are not included as structural data",
        "nodes when they have been superseded by precise path objects.",
        "",
        "## Hub Nodes",
        "",
        "`callable_hub_nodes.csv` and `data_hub_nodes.csv` are neutral structural",
        "analysis artifacts. They identify high-degree, zero-in/high-out",
        "entrypoint, and low-in/high-out orchestrator hub nodes but do not decide",
        "whether a node should be dropped from clustering. Drop/keep policy",
        "remains part of `microservice-pipeline structural-cluster`.",
    ]
    write_markdown(path, lines, trailing_newline=True)


def write_outputs(outdir: Path, graph: Mapping[str, Any]) -> None:
    ensure_dir(outdir)
    write_csv_rows(outdir / "nodes.csv", NODE_FIELDNAMES, graph["nodes"])
    write_csv_rows(outdir / "edges.csv", EDGE_FIELDNAMES, graph["edges"])
    write_csv_rows(
        outdir / "callable_hub_nodes.csv",
        CALLABLE_HUB_NODE_FIELDNAMES,
        graph.get("callable_hub_nodes", []),
    )
    write_csv_rows(
        outdir / "data_hub_nodes.csv",
        DATA_HUB_NODE_FIELDNAMES,
        graph.get("data_hub_nodes", []),
    )
    write_json(outdir / "structural_graph.json", graph)
    write_artifact_guide(outdir / "README.md")


def _resolve_config_path(project_root: Path, value: Any) -> str | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else project_root / path)


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    return value if isinstance(value, Mapping) else {}


def _config_defaults(config_path: Path, project_root: Path) -> Dict[str, Any]:
    payload = load_jsonc(config_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Structural graph config must be a JSON object: {config_path}")
    paths = _section(payload, "paths")
    weighting = _section(payload, "weighting")
    hubs = _section(payload, "hub_detection") or _section(payload, "hub_policy")
    defaults: Dict[str, Any] = {}
    for source_key, dest_key in {
        "call_graph": "call_graph",
        "data_access": "data_access",
        "outdir": "outdir",
    }.items():
        value = _resolve_config_path(project_root, paths.get(source_key))
        if value is not None:
            defaults[dest_key] = value
    weight_config = resolve_weight_config_reference(weighting.get("weight_config"), project_root=project_root)
    if weight_config is not None:
        defaults["weight_config"] = weight_config
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
        if key in hubs and hubs[key] is not None:
            defaults[key] = hubs[key]
    return defaults


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a heterogeneous structural dependency graph")
    parser.add_argument("--project-root", default=".", help="Project root used to resolve config-relative paths")
    parser.add_argument("--config", default=None, type=Path, help="Optional structural graph JSON/JSONC config")
    parser.add_argument(
        "--call-graph",
        default="artifacts/call_graph/call_graph.json",
        help="Path to call_graph.json",
    )
    parser.add_argument(
        "--data-access",
        default="artifacts/data_access_inferred/data_access.json",
        help="Path to data_access.json",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/structural_dependency_graph",
        help="Output directory",
    )
    parser.add_argument(
        "--weight-config",
        default=None,
        help="Optional structural weight profile JSON or builtin:NAME alias. Defaults to builtin:default.",
    )
    parser.add_argument(
        "--hub-callable-degree-percentile",
        type=float,
        default=95.0,
        help="Percentile used with --hub-callable-min-degree to detect callable hub nodes",
    )
    parser.add_argument(
        "--hub-callable-min-degree",
        type=int,
        default=25,
        help="Minimum structural degree for degree-based callable hub-node detection",
    )
    parser.add_argument(
        "--hub-callable-min-in-degree",
        type=int,
        default=2,
        help="Minimum incoming structural degree for degree-based callable hub-node detection",
    )
    parser.add_argument(
        "--hub-callable-min-out-degree",
        type=int,
        default=2,
        help="Minimum outgoing structural degree for degree-based callable hub-node detection",
    )
    parser.add_argument(
        "--hub-entrypoint-min-out-degree",
        type=int,
        default=12,
        help="Minimum outgoing structural degree for zero-incoming callable entrypoint hub-node detection",
    )
    parser.add_argument(
        "--hub-orchestrator-max-in-degree",
        type=int,
        default=1,
        help="Maximum incoming structural degree for callable orchestrator hub-node detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-out-degree",
        type=int,
        default=12,
        help="Minimum outgoing structural degree for callable orchestrator hub-node detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-out-call-degree",
        type=int,
        default=4,
        help="Minimum outgoing call edges for callable orchestrator hub-node detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-target-modules",
        type=int,
        default=3,
        help="Minimum distinct target callable modules for callable orchestrator hub-node detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-target-callables",
        type=int,
        default=4,
        help="Minimum distinct target callables for callable orchestrator hub-node detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-target-data",
        type=int,
        default=4,
        help="Minimum distinct target data objects for callable orchestrator hub-node detection",
    )
    parser.add_argument(
        "--hub-orchestrator-min-data-to-call-ratio",
        type=float,
        default=1.0,
        help="Minimum distinct target data/callable ratio for callable orchestrator hub-node detection",
    )
    parser.add_argument("--hub-data-min-degree", type=int, default=20, help="Data hub-node structural degree threshold")
    parser.add_argument(
        "--hub-data-min-callable-count",
        type=int,
        default=10,
        help="Data hub-node callable_count threshold",
    )
    parser.add_argument(
        "--hub-data-min-access-count",
        type=int,
        default=100,
        help="Data hub-node access_count threshold",
    )
    args = parser.parse_args(argv)
    if args.config:
        project_root = Path(args.project_root).resolve()
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        for key, value in _config_defaults(config_path, project_root).items():
            setattr(args, key, value)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    call_graph_path = Path(args.call_graph).resolve()
    data_access_path = Path(args.data_access).resolve()
    outdir = Path(args.outdir).resolve()
    weight_config = load_weight_config(args.weight_config)

    graph = build_structural_graph(
        call_graph_payload=_load_json(call_graph_path),
        data_access_payload=_load_json(data_access_path),
        weight_config=weight_config,
        hub_options=HubDetectionOptions(
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
        ),
    )
    write_outputs(outdir, graph)

    stats = graph["stats"]
    print(f"Structural dependency graph written to: {outdir}")
    print(f"Weight profile: {weight_config.name or '(unnamed)'}")
    print(f"Nodes: {stats['node_count']} (callable: {stats['callable_node_count']}, data: {stats['data_node_count']})")
    print(f"Edges: {stats['edge_count']}")
    print(
        "Hub nodes: "
        f"callable={stats['callable_hub_node_count']}, "
        f"data={stats['data_hub_node_count']}"
    )


if __name__ == "__main__":
    main()
