"""Artifact writers for the static data-access extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown
from microservice_pipeline.call_graph.generate_call_graph_ast import CallableDef
from microservice_pipeline.data_access.models import AccessEdge, DataObject, LineageEdge, confidence_weight


def _callables_payload(callable_map: Dict[str, CallableDef]) -> List[dict]:
    return [c.__dict__ for c in sorted(callable_map.values(), key=lambda x: x.id)]


def _objects_payload(objects: Dict[str, DataObject]) -> List[dict]:
    payload: List[dict] = []
    for data_object in sorted(objects.values(), key=lambda x: x.id):
        row = data_object.__dict__.copy()
        payload.append(row)
    return payload


def _edges_payload(edges: Sequence[AccessEdge], objects: Dict[str, DataObject]) -> List[dict]:
    return [
        {
            **e.__dict__,
            "confidence_weight": confidence_weight(e.confidence),
        }
        for e in sorted(
            edges,
            key=lambda x: (x.callable, x.object_id, x.access, x.lineno, x.operation),
        )
    ]


def _lineage_payload(lineage_edges: Sequence[LineageEdge]) -> List[dict]:
    return [
        edge.__dict__.copy()
        for edge in sorted(
            lineage_edges,
            key=lambda x: (x.src_object_id, x.dst_object_id, x.relation, x.lineno, x.slot),
        )
    ]


def write_outputs(
    outdir: Path,
    callable_map: Dict[str, CallableDef],
    objects: Dict[str, DataObject],
    edges: List[AccessEdge],
    lineage_edges: List[LineageEdge],
) -> None:
    ensure_dir(outdir)

    write_csv_rows(
        outdir / "data_objects.csv",
        [
            "id",
            "kind",
            "display_name",
            "scope",
            "owner",
            "container",
            "field",
            "file",
            "lineno",
            "inferred_type",
            "confidence",
            "alias_of",
            "access_path",
            "structural_role",
        ],
        _objects_payload(objects),
    )
    write_csv_rows(
        outdir / "access_edges.csv",
        [
            "callable",
            "object_id",
            "access",
            "operation",
            "file",
            "lineno",
            "confidence",
            "confidence_weight",
            "evidence",
        ],
        _edges_payload(edges, objects),
    )

    denormalized_rows = []
    for edge in sorted(edges, key=lambda x: (x.callable, x.object_id, x.lineno, x.operation)):
        obj = objects.get(edge.object_id)
        denormalized_rows.append(
            {
                "callable": edge.callable,
                "access": edge.access,
                "operation": edge.operation,
                "object_id": edge.object_id,
                "object_kind": obj.kind if obj else "",
                "display_name": obj.display_name if obj else edge.object_id,
                "scope": obj.scope if obj else "",
                "owner": obj.owner if obj else "",
                "field": obj.field if obj else "",
                "inferred_type": obj.inferred_type if obj else "",
                "access_path": obj.access_path if obj else "",
                "structural_role": obj.structural_role if obj else "",
                "confidence": edge.confidence,
                "confidence_weight": confidence_weight(edge.confidence),
                "file": edge.file,
                "lineno": edge.lineno,
                "evidence": edge.evidence,
            }
        )
    write_csv_rows(
        outdir / "callable_data_access.csv",
        [
            "callable",
            "access",
            "operation",
            "object_id",
            "object_kind",
            "display_name",
            "scope",
            "owner",
            "field",
            "inferred_type",
            "access_path",
            "structural_role",
            "confidence",
            "confidence_weight",
            "file",
            "lineno",
            "evidence",
        ],
        denormalized_rows,
    )

    payload = {
        "callables": _callables_payload(callable_map),
        "objects": _objects_payload(objects),
        "edges": _edges_payload(edges, objects),
        "lineage_edges": _lineage_payload(lineage_edges),
    }
    write_json(outdir / "data_access.json", payload)
    write_artifact_guide(outdir / "README.md")
    write_report(outdir / "data_access_report.md", callable_map, objects, edges)


def _edge_summary(edges: Iterable[AccessEdge]) -> Dict[Tuple[str, str], int]:
    summary: Dict[Tuple[str, str], int] = {}
    for edge in edges:
        key = (edge.access, edge.object_id)
        summary[key] = summary.get(key, 0) + 1
    return summary


def write_artifact_guide(doc_path: Path) -> None:
    lines = [
        "# Data Access Artifacts",
        "",
        "This folder contains the static data-access view generated by",
        "`microservice-pipeline data-access`.",
        "",
        "The extractor records which callable reads, writes, creates, or mutates",
        "service-relevant data objects such as parameters, object state, globals,",
        "DataFrame columns, dictionary keys, unknown-family container fields,",
        "files, and exposed local containers.",
        "",
        "## Files",
        "",
        "- `data_objects.csv`: one row per data object discovered by the extractor.",
        "- `access_edges.csv`: one row per callable-to-object access edge.",
        "- `callable_data_access.csv`: denormalized join of edge rows with object metadata.",
        "- `data_access.json`: JSON payload containing callables, objects, access edges, and lineage edges.",
        "- `data_access_report.md`: human-readable summaries by callable and data object.",
        "- `README.md`: this schema and interpretation guide.",
        "",
        "## Data Object Identity",
        "",
        "The extractor exposes one data-object identity:",
        "",
        "- `object_id`: the data object touched by source evidence.",
        "  Example: `df_col:sample.summarize:Results_extended:Compartment`.",
        "",
        "Nested non-class object paths are kept precise when a callable touches",
        "a specific attribute/key chain. For example, a whole `model` parameter",
        "can coexist with precise paths such as `model.R['mass_g']` and",
        "`model.system_particle_object_list[].RateConstants['k_fragmentation']`.",
        "",
        "There is no separate clustering identity. Downstream graph construction uses",
        "`object_id` directly. Alias and flow relationships are preserved separately",
        "in `alias_of` and `lineage_edges` instead of being hidden behind an ID roll-up.",
        "",
        "By default, extraction is raw: same-name containers are not globalized across",
        "callables unless you provide `--shared-containers-config`.",
        "",
        "## Access Semantics",
        "",
        "- `read`: the callable uses the object in an expression.",
        "- `write`: the callable assigns into the object or overwrites part of it.",
        "- `create`: the callable creates the object in the current scope.",
        "- `read_write`: the callable mutates the object or both reads and writes it.",
        "",
        "## Confidence",
        "",
        "Confidence describes how directly the object was identified from the AST:",
        "",
        "- `high`: direct evidence such as `self.data['key']` or `df['col']`.",
        "- `medium`: propagated alias or obvious container inference.",
        "- `low`: dynamic or weakly resolved identity.",
        "",
        "`confidence_weight` is the numeric form used by downstream clustering:",
        "",
        "- `high = 1.0`",
        "- `medium = 0.6`",
        "- `low = 0.25`",
        "",
        "## Object Kinds",
        "",
        "- `param`: callable parameter.",
        "- `module_global`: module-level global or constant.",
        "- `class_state`: object state on `self` / `cls`, rolled up to the class level.",
        "- `class_attr_state`: top-level class attribute state for coordinator-style classes that are split selectively.",
        "- `object_state`: non-class object state reached through a parameter/local.",
        "- `df_col`: DataFrame field or column access.",
        "- `dict_key`: dictionary key access.",
        "- `container_field`: keyed or column-like access where the base container family remained unknown.",
        "- `file`: file path or file handle target.",
        "- `local_exposed`: local container that escaped by being returned, passed, or assigned into externally reachable state.",
        "- `unknown`: fallback object created when an edge exists but the object metadata was not materialized earlier.",
        "",
        "## Column Guide",
        "",
        "### `data_objects.csv`",
        "",
        "- `id`: data object ID used by access edges and downstream structural graph nodes.",
        "- `kind`: object category.",
        "- `display_name`: human-readable form, usually close to source syntax.",
        "- `scope`: where the object lives, such as `callable`, `field`, `class`, `module`, `external`, or `object`.",
        "- `owner`: owning callable/object/container context for the raw object.",
        "- `container`: parent container ID when this is a field-level object.",
        "- `field`: field/key/parameter/local name when applicable.",
        "- `file`: source file where the object was first registered.",
        "- `lineno`: source line where the object was first registered.",
        "- `inferred_type`: coarse family from Pyright or syntax-certain inference.",
        "- `confidence`: categorical confidence label.",
        "- `alias_of`: underlying object ID if this object aliases another object.",
        "- `access_path`: source-like attribute/key path represented by this object, when available.",
        "- `structural_role`: `primary`, `precise`, or `coarse`.",
        "",
        "### `access_edges.csv`",
        "",
        "- `callable`: callable ID using the object.",
        "- `object_id`: data object ID used by the callable.",
        "- `access`: one of `read`, `write`, `create`, or `read_write`.",
        "- `operation`: AST-level operation label such as `load`, `subscript_load`, `assign`, `open`, or `method:append`.",
        "- `file`: source file where the access occurs.",
        "- `lineno`: source line where the access occurs.",
        "- `confidence`: categorical confidence for this edge.",
        "- `confidence_weight`: numeric form of the confidence label.",
        "- `evidence`: short source-like snippet showing why the edge exists.",
        "",
        "### `callable_data_access.csv`",
        "",
        "This is `access_edges.csv` enriched with object metadata so you can filter or",
        "group without joining manually.",
        "",
        "- `callable`, `access`, `operation`, `object_id`: same meaning as above.",
        "- `object_kind`: copied from the object.",
        "- `display_name`: copied from the object.",
        "- `scope`, `owner`, `field`, `inferred_type`, `access_path`, `structural_role`: copied from the object.",
        "- `confidence`, `confidence_weight`, `file`, `lineno`, `evidence`: copied from the edge.",
        "",
        "### `data_access.json`",
        "",
        "Top-level keys:",
        "",
        "- `callables`: callable metadata using the same callable IDs as the call graph extractor.",
        "- `objects`: same information as `data_objects.csv`.",
        "- `edges`: same information as `access_edges.csv`.",
        "- `lineage_edges`: object-to-object flow edges used to solve interprocedural aliases and to support shared-container inference.",
        "",
        "### `data_access_report.md`",
        "",
        "Report sections:",
        "",
        "- `By Callable`: what each callable touches.",
        "- `By Data Object`: object-level view for audit/debugging.",
        "",
        "## Reading the Output",
        "",
        "A useful workflow is:",
        "",
        "1. Start with the raw extractor output with no shared-container config.",
        "2. Use `lineage_edges` and `callable_data_access.csv` to understand which containers are genuinely connected.",
        "3. Generate a draft config with `microservice-pipeline infer-shared-containers` and review the report.",
        "4. Re-run the extractor with `microservice-pipeline data-access --shared-containers-config` once the mappings look safe.",
        "5. Use `object_id` as the data-node identity for clustering input.",
        "6. For evaluation sets focused on ownership, filter to objects with at least one `create`, `write`, or `read_write` edge.",
        "7. Fall back to `evidence` and `display_name` when a row looks surprising.",
        "",
    ]
    write_markdown(doc_path, lines)


def write_report(
    report_path: Path,
    callable_map: Dict[str, CallableDef],
    objects: Dict[str, DataObject],
    edges: List[AccessEdge],
) -> None:
    edges_by_callable: Dict[str, List[AccessEdge]] = {}
    edges_by_object: Dict[str, List[AccessEdge]] = {}
    for edge in edges:
        edges_by_callable.setdefault(edge.callable, []).append(edge)
        edges_by_object.setdefault(edge.object_id, []).append(edge)

    lines = [
        "# Data Access View",
        "",
        f"- Callables with access: `{len(edges_by_callable)}`",
        f"- Data objects: `{len(objects)}`",
        f"- Access edges: `{len(edges)}`",
        "- Confidence weights: `high=1.0`, `medium=0.6`, `low=0.25`",
        "",
        "## By Callable",
        "",
    ]

    for callable_id in sorted(edges_by_callable):
        meta = callable_map.get(callable_id)
        title = meta.qualname if meta else callable_id
        lines.extend([f"### {callable_id}", ""])
        if meta:
            lines.append(f"- Location: `{meta.file}:{meta.lineno}`")
            lines.append(f"- Qualname: `{title}`")
        lines.extend(
            [
                "",
                "| Access | Object | Kind | Count | Confidence | Weight |",
                "| --- | --- | --- | ---: | --- | ---: |",
            ]
        )
        summary = _edge_summary(edges_by_callable[callable_id])
        for (access, object_id), count in sorted(summary.items(), key=lambda item: (item[0][1], item[0][0])):
            obj = objects.get(object_id)
            obj_name = obj.display_name if obj else object_id
            kind = obj.kind if obj else ""
            confidence = obj.confidence if obj else ""
            weight = confidence_weight(confidence) if confidence else ""
            lines.append(
                f"| {access} | `{obj_name}` | {kind} | {count} | {confidence} | {weight} |"
            )
        lines.append("")

    lines.extend(["## By Data Object", ""])
    for object_id in sorted(edges_by_object):
        obj = objects.get(object_id)
        if not obj:
            continue
        access_counts: Dict[str, int] = {}
        callable_ids: set[str] = set()
        for edge in edges_by_object[object_id]:
            access_counts[edge.access] = access_counts.get(edge.access, 0) + 1
            callable_ids.add(edge.callable)
        count_text = ", ".join(f"{access}={count}" for access, count in sorted(access_counts.items()))
        callable_preview = "; ".join(sorted(callable_ids)[:8])
        if len(callable_ids) > 8:
            callable_preview += f"; ... ({len(callable_ids)} total)"
        lines.extend(
            [
                f"### {object_id}",
                "",
                f"- Display: `{obj.display_name}`",
                f"- Kind: `{obj.kind}`",
                f"- Access path: `{obj.access_path}`",
                f"- Structural role: `{obj.structural_role}`",
                f"- Access counts: {count_text}",
                f"- Callables: {callable_preview}",
                "",
            ]
        )

    write_markdown(report_path, lines)
