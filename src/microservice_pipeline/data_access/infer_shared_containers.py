#!/usr/bin/env python3
"""Infer a draft shared-containers config from raw data-access artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.artifact_io import write_json, write_markdown
    from microservice_pipeline.config import load_extraction_config
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import write_json, write_markdown  # type: ignore
    from microservice_pipeline.config import load_extraction_config  # type: ignore

try:
    from microservice_pipeline.call_graph.generate_call_graph_ast import path_to_module
    from microservice_pipeline.data_access.pyright_type_probe import (
        FAMILY_UNKNOWN,
        PROBE_RE,
        PyrightProbeTarget,
        discover_project_root,
        probe_pyright_targets,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from microservice_pipeline.call_graph.generate_call_graph_ast import path_to_module  # type: ignore
    from microservice_pipeline.data_access.pyright_type_probe import (  # type: ignore
        FAMILY_UNKNOWN,
        PROBE_RE,
        PyrightProbeTarget,
        discover_project_root,
        probe_pyright_targets,
    )

try:
    from microservice_pipeline.data_access.models import IDENTITY_RELATIONS
except ImportError:  # pragma: no cover - supports direct script execution
    from data_access.models import IDENTITY_RELATIONS  # type: ignore


GENERIC_CONTAINER_NAMES = {"df", "data", "results", "table", "mapping", "dict", "items"}


@dataclass
class CandidateContainer:
    kind: str
    container_id: str
    leaf_name: str
    normalized_leaf: str
    object_kind: str
    fields: Set[str] = field(default_factory=set)
    callables: Set[str] = field(default_factory=set)
    roots: Set[str] = field(default_factory=set)
    file: str = ""
    lineno: int = 0
    inferred_type: str = ""
    pyright_family: str = ""


@dataclass
class PairDecision:
    kind: str
    normalized_leaf: str
    left: str
    right: str
    accepted: bool
    reasons: List[str]
    jaccard: float
    overlap: float
    shared_lineage: bool
    direct_call: bool
    common_parent: bool


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_call_graph_edges(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _display_base(display_name: str) -> str:
    return display_name.split("[", 1)[0].strip()


def _normalized_leaf(name: str) -> str:
    return name.strip().lower()


def _candidate_kind(object_kind: str) -> str:
    return "dataframe" if object_kind == "df_col" else "dict"


def _lineage_root_ids_with_cycle_flag(
    object_id: str,
    incoming: Dict[str, Set[str]],
    known_ids: Set[str],
    cache: Dict[str, Set[str]],
    seen: Set[str],
) -> Tuple[Set[str], bool]:
    """Reaching roots, and whether a live cycle truncated them.

    The same function and the same defect as
    ``generate_data_access_ast._lineage_roots``: the cycle guard's empty set is
    correct only for the traversal that produced it, and caching a value
    computed above one made every later answer depend on which node was asked
    first. Fixed here as well as there, because a fix applied to one copy of a
    duplicated function and not the other is how the two drift apart.
    """
    if object_id in cache:
        return cache[object_id], False
    if object_id in seen:
        return set(), True
    seen.add(object_id)

    roots: Set[str] = set()
    truncated = False
    for parent_id in sorted(incoming.get(object_id, set())):
        if parent_id == object_id:
            continue
        parent_roots, parent_truncated = _lineage_root_ids_with_cycle_flag(
            parent_id, incoming, known_ids, cache, seen
        )
        truncated = truncated or parent_truncated
        if parent_roots:
            roots.update(parent_roots)
        elif parent_id in known_ids:
            roots.add(parent_id)

    seen.remove(object_id)
    if not truncated:
        cache[object_id] = roots
    return roots, truncated


def _lineage_root_ids(
    object_id: str,
    incoming: Dict[str, Set[str]],
    known_ids: Set[str],
    cache: Dict[str, Set[str]],
    seen: Optional[Set[str]] = None,
) -> Set[str]:
    roots, _truncated = _lineage_root_ids_with_cycle_flag(
        object_id, incoming, known_ids, cache, seen if seen is not None else set()
    )
    return roots


def _lineage_roots_for_objects(data_access_payload: dict) -> Dict[str, Set[str]]:
    objects = {row["id"]: row for row in data_access_payload.get("objects", [])}
    incoming: Dict[str, Set[str]] = defaultdict(set)
    for obj in objects.values():
        alias_of = (obj.get("alias_of") or "").strip()
        if alias_of:
            incoming[obj["id"]].add(alias_of)
    for edge in data_access_payload.get("lineage_edges", []):
        # Only identity relations. A ``derived_from`` edge says the value was
        # made from its source, not that it *is* its source, so following one to
        # a root would put back exactly the merges Step 4a removed -- the
        # section 4.3 drift this file has already suffered once, when its copy
        # of the root walk inherited section 1.3's caching bug.
        if str(edge.get("relation", "")) not in IDENTITY_RELATIONS:
            continue
        src = (edge.get("src_object_id") or "").strip()
        dst = (edge.get("dst_object_id") or "").strip()
        if src and dst:
            incoming[dst].add(src)

    cache: Dict[str, Set[str]] = {}
    roots: Dict[str, Set[str]] = {}
    known_ids = set(objects)
    for object_id in sorted(objects):
        roots[object_id] = {
            root_id
            for root_id in _lineage_root_ids(object_id, incoming, known_ids, cache)
            if root_id and root_id != object_id and root_id in objects
        }
    return roots


def _build_candidates(data_access_payload: dict) -> Dict[Tuple[str, str], CandidateContainer]:
    objects = {row["id"]: row for row in data_access_payload.get("objects", [])}
    lineage_roots = _lineage_roots_for_objects(data_access_payload)
    candidates: Dict[Tuple[str, str], CandidateContainer] = {}

    field_rows = [row for row in data_access_payload.get("objects", []) if row.get("kind") in {"df_col", "dict_key"}]
    callable_by_object: Dict[str, Set[str]] = defaultdict(set)
    for edge in data_access_payload.get("edges", []):
        callable_by_object[(edge.get("object_id") or "").strip()].add((edge.get("callable") or "").strip())

    for row in field_rows:
        container_id = (row.get("container") or row.get("owner") or "").strip()
        if not container_id:
            continue
        key = (row["kind"], container_id)
        container_obj = objects.get(container_id, {})
        display_name = container_obj.get("display_name") or _display_base(row.get("display_name", ""))
        leaf_name = display_name.split(".")[-1].strip()
        candidate = candidates.setdefault(
            key,
            CandidateContainer(
                kind=_candidate_kind(row["kind"]),
                container_id=container_id,
                leaf_name=leaf_name,
                normalized_leaf=_normalized_leaf(leaf_name),
                object_kind=row["kind"],
                file=container_obj.get("file", ""),
                lineno=int(container_obj.get("lineno") or 0),
                inferred_type=(container_obj.get("inferred_type") or "").strip(),
            ),
        )
        field_name = (row.get("field") or "").strip()
        if field_name:
            candidate.fields.add(field_name)
        candidate.roots.update(lineage_roots.get(container_id, set()))
        callable_ids = callable_by_object.get(row["id"], set())
        candidate.callables.update(callable_id for callable_id in callable_ids if callable_id)
    return candidates


def _build_call_graph_views(call_graph_edges: Sequence[dict]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    incoming: Dict[str, Set[str]] = defaultdict(set)
    for row in call_graph_edges:
        if str(row.get("resolved", "")).strip() not in {"1", "True", "true"}:
            continue
        caller = (row.get("caller") or "").strip()
        callee = (row.get("callee") or "").strip()
        if not caller or not callee:
            continue
        outgoing[caller].add(callee)
        incoming[callee].add(caller)
    return outgoing, incoming


def _has_direct_call(
    left_callables: Set[str],
    right_callables: Set[str],
    outgoing: Dict[str, Set[str]],
) -> bool:
    for caller in left_callables:
        if outgoing.get(caller, set()) & right_callables:
            return True
    for caller in right_callables:
        if outgoing.get(caller, set()) & left_callables:
            return True
    return False


def _has_common_parent(
    left_callables: Set[str],
    right_callables: Set[str],
    incoming: Dict[str, Set[str]],
) -> bool:
    for left in left_callables:
        for right in right_callables:
            if incoming.get(left, set()) & incoming.get(right, set()):
                return True
    return False


def _similarity(left_fields: Set[str], right_fields: Set[str]) -> Tuple[float, float]:
    if not left_fields or not right_fields:
        return 0.0, 0.0
    overlap = len(left_fields & right_fields)
    union = len(left_fields | right_fields)
    jaccard = overlap / union if union else 0.0
    overlap_coeff = overlap / min(len(left_fields), len(right_fields)) if min(len(left_fields), len(right_fields)) else 0.0
    return jaccard, overlap_coeff


def _classify_family(candidate: CandidateContainer) -> str:
    if candidate.pyright_family:
        return candidate.pyright_family
    return FAMILY_UNKNOWN


def infer_shared_containers_from_payload(
    data_access_payload: dict,
    call_graph_edges: Sequence[dict],
    pyright_families: Optional[Dict[str, str]] = None,
) -> Tuple[dict, List[PairDecision]]:
    pyright_families = pyright_families or {}
    candidates = _build_candidates(data_access_payload)
    outgoing, incoming = _build_call_graph_views(call_graph_edges)

    for candidate in candidates.values():
        candidate.pyright_family = pyright_families.get(candidate.container_id, "")

    grouped: Dict[Tuple[str, str], List[CandidateContainer]] = defaultdict(list)
    for candidate in candidates.values():
        grouped[(candidate.object_kind, candidate.normalized_leaf)].append(candidate)

    decisions: List[PairDecision] = []
    inferred_config = {"df_col": {}, "dict_key": {}}

    for (object_kind, normalized_leaf), members in sorted(grouped.items()):
        members.sort(key=lambda item: (item.leaf_name.lower(), item.container_id))
        if len(members) < 2:
            continue

        leaf_accepted = True
        for idx, left in enumerate(members):
            for right in members[idx + 1 :]:
                jaccard, overlap_coeff = _similarity(left.fields, right.fields)
                shared_lineage = bool(
                    (set(left.roots) | {left.container_id})
                    & (set(right.roots) | {right.container_id})
                )
                direct_call = _has_direct_call(left.callables, right.callables, outgoing)
                common_parent = _has_common_parent(left.callables, right.callables, incoming)
                reasons: List[str] = []
                accepted = True

                if left.normalized_leaf != right.normalized_leaf:
                    accepted = False
                    reasons.append("name_mismatch")
                if len(left.fields) < 2 or len(right.fields) < 2:
                    accepted = False
                    reasons.append("insufficient_fields")
                if jaccard < 0.6 and overlap_coeff < 0.8:
                    accepted = False
                    reasons.append("insufficient_overlap")
                if not (shared_lineage or direct_call or common_parent):
                    accepted = False
                    reasons.append("no_identity_support")
                if normalized_leaf in GENERIC_CONTAINER_NAMES and not shared_lineage:
                    accepted = False
                    reasons.append("generic_name_without_lineage")
                left_family = _classify_family(left)
                right_family = _classify_family(right)
                if left_family == FAMILY_UNKNOWN or right_family == FAMILY_UNKNOWN:
                    accepted = False
                    reasons.append("pyright_unknown")
                if left_family != FAMILY_UNKNOWN and left_family != left.kind:
                    accepted = False
                    reasons.append("pyright_field_kind_conflict")
                if right_family != FAMILY_UNKNOWN and right_family != right.kind:
                    accepted = False
                    reasons.append("pyright_field_kind_conflict")
                if left_family and right_family and left_family != right_family:
                    accepted = False
                    reasons.append("pyright_conflict")

                leaf_accepted = leaf_accepted and accepted
                decisions.append(
                    PairDecision(
                        kind=object_kind,
                        normalized_leaf=normalized_leaf,
                        left=left.container_id,
                        right=right.container_id,
                        accepted=accepted,
                        reasons=reasons,
                        jaccard=jaccard,
                        overlap=overlap_coeff,
                        shared_lineage=shared_lineage,
                        direct_call=direct_call,
                        common_parent=common_parent,
                    )
                )

        if not leaf_accepted:
            continue

        canonical_leaf = sorted(
            {candidate.leaf_name for candidate in members},
            key=lambda value: (value.lower(), value),
        )[0]
        for candidate in members:
            inferred_config[object_kind][candidate.leaf_name] = canonical_leaf

    return inferred_config, decisions


def probe_pyright_families(
    project_root: Path,
    source_root: Path,
    candidates: Sequence[CandidateContainer],
    pyright_bin: str,
) -> Dict[str, str]:
    targets: List[PyrightProbeTarget] = []
    for candidate in candidates:
        if not candidate.file or candidate.container_id.startswith(("object_state:", "class_state:", "module_global:")):
            continue
        file_path = Path(candidate.file).resolve()
        try:
            module = path_to_module(file_path, source_root.resolve())
        except Exception:
            try:
                module = path_to_module(file_path, (project_root / "src").resolve())
            except Exception:
                continue
        if candidate.container_id.startswith("param:"):
            callable_id, _, variable_name = candidate.container_id.removeprefix("param:").rpartition(":")
            targets.append(
                PyrightProbeTarget(
                    target_id=candidate.container_id,
                    expression=variable_name,
                    file=file_path,
                    module=module,
                    mode="callable_entry",
                    callable_id=callable_id,
                    lineno=candidate.lineno,
                )
            )
        elif candidate.container_id.startswith("local_exposed:"):
            _callable_id, _, variable_name = candidate.container_id.removeprefix("local_exposed:").rpartition(":")
            targets.append(
                PyrightProbeTarget(
                    target_id=candidate.container_id,
                    expression=variable_name,
                    file=file_path,
                    module=module,
                    mode="after_line",
                    lineno=candidate.lineno,
                )
            )
    # ``probe_pyright_targets`` returns a report, not a mapping: the counts are
    # what tell a failed probe apart from a correctly-typed unknown. Only the
    # families are wanted here, as in ``generate_data_access_ast``.
    return probe_pyright_targets(project_root, targets, pyright_bin).families


def render_report(
    out_path: Path,
    inferred_config: dict,
    pair_decisions: Sequence[PairDecision],
) -> None:
    lines = [
        "# Inferred Shared Containers",
        "",
        "This report summarizes which same-name raw containers were promoted into",
        "a draft shared-containers config and which pairs were rejected.",
        "",
        "## Accepted Leaves",
        "",
    ]

    accepted_items = [
        (kind, source, canonical)
        for kind, mapping in sorted(inferred_config.items())
        for source, canonical in sorted(mapping.items())
    ]
    if accepted_items:
        for kind, source, canonical in accepted_items:
            lines.append(f"- `{kind}`: `{source}` -> `{canonical}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Pair Decisions", ""])
    if not pair_decisions:
        lines.append("- None")
    else:
        for decision in pair_decisions:
            status = "accepted" if decision.accepted else "rejected"
            reasons = ", ".join(decision.reasons) if decision.reasons else "(none)"
            lines.extend(
                [
                    f"### {decision.kind}:{decision.normalized_leaf}",
                    "",
                    f"- Pair: `{decision.left}` <-> `{decision.right}`",
                    f"- Status: `{status}`",
                    f"- Reasons: {reasons}",
                    f"- Similarity: `jaccard={decision.jaccard:.3f}`, `overlap={decision.overlap:.3f}`",
                    f"- Identity support: `shared_lineage={decision.shared_lineage}`, `direct_call={decision.direct_call}`, `common_parent={decision.common_parent}`",
                    "",
                ]
            )
    write_markdown(out_path, lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer a draft shared-containers config")
    parser.add_argument("--config", default=None, type=Path, help="Optional extraction JSON/JSONC config")
    parser.add_argument("--data-access", default=None, help="Path to data_access.json from the raw extractor run")
    parser.add_argument("--call-graph-edges", default=None, help="Path to call-graph edges.csv")
    parser.add_argument("--root", default=None, help="Source root used for Pyright probing (e.g. src/shop)")
    parser.add_argument("--pyright-bin", default=None, help="Pyright binary or command name")
    parser.add_argument("--out", default=None, help="Path to write inferred shared-containers JSON")
    parser.add_argument("--report", default=None, help="Path to write Markdown review report")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.config:
        config = load_extraction_config(args.config)
        data_access_path = Path(args.data_access).resolve() if args.data_access else config.data_access.outdir / "data_access.json"
        call_graph_edges_path = (
            Path(args.call_graph_edges).resolve()
            if args.call_graph_edges
            else config.call_graph.outdir / "edges.csv"
        )
        source_root = Path(args.root).resolve() if args.root else config.source_roots[0].path
        project_root = config.project_root
        pyright_bin = args.pyright_bin or config.data_access.pyright.bin
        out_path = Path(args.out).resolve() if args.out else config.data_access.inferred_shared_containers_config
        report_path = Path(args.report).resolve() if args.report else config.data_access.inferred_shared_containers_report
    else:
        if not args.data_access or not args.call_graph_edges or not args.root or not args.out or not args.report:
            raise SystemExit("--data-access, --call-graph-edges, --root, --out, and --report are required unless --config is provided")
        data_access_path = Path(args.data_access).resolve()
        call_graph_edges_path = Path(args.call_graph_edges).resolve()
        source_root = Path(args.root).resolve()
        project_root = discover_project_root(source_root)
        pyright_bin = args.pyright_bin or "pyright"
        out_path = Path(args.out).resolve()
        report_path = Path(args.report).resolve()

    data_access_payload = _load_json(data_access_path)
    call_graph_edges = _load_call_graph_edges(call_graph_edges_path)
    candidates = list(_build_candidates(data_access_payload).values())
    if args.config and not config.data_access.pyright.enabled:
        pyright_families = {}
    else:
        try:
            pyright_families = probe_pyright_families(project_root, source_root, candidates, pyright_bin)
        except RuntimeError as exc:
            if args.config and config.data_access.pyright.fail_policy == "fallback_unknown":
                print(f"Pyright failed; continuing with unknown families: {exc}", file=sys.stderr)
                pyright_families = {}
            else:
                raise
    inferred_config, pair_decisions = infer_shared_containers_from_payload(
        data_access_payload,
        call_graph_edges,
        pyright_families=pyright_families,
    )

    write_json(out_path, inferred_config)
    render_report(report_path, inferred_config, pair_decisions)

    accepted_count = sum(len(mapping) for mapping in inferred_config.values())
    print(f"Inferred shared-container config written to {out_path}")
    print(f"Accepted mappings: {accepted_count}")
    print(f"Pair decisions: {len(pair_decisions)}")


if __name__ == "__main__":
    main()
