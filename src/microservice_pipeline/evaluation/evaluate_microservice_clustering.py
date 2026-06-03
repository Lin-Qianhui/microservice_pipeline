#!/usr/bin/env python3
"""Evaluate clustering output against manual microservice labels.

The evaluator treats clustering as a partition-matching problem: cluster IDs are
arbitrary, and the number of predicted clusters does not need to match the
number of manual microservices.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Container, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown
    from microservice_pipeline.jsonc_config import load_jsonc
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown  # type: ignore
    from jsonc_config import load_jsonc  # type: ignore


DEFAULT_MANUAL = Path("artifacts/manual_results_mapping/manual_results_mapping.csv")
DEFAULT_CLUSTERS = Path("artifacts/structural_microservice_candidates/cluster_assignments.csv")
DEFAULT_OUTDIR = Path("artifacts/microservice_clustering_evaluation")
NA_DEFAULTS = ("", "NA", "N/A", "nan", "None")
DEFAULT_EVALUATION_NODE_TYPES = ("callable",)
DEFAULT_EVALUATION_KIND_TOKENS = ("class_attr_state",)


@dataclass(frozen=True)
class JoinedAssignment:
    node: str
    normalized_node: str
    manual_label: str
    cluster_id: str
    node_type: str = ""
    label: str = ""
    kind: str = ""
    module: str = ""


@dataclass(frozen=True)
class EvaluationInput:
    joined: list[JoinedAssignment]
    manual_rows: list[dict[str, str]]
    cluster_rows: list[dict[str, str]]
    raw_manual_row_count: int
    raw_cluster_row_count: int
    manual_label_column: str
    node_mode: str
    evaluation_node_types: tuple[str, ...]
    evaluation_kind_tokens: tuple[str, ...]
    all_evaluation_nodes: bool
    unmatched_manual_keys: list[str]
    unmatched_cluster_keys: list[str]


@dataclass(frozen=True)
class EvaluationComputation:
    payload: dict[str, Any]
    known_assignments: list[JoinedAssignment]
    primary_best_matches: list[dict[str, Any]]
    all_assignments_metrics: dict[str, Any]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _comb2(value: int) -> int:
    return value * (value - 1) // 2


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def parse_evaluation_tokens(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = value
    return tuple(_text(part).lower() for part in parts if _text(part))


def _kind_tokens(row: Mapping[str, Any]) -> set[str]:
    return set(parse_evaluation_tokens(_text(row.get("kind")).replace(";", ",")))


def _row_node_type(row: Mapping[str, Any]) -> str:
    node_type = _text(row.get("node_type")).lower()
    if node_type:
        return node_type

    node = _text(row.get("node"))
    if node.startswith("callable:"):
        return "callable"
    if node.startswith("data:"):
        return "data"

    kind_tokens = _kind_tokens(row)
    if kind_tokens & {"function", "method", "module"}:
        return "callable"
    return ""


def _filter_evaluation_rows(
    rows: Sequence[dict[str, str]],
    node_types: tuple[str, ...],
    kind_tokens: tuple[str, ...],
    all_nodes: bool,
) -> list[dict[str, str]]:
    if all_nodes:
        return list(rows)

    allowed_node_types = set(node_types)
    allowed_kind_tokens = set(kind_tokens)
    return [
        row
        for row in rows
        if _row_node_type(row) in allowed_node_types
        or bool(_kind_tokens(row) & allowed_kind_tokens)
    ]


def _detect_manual_label_column(fieldnames: Sequence[str]) -> str:
    candidates = {
        "microservice_id",
        "manual_microservice_id",
        "manual_label",
        "service_id",
    }
    for fieldname in fieldnames:
        cleaned = fieldname.lstrip("\ufeff").strip().lower()
        if cleaned in candidates:
            return fieldname
    raise ValueError(
        "Could not find the manual microservice label column. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _normalize_node(row: Mapping[str, str], mode: str, source: str) -> str | None:
    node = _text(row.get("node"))
    if not node:
        return None

    if mode == "exact":
        return node

    if mode != "callable":
        raise ValueError(f"Unsupported node normalization mode: {mode}")

    node_type = _text(row.get("node_type")).lower()
    if source == "manual" and node_type and node_type != "callable":
        return None
    if node.startswith("data:"):
        return None
    if node.startswith("callable:"):
        return node.removeprefix("callable:")
    return node


def _index_by_normalized_node(
    rows: Iterable[dict[str, str]],
    mode: str,
    source: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = _normalize_node(row, mode, source)
        if key is None:
            continue
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = row
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate normalized {source} node keys: {preview}")
    return indexed


def _choose_node_mode(
    manual_rows: Sequence[dict[str, str]],
    cluster_rows: Sequence[dict[str, str]],
) -> str:
    exact_manual = set(_index_by_normalized_node(manual_rows, "exact", "manual"))
    exact_clusters = set(_index_by_normalized_node(cluster_rows, "exact", "cluster"))
    callable_manual = set(_index_by_normalized_node(manual_rows, "callable", "manual"))
    callable_clusters = set(_index_by_normalized_node(cluster_rows, "callable", "cluster"))

    exact_overlap = len(exact_manual & exact_clusters)
    callable_overlap = len(callable_manual & callable_clusters)
    if exact_overlap >= callable_overlap and exact_overlap > 0:
        return "exact"
    if callable_overlap > 0:
        return "callable"
    return "exact"


def load_evaluation_input(
    manual_path: Path,
    clusters_path: Path,
    manual_label_column: str | None = None,
    node_mode: str = "auto",
    evaluation_node_types: Sequence[str] = DEFAULT_EVALUATION_NODE_TYPES,
    evaluation_kind_tokens: Sequence[str] = DEFAULT_EVALUATION_KIND_TOKENS,
    all_evaluation_nodes: bool = False,
) -> EvaluationInput:
    manual_rows, manual_fields = _read_csv(manual_path)
    cluster_rows, cluster_fields = _read_csv(clusters_path)
    return build_evaluation_input_from_rows(
        manual_rows=manual_rows,
        manual_fields=manual_fields,
        cluster_rows=cluster_rows,
        cluster_fields=cluster_fields,
        manual_label_column=manual_label_column,
        node_mode=node_mode,
        evaluation_node_types=evaluation_node_types,
        evaluation_kind_tokens=evaluation_kind_tokens,
        all_evaluation_nodes=all_evaluation_nodes,
        manual_source=str(manual_path),
        cluster_source=str(clusters_path),
    )


def build_evaluation_input_from_rows(
    manual_rows: Sequence[Mapping[str, Any]],
    manual_fields: Sequence[str],
    cluster_rows: Sequence[Mapping[str, Any]],
    cluster_fields: Sequence[str],
    manual_label_column: str | None = None,
    node_mode: str = "auto",
    evaluation_node_types: Sequence[str] = DEFAULT_EVALUATION_NODE_TYPES,
    evaluation_kind_tokens: Sequence[str] = DEFAULT_EVALUATION_KIND_TOKENS,
    all_evaluation_nodes: bool = False,
    manual_source: str = "manual rows",
    cluster_source: str = "cluster rows",
) -> EvaluationInput:
    if not manual_rows:
        raise ValueError(f"Manual mapping CSV has no rows: {manual_source}")
    if not cluster_rows:
        raise ValueError(f"Cluster assignments CSV has no rows: {cluster_source}")
    if "node" not in manual_fields:
        raise ValueError(f"Manual mapping CSV is missing required column 'node': {manual_source}")
    if "node" not in cluster_fields:
        raise ValueError(f"Cluster assignments CSV is missing required column 'node': {cluster_source}")
    if "cluster_id" not in cluster_fields:
        raise ValueError(f"Cluster assignments CSV is missing required column 'cluster_id': {cluster_source}")

    raw_manual_row_dicts = [dict(row) for row in manual_rows]
    raw_cluster_row_dicts = [dict(row) for row in cluster_rows]
    resolved_node_types = parse_evaluation_tokens(evaluation_node_types)
    resolved_kind_tokens = parse_evaluation_tokens(evaluation_kind_tokens)
    manual_row_dicts = _filter_evaluation_rows(
        raw_manual_row_dicts,
        resolved_node_types,
        resolved_kind_tokens,
        all_evaluation_nodes,
    )
    cluster_row_dicts = _filter_evaluation_rows(
        raw_cluster_row_dicts,
        resolved_node_types,
        resolved_kind_tokens,
        all_evaluation_nodes,
    )
    label_column = manual_label_column or _detect_manual_label_column(manual_fields)
    if label_column not in manual_fields:
        raise ValueError(f"Manual mapping CSV is missing label column '{label_column}'")

    resolved_mode = (
        _choose_node_mode(manual_row_dicts, cluster_row_dicts)
        if node_mode == "auto"
        else node_mode
    )
    manual_by_key = _index_by_normalized_node(manual_row_dicts, resolved_mode, "manual")
    cluster_by_key = _index_by_normalized_node(cluster_row_dicts, resolved_mode, "cluster")
    joined_keys = sorted(set(manual_by_key) & set(cluster_by_key))

    joined: list[JoinedAssignment] = []
    for key in joined_keys:
        manual = manual_by_key[key]
        cluster = cluster_by_key[key]
        joined.append(
            JoinedAssignment(
                node=_text(manual.get("node")),
                normalized_node=key,
                manual_label=_text(manual.get(label_column)),
                cluster_id=_text(cluster.get("cluster_id")),
                node_type=_text(manual.get("node_type") or cluster.get("node_type")),
                label=_text(manual.get("label") or cluster.get("label") or cluster.get("qualname")),
                kind=_text(manual.get("kind") or cluster.get("kind")),
                module=_text(manual.get("module") or cluster.get("module")),
            )
        )

    return EvaluationInput(
        joined=joined,
        manual_rows=manual_row_dicts,
        cluster_rows=cluster_row_dicts,
        raw_manual_row_count=len(raw_manual_row_dicts),
        raw_cluster_row_count=len(raw_cluster_row_dicts),
        manual_label_column=label_column,
        node_mode=resolved_mode,
        evaluation_node_types=resolved_node_types,
        evaluation_kind_tokens=resolved_kind_tokens,
        all_evaluation_nodes=all_evaluation_nodes,
        unmatched_manual_keys=sorted(set(manual_by_key) - set(cluster_by_key)),
        unmatched_cluster_keys=sorted(set(cluster_by_key) - set(manual_by_key)),
    )


def evaluate_assignment_rows(
    manual_rows: Sequence[Mapping[str, Any]],
    manual_fields: Sequence[str],
    cluster_rows: Sequence[Mapping[str, Any]],
    cluster_fields: Sequence[str],
    manual_label_column: str | None = None,
    node_mode: str = "auto",
    na_labels: Container[str] = NA_DEFAULTS,
    evaluation_node_types: Sequence[str] = DEFAULT_EVALUATION_NODE_TYPES,
    evaluation_kind_tokens: Sequence[str] = DEFAULT_EVALUATION_KIND_TOKENS,
    all_evaluation_nodes: bool = False,
) -> dict[str, Any]:
    evaluation = build_evaluation_input_from_rows(
        manual_rows=manual_rows,
        manual_fields=manual_fields,
        cluster_rows=cluster_rows,
        cluster_fields=cluster_fields,
        manual_label_column=manual_label_column,
        node_mode=node_mode,
        evaluation_node_types=evaluation_node_types,
        evaluation_kind_tokens=evaluation_kind_tokens,
        all_evaluation_nodes=all_evaluation_nodes,
    )
    return build_evaluation_payload(
        evaluation,
        na_labels=na_labels,
        include_sensitivity=False,
    )


def _contingency(assignments: Sequence[JoinedAssignment]) -> Counter[tuple[str, str]]:
    return Counter((item.manual_label, item.cluster_id) for item in assignments)


def _labels(assignments: Sequence[JoinedAssignment]) -> tuple[Counter[str], Counter[str]]:
    manual = Counter(item.manual_label for item in assignments)
    predicted = Counter(item.cluster_id for item in assignments)
    return manual, predicted


def _max_weight_matching_sum(weights: Sequence[Sequence[int]]) -> int:
    if not weights or not weights[0]:
        return 0

    matrix = [list(row) for row in weights]
    if len(matrix) > len(matrix[0]):
        matrix = [list(col) for col in zip(*matrix)]

    n = len(matrix)
    m = len(matrix[0])
    max_weight = max(max(row) for row in matrix)
    costs = [[max_weight - value for value in row] for row in matrix]

    # Hungarian algorithm for rectangular minimization with n <= m.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: dict[int, int] = {}
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return sum(matrix[row][col] for row, col in assignment.items())


def compute_metrics(assignments: Sequence[JoinedAssignment]) -> dict[str, float | int]:
    n = len(assignments)
    if n == 0:
        return {
            "n": 0,
            "manual_cluster_count": 0,
            "predicted_cluster_count": 0,
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "pairwise_f1": 0.0,
            "adjusted_rand_index": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "nmi": 0.0,
            "purity": 0.0,
            "inverse_purity": 0.0,
            "purity_f1": 0.0,
            "macro_purity_precision": 0.0,
            "macro_purity_recall": 0.0,
            "macro_purity_f1": 0.0,
            "predicted_match_precision": 0.0,
            "predicted_match_recall": 0.0,
            "predicted_match_f1": 0.0,
            "predicted_match_pair_macro_f1": 0.0,
            "bcubed_precision": 0.0,
            "bcubed_recall": 0.0,
            "bcubed_f1": 0.0,
            "hungarian_accuracy": 0.0,
        }

    contingency = _contingency(assignments)
    manual_counts, predicted_counts = _labels(assignments)

    true_positive_pairs = sum(_comb2(value) for value in contingency.values())
    manual_pairs = sum(_comb2(value) for value in manual_counts.values())
    predicted_pairs = sum(_comb2(value) for value in predicted_counts.values())
    false_positive_pairs = predicted_pairs - true_positive_pairs
    false_negative_pairs = manual_pairs - true_positive_pairs

    pairwise_precision = _safe_ratio(
        true_positive_pairs,
        true_positive_pairs + false_positive_pairs,
        default=1.0,
    )
    pairwise_recall = _safe_ratio(
        true_positive_pairs,
        true_positive_pairs + false_negative_pairs,
        default=1.0,
    )
    pairwise_f1 = _harmonic_mean(pairwise_precision, pairwise_recall)

    total_pairs = _comb2(n)
    expected_index = _safe_ratio(manual_pairs * predicted_pairs, total_pairs, default=0.0)
    max_index = 0.5 * (manual_pairs + predicted_pairs)
    denominator = max_index - expected_index
    if denominator == 0:
        # Both partitions are trivial in the same way (all-in-one or all-singletons
        # on both sides) -> perfect agreement. Otherwise one side is trivial while
        # the other has structure -> no agreement beyond chance.
        both_all_in_one = (
            manual_pairs == total_pairs and predicted_pairs == total_pairs
        )
        both_all_singletons = manual_pairs == 0 and predicted_pairs == 0
        adjusted_rand = 1.0 if (both_all_in_one or both_all_singletons) else 0.0
    else:
        adjusted_rand = (true_positive_pairs - expected_index) / denominator

    manual_entropy = -sum((count / n) * math.log(count / n) for count in manual_counts.values())
    predicted_entropy = -sum((count / n) * math.log(count / n) for count in predicted_counts.values())
    mutual_information = sum(
        (count / n) * math.log((count * n) / (manual_counts[manual] * predicted_counts[predicted]))
        for (manual, predicted), count in contingency.items()
        if count
    )
    homogeneity = _safe_ratio(mutual_information, manual_entropy, default=1.0)
    completeness = _safe_ratio(mutual_information, predicted_entropy, default=1.0)
    v_measure = _harmonic_mean(homogeneity, completeness)
    nmi = _safe_ratio(2.0 * mutual_information, manual_entropy + predicted_entropy, default=1.0)

    purity = sum(
        max(contingency[(manual, predicted)] for manual in manual_counts)
        for predicted in predicted_counts
    ) / n
    inverse_purity = sum(
        max(contingency[(manual, predicted)] for predicted in predicted_counts)
        for manual in manual_counts
    ) / n
    purity_f1 = _harmonic_mean(purity, inverse_purity)
    macro_purity_precision = sum(
        max(contingency[(manual, predicted)] for manual in manual_counts)
        / predicted_counts[predicted]
        for predicted in predicted_counts
    ) / len(predicted_counts)
    macro_purity_recall = sum(
        max(contingency[(manual, predicted)] for predicted in predicted_counts)
        / manual_counts[manual]
        for manual in manual_counts
    ) / len(manual_counts)
    macro_purity_f1 = _harmonic_mean(macro_purity_precision, macro_purity_recall)
    predicted_match_precisions: list[float] = []
    predicted_match_recalls: list[float] = []
    predicted_match_pair_f1s: list[float] = []
    for predicted in predicted_counts:
        best_manual = max(
            manual_counts,
            key=lambda manual: (
                contingency[(manual, predicted)],
                _safe_ratio(contingency[(manual, predicted)], manual_counts[manual]),
                manual,
            ),
        )
        intersection = contingency[(best_manual, predicted)]
        precision = intersection / predicted_counts[predicted]
        recall = intersection / manual_counts[best_manual]
        predicted_match_precisions.append(precision)
        predicted_match_recalls.append(recall)
        predicted_match_pair_f1s.append(_harmonic_mean(precision, recall))
    predicted_match_precision = sum(predicted_match_precisions) / len(predicted_match_precisions)
    predicted_match_recall = sum(predicted_match_recalls) / len(predicted_match_recalls)
    predicted_match_f1 = _harmonic_mean(predicted_match_precision, predicted_match_recall)
    predicted_match_pair_macro_f1 = sum(predicted_match_pair_f1s) / len(predicted_match_pair_f1s)

    bcubed_precision = sum(
        contingency[(item.manual_label, item.cluster_id)] / predicted_counts[item.cluster_id]
        for item in assignments
    ) / n
    bcubed_recall = sum(
        contingency[(item.manual_label, item.cluster_id)] / manual_counts[item.manual_label]
        for item in assignments
    ) / n
    bcubed_f1 = _harmonic_mean(bcubed_precision, bcubed_recall)

    manual_labels = sorted(manual_counts)
    predicted_labels = sorted(predicted_counts)
    matching_matrix = [
        [contingency[(manual, predicted)] for predicted in predicted_labels]
        for manual in manual_labels
    ]
    hungarian_accuracy = _max_weight_matching_sum(matching_matrix) / n

    return {
        "n": n,
        "manual_cluster_count": len(manual_counts),
        "predicted_cluster_count": len(predicted_counts),
        "pairwise_precision": pairwise_precision,
        "pairwise_recall": pairwise_recall,
        "pairwise_f1": pairwise_f1,
        "adjusted_rand_index": adjusted_rand,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "nmi": nmi,
        "purity": purity,
        "inverse_purity": inverse_purity,
        "purity_f1": purity_f1,
        "macro_purity_precision": macro_purity_precision,
        "macro_purity_recall": macro_purity_recall,
        "macro_purity_f1": macro_purity_f1,
        "predicted_match_precision": predicted_match_precision,
        "predicted_match_recall": predicted_match_recall,
        "predicted_match_f1": predicted_match_f1,
        "predicted_match_pair_macro_f1": predicted_match_pair_macro_f1,
        "bcubed_precision": bcubed_precision,
        "bcubed_recall": bcubed_recall,
        "bcubed_f1": bcubed_f1,
        "hungarian_accuracy": hungarian_accuracy,
    }


def filter_known_labels(
    assignments: Sequence[JoinedAssignment],
    na_labels: Container[str],
) -> list[JoinedAssignment]:
    return [item for item in assignments if item.manual_label not in na_labels]


def build_metadata(
    evaluation: EvaluationInput,
    known_assignments: Sequence[JoinedAssignment],
) -> dict[str, Any]:
    return {
        "manual_rows": evaluation.raw_manual_row_count,
        "cluster_rows": evaluation.raw_cluster_row_count,
        "scoped_manual_rows": len(evaluation.manual_rows),
        "scoped_cluster_rows": len(evaluation.cluster_rows),
        "joined_rows": len(evaluation.joined),
        "known_joined_rows": len(known_assignments),
        "na_joined_rows": len(evaluation.joined) - len(known_assignments),
        "known_coverage_of_joined": _safe_ratio(len(known_assignments), len(evaluation.joined)),
        "manual_label_column": evaluation.manual_label_column.lstrip("\ufeff"),
        "node_mode": evaluation.node_mode,
        "all_evaluation_nodes": evaluation.all_evaluation_nodes,
        "evaluation_node_types": list(evaluation.evaluation_node_types),
        "evaluation_kind_tokens": list(evaluation.evaluation_kind_tokens),
        "unmatched_manual_rows": len(evaluation.unmatched_manual_keys),
        "unmatched_cluster_rows": len(evaluation.unmatched_cluster_keys),
    }


def build_contingency_rows(assignments: Sequence[JoinedAssignment]) -> tuple[list[str], list[dict[str, Any]]]:
    contingency = _contingency(assignments)
    manual_counts, predicted_counts = _labels(assignments)
    cluster_ids = sorted(predicted_counts)
    rows: list[dict[str, Any]] = []
    for manual_label in sorted(manual_counts):
        row: dict[str, Any] = {
            "manual_microservice_id": manual_label,
            "manual_size": manual_counts[manual_label],
        }
        for cluster_id in cluster_ids:
            row[cluster_id] = contingency[(manual_label, cluster_id)]
        rows.append(row)
    return ["manual_microservice_id", "manual_size", *cluster_ids], rows


def build_best_match_rows(assignments: Sequence[JoinedAssignment]) -> list[dict[str, Any]]:
    contingency = _contingency(assignments)
    manual_counts, predicted_counts = _labels(assignments)
    rows: list[dict[str, Any]] = []
    for manual_label in sorted(manual_counts):
        best: dict[str, Any] | None = None
        for cluster_id in sorted(predicted_counts):
            intersection = contingency[(manual_label, cluster_id)]
            if intersection == 0:
                continue
            precision = intersection / predicted_counts[cluster_id]
            recall = intersection / manual_counts[manual_label]
            f1 = _harmonic_mean(precision, recall)
            jaccard = intersection / (
                manual_counts[manual_label] + predicted_counts[cluster_id] - intersection
            )
            candidate = {
                "manual_microservice_id": manual_label,
                "best_cluster_id": cluster_id,
                "intersection": intersection,
                "manual_size": manual_counts[manual_label],
                "cluster_size_evaluated": predicted_counts[cluster_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": jaccard,
            }
            if best is None or (
                candidate["f1"],
                candidate["jaccard"],
                candidate["intersection"],
            ) > (
                best["f1"],
                best["jaccard"],
                best["intersection"],
            ):
                best = candidate
        if best is not None:
            rows.append(best)
    return rows


def summarize_best_matches(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "best_match_macro_f1": 0.0,
            "best_match_weighted_f1": 0.0,
            "best_match_macro_jaccard": 0.0,
            "best_match_weighted_jaccard": 0.0,
        }
    total_size = sum(int(row["manual_size"]) for row in rows)
    return {
        "best_match_macro_f1": sum(float(row["f1"]) for row in rows) / len(rows),
        "best_match_weighted_f1": sum(float(row["f1"]) * int(row["manual_size"]) for row in rows) / total_size,
        "best_match_macro_jaccard": sum(float(row["jaccard"]) for row in rows) / len(rows),
        "best_match_weighted_jaccard": (
            sum(float(row["jaccard"]) * int(row["manual_size"]) for row in rows) / total_size
        ),
    }


def build_na_review_rows(
    assignments: Sequence[JoinedAssignment],
    known_assignments: Sequence[JoinedAssignment],
    na_labels: Container[str],
) -> list[dict[str, Any]]:
    all_cluster_counts = Counter(item.cluster_id for item in assignments)
    na_cluster_counts = Counter(item.cluster_id for item in assignments if item.manual_label in na_labels)
    known_by_cluster: dict[str, Counter[str]] = defaultdict(Counter)
    for item in known_assignments:
        known_by_cluster[item.cluster_id][item.manual_label] += 1

    rows: list[dict[str, Any]] = []
    for cluster_id, na_count in sorted(na_cluster_counts.items(), key=lambda item: (-item[1], item[0])):
        known_counter = known_by_cluster.get(cluster_id, Counter())
        dominant_known_label = ""
        dominant_known_count = 0
        if known_counter:
            dominant_known_label, dominant_known_count = sorted(
                known_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
        rows.append(
            {
                "cluster_id": cluster_id,
                "na_count": na_count,
                "cluster_size_all_joined": all_cluster_counts[cluster_id],
                "na_share": na_count / all_cluster_counts[cluster_id],
                "known_count": all_cluster_counts[cluster_id] - na_count,
                "dominant_known_microservice": dominant_known_label,
                "dominant_known_count": dominant_known_count,
                "known_label_distribution": ";".join(
                    f"{label}:{count}" for label, count in sorted(known_counter.items())
                ),
            }
        )
    return rows


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    formatted_rows = [
        {key: _format_value(row.get(key, "")) for key in fieldnames}
        for row in rows
    ]
    write_csv_rows(path, fieldnames, formatted_rows)


def _metrics_rows(scenario: str, metrics: Mapping[str, float | int]) -> list[dict[str, Any]]:
    return [
        {
            "scenario": scenario,
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


def compute_evaluation(
    evaluation: EvaluationInput,
    na_labels: Container[str],
    include_sensitivity: bool = True,
) -> EvaluationComputation:
    """Compute reusable evaluation payload plus detailed rows for writers."""
    known_assignments = filter_known_labels(evaluation.joined, na_labels)
    primary_metrics = compute_metrics(known_assignments)
    primary_best_matches = build_best_match_rows(known_assignments)
    primary_best_summary = summarize_best_matches(primary_best_matches)
    primary_metrics = {**primary_metrics, **primary_best_summary}

    all_assignments_metrics = compute_metrics(evaluation.joined)
    all_best_matches = build_best_match_rows(evaluation.joined)
    all_assignments_metrics = {
        **all_assignments_metrics,
        **summarize_best_matches(all_best_matches),
    }
    na_review_rows = build_na_review_rows(evaluation.joined, known_assignments, na_labels)
    payload = {
        "metadata": build_metadata(evaluation, known_assignments),
        "primary_exclude_na": primary_metrics,
        "sensitivity_treat_na_as_class": (
            all_assignments_metrics if include_sensitivity else None
        ),
        "na_cluster_review": na_review_rows,
    }
    return EvaluationComputation(
        payload=payload,
        known_assignments=known_assignments,
        primary_best_matches=primary_best_matches,
        all_assignments_metrics=all_assignments_metrics,
    )


def build_evaluation_payload(
    evaluation: EvaluationInput,
    na_labels: Container[str],
    include_sensitivity: bool = True,
) -> dict[str, Any]:
    return compute_evaluation(
        evaluation,
        na_labels=na_labels,
        include_sensitivity=include_sensitivity,
    ).payload


def evaluation_summary_row(
    payload: Mapping[str, Any],
    prefix: str = "evaluation_",
) -> dict[str, Any]:
    metadata = payload["metadata"]
    primary = payload["primary_exclude_na"]
    row: dict[str, Any] = {
        f"{prefix}joined_rows": metadata["joined_rows"],
        f"{prefix}known_joined_rows": metadata["known_joined_rows"],
        f"{prefix}known_coverage": metadata["known_coverage_of_joined"],
        f"{prefix}unmatched_manual_rows": metadata["unmatched_manual_rows"],
        f"{prefix}unmatched_cluster_rows": metadata["unmatched_cluster_rows"],
    }
    for metric, value in primary.items():
        row[f"{prefix}{metric}"] = value
    return row


def write_outputs(
    evaluation: EvaluationInput,
    outdir: Path,
    na_labels: Container[str],
    write_sensitivity: bool,
) -> dict[str, Any]:
    ensure_dir(outdir)

    computation = compute_evaluation(
        evaluation,
        na_labels=na_labels,
        include_sensitivity=write_sensitivity,
    )
    payload = computation.payload
    primary_metrics = payload["primary_exclude_na"]
    all_assignments_metrics = computation.all_assignments_metrics

    metric_rows = _metrics_rows("primary_exclude_na", primary_metrics)
    if write_sensitivity:
        metric_rows.extend(_metrics_rows("sensitivity_treat_na_as_class", all_assignments_metrics))
    _write_csv(outdir / "metrics_summary.csv", ["scenario", "metric", "value"], metric_rows)

    contingency_fields, contingency_rows = build_contingency_rows(computation.known_assignments)
    _write_csv(outdir / "contingency_table.csv", contingency_fields, contingency_rows)
    contingency_all_fields, contingency_all_rows = build_contingency_rows(evaluation.joined)
    _write_csv(outdir / "contingency_table_with_na.csv", contingency_all_fields, contingency_all_rows)

    best_match_fields = [
        "manual_microservice_id",
        "best_cluster_id",
        "intersection",
        "manual_size",
        "cluster_size_evaluated",
        "precision",
        "recall",
        "f1",
        "jaccard",
    ]
    _write_csv(
        outdir / "per_microservice_best_match.csv",
        best_match_fields,
        computation.primary_best_matches,
    )

    _write_csv(
        outdir / "na_cluster_review.csv",
        [
            "cluster_id",
            "na_count",
            "cluster_size_all_joined",
            "na_share",
            "known_count",
            "dominant_known_microservice",
            "dominant_known_count",
            "known_label_distribution",
        ],
        payload["na_cluster_review"],
    )

    joined_fields = [
        "node",
        "normalized_node",
        "manual_microservice_id",
        "cluster_id",
        "node_type",
        "label",
        "kind",
        "module",
    ]
    _write_csv(
        outdir / "joined_assignments.csv",
        joined_fields,
        [
            {
                "node": item.node,
                "normalized_node": item.normalized_node,
                "manual_microservice_id": item.manual_label,
                "cluster_id": item.cluster_id,
                "node_type": item.node_type,
                "label": item.label,
                "kind": item.kind,
                "module": item.module,
            }
            for item in evaluation.joined
        ],
    )

    write_json(outdir / "evaluation.json", payload)

    write_markdown_summary(
        outdir / "metrics_summary.md",
        payload,
        primary_metrics,
        all_assignments_metrics,
        write_sensitivity,
    )
    return payload


def write_markdown_summary(
    path: Path,
    payload: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
    sensitivity_metrics: Mapping[str, Any],
    write_sensitivity: bool,
) -> None:
    metadata = payload["metadata"]
    lines = [
        "# Microservice Clustering Evaluation",
        "",
        "## Coverage",
        "",
        f"- Manual rows: {metadata['manual_rows']}",
        f"- Cluster rows: {metadata['cluster_rows']}",
        f"- Scoped manual rows: {metadata['scoped_manual_rows']}",
        f"- Scoped cluster rows: {metadata['scoped_cluster_rows']}",
        f"- Joined rows: {metadata['joined_rows']}",
        f"- Known joined rows used for primary metrics: {metadata['known_joined_rows']}",
        f"- NA/unadjudicated joined rows: {metadata['na_joined_rows']}",
        f"- Known coverage of joined rows: {metadata['known_coverage_of_joined']:.3%}",
        f"- Node matching mode: `{metadata['node_mode']}`",
        f"- All evaluation nodes: `{metadata['all_evaluation_nodes']}`",
        f"- Evaluation node types: `{', '.join(metadata['evaluation_node_types'])}`",
        f"- Evaluation kind tokens: `{', '.join(metadata['evaluation_kind_tokens'])}`",
        f"- Unmatched manual rows: {metadata['unmatched_manual_rows']}",
        f"- Unmatched cluster rows: {metadata['unmatched_cluster_rows']}",
        "",
        "## Primary Metrics",
        "",
        "Rows with manual label `NA` are excluded from the primary metrics.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric in (
        "adjusted_rand_index",
        "v_measure",
        "homogeneity",
        "completeness",
        "nmi",
        "pairwise_precision",
        "pairwise_recall",
        "pairwise_f1",
        "bcubed_precision",
        "bcubed_recall",
        "bcubed_f1",
        "purity",
        "inverse_purity",
        "purity_f1",
        "macro_purity_precision",
        "macro_purity_recall",
        "macro_purity_f1",
        "predicted_match_precision",
        "predicted_match_recall",
        "predicted_match_f1",
        "predicted_match_pair_macro_f1",
        "hungarian_accuracy",
        "best_match_macro_f1",
        "best_match_weighted_f1",
        "best_match_macro_jaccard",
        "best_match_weighted_jaccard",
    ):
        lines.append(f"| `{metric}` | {_format_value(primary_metrics[metric])} |")

    if write_sensitivity:
        lines.extend(
            [
                "",
                "## Sensitivity",
                "",
                "This conservative sensitivity run treats `NA` as one ordinary manual class.",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for metric in (
            "adjusted_rand_index",
            "v_measure",
            "pairwise_precision",
            "pairwise_recall",
            "pairwise_f1",
            "macro_purity_precision",
            "macro_purity_recall",
            "macro_purity_f1",
            "predicted_match_precision",
            "predicted_match_recall",
            "predicted_match_f1",
            "predicted_match_pair_macro_f1",
            "bcubed_f1",
            "hungarian_accuracy",
        ):
            lines.append(f"| `{metric}` | {_format_value(sensitivity_metrics[metric])} |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `metrics_summary.csv`: primary and sensitivity metric values.",
            "- `contingency_table.csv`: manual microservice by predicted cluster counts, excluding `NA`.",
            "- `contingency_table_with_na.csv`: same table including `NA`.",
            "- `per_microservice_best_match.csv`: best predicted cluster for each manual microservice.",
            "- `na_cluster_review.csv`: where unadjudicated nodes landed.",
            "- `joined_assignments.csv`: joined manual and predicted labels by node.",
            "- `evaluation.json`: combined machine-readable payload.",
            "",
        ]
    )
    write_markdown(path, lines)


def _resolve_config_path(project_root: Path, value: Any) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    return value if isinstance(value, Mapping) else {}


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _evaluation_config_defaults(config_path: Path, project_root: Path) -> dict[str, Any]:
    payload = load_jsonc(config_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evaluation config must be a JSON object: {config_path}")

    defaults: dict[str, Any] = {}
    paths = _section(payload, "paths")
    mapping = _section(payload, "mapping")
    matching = _section(payload, "matching")
    scope = _section(payload, "scope")
    outputs = _section(payload, "outputs")

    for key in ("manual", "clusters", "outdir"):
        value = _resolve_config_path(project_root, paths.get(key))
        if value is not None:
            defaults[key] = value

    if "manual_label_column" in mapping:
        defaults["manual_label_column"] = mapping.get("manual_label_column")

    na_labels = _string_list(mapping.get("na_labels"))
    if na_labels:
        defaults["na_label"] = na_labels

    if matching.get("node_mode") is not None:
        defaults["node_mode"] = matching.get("node_mode")

    if scope.get("evaluation_node_types") is not None:
        defaults["evaluation_node_types"] = scope.get("evaluation_node_types")
    if scope.get("evaluation_kind_tokens") is not None:
        defaults["evaluation_kind_tokens"] = scope.get("evaluation_kind_tokens")
    if "all_evaluation_nodes" in scope:
        defaults["all_evaluation_nodes"] = _bool(scope.get("all_evaluation_nodes"), False)

    if "write_sensitivity" in outputs:
        defaults["write_sensitivity"] = _bool(outputs.get("write_sensitivity"), True)

    return defaults


def _preparse_project_and_config(argv: Sequence[str] | None) -> tuple[Path, Path | None, dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=None, type=Path)
    args, _remainder = parser.parse_known_args(argv)

    project_root = Path(args.project_root).resolve()
    config_path = args.config
    if config_path is not None and not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve() if config_path is not None else None

    defaults = _evaluation_config_defaults(config_path, project_root) if config_path else {}
    defaults["project_root"] = str(project_root)
    defaults["config"] = config_path
    return project_root, config_path, defaults


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    _project_root, _config_path, config_defaults = _preparse_project_and_config(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root used to resolve config-relative paths")
    parser.add_argument("--config", default=None, type=Path, help="Optional evaluation JSON/JSONC config")
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL, help="Manual mapping CSV.")
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS, help="Cluster assignments CSV.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for evaluation artifacts.")
    parser.add_argument("--manual-label-column", default=None, help="Manual microservice label column.")
    parser.add_argument(
        "--node-mode",
        choices=("auto", "exact", "callable"),
        default="auto",
        help=(
            "Node matching mode. 'auto' prefers exact joins, then callable-only joins "
            "with the callable: prefix stripped."
        ),
    )
    parser.add_argument(
        "--na-label",
        action="append",
        default=None,
        help="Manual label to treat as unknown/unadjudicated. Can be repeated.",
    )
    parser.add_argument(
        "--evaluation-node-types",
        default=",".join(DEFAULT_EVALUATION_NODE_TYPES),
        help=(
            "Comma-separated node_type values to include in evaluation. "
            "Use an empty value with --evaluation-kind-tokens for kind-only evaluation."
        ),
    )
    parser.add_argument(
        "--evaluation-kind-tokens",
        default=",".join(DEFAULT_EVALUATION_KIND_TOKENS),
        help=(
            "Comma-separated semicolon-tokenized kind values to include in evaluation. "
            "Use an empty value for node-type-only evaluation."
        ),
    )
    parser.add_argument(
        "--all-evaluation-nodes",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Evaluate all joined manual and cluster rows, preserving the previous unfiltered behavior.",
    )
    parser.add_argument(
        "--write-sensitivity",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Write the sensitivity run that treats NA as a normal class.",
    )
    parser.add_argument(
        "--no-sensitivity",
        dest="write_sensitivity",
        action="store_false",
        help="Do not write the sensitivity run that treats NA as a normal class.",
    )
    parser.set_defaults(**config_defaults)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    na_labels = set(args.na_label or NA_DEFAULTS)
    evaluation = load_evaluation_input(
        args.manual,
        args.clusters,
        manual_label_column=args.manual_label_column,
        node_mode=args.node_mode,
        evaluation_node_types=parse_evaluation_tokens(args.evaluation_node_types),
        evaluation_kind_tokens=parse_evaluation_tokens(args.evaluation_kind_tokens),
        all_evaluation_nodes=args.all_evaluation_nodes,
    )
    payload = write_outputs(
        evaluation,
        args.outdir,
        na_labels=na_labels,
        write_sensitivity=args.write_sensitivity,
    )
    metadata = payload["metadata"]
    primary = payload["primary_exclude_na"]
    print(f"Wrote evaluation artifacts to {args.outdir}")
    print(
        "Primary excluding NA: "
        f"ARI={primary['adjusted_rand_index']:.3f}, "
        f"V-measure={primary['v_measure']:.3f}, "
        f"Pairwise F1={primary['pairwise_f1']:.3f}, "
        f"BCubed F1={primary['bcubed_f1']:.3f}"
    )
    print(
        "Coverage: "
        f"{metadata['known_joined_rows']}/{metadata['joined_rows']} "
        f"known joined rows ({metadata['known_coverage_of_joined']:.1%}); "
        f"node mode={metadata['node_mode']}; "
        f"scope={'all' if metadata['all_evaluation_nodes'] else 'filtered'}"
    )


if __name__ == "__main__":
    main()
