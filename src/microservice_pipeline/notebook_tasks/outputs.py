"""Output schemas and writers for notebook task analysis artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown
    from microservice_pipeline.notebook_tasks.pruning import (
        NOTEBOOK_UNOBSERVED_EXCLUDED_FIELDS,
        NotebookPruningResult,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown  # type: ignore
    from notebook_tasks.pruning import (  # type: ignore
        NOTEBOOK_UNOBSERVED_EXCLUDED_FIELDS,
        NotebookPruningResult,
    )


NOTEBOOK_TASK_FIELDS = [
    "task_id",
    "notebook",
    "notebook_path",
    "cell_index",
    "heading_level",
    "heading_text",
    "heading_path",
    "parent_task_id",
    "major_task_id",
    "major_task_title",
    "classification",
    "normalized_label",
]

TASK_CALLABLE_USAGE_FIELDS = [
    "usage_id",
    "notebook",
    "notebook_path",
    "cell_index",
    "task_granularity",
    "task_id",
    "task_title",
    "task_classification",
    "scenario_task_id",
    "major_task_id",
    "major_task_title",
    "leaf_task_id",
    "leaf_task_title",
    "refinement_task_id",
    "refinement_task_title",
    "refinement_task_classification",
    "caller",
    "callee",
    "callable_node",
    "resolved",
    "relation",
    "lineno",
]

TASK_CLUSTER_OVERLAY_FIELDS = [
    "task_id",
    "task_title",
    "task_classification",
    "task_granularity",
    "notebooks",
    "cluster_id",
    "cluster_size",
    "occurrences",
    "callable_count",
    "callables_preview",
]

CLUSTER_TASK_DIAGNOSTIC_FIELDS = [
    "cluster_id",
    "cluster_size",
    "domain_occurrences",
    "domain_callable_count",
    "domain_task_count",
    "dominant_task_id",
    "dominant_task_title",
    "dominant_task_share",
    "task_entropy",
    "support_occurrences",
    "ignored_occurrences",
    "warnings",
    "recommended_action",
]

TASK_SCATTER_FIELDS = [
    "task_id",
    "task_title",
    "task_classification",
    "task_granularity",
    "notebooks",
    "cluster_count",
    "occurrences",
    "callable_count",
    "dominant_cluster_id",
    "dominant_cluster_share",
    "clusters_preview",
    "warnings",
]

REFINEMENT_RECOMMENDATION_FIELDS = [
    "kind",
    "cluster_id",
    "refined_group",
    "task_id",
    "task_label",
    "reason",
    "node_count",
    "occurrences",
    "callable_count",
    "accepted",
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def write_report(
    path: Path,
    *,
    notebooks: Sequence[Path],
    task_rows: Sequence[Mapping[str, Any]],
    usage_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    recommendations: Sequence[Mapping[str, Any]],
    pruning_result: NotebookPruningResult,
    prune_notebook_unobserved: bool,
) -> None:
    split_candidates = [
        row for row in diagnostics if _text(row.get("recommended_action")) == "split_candidate"
    ]
    lines = [
        "# Notebook Task Analysis",
        "",
        f"- Notebooks analyzed: {len(notebooks)}",
        f"- Extracted tasks/headings: {len(task_rows)}",
        f"- Callable usage rows: {len(usage_rows)}",
        f"- Split candidate clusters: {len(split_candidates)}",
        f"- Refinement recommendations: {len(recommendations)}",
        (
            "- Notebook-unobserved pruning: "
            f"{'enabled' if prune_notebook_unobserved else 'disabled'} "
            f"({pruning_result.pruned_callable_count} callables, "
            f"{pruning_result.pruned_data_count} data rows)"
        ),
        "",
        "## Notebooks",
        "",
    ]
    lines.extend(f"- `{path}`" for path in notebooks)
    lines.extend(["", "## Split Candidates", ""])
    if split_candidates:
        for row in split_candidates[:20]:
            lines.append(
                "- "
                f"{row['cluster_id']}: dominant `{row['dominant_task_title']}` "
                f"share {row['dominant_task_share']}, "
                f"{row['domain_task_count']} domain tasks"
            )
    else:
        lines.append("No split candidates found.")
    write_markdown(path, lines, trailing_newline=True)


def write_analysis_outputs(
    *,
    outdir: Path,
    reusable_outdir: Path | None = None,
    cluster_fields: Sequence[str],
    task_rows: Sequence[Mapping[str, Any]],
    usage_rows: Sequence[Mapping[str, Any]],
    overlay_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    scatter_rows: Sequence[Mapping[str, Any]],
    recommendations: Sequence[Mapping[str, Any]],
    refined_rows: Sequence[Mapping[str, Any]],
    pruning_result: NotebookPruningResult,
    metadata: Mapping[str, Any],
    notebooks: Sequence[Path],
    prune_notebook_unobserved: bool,
) -> None:
    reusable_dir = reusable_outdir or outdir
    ensure_dir(outdir)
    ensure_dir(reusable_dir)
    write_csv_rows(reusable_dir / "notebook_tasks.csv", NOTEBOOK_TASK_FIELDS, task_rows)
    write_csv_rows(reusable_dir / "task_callable_usage.csv", TASK_CALLABLE_USAGE_FIELDS, usage_rows)
    write_csv_rows(outdir / "task_cluster_overlay.csv", TASK_CLUSTER_OVERLAY_FIELDS, overlay_rows)
    write_csv_rows(outdir / "cluster_task_diagnostics.csv", CLUSTER_TASK_DIAGNOSTIC_FIELDS, diagnostics)
    write_csv_rows(outdir / "task_scatter.csv", TASK_SCATTER_FIELDS, scatter_rows)
    write_csv_rows(outdir / "refinement_recommendations.csv", REFINEMENT_RECOMMENDATION_FIELDS, recommendations)
    write_csv_rows(
        outdir / "notebook_unobserved_excluded_nodes.csv",
        NOTEBOOK_UNOBSERVED_EXCLUDED_FIELDS,
        pruning_result.excluded_rows,
    )
    refined_fields = [
        *cluster_fields,
        "original_cluster_id",
        "original_cluster_size",
        "refined_cluster_id",
        "refined_cluster_size",
        "refinement_action",
        "task_id",
        "task_label",
    ]
    write_csv_rows(outdir / "refined_cluster_assignments.csv", refined_fields, refined_rows)
    write_json(outdir / "notebook_task_analysis.json", dict(metadata))
    write_report(
        outdir / "notebook_task_report.md",
        notebooks=notebooks,
        task_rows=task_rows,
        usage_rows=usage_rows,
        diagnostics=diagnostics,
        recommendations=recommendations,
        pruning_result=pruning_result,
        prune_notebook_unobserved=prune_notebook_unobserved,
    )
