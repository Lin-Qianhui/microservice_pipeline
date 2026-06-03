import json
from pathlib import Path

from microservice_pipeline.notebook_tasks.analyze_notebook_tasks import (
    HeadingClassification,
    RefinementAcceptance,
    build_resolver,
    classify_heading,
    compute_cluster_diagnostics,
    annotate_and_roll_up_usage_rows,
    expand_task_extraction_candidates,
    extract_notebook_tasks_and_usage,
    identify_task_extraction_candidates,
    load_notebook_task_config_defaults,
    notebook_config_for_runtime,
    notebook_task_config_defaults,
    parse_args,
    refine_assignments,
    refinement_acceptance_from_config,
    select_notebooks,
)
from microservice_pipeline.notebook_tasks.outputs import write_analysis_outputs
from microservice_pipeline.notebook_tasks.pruning import (
    CALLABLE_PRUNE_REASON,
    NotebookPruningResult,
    ORPHAN_DATA_PRUNE_REASON,
    prune_notebook_unobserved_assignments,
)


def _write_notebook(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def _markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _code(source):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def _sample_resolver(tmp_path):
    source_root = tmp_path / "src" / "pkg"
    source_root.mkdir(parents=True)
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "service.py").write_text(
        """
class Processor:
    def __init__(self):
        pass

    def run(self):
        pass

    def analyze(self):
        pass
""",
        encoding="utf-8",
    )
    return build_resolver(
        source_root=source_root,
        module_prefix="pkg",
        package="pkg",
        module="notebook_tasks.sample",
        file=tmp_path / "docs" / "tutorial.ipynb",
    )


def _assignment_row(node, node_type, *, in_degree=0, out_degree=1, cluster_id="R001"):
    label = node.rsplit(".", 1)[-1].removeprefix("data:")
    return {
        "cluster_id": cluster_id,
        "cluster_size": "1",
        "node": node,
        "node_type": node_type,
        "label": label,
        "kind": "function" if node_type == "callable" else "local_exposed",
        "module": "pkg",
        "qualname": node.removeprefix("callable:"),
        "owner": "",
        "in_degree": str(in_degree),
        "out_degree": str(out_degree),
        "total_degree": str(in_degree + out_degree),
        "weighted_degree": str(in_degree + out_degree),
        "must_link_group": "",
        "warnings": "",
        "file": "pkg/service.py",
        "lineno": "1",
        "original_cluster_id": cluster_id,
        "original_cluster_size": "1",
        "refined_cluster_id": cluster_id,
        "refined_cluster_size": "1",
        "refinement_action": "unchanged",
        "task_id": "",
        "task_label": "",
    }


def test_notebook_task_config_defaults_resolve_paths_and_nested_sections(tmp_path, monkeypatch):
    config = {
        "paths": {
            "call_graph": "artifacts/call_graph.json",
            "structural_nodes": "graph/nodes.csv",
            "structural_edges": "graph/edges.csv",
            "clusters": "clusters/cluster_assignments.csv",
            "outdir": "notebook_tasks",
            "reusable_outdir": "notebook_tasks_reusable",
        },
        "source": {
            "source_root": "src/pkg",
            "module_prefix": "pkg",
            "package": "pkg",
        },
        "notebooks": {
            "include": ["docs/tutorial.ipynb"],
            "include_globs": ["docs/*_tutorial.ipynb"],
            "exclude": ["docs/skip.ipynb"],
        },
        "tasks": {
            "granularity": "major-heading",
            "refinement_heading_level": 3,
            "task_extract_call_depth": 4,
        },
        "refinement": {
            "mode": "selected",
            "accepted_refinements": ["refined-group-1"],
        },
        "pruning": {
            "notebook_unobserved": False,
        },
        "heading_classification": {
            "ignored_patterns": [r"ignore me"],
            "support_patterns": [r"support me"],
        },
    }

    defaults = notebook_task_config_defaults(config, tmp_path)
    runtime = notebook_config_for_runtime(config)

    assert defaults["call_graph"] == str((tmp_path / "artifacts/call_graph.json").resolve())
    assert defaults["structural_nodes"] == str((tmp_path / "graph/nodes.csv").resolve())
    assert defaults["structural_edges"] == str((tmp_path / "graph/edges.csv").resolve())
    assert defaults["clusters"] == str((tmp_path / "clusters/cluster_assignments.csv").resolve())
    assert defaults["outdir"] == str((tmp_path / "notebook_tasks").resolve())
    assert defaults["reusable_outdir"] == str((tmp_path / "notebook_tasks_reusable").resolve())
    assert defaults["source_root"] == str((tmp_path / "src/pkg").resolve())
    assert defaults["module_prefix"] == "pkg"
    assert defaults["package"] == "pkg"
    assert defaults["notebook"] == ["docs/tutorial.ipynb"]
    assert defaults["notebook_glob"] == ["docs/*_tutorial.ipynb"]
    assert defaults["exclude_notebook"] == ["docs/skip.ipynb"]
    assert defaults["task_granularity"] == "major-heading"
    assert defaults["refinement_heading_level"] == 3
    assert defaults["task_extract_call_depth"] == 4
    assert defaults["accept_refinements"] == "selected"
    assert defaults["accept_refinement"] == ["refined-group-1"]
    assert defaults["prune_notebook_unobserved"] is False
    assert runtime["include_notebooks"] == ["docs/tutorial.ipynb"]
    assert runtime["exclude_notebooks"] == ["docs/skip.ipynb"]
    assert runtime["heading_classification"]["support_patterns"] == [r"support me"]


def test_notebook_task_parse_args_uses_jsonc_config_and_cli_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "notebook_task_analysis.jsonc"
    config_path.write_text(
        """
        {
          // Comments and trailing commas are allowed.
          "paths": {
            "call_graph": "artifacts/call_graph.json",
            "clusters": "configured/cluster_assignments.csv",
            "outdir": "configured/run",
            "reusable_outdir": "configured/reusable",
          },
          "tasks": {
            "granularity": "major-heading",
          },
          "refinement": {
            "mode": "none",
          },
          "pruning": {
            "notebook_unobserved": true,
          },
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_notebook_tasks.py",
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--clusters",
            "override/cluster_assignments.csv",
            "--task-granularity",
            "leaf-heading",
            "--reusable-outdir",
            "override/reusable",
            "--no-prune-notebook-unobserved",
        ],
    )

    args = parse_args()

    assert args.call_graph == str((tmp_path / "artifacts/call_graph.json").resolve())
    assert args.clusters == "override/cluster_assignments.csv"
    assert args.outdir == str((tmp_path / "configured/run").resolve())
    assert args.reusable_outdir == "override/reusable"
    assert args.source_root == "src"
    assert args.module_prefix is None
    assert args.package is None
    assert args.task_granularity == "leaf-heading"
    assert args.accept_refinements == "none"
    assert args.prune_notebook_unobserved is False


def test_load_notebook_task_config_defaults_reads_jsonc_file(tmp_path):
    config_path = tmp_path / "notebook_task_analysis.jsonc"
    config_path.write_text(
        """
        {
          "tasks": {
            "task_extract_call_depth": 5,
          },
        }
        """,
        encoding="utf-8",
    )

    defaults = load_notebook_task_config_defaults(config_path, tmp_path)

    assert defaults["task_extract_call_depth"] == 5


def test_write_analysis_outputs_splits_reusable_and_cluster_specific_files(tmp_path):
    reusable_dir = tmp_path / "reusable"
    run_dir = tmp_path / "run"
    pruning_result = NotebookPruningResult(
        refined_rows=[],
        excluded_rows=[],
        pruned_callable_count=0,
        pruned_data_count=0,
    )

    write_analysis_outputs(
        outdir=run_dir,
        reusable_outdir=reusable_dir,
        cluster_fields=["cluster_id", "cluster_size", "node"],
        notebooks=[],
        task_rows=[{"task_id": "T1"}],
        usage_rows=[{"usage_id": "U1"}],
        overlay_rows=[],
        diagnostics=[],
        scatter_rows=[],
        recommendations=[],
        refined_rows=[{"cluster_id": "C001", "cluster_size": "1", "node": "callable:a"}],
        pruning_result=pruning_result,
        metadata={"schema": "test"},
        prune_notebook_unobserved=False,
    )

    assert (reusable_dir / "notebook_tasks.csv").exists()
    assert (reusable_dir / "task_callable_usage.csv").exists()
    assert not (run_dir / "notebook_tasks.csv").exists()
    assert not (run_dir / "task_callable_usage.csv").exists()
    assert (run_dir / "task_cluster_overlay.csv").exists()
    assert (run_dir / "task_scatter.csv").exists()
    assert (run_dir / "refinement_recommendations.csv").exists()
    assert (run_dir / "refined_cluster_assignments.csv").exists()
    assert not (reusable_dir / "task_cluster_overlay.csv").exists()
    assert not (reusable_dir / "refined_cluster_assignments.csv").exists()

    fallback_dir = tmp_path / "fallback"
    write_analysis_outputs(
        outdir=fallback_dir,
        cluster_fields=["cluster_id", "cluster_size", "node"],
        notebooks=[],
        task_rows=[],
        usage_rows=[],
        overlay_rows=[],
        diagnostics=[],
        scatter_rows=[],
        recommendations=[],
        refined_rows=[],
        pruning_result=pruning_result,
        metadata={"schema": "test"},
        prune_notebook_unobserved=False,
    )

    assert (fallback_dir / "notebook_tasks.csv").exists()
    assert (fallback_dir / "task_cluster_overlay.csv").exists()


def test_select_notebooks_uses_generic_default_globs(tmp_path):
    docs = tmp_path / "docs"
    _write_notebook(docs / "model_tutorial.ipynb", [])
    _write_notebook(docs / "example.ipynb", [])
    _write_notebook(docs / "call_graph_preprocessing.ipynb", [])
    _write_notebook(tmp_path / "notebooks" / "analysis.ipynb", [])

    selected = select_notebooks(
        tmp_path,
        config={},
        notebooks=[],
        notebook_globs=[],
        exclude_notebooks=[],
    )

    assert {path.name for path in selected} == {
        "analysis.ipynb",
        "call_graph_preprocessing.ipynb",
        "example.ipynb",
        "model_tutorial.ipynb",
    }


def test_notebook_unobserved_pruning_keeps_guardrailed_callables_and_shared_data():
    used = "callable:pkg.service.used"
    unused = "callable:pkg.service.unused"
    callback = "callable:pkg.service.used.<locals>.callback"
    dunder = "callable:pkg.service.Model.__repr__"
    worker = "callable:pkg.service.worker"
    orphan_data = "data:pkg.service.unused:state"
    shared_data = "data:pkg.service.shared:state"
    rows = [
        _assignment_row(used, "callable", in_degree=0),
        _assignment_row(unused, "callable", in_degree=0),
        _assignment_row(callback, "callable", in_degree=0),
        _assignment_row(dunder, "callable", in_degree=0),
        _assignment_row(worker, "callable", in_degree=1),
        _assignment_row(orphan_data, "data", in_degree=1),
        _assignment_row(shared_data, "data", in_degree=2),
    ]
    structural_edges = [
        {"src": unused, "dst": orphan_data, "edge_type": "data_access", "access": "read"},
        {"src": unused, "dst": shared_data, "edge_type": "data_access", "access": "read"},
        {"src": worker, "dst": shared_data, "edge_type": "data_access", "access": "read"},
        {"src": callback, "dst": "data:pkg.service.callback:state", "edge_type": "data_access", "access": "read"},
    ]
    usage_rows = [
        {
            "callable_node": used,
            "resolved": "1",
            "task_id": "T1",
            "task_classification": "ignored",
        }
    ]

    result = prune_notebook_unobserved_assignments(
        refined_rows=rows,
        structural_edges=structural_edges,
        usage_rows=usage_rows,
    )

    kept_nodes = {row["node"] for row in result.refined_rows}
    excluded = {row["node"]: row["reason"] for row in result.excluded_rows}
    assert used in kept_nodes
    assert callback in kept_nodes
    assert dunder in kept_nodes
    assert worker in kept_nodes
    assert shared_data in kept_nodes
    assert unused not in kept_nodes
    assert orphan_data not in kept_nodes
    assert excluded == {
        unused: CALLABLE_PRUNE_REASON,
        orphan_data: ORPHAN_DATA_PRUNE_REASON,
    }
    assert result.pruned_callable_count == 1
    assert result.pruned_data_count == 1
    assert {row["cluster_size"] for row in result.refined_rows} == {5}
    assert {row["refined_cluster_size"] for row in result.refined_rows} == {5}


def test_heading_classification_is_rule_based_and_configurable():
    assert classify_heading("Import the libraries", HeadingClassification()) == "ignored"
    assert classify_heading("Plot sensitivity indices", HeadingClassification()) == "support"
    assert classify_heading("Process rate constants", HeadingClassification()) == "support"
    assert classify_heading("General results: Heatmaps of mass distribution", HeadingClassification()) == "support"
    assert classify_heading("Load the default configuration and data", HeadingClassification()) == "support"
    assert classify_heading("Step 4: Output and Results", HeadingClassification()) == "support"
    assert classify_heading("Exposure Indicators", HeadingClassification()) == "domain"
    assert classify_heading("Run the model", HeadingClassification()) == "domain"

    custom = HeadingClassification(
        ignored_patterns=(),
        support_patterns=(r"\bmodel\b",),
    )
    assert classify_heading("Run the model", custom) == "support"


def test_extract_notebook_tasks_uses_leaf_heading_and_resolves_cell_calls(tmp_path):
    notebook = tmp_path / "docs" / "tutorial.ipynb"
    _write_notebook(
        notebook,
        [
            _markdown("## Scenario\n"),
            _markdown("### Run Task\n"),
            _code(
                """
from pkg.service import Processor

processor = Processor()
processor.run()
"""
            ),
            _markdown("#### Analyze Subtask\n"),
            _code("processor.analyze()\n"),
        ],
    )

    resolver = _sample_resolver(tmp_path)
    structural_nodes = {
        "callable:pkg.service.Processor.__init__",
        "callable:pkg.service.Processor.run",
        "callable:pkg.service.Processor.analyze",
    }
    task_rows, usage_rows, _usage_index = extract_notebook_tasks_and_usage(
        notebook_path=notebook,
        project_root=tmp_path,
        granularity="leaf-heading",
        heading_rules=HeadingClassification(),
        resolver=resolver,
        structural_nodes=structural_nodes,
        usage_start=0,
    )

    task_titles = {row["task_id"]: row["heading_text"] for row in task_rows}
    resolved = [row for row in usage_rows if row["callable_node"]]

    assert any(row["task_title"] == "Run Task" for row in resolved)
    assert any(row["task_title"] == "Analyze Subtask" for row in resolved)
    assert any(row["callable_node"] == "callable:pkg.service.Processor.__init__" for row in resolved)
    assert any(row["callable_node"] == "callable:pkg.service.Processor.run" for row in resolved)
    assert any(row["callable_node"] == "callable:pkg.service.Processor.analyze" for row in resolved)
    assert task_titles


def test_child_heading_classification_is_not_inherited_from_import_parent(tmp_path):
    notebook = tmp_path / "docs" / "tutorial.ipynb"
    _write_notebook(
        notebook,
        [
            _markdown("### Import the necessary libraries\n"),
            _markdown("#### Exposure Indicators\n"),
            _code("from pkg.service import Processor\nprocessor = Processor()\nprocessor.analyze()\n"),
        ],
    )

    resolver = _sample_resolver(tmp_path)
    _task_rows, usage_rows, _usage_index = extract_notebook_tasks_and_usage(
        notebook_path=notebook,
        project_root=tmp_path,
        granularity="leaf-heading",
        heading_rules=HeadingClassification(),
        resolver=resolver,
        structural_nodes={
            "callable:pkg.service.Processor.__init__",
            "callable:pkg.service.Processor.analyze",
        },
        usage_start=0,
    )

    resolved = [row for row in usage_rows if row["callable_node"]]
    assert resolved
    assert {row["task_title"] for row in resolved} == {"Exposure Indicators"}
    assert {row["task_classification"] for row in resolved} == {"domain"}


def test_refinement_rolls_leaf_subtask_up_to_domain_parent_heading(tmp_path):
    notebook = tmp_path / "docs" / "tutorial.ipynb"
    _write_notebook(
        notebook,
        [
            _markdown("#### Exposure Indicators\n"),
            _code("from pkg.service import Processor\nprocessor = Processor()\nprocessor.analyze()\n"),
            _markdown("##### Dispersed mass fraction\n"),
            _code("processor.run()\n"),
        ],
    )

    resolver = _sample_resolver(tmp_path)
    task_rows, usage_rows, _usage_index = extract_notebook_tasks_and_usage(
        notebook_path=notebook,
        project_root=tmp_path,
        granularity="leaf-heading",
        heading_rules=HeadingClassification(),
        resolver=resolver,
        structural_nodes={
            "callable:pkg.service.Processor.__init__",
            "callable:pkg.service.Processor.analyze",
            "callable:pkg.service.Processor.run",
        },
        usage_start=0,
    )

    annotated_rows, refinement_rows = annotate_and_roll_up_usage_rows(
        usage_rows,
        task_rows,
        max_heading_level=4,
    )
    run_rows = [
        row
        for row in refinement_rows
        if row["callable_node"] == "callable:pkg.service.Processor.run"
    ]

    assert run_rows
    assert {row["task_title"] for row in run_rows} == {"Exposure Indicators"}
    assert any(row["leaf_task_title"] == "Dispersed mass fraction" for row in run_rows)
    assert any(
        row["refinement_task_title"] == "Exposure Indicators" for row in annotated_rows
    )


def test_extract_notebook_tasks_can_use_notebook_as_task_granularity(tmp_path):
    notebook = tmp_path / "docs" / "tutorial.ipynb"
    _write_notebook(
        notebook,
        [
            _markdown("## Scenario\n### Run Task\n"),
            _code("from pkg.service import Processor\nprocessor = Processor()\n"),
        ],
    )

    resolver = _sample_resolver(tmp_path)
    _task_rows, usage_rows, _usage_index = extract_notebook_tasks_and_usage(
        notebook_path=notebook,
        project_root=tmp_path,
        granularity="notebook",
        heading_rules=HeadingClassification(),
        resolver=resolver,
        structural_nodes={"callable:pkg.service.Processor.__init__"},
        usage_start=0,
    )

    resolved = [row for row in usage_rows if row["callable_node"]]
    assert resolved
    assert {row["task_title"] for row in resolved} == {"tutorial"}


def test_diagnostics_and_refinement_split_mixed_domain_cluster_and_attach_data():
    cluster_rows = [
        {"cluster_id": "C001", "cluster_size": "5", "node": "callable:a"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "callable:b"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "callable:c"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "callable:d"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "data:shared"},
    ]
    structural_edges = [
        {"src": "data:shared", "dst": "callable:a", "weight": "2.0"},
    ]
    usage_rows = [
        {
            "callable_node": "callable:a",
            "task_id": "T1",
            "task_title": "Run",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        },
        {
            "callable_node": "callable:b",
            "task_id": "T1",
            "task_title": "Run",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        },
        {
            "callable_node": "callable:c",
            "task_id": "T2",
            "task_title": "Analyze",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        },
        {
            "callable_node": "callable:d",
            "task_id": "T2",
            "task_title": "Analyze",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        },
    ]
    cluster_of = {row["node"]: row["cluster_id"] for row in cluster_rows}
    diagnostics = compute_cluster_diagnostics(
        usage_rows=usage_rows,
        cluster_of=cluster_of,
        cluster_size={"C001": 5},
        task_titles={"T1": "Run", "T2": "Analyze"},
    )

    assert diagnostics[0]["recommended_action"] == "split_candidate"

    refined_rows, recommendations = refine_assignments(
        cluster_rows=cluster_rows,
        structural_edges=structural_edges,
        usage_rows=usage_rows,
        diagnostics=diagnostics,
        task_titles={"T1": "Run", "T2": "Analyze"},
        task_classification={"T1": "domain", "T2": "domain"},
        task_granularity="leaf-heading",
    )

    refined_by_node = {row["node"]: row for row in refined_rows}
    assert refined_by_node["callable:a"]["refined_cluster_id"] == refined_by_node["callable:b"]["refined_cluster_id"]
    assert refined_by_node["callable:c"]["refined_cluster_id"] == refined_by_node["callable:d"]["refined_cluster_id"]
    assert refined_by_node["callable:a"]["refined_cluster_id"] != refined_by_node["callable:c"]["refined_cluster_id"]
    assert refined_by_node["data:shared"]["refinement_action"] == "split_by_task_structural_attach"
    assert refined_by_node["data:shared"]["task_label"] == "Run"
    assert {row["kind"] for row in recommendations} == {"split"}


def test_small_domain_task_can_be_extracted_from_large_cluster():
    cluster_rows = [
        {"cluster_id": "C001", "cluster_size": "40", "node": "callable:task"},
        {"cluster_id": "C001", "cluster_size": "40", "node": "callable:other"},
    ]
    usage_rows = [
        {
            "callable_node": "callable:task",
            "task_id": "T1",
            "task_title": "Exposure Indicators",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        }
    ]
    cluster_of = {row["node"]: row["cluster_id"] for row in cluster_rows}
    candidates = identify_task_extraction_candidates(
        usage_rows=usage_rows,
        cluster_of=cluster_of,
        cluster_size={"C001": 40},
        task_classification={"T1": "domain"},
    )

    refined_rows, recommendations = refine_assignments(
        cluster_rows=cluster_rows,
        structural_edges=[],
        usage_rows=usage_rows,
        diagnostics=[],
        task_titles={"T1": "Exposure Indicators"},
        task_classification={"T1": "domain"},
        task_granularity="leaf-heading",
        task_extraction_candidates=candidates,
    )

    refined_by_node = {row["node"]: row for row in refined_rows}
    assert refined_by_node["callable:task"]["refinement_action"] == "extracted_task_usage"
    assert refined_by_node["callable:task"]["task_label"] == "Exposure Indicators"
    assert refined_by_node["callable:task"]["refined_cluster_id"] != refined_by_node["callable:other"]["refined_cluster_id"]
    assert {row["kind"] for row in recommendations} == {"extract_task"}
    assert recommendations[0]["accepted"] == "1"


def test_refinement_acceptance_none_keeps_baseline_assignments_with_recommendations():
    cluster_rows = [
        {"cluster_id": "C001", "cluster_size": "40", "node": "callable:task"},
        {"cluster_id": "C001", "cluster_size": "40", "node": "callable:other"},
    ]
    usage_rows = [
        {
            "callable_node": "callable:task",
            "task_id": "T1",
            "task_title": "Exposure Indicators",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        }
    ]
    cluster_of = {row["node"]: row["cluster_id"] for row in cluster_rows}
    candidates = identify_task_extraction_candidates(
        usage_rows=usage_rows,
        cluster_of=cluster_of,
        cluster_size={"C001": 40},
        task_classification={"T1": "domain"},
    )

    refined_rows, recommendations = refine_assignments(
        cluster_rows=cluster_rows,
        structural_edges=[],
        usage_rows=usage_rows,
        diagnostics=[],
        task_titles={"T1": "Exposure Indicators"},
        task_classification={"T1": "domain"},
        task_granularity="leaf-heading",
        task_extraction_candidates=candidates,
        refinement_acceptance=RefinementAcceptance(mode="none"),
    )

    refined_by_node = {row["node"]: row for row in refined_rows}
    assert refined_by_node["callable:task"]["refined_cluster_id"] == refined_by_node["callable:other"]["refined_cluster_id"]
    assert refined_by_node["callable:task"]["refinement_action"] == "unchanged"
    assert refined_by_node["callable:task"]["task_label"] == ""
    assert recommendations[0]["accepted"] == "0"


def test_refinement_acceptance_selected_applies_only_named_refined_group():
    cluster_rows = [
        {"cluster_id": "C001", "cluster_size": "80", "node": "callable:task_a"},
        {"cluster_id": "C001", "cluster_size": "80", "node": "callable:task_b"},
        {"cluster_id": "C001", "cluster_size": "80", "node": "callable:other"},
    ]
    usage_rows = [
        {
            "callable_node": "callable:task_a",
            "task_id": "T1",
            "task_title": "Exposure Indicators",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        },
        {
            "callable_node": "callable:task_b",
            "task_id": "T2",
            "task_title": "Run Model",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        },
    ]
    cluster_of = {row["node"]: row["cluster_id"] for row in cluster_rows}
    candidates = identify_task_extraction_candidates(
        usage_rows=usage_rows,
        cluster_of=cluster_of,
        cluster_size={"C001": 80},
        task_classification={"T1": "domain", "T2": "domain"},
    )

    refined_rows, recommendations = refine_assignments(
        cluster_rows=cluster_rows,
        structural_edges=[],
        usage_rows=usage_rows,
        diagnostics=[],
        task_titles={"T1": "Exposure Indicators", "T2": "Run Model"},
        task_classification={"T1": "domain", "T2": "domain"},
        task_granularity="leaf-heading",
        task_extraction_candidates=candidates,
        refinement_acceptance=RefinementAcceptance(
            mode="selected",
            refined_groups=frozenset({"extract::T1"}),
        ),
    )

    refined_by_node = {row["node"]: row for row in refined_rows}
    assert refined_by_node["callable:task_a"]["refinement_action"] == "extracted_task_usage"
    assert refined_by_node["callable:task_b"]["refinement_action"] == "unchanged"
    assert refined_by_node["callable:task_b"]["refined_cluster_id"] == refined_by_node["callable:other"]["refined_cluster_id"]
    accepted_by_group = {row["refined_group"]: row["accepted"] for row in recommendations}
    assert accepted_by_group["extract::T1"] == "1"
    assert accepted_by_group["extract::T2"] == "0"


def test_refinement_acceptance_config_defaults_to_selected_when_ids_are_given():
    acceptance = refinement_acceptance_from_config(
        {"refinement_acceptance": {"accepted_refinements": ["extract::T1"]}},
        mode=None,
        accepted_refinements=[],
    )

    assert acceptance.mode == "selected"
    assert acceptance.refined_groups == frozenset({"extract::T1"})


def test_refined_callable_group_moves_owned_data_but_not_externally_created_data():
    cluster_rows = [
        {"cluster_id": "C001", "cluster_size": "5", "node": "callable:task"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "callable:other"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "data:created_by_task"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "data:mutated_by_task"},
        {"cluster_id": "C001", "cluster_size": "5", "node": "data:created_elsewhere"},
    ]
    structural_edges = [
        {
            "src": "callable:task",
            "dst": "data:created_by_task",
            "edge_type": "data_access",
            "access": "create",
            "operation": "assign",
            "weight": "1.0",
        },
        {
            "src": "callable:task",
            "dst": "data:mutated_by_task",
            "edge_type": "data_access",
            "access": "read_write",
            "operation": "method:append",
            "weight": "1.0",
        },
        {
            "src": "callable:other",
            "dst": "data:created_elsewhere",
            "edge_type": "data_access",
            "access": "create",
            "operation": "assign",
            "weight": "1.0",
        },
        {
            "src": "callable:task",
            "dst": "data:created_elsewhere",
            "edge_type": "data_access",
            "access": "write",
            "operation": "assign",
            "weight": "1.0",
        },
    ]
    usage_rows = [
        {
            "callable_node": "callable:task",
            "task_id": "T1",
            "task_title": "Exposure Indicators",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        }
    ]
    cluster_of = {row["node"]: row["cluster_id"] for row in cluster_rows}
    candidates = identify_task_extraction_candidates(
        usage_rows=usage_rows,
        cluster_of=cluster_of,
        cluster_size={"C001": 40},
        task_classification={"T1": "domain"},
    )

    refined_rows, recommendations = refine_assignments(
        cluster_rows=cluster_rows,
        structural_edges=structural_edges,
        usage_rows=usage_rows,
        diagnostics=[],
        task_titles={"T1": "Exposure Indicators"},
        task_classification={"T1": "domain"},
        task_granularity="leaf-heading",
        task_extraction_candidates=candidates,
    )

    refined_by_node = {row["node"]: row for row in refined_rows}
    task_cluster = refined_by_node["callable:task"]["refined_cluster_id"]
    assert refined_by_node["data:created_by_task"]["refined_cluster_id"] == task_cluster
    assert refined_by_node["data:mutated_by_task"]["refined_cluster_id"] == task_cluster
    assert refined_by_node["data:created_by_task"]["refinement_action"] == "attached_owned_data"
    assert refined_by_node["data:mutated_by_task"]["refinement_action"] == "attached_owned_data"
    assert refined_by_node["data:created_elsewhere"]["refined_cluster_id"] != task_cluster
    assert refined_by_node["data:created_elsewhere"]["refinement_action"] == "unchanged"
    assert recommendations[0]["node_count"] == 3
    assert recommendations[0]["callable_count"] == 1


def test_extracted_task_can_include_bounded_internal_call_expansion():
    cluster_rows = [
        {"cluster_id": "C001", "cluster_size": "40", "node": "callable:task"},
        {"cluster_id": "C001", "cluster_size": "40", "node": "callable:helper"},
        {"cluster_id": "C001", "cluster_size": "40", "node": "callable:leaf"},
        {"cluster_id": "C002", "cluster_size": "2", "node": "callable:other_cluster"},
    ]
    usage_rows = [
        {
            "callable_node": "callable:task",
            "task_id": "T1",
            "task_title": "Exposure Indicators",
            "task_classification": "domain",
            "task_granularity": "leaf-heading",
            "relation": "direct",
            "resolved": "1",
        }
    ]
    structural_edges = [
        {
            "src": "callable:task",
            "dst": "callable:helper",
            "edge_type": "call",
            "relation": "imported",
            "weight": "1.0",
        },
        {
            "src": "callable:helper",
            "dst": "callable:leaf",
            "edge_type": "call",
            "relation": "direct",
            "weight": "1.0",
        },
        {
            "src": "callable:task",
            "dst": "callable:other_cluster",
            "edge_type": "call",
            "relation": "direct",
            "weight": "1.0",
        },
    ]
    cluster_of = {row["node"]: row["cluster_id"] for row in cluster_rows}
    candidates = identify_task_extraction_candidates(
        usage_rows=usage_rows,
        cluster_of=cluster_of,
        cluster_size={"C001": 40, "C002": 2},
        task_classification={"T1": "domain"},
    )
    expanded_candidates = expand_task_extraction_candidates(
        task_extraction_candidates=candidates,
        structural_edges=structural_edges,
        cluster_of=cluster_of,
        max_depth=2,
    )

    refined_rows, _recommendations = refine_assignments(
        cluster_rows=cluster_rows,
        structural_edges=structural_edges,
        usage_rows=usage_rows,
        diagnostics=[],
        task_titles={"T1": "Exposure Indicators"},
        task_classification={"T1": "domain"},
        task_granularity="leaf-heading",
        task_extraction_candidates=expanded_candidates,
    )

    refined_by_node = {row["node"]: row for row in refined_rows}
    assert expanded_candidates["T1"] == {"callable:task", "callable:helper", "callable:leaf"}
    assert refined_by_node["callable:helper"]["refinement_action"] == "extracted_task_call_expansion"
    assert refined_by_node["callable:leaf"]["refinement_action"] == "extracted_task_call_expansion"
    assert refined_by_node["callable:other_cluster"]["refinement_action"] == "unchanged"
