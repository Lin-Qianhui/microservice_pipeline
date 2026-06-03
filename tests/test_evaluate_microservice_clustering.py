import csv
import json

import pytest

from microservice_pipeline.evaluation.evaluate_microservice_clustering import (
    JoinedAssignment,
    build_evaluation_payload,
    build_evaluation_input_from_rows,
    build_best_match_rows,
    compute_metrics,
    evaluation_summary_row,
    evaluate_assignment_rows,
    filter_known_labels,
    load_evaluation_input,
    main as evaluate_main,
)


def _assignment(node, manual_label, cluster_id, node_type="callable"):
    return JoinedAssignment(
        node=node,
        normalized_node=node,
        manual_label=manual_label,
        cluster_id=cluster_id,
        node_type=node_type,
    )


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_basic_evaluation_inputs(project_root, cluster_id="C001", cluster_path=None):
    manual = project_root / "configs/microservice_pipeline/manual_mapping.csv"
    clusters = cluster_path or project_root / "artifacts/structural_microservice_candidates/cluster_assignments.csv"
    _write_csv(
        manual,
        ["microservice_id", "node", "node_type", "label", "kind", "module"],
        [
            {
                "microservice_id": "inventory",
                "node": "callable:shop.inventory.reserve",
                "node_type": "callable",
                "label": "reserve",
                "kind": "function",
                "module": "shop.inventory",
            },
            {
                "microservice_id": "NA",
                "node": "callable:shop.inventory.debug",
                "node_type": "callable",
                "label": "debug",
                "kind": "function",
                "module": "shop.inventory",
            },
        ],
    )
    _write_csv(
        clusters,
        ["cluster_id", "node", "node_type", "label", "kind", "module"],
        [
            {
                "cluster_id": cluster_id,
                "node": "callable:shop.inventory.reserve",
                "node_type": "callable",
                "label": "reserve",
                "kind": "function",
                "module": "shop.inventory",
            },
            {
                "cluster_id": "C999",
                "node": "callable:shop.inventory.debug",
                "node_type": "callable",
                "label": "debug",
                "kind": "function",
                "module": "shop.inventory",
            },
        ],
    )
    return manual, clusters


def test_perfect_partition_scores_one_even_when_cluster_ids_differ():
    assignments = [
        _assignment("n1", "service-a", "C002"),
        _assignment("n2", "service-a", "C002"),
        _assignment("n3", "service-b", "C001"),
        _assignment("n4", "service-b", "C001"),
    ]

    metrics = compute_metrics(assignments)

    assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert metrics["v_measure"] == pytest.approx(1.0)
    assert metrics["pairwise_f1"] == pytest.approx(1.0)
    assert metrics["bcubed_f1"] == pytest.approx(1.0)
    assert metrics["macro_purity_precision"] == pytest.approx(1.0)
    assert metrics["macro_purity_recall"] == pytest.approx(1.0)
    assert metrics["macro_purity_f1"] == pytest.approx(1.0)
    assert metrics["predicted_match_precision"] == pytest.approx(1.0)
    assert metrics["predicted_match_recall"] == pytest.approx(1.0)
    assert metrics["predicted_match_f1"] == pytest.approx(1.0)
    assert metrics["predicted_match_pair_macro_f1"] == pytest.approx(1.0)
    assert metrics["hungarian_accuracy"] == pytest.approx(1.0)


def test_split_manual_service_preserves_precision_but_lowers_recall():
    assignments = [
        _assignment("n1", "service-a", "C001"),
        _assignment("n2", "service-a", "C001"),
        _assignment("n3", "service-a", "C002"),
        _assignment("n4", "service-a", "C002"),
    ]

    metrics = compute_metrics(assignments)

    assert metrics["pairwise_precision"] == pytest.approx(1.0)
    assert metrics["pairwise_recall"] == pytest.approx(2 / 6)
    assert metrics["pairwise_f1"] == pytest.approx(0.5)
    assert metrics["homogeneity"] == pytest.approx(1.0)
    assert metrics["completeness"] == pytest.approx(0.0)


def test_na_rows_are_excluded_from_primary_metric_input():
    assignments = [
        _assignment("n1", "service-a", "C001"),
        _assignment("n2", "service-a", "C001"),
        _assignment("n3", "NA", "C002"),
    ]

    known = filter_known_labels(assignments, {"NA"})
    primary_metrics = compute_metrics(known)
    sensitivity_metrics = compute_metrics(assignments)

    assert len(known) == 2
    assert primary_metrics["n"] == 2
    assert primary_metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert sensitivity_metrics["n"] == 3
    assert sensitivity_metrics["manual_cluster_count"] == 2


def test_evaluation_payload_and_summary_row_are_reusable():
    manual_fields = ["Microservice_id", "node", "node_type", "label", "kind", "module"]
    cluster_fields = ["cluster_id", "node", "node_type", "label", "kind", "module"]
    manual_rows = [
        {
            "Microservice_id": "service-a",
            "node": "callable:pkg.a",
            "node_type": "callable",
            "label": "a",
            "kind": "function",
            "module": "pkg",
        },
        {
            "Microservice_id": "NA",
            "node": "callable:pkg.b",
            "node_type": "callable",
            "label": "b",
            "kind": "function",
            "module": "pkg",
        },
    ]
    cluster_rows = [
        {
            "cluster_id": "C001",
            "node": "callable:pkg.a",
            "node_type": "callable",
            "label": "a",
            "kind": "function",
            "module": "pkg",
        },
        {
            "cluster_id": "C002",
            "node": "callable:pkg.b",
            "node_type": "callable",
            "label": "b",
            "kind": "function",
            "module": "pkg",
        },
    ]

    evaluation = build_evaluation_input_from_rows(
        manual_rows,
        manual_fields,
        cluster_rows,
        cluster_fields,
        node_mode="exact",
    )
    payload = build_evaluation_payload(evaluation, na_labels={"NA"}, include_sensitivity=False)
    row = evaluation_summary_row(payload)

    assert payload["primary_exclude_na"]["n"] == 1
    assert payload["sensitivity_treat_na_as_class"] is None
    assert row["evaluation_joined_rows"] == 2
    assert row["evaluation_known_joined_rows"] == 1
    assert row["evaluation_known_coverage"] == pytest.approx(0.5)
    assert row["evaluation_adjusted_rand_index"] == pytest.approx(1.0)


def test_evaluate_cli_loads_config_and_resolves_paths_from_project_root(tmp_path):
    project_root = tmp_path / "project"
    config_path = project_root / "configs/microservice_pipeline/evaluation.jsonc"
    _write_basic_evaluation_inputs(project_root, cluster_id="C123")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
{
  "paths": {
    "manual": "configs/microservice_pipeline/manual_mapping.csv",
    "clusters": "artifacts/structural_microservice_candidates/cluster_assignments.csv",
    "outdir": "artifacts/configured_evaluation"
  },
  "matching": {
    "node_mode": "exact"
  },
  "scope": {
    "evaluation_node_types": "callable",
    "evaluation_kind_tokens": "",
    "all_evaluation_nodes": false
  }
}
""",
        encoding="utf-8",
    )

    evaluate_main(
        [
            "--project-root",
            str(project_root),
            "--config",
            "configs/microservice_pipeline/evaluation.jsonc",
        ]
    )

    outdir = project_root / "artifacts/configured_evaluation"
    payload = json.loads((outdir / "evaluation.json").read_text(encoding="utf-8"))
    assert (outdir / "metrics_summary.md").is_file()
    assert payload["metadata"]["node_mode"] == "exact"
    assert payload["metadata"]["known_joined_rows"] == 1


def test_evaluate_config_allows_cli_path_overrides_and_disables_sensitivity(tmp_path):
    project_root = tmp_path / "project"
    config_path = project_root / "configs/microservice_pipeline/evaluation.jsonc"
    _manual, _base_clusters = _write_basic_evaluation_inputs(project_root, cluster_id="BASE")
    override_clusters = project_root / "artifacts/refined/cluster_assignments.csv"
    _write_basic_evaluation_inputs(project_root, cluster_id="REFINED", cluster_path=override_clusters)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
{
  "paths": {
    "manual": "configs/microservice_pipeline/manual_mapping.csv",
    "clusters": "artifacts/structural_microservice_candidates/cluster_assignments.csv",
    "outdir": "artifacts/configured_evaluation"
  },
  "mapping": {
    "na_labels": ["NA"]
  },
  "outputs": {
    "write_sensitivity": false
  }
}
""",
        encoding="utf-8",
    )

    override_outdir = project_root / "artifacts/override_evaluation"
    evaluate_main(
        [
            "--project-root",
            str(project_root),
            "--config",
            "configs/microservice_pipeline/evaluation.jsonc",
            "--clusters",
            str(override_clusters),
            "--outdir",
            str(override_outdir),
        ]
    )

    payload = json.loads((override_outdir / "evaluation.json").read_text(encoding="utf-8"))
    with (override_outdir / "joined_assignments.csv").open(encoding="utf-8") as f:
        joined_rows = list(csv.DictReader(f))
    with (override_outdir / "metrics_summary.csv").open(encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))
    assert not (project_root / "artifacts/configured_evaluation").exists()
    assert payload["sensitivity_treat_na_as_class"] is None
    assert {row["cluster_id"] for row in joined_rows} == {"REFINED", "C999"}
    assert {row["scenario"] for row in metric_rows} == {"primary_exclude_na"}


def test_best_match_reports_cluster_level_precision_and_recall():
    assignments = [
        _assignment("n1", "service-a", "C001"),
        _assignment("n2", "service-a", "C001"),
        _assignment("n3", "service-a", "C002"),
        _assignment("n4", "service-b", "C001"),
    ]

    rows = build_best_match_rows(assignments)
    service_a = next(row for row in rows if row["manual_microservice_id"] == "service-a")

    assert service_a["best_cluster_id"] == "C001"
    assert service_a["intersection"] == 2
    assert service_a["precision"] == pytest.approx(2 / 3)
    assert service_a["recall"] == pytest.approx(2 / 3)
    assert service_a["jaccard"] == pytest.approx(0.5)


def test_macro_purity_scores_average_clusters_and_services_equally():
    assignments = [
        _assignment("n1", "service-a", "C001"),
        _assignment("n2", "service-a", "C001"),
        _assignment("n3", "service-a", "C002"),
        _assignment("n4", "service-b", "C001"),
    ]

    metrics = compute_metrics(assignments)

    assert metrics["purity"] == pytest.approx(3 / 4)
    assert metrics["inverse_purity"] == pytest.approx(3 / 4)
    assert metrics["macro_purity_precision"] == pytest.approx(5 / 6)
    assert metrics["macro_purity_recall"] == pytest.approx(5 / 6)
    assert metrics["macro_purity_f1"] == pytest.approx(5 / 6)
    assert metrics["predicted_match_precision"] == pytest.approx(5 / 6)
    assert metrics["predicted_match_recall"] == pytest.approx(1 / 2)
    assert metrics["predicted_match_f1"] == pytest.approx(5 / 8)
    assert metrics["predicted_match_pair_macro_f1"] == pytest.approx(7 / 12)


def test_auto_node_mode_falls_back_to_callable_prefix_normalization(tmp_path):
    manual = tmp_path / "manual.csv"
    clusters = tmp_path / "clusters.csv"
    _write_csv(
        manual,
        ["Microservice_id", "node", "node_type", "label", "kind", "module"],
        [
            {
                "Microservice_id": "service-a",
                "node": "callable:pkg.mod.fn",
                "node_type": "callable",
                "label": "fn",
                "kind": "function",
                "module": "pkg.mod",
            },
            {
                "Microservice_id": "service-a",
                "node": "data:pkg.mod:state",
                "node_type": "data",
                "label": "state",
                "kind": "class_attr_state",
                "module": "",
            },
        ],
    )
    _write_csv(
        clusters,
        ["cluster_id", "node", "module", "qualname", "kind"],
        [
            {
                "cluster_id": "C001",
                "node": "pkg.mod.fn",
                "module": "pkg.mod",
                "qualname": "fn",
                "kind": "function",
            }
        ],
    )

    evaluation = load_evaluation_input(manual, clusters, node_mode="auto")

    assert evaluation.node_mode == "callable"
    assert len(evaluation.joined) == 1
    assert evaluation.joined[0].node == "callable:pkg.mod.fn"
    assert evaluation.joined[0].normalized_node == "pkg.mod.fn"


def test_generated_manual_mapping_columns_join_cluster_assignments(tmp_path):
    manual = tmp_path / "manual_mapping.csv"
    clusters = tmp_path / "cluster_assignments.csv"
    _write_csv(
        manual,
        ["microservice_id", "node", "node_type", "label", "kind", "module"],
        [
            {
                "microservice_id": "inventory",
                "node": "callable:shop.inventory.Inventory.reserve",
                "node_type": "callable",
                "label": "Inventory.reserve",
                "kind": "method",
                "module": "shop.inventory",
            },
        ],
    )
    _write_csv(
        clusters,
        ["cluster_id", "node", "node_type", "label", "kind", "module"],
        [
            {
                "cluster_id": "C001",
                "node": "callable:shop.inventory.Inventory.reserve",
                "node_type": "callable",
                "label": "Inventory.reserve",
                "kind": "method",
                "module": "shop.inventory",
            },
        ],
    )

    evaluation = load_evaluation_input(manual, clusters)

    assert evaluation.manual_label_column == "microservice_id"
    assert evaluation.node_mode == "exact"
    assert evaluation.joined[0].manual_label == "inventory"


def test_default_evaluation_scope_keeps_callables_and_class_attr_state_tokens():
    manual_fields = ["Microservice_id", "node", "node_type", "label", "kind", "module"]
    cluster_fields = ["cluster_id", "node", "node_type", "label", "kind", "module"]
    manual_rows = [
        {
            "Microservice_id": "service-a",
            "node": "callable:pkg.mod.fn",
            "node_type": "callable",
            "label": "fn",
            "kind": "function",
            "module": "pkg.mod",
        },
        {
            "Microservice_id": "service-a",
            "node": "data:pkg.Service:state",
            "node_type": "data",
            "label": "state",
            "kind": "class_attr_state",
            "module": "",
        },
        {
            "Microservice_id": "service-a",
            "node": "data:pkg.Service:composite",
            "node_type": "data",
            "label": "composite",
            "kind": "class_attr_state; local_exposed",
            "module": "",
        },
        {
            "Microservice_id": "service-a",
            "node": "data:pkg.fn:local",
            "node_type": "data",
            "label": "local",
            "kind": "local_exposed",
            "module": "",
        },
    ]
    cluster_rows = [
        {**row, "cluster_id": "C001"}
        for row in manual_rows
    ]

    evaluation = build_evaluation_input_from_rows(
        manual_rows,
        manual_fields,
        cluster_rows,
        cluster_fields,
        node_mode="exact",
    )
    payload = evaluate_assignment_rows(
        manual_rows,
        manual_fields,
        cluster_rows,
        cluster_fields,
        node_mode="exact",
    )

    assert {item.node for item in evaluation.joined} == {
        "callable:pkg.mod.fn",
        "data:pkg.Service:state",
        "data:pkg.Service:composite",
    }
    assert evaluation.raw_manual_row_count == 4
    assert len(evaluation.manual_rows) == 3
    assert payload["metadata"]["manual_rows"] == 4
    assert payload["metadata"]["scoped_manual_rows"] == 3
    assert payload["metadata"]["evaluation_node_types"] == ["callable"]
    assert payload["metadata"]["evaluation_kind_tokens"] == ["class_attr_state"]


def test_callable_only_scope_uses_empty_kind_tokens():
    fields = ["Microservice_id", "node", "node_type", "label", "kind", "module"]
    cluster_fields = ["cluster_id", "node", "node_type", "label", "kind", "module"]
    manual_rows = [
        {
            "Microservice_id": "service-a",
            "node": "callable:pkg.mod.fn",
            "node_type": "callable",
            "label": "fn",
            "kind": "function",
            "module": "pkg.mod",
        },
        {
            "Microservice_id": "service-a",
            "node": "data:pkg.Service:state",
            "node_type": "data",
            "label": "state",
            "kind": "class_attr_state",
            "module": "",
        },
    ]
    cluster_rows = [{**row, "cluster_id": "C001"} for row in manual_rows]

    evaluation = build_evaluation_input_from_rows(
        manual_rows,
        fields,
        cluster_rows,
        cluster_fields,
        node_mode="exact",
        evaluation_kind_tokens=(),
    )

    assert [item.node for item in evaluation.joined] == ["callable:pkg.mod.fn"]
    assert evaluation.evaluation_kind_tokens == ()


def test_all_evaluation_nodes_preserves_unfiltered_exact_join_behavior():
    fields = ["Microservice_id", "node", "node_type", "label", "kind", "module"]
    cluster_fields = ["cluster_id", "node", "node_type", "label", "kind", "module"]
    manual_rows = [
        {
            "Microservice_id": "service-a",
            "node": "callable:pkg.mod.fn",
            "node_type": "callable",
            "label": "fn",
            "kind": "function",
            "module": "pkg.mod",
        },
        {
            "Microservice_id": "service-a",
            "node": "data:pkg.fn:local",
            "node_type": "data",
            "label": "local",
            "kind": "local_exposed",
            "module": "",
        },
    ]
    cluster_rows = [{**row, "cluster_id": "C001"} for row in manual_rows]

    evaluation = build_evaluation_input_from_rows(
        manual_rows,
        fields,
        cluster_rows,
        cluster_fields,
        node_mode="exact",
        all_evaluation_nodes=True,
    )

    assert {item.node for item in evaluation.joined} == {
        "callable:pkg.mod.fn",
        "data:pkg.fn:local",
    }
    assert evaluation.all_evaluation_nodes is True
