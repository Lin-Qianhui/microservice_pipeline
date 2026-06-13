import argparse
import json
import sys
import types
from pathlib import Path

import pytest

from microservice_pipeline.cluster_structural_graph import (
    ClusterOptions,
    Edge,
    LOCAL_CALLABLE_MUST_LINK_RELATION,
    SweepBestSelectionOptions,
    _cluster_stats_for_sweep,
    _default_sweep_hac_n_clusters,
    _default_sweep_markov_times,
    _default_sweep_resolutions,
    _parse_sweep_range,
    _parse_sweep_resolutions,
    _resolve_clustering_weight_scales,
    build_contracted_edge_views,
    callable_hub_decisions_from_json,
    cluster_structural_graph,
    local_callable_must_link_pairs,
    local_callable_parent_id,
    load_structural_config_defaults,
    materialize_sweep_best_cluster,
    parse_args,
    run_parameter_sweep,
    scale_edges_for_clustering,
    select_sweep_best_row,
    structural_config_defaults,
    sweep_options_from_row,
    write_outputs,
    write_sweep_outputs,
)
from microservice_pipeline.jsonc_config import load_jsonc, loads_jsonc
from microservice_pipeline.structural_dependency_graph.clustering.common import StructuralClusteringInput
from microservice_pipeline.structural_dependency_graph.clustering.postprocess import (
    postprocess_data_only_clusters,
)
from microservice_pipeline.structural_dependency_graph.clustering.registry import (
    algorithm_choices,
    cluster_with_algorithm,
)
from microservice_pipeline.structural_dependency_graph.weight_config import load_weight_config


def _callable(name):
    node = f"callable:sample.{name}"
    return node, {
        "id": node,
        "node_type": "callable",
        "label": name,
        "kind": "function",
        "module": "sample",
        "qualname": name,
        "class_name": "",
        "display_name": name,
        "scope": "",
        "owner": "",
        "file": "sample.py",
        "lineno": "1",
    }


def _module_callable(module="sample"):
    node = f"callable:{module}.<module>"
    return node, {
        "id": node,
        "node_type": "callable",
        "label": "<module>",
        "kind": "module",
        "module": module,
        "qualname": "<module>",
        "class_name": "",
        "display_name": "<module>",
        "scope": "",
        "owner": "",
        "file": f"{module}.py",
        "lineno": "1",
    }


def _data(object_id, kind="local_exposed", callable_count=1, access_count=1):
    node = f"data:{object_id}"
    return node, {
        "id": node,
        "node_type": "data",
        "label": object_id.rsplit(":", 1)[-1],
        "kind": kind,
        "module": "",
        "qualname": "",
        "class_name": "",
        "display_name": object_id.rsplit(":", 1)[-1],
        "scope": "callable" if "local_exposed" in kind else "class",
        "owner": "sample.Service",
        "file": "sample.py",
        "lineno": "2",
        "raw_object_count": "1",
        "callable_count": str(callable_count),
        "access_count": str(access_count),
    }


def _edge(src, dst, edge_type="data_access", relation="", access="", weight=1.0):
    return Edge(
        src=src,
        dst=dst,
        edge_type=edge_type,
        relation=relation,
        access=access,
        operation="load" if access == "read" else "",
        weight=weight,
    )


def _install_fake_sklearn(monkeypatch):
    class FakeAgglomerativeClustering:
        calls = []

        def __init__(self, n_clusters, metric=None, affinity=None, linkage=None):
            self.n_clusters = n_clusters
            self.metric = metric
            self.affinity = affinity
            self.linkage = linkage
            self.fit_matrix = None
            FakeAgglomerativeClustering.calls.append(self)

        def fit_predict(self, distance_matrix):
            self.fit_matrix = distance_matrix
            groups = [{idx} for idx in range(len(distance_matrix))]
            while len(groups) > self.n_clusters:
                best = None
                for left_idx in range(len(groups)):
                    for right_idx in range(left_idx + 1, len(groups)):
                        distances = [
                            distance_matrix[left][right]
                            for left in groups[left_idx]
                            for right in groups[right_idx]
                        ]
                        average = sum(distances) / len(distances)
                        candidate = (
                            average,
                            min(groups[left_idx]),
                            min(groups[right_idx]),
                            left_idx,
                            right_idx,
                        )
                        if best is None or candidate < best:
                            best = candidate
                _average, _left_min, _right_min, left_idx, right_idx = best
                groups[left_idx] = groups[left_idx] | groups[right_idx]
                groups.pop(right_idx)

            labels = [0] * len(distance_matrix)
            for label, group in enumerate(sorted(groups, key=lambda values: min(values))):
                for idx in group:
                    labels[idx] = label
            return labels

    sklearn_module = types.ModuleType("sklearn")
    cluster_module = types.ModuleType("sklearn.cluster")
    cluster_module.AgglomerativeClustering = FakeAgglomerativeClustering
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", cluster_module)
    return FakeAgglomerativeClustering


def test_jsonc_loader_supports_comments_and_trailing_commas(tmp_path):
    config_path = tmp_path / "config.jsonc"
    config_path.write_text(
        """
        {
          // Line comment
          "path": "src//not-a-comment",
          "nested": {
            "values": [1, 2, 3,],
          },
          /*
            Block comment
          */
          "enabled": true,
        }
        """,
        encoding="utf-8",
    )

    payload = load_jsonc(config_path)

    assert payload == {
        "path": "src//not-a-comment",
        "nested": {"values": [1, 2, 3]},
        "enabled": True,
    }


def test_jsonc_loader_reports_path_for_invalid_config(tmp_path):
    config_path = tmp_path / "bad.jsonc"
    config_path.write_text('{"missing": }', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid JSONC config .*bad\.jsonc"):
        load_jsonc(config_path)


def test_structural_config_defaults_resolve_paths_and_sections(tmp_path):
    config = loads_jsonc(
        """
        {
          "paths": {
            "nodes": "graph/nodes.csv",
            "edges": "graph/edges.csv",
            "outdir": "clusters",
            "sweep_outdir": "sweep",
            "manual_mapping": "manual.csv",
          },
          "algorithm": {
            "algorithm": "leiden",
            "resolution": 0.2,
            "seed": 7,
          },
          "sweep": {
            "run_sweep": true,
            "range": "0.1:0.3:0.1",
            "evaluation_node_types": ["callable", "data"],
            "na_labels": ["unknown"],
            "all_evaluation_nodes": true,
          },
          "sweep_best": {
            "enabled": true,
            "metric": "coupling",
            "metric_direction": "min",
            "data_hub_policy": "keep_data_hubs",
            "min_metric": "best_match_macro_f1",
            "min_value": 0.4,
          },
          "hub_policy": {
            "drop_data_hubs": true,
            "drop_callable_hub": ["callable:sample.run"],
          },
          "weighting": {
            "call_weight_scale": 2.0,
          },
        }
        """,
        path="test_config",
    )

    defaults = structural_config_defaults(config, tmp_path)

    assert defaults["nodes"] == str((tmp_path / "graph/nodes.csv").resolve())
    assert defaults["edges"] == str((tmp_path / "graph/edges.csv").resolve())
    assert defaults["outdir"] == str((tmp_path / "clusters").resolve())
    assert defaults["sweep_outdir"] == str((tmp_path / "sweep").resolve())
    assert defaults["sweep_manual"] == str((tmp_path / "manual.csv").resolve())
    assert defaults["algorithm"] == "leiden"
    assert defaults["resolution"] == 0.2
    assert defaults["seed"] == 7
    assert defaults["run_sweep"] is True
    assert defaults["sweep_range"] == "0.1:0.3:0.1"
    assert defaults["sweep_evaluation_node_types"] == "callable,data"
    assert defaults["sweep_na_label"] == ["unknown"]
    assert defaults["sweep_all_evaluation_nodes"] is True
    assert defaults["select_sweep_best"] is True
    assert defaults["sweep_best_metric"] == "coupling"
    assert defaults["sweep_best_metric_direction"] == "min"
    assert defaults["sweep_best_data_hub_policy"] == "keep_data_hubs"
    assert defaults["sweep_best_min_metric"] == "best_match_macro_f1"
    assert defaults["sweep_best_min_value"] == 0.4
    assert defaults["drop_data_hubs"] is True
    assert defaults["drop_callable_hub"] == ["callable:sample.run"]
    assert defaults["call_weight_scale"] == 2.0


def test_load_structural_config_defaults_reads_jsonc_file(tmp_path):
    config_path = tmp_path / "structural.jsonc"
    config_path.write_text(
        """
        {
          // Comments and trailing commas are allowed.
          "algorithm": {
            "resolution": 0.35,
          },
        }
        """,
        encoding="utf-8",
    )

    defaults = load_structural_config_defaults(config_path, tmp_path)

    assert defaults["resolution"] == 0.35


def test_parse_args_uses_structural_config_and_cli_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "structural.jsonc"
    config_path.write_text(
        """
        {
          "paths": {
            "nodes": "graph/nodes.csv",
            "outdir": "configured_clusters",
          },
          "algorithm": {
            "algorithm": "leiden",
            "resolution": 0.2,
          },
          "sweep": {
            "run_sweep": true,
            "range": "0.1:0.3:0.1",
          },
          "sweep_best": {
            "enabled": true,
            "metric": "coupling",
          },
          "hub_policy": {
            "drop_data_hubs": true,
            "callable_hub_decisions": null,
          },
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "cluster_structural_graph.py",
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--resolution",
            "0.8",
            "--evaluation-node-mode",
            "callable",
            "--evaluation-na-label",
            "unknown",
            "--no-select-sweep-best",
        ],
    )

    args = parse_args()

    assert args.nodes == str((tmp_path / "graph/nodes.csv").resolve())
    assert args.outdir == str((tmp_path / "configured_clusters").resolve())
    assert args.run_sweep is True
    assert args.sweep_range == "0.1:0.3:0.1"
    assert args.sweep_best_metric == "coupling"
    assert args.drop_data_hubs is True
    assert args.callable_hub_decisions is None
    assert args.resolution == 0.8
    assert args.select_sweep_best is False


def test_parse_args_expands_algorithm_placeholder_in_output_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "structural.jsonc"
    config_path.write_text(
        """
        {
          "paths": {
            "outdir": "clusters_{algorithm}",
            "sweep_outdir": "sweep_${algorithm}",
          },
          "algorithm": {
            "algorithm": "leiden",
          },
          "sweep_best": {
            "outdir": "best_{algorithm}",
          },
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "cluster_structural_graph.py",
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--algorithm",
            "infomap",
        ],
    )

    args = parse_args()

    assert args.outdir == str((tmp_path / "clusters_infomap").resolve())
    assert args.sweep_outdir == str((tmp_path / "sweep_infomap").resolve())
    assert args.sweep_best_outdir == str((tmp_path / "best_infomap").resolve())



def test_structural_algorithm_registry_exposes_cli_choices_and_errors(monkeypatch):
    assert algorithm_choices() == (
        "leiden",
        "leiden_reweighted",
        "leiden_multiplex",
        "infomap",
        "label_propagation",
        "hac_callable_projection",
    )
    for algorithm in algorithm_choices():
        monkeypatch.setattr("sys.argv", ["cluster_structural_graph.py", "--algorithm", algorithm])
        assert parse_args().algorithm == algorithm

    monkeypatch.setattr("sys.argv", ["cluster_structural_graph.py", "--exclude-module-callables"])
    assert parse_args().exclude_module_callables is True
    monkeypatch.setattr("sys.argv", ["cluster_structural_graph.py", "--no-exclude-module-callables"])
    assert parse_args().exclude_module_callables is False
    monkeypatch.setattr(
        "sys.argv",
        [
            "cluster_structural_graph.py",
            "--algorithm",
            "hac_callable_projection",
            "--hac-n-clusters",
            "7",
            "--sweep-hac-n-clusters",
            "10,11",
        ],
    )
    hac_args = parse_args()
    assert hac_args.algorithm == "hac_callable_projection"
    assert hac_args.hac_n_clusters == 7
    assert hac_args.sweep_hac_n_clusters == "10,11"

    monkeypatch.setattr(
        "sys.argv",
        [
            "cluster_structural_graph.py",
            "--run-sweep",
            "--no-select-sweep-best",
            "--sweep-best-metric",
            "coupling",
            "--sweep-best-metric-direction",
            "min",
            "--sweep-best-resolution",
            "0.2",
            "--sweep-best-data-hub-policy",
            "keep_data_hubs",
            "--sweep-best-min-metric",
            "best_match_macro_f1",
            "--sweep-best-min-value",
            "0.4",
            "--sweep-best-outdir",
            "artifacts/sweep/best",
        ],
    )
    sweep_best_args = parse_args()
    assert sweep_best_args.select_sweep_best is False
    assert sweep_best_args.sweep_best_metric == "coupling"
    assert sweep_best_args.sweep_best_metric_direction == "min"
    assert sweep_best_args.sweep_best_resolution == 0.2
    assert sweep_best_args.sweep_best_data_hub_policy == "keep_data_hubs"
    assert sweep_best_args.sweep_best_min_metric == "best_match_macro_f1"
    assert sweep_best_args.sweep_best_min_value == 0.4
    assert sweep_best_args.sweep_best_outdir == "artifacts/sweep/best"

    monkeypatch.setattr(
        "sys.argv",
        [
            "cluster_structural_graph.py",
            "--callable-hub-policy",
            "drop-configured",
            "--callable-hub-decisions",
            "hub_decisions.json",
            "--callable-hub-nodes",
            "callable_hub_nodes.csv",
            "--data-hub-nodes",
            "data_hub_nodes.csv",
            "--drop-callable-hub",
            "callable:sample.run",
            "--keep-callable-hub",
            "callable:sample.factory",
            "--hub-orchestrator-min-out-degree",
            "8",
            "--hub-callable-min-in-degree",
            "3",
            "--hub-entrypoint-min-out-degree",
            "9",
            "--hub-orchestrator-min-data-to-call-ratio",
            "0.5",
        ],
    )
    args = parse_args()
    assert args.callable_hub_policy == "drop-configured"
    assert args.callable_hub_decisions == Path("hub_decisions.json")
    assert args.callable_hub_nodes == Path("callable_hub_nodes.csv")
    assert args.data_hub_nodes == Path("data_hub_nodes.csv")
    assert args.drop_callable_hub == ["callable:sample.run"]
    assert args.keep_callable_hub == ["callable:sample.factory"]
    assert args.hub_orchestrator_min_out_degree == 8
    assert args.hub_callable_min_in_degree == 3
    assert args.hub_entrypoint_min_out_degree == 9
    assert args.hub_orchestrator_min_data_to_call_ratio == 0.5

    input_data = StructuralClusteringInput(
        supernodes=set(),
        directed_edges={},
        undirected_edges={},
        members_of={},
        seed=42,
        resolution=1.0,
        markov_time=1.0,
        max_iter=100,
        leiden_quality="rb_configuration",
    )
    with pytest.raises(ValueError, match="Unsupported algorithm: not_real"):
        cluster_with_algorithm("not_real", input_data)


def test_hac_callable_projection_uses_sklearn_precomputed_average(monkeypatch):
    fake_sklearn = _install_fake_sklearn(monkeypatch)
    callable_a, _row_a = _callable("a")
    callable_b, _row_b = _callable("b")
    callable_c, _row_c = _callable("c")
    input_data = StructuralClusteringInput(
        supernodes={callable_a, callable_b, callable_c},
        directed_edges={},
        undirected_edges={},
        members_of={
            callable_a: [callable_a],
            callable_b: [callable_b],
            callable_c: [callable_c],
        },
        seed=42,
        resolution=1.0,
        markov_time=1.0,
        max_iter=100,
        leiden_quality="rb_configuration",
        hac_n_clusters=2,
        typed_undirected_edges={
            "call": {
                tuple(sorted((callable_a, callable_b))): 2.0,
            }
        },
        edge_type_layer_weights={"call": 2.0, "data_access": 1.0},
    )

    result = cluster_with_algorithm("hac_callable_projection", input_data)

    assert fake_sklearn.calls
    call = fake_sklearn.calls[-1]
    assert call.metric == "precomputed"
    assert call.linkage == "average"
    assert call.fit_matrix[0][0] == 0.0
    assert call.fit_matrix[0][1] < call.fit_matrix[0][2]
    assert result[callable_a] == result[callable_b]
    assert result[callable_c] != result[callable_a]


def test_hac_shared_read_data_clusters_callables_but_excludes_read_only_data(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    callable_a, row_a = _callable("a")
    callable_b, row_b = _callable("b")
    callable_c, row_c = _callable("c")
    shared_data, shared_row = _data("local_exposed:sample.shared:data")
    c_data, c_data_row = _data("local_exposed:sample.c:data")
    nodes = {
        callable_a: row_a,
        callable_b: row_b,
        callable_c: row_c,
        shared_data: shared_row,
        c_data: c_data_row,
    }
    edges = [
        _edge(callable_a, shared_data, access="read", weight=4.0),
        _edge(callable_b, shared_data, access="read", weight=4.0),
        _edge(callable_c, c_data, access="read", weight=4.0),
    ]

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="hac_callable_projection",
            hac_n_clusters=2,
            run_sweep=False,
            exclude_module_callables=False,
            drop_callable_hubs=False,
        ),
    )

    assert result.cluster_of[callable_a] == result.cluster_of[callable_b]
    assert result.cluster_of[callable_c] != result.cluster_of[callable_a]
    assert shared_data not in result.cluster_of
    assert c_data not in result.cluster_of
    excluded = {row["node"]: row["reason"] for row in result.excluded_nodes}
    assert excluded[shared_data] == "read_only_data"
    assert excluded[c_data] == "read_only_data"


def test_hac_direct_call_evidence_clusters_callables(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    callable_a, row_a = _callable("a")
    callable_b, row_b = _callable("b")
    callable_c, row_c = _callable("c")
    c_data, c_data_row = _data("local_exposed:sample.c:data")
    nodes = {
        callable_a: row_a,
        callable_b: row_b,
        callable_c: row_c,
        c_data: c_data_row,
    }
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct", weight=4.0),
        _edge(callable_c, c_data, access="write", weight=4.0),
    ]

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="hac_callable_projection",
            hac_n_clusters=2,
            run_sweep=False,
            exclude_module_callables=False,
            drop_callable_hubs=False,
        ),
    )

    assert result.cluster_of[callable_a] == result.cluster_of[callable_b]
    assert result.cluster_of[callable_c] != result.cluster_of[callable_a]
    assert result.cluster_of[c_data] == result.cluster_of[callable_c]


def test_hac_assigns_mutating_data_to_strongest_writer_cluster(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    writer, writer_row = _callable("writer")
    other, other_row = _callable("other")
    owned_data, owned_row = _data("class_attr_state:sample.Service:state", kind="class_attr_state")
    read_data, read_row = _data("local_exposed:sample.reader:input")
    nodes = {
        writer: writer_row,
        other: other_row,
        owned_data: owned_row,
        read_data: read_row,
    }
    edges = [
        _edge(writer, owned_data, access="write", weight=5.0),
        _edge(other, owned_data, access="write", weight=1.0),
        _edge(other, read_data, access="read", weight=5.0),
    ]

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="hac_callable_projection",
            hac_n_clusters=2,
            run_sweep=False,
            exclude_module_callables=False,
            drop_callable_hubs=False,
        ),
    )

    assert result.cluster_of[owned_data] == result.cluster_of[writer]
    assert read_data not in result.cluster_of
    excluded = {row["node"]: row["reason"] for row in result.excluded_nodes}
    assert excluded[read_data] == "read_only_data"


def test_hac_excludes_ambiguous_mutating_data(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    writer_a, row_a = _callable("writer_a")
    writer_b, row_b = _callable("writer_b")
    shared_data, shared_row = _data("class_attr_state:sample.Service:shared", kind="class_attr_state")
    nodes = {
        writer_a: row_a,
        writer_b: row_b,
        shared_data: shared_row,
    }
    edges = [
        _edge(writer_a, shared_data, access="write", weight=3.0),
        _edge(writer_b, shared_data, access="create", weight=3.0),
    ]

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="hac_callable_projection",
            hac_n_clusters=2,
            run_sweep=False,
            exclude_module_callables=False,
            drop_callable_hubs=False,
        ),
    )

    assert shared_data not in result.cluster_of
    excluded = {row["node"]: row["reason"] for row in result.excluded_nodes}
    assert excluded[shared_data] == "ambiguous_mutating_data"


def test_hac_preserves_data_lineage_must_link_groups_during_assignment(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    writer, writer_row = _callable("writer")
    data_a, row_a = _data("class_attr_state:sample.Service:state", kind="class_attr_state")
    data_b, row_b = _data("local_exposed:sample.writer:derived")
    nodes = {writer: writer_row, data_a: row_a, data_b: row_b}
    edges = [
        _edge(writer, data_a, access="write", weight=3.0),
        _edge(data_a, data_b, edge_type="data_lineage", relation="local_assign", weight=1.0),
    ]

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="hac_callable_projection",
            hac_n_clusters=1,
            run_sweep=False,
            exclude_module_callables=False,
            drop_callable_hubs=False,
        ),
    )

    assert result.cluster_of[data_a] == result.cluster_of[writer]
    assert result.cluster_of[data_b] == result.cluster_of[writer]
    assert any(data_a in row["members"] and data_b in row["members"] for row in result.must_link_groups)


def test_hac_tie_output_is_deterministic(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    nodes = {}
    edges = []
    for name in ("a", "b", "c", "d"):
        callable_node, callable_row = _callable(name)
        data_node, data_row = _data(f"local_exposed:sample.{name}:data")
        nodes[callable_node] = callable_row
        nodes[data_node] = data_row
        edges.append(_edge(callable_node, data_node, access="write", weight=1.0))
    options = ClusterOptions(
        algorithm="hac_callable_projection",
        hac_n_clusters=2,
        run_sweep=False,
        exclude_module_callables=False,
        drop_callable_hubs=False,
    )

    first = cluster_structural_graph(nodes, edges, options)
    second = cluster_structural_graph(nodes, edges, options)

    assert first.cluster_of == second.cluster_of


def test_data_only_cluster_postprocess_reassigns_mutated_data_and_removes_read_only():
    writer, _writer_row = _callable("writer")
    reader, _reader_row = _callable("reader")
    created_data, _created_row = _data("sample.writer:state")
    read_data, _read_row = _data("sample.reader:input")
    cluster_of = {
        writer: "C001",
        reader: "C002",
        created_data: "C003",
        read_data: "C003",
    }
    edges = [
        _edge(writer, created_data, access="write", weight=2.0),
        _edge(reader, read_data, access="read", weight=5.0),
    ]

    result = postprocess_data_only_clusters(cluster_of, edges)

    assert result.cluster_of[created_data] == "C001"
    assert read_data not in result.cluster_of
    assert result.reassigned_nodes == {created_data: writer}
    assert result.removed_nodes == (read_data,)


def test_cluster_structural_graph_repairs_data_only_clusters_before_outputs(monkeypatch):
    writer, writer_row = _callable("writer")
    reader, reader_row = _callable("reader")
    created_data, created_row = _data("sample.writer:state")
    read_data, read_row = _data("sample.reader:input")
    nodes = {
        writer: writer_row,
        reader: reader_row,
        created_data: created_row,
        read_data: read_row,
    }
    edges = [
        _edge(writer, created_data, access="write", weight=2.0),
        _edge(reader, read_data, access="read", weight=2.0),
    ]

    def fake_cluster_supernodes(supernodes, *_args, **_kwargs):
        return {
            node: ("data-only" if node in {created_data, read_data} else node)
            for node in supernodes
        }

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_supernodes",
        fake_cluster_supernodes,
    )

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(exclude_module_callables=False),
    )

    assert created_data in result.cluster_of
    assert result.cluster_of[created_data] == result.cluster_of[writer]
    assert read_data not in result.cluster_of
    assert {
        row["node"]: row["reason"]
        for row in result.excluded_nodes
        if row["node"] == read_data
    } == {read_data: "data_only_read_only"}
    assert all(
        row["callable_count"] != 0 or row["data_count"] != row["size"]
        for row in result.cluster_summary
    )


def test_callable_hub_decision_json_supports_flat_and_nested_shapes(tmp_path):
    flat_path = tmp_path / "flat.json"
    flat_path.write_text(
        json.dumps(
            {
                "drop": ["callable:sample.run"],
                "keep": ["callable:sample.factory"],
            }
        ),
        encoding="utf-8",
    )
    nested_path = tmp_path / "nested.json"
    nested_path.write_text(
        json.dumps(
            {
                "callable_hubs": {
                    "drop": ["callable:sample.process_all"],
                    "keep": [],
                }
            }
        ),
        encoding="utf-8",
    )

    assert callable_hub_decisions_from_json(flat_path) == (
        ("callable:sample.run",),
        ("callable:sample.factory",),
    )
    assert callable_hub_decisions_from_json(nested_path) == (
        ("callable:sample.process_all",),
        tuple(),
    )


def test_weight_scales_are_applied_by_edge_type():
    callable_a, _row_a = _callable("a")
    callable_b, _row_b = _callable("b")
    data_a, _row_data_a = _data("local_exposed:sample.a:data")
    data_b, _row_data_b = _data("local_exposed:sample.b:data")
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct", weight=2.0),
        _edge(callable_a, data_a, edge_type="data_access", access="read", weight=3.0),
        _edge(data_a, data_b, edge_type="data_lineage", relation="local_assign", weight=5.0),
    ]
    options = ClusterOptions(
        call_weight_scale=2.0,
        data_access_weight_scale=0.5,
        data_lineage_weight_scale=3.0,
    )

    scaled = scale_edges_for_clustering(edges, options)

    assert [edge.weight for edge in scaled] == [4.0, 1.5, 15.0]
    assert [edge.weight for edge in edges] == [2.0, 3.0, 5.0]


def test_multiplex_preserves_edge_weights_and_uses_layer_weights(monkeypatch):
    callable_a, row_a = _callable("a")
    callable_b, row_b = _callable("b")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    data_b, row_data_b = _data("local_exposed:sample.b:data")
    nodes = {
        callable_a: row_a,
        callable_b: row_b,
        data_a: row_data_a,
        data_b: row_data_b,
    }
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct", weight=2.0),
        _edge(callable_a, data_a, edge_type="data_access", access="read", weight=3.0),
        _edge(data_a, data_b, edge_type="data_lineage", relation="arg_to_param", weight=5.0),
    ]
    options = ClusterOptions(
        algorithm="leiden_multiplex",
        call_weight_scale=4.0,
        data_access_weight_scale=5.0,
        data_lineage_weight_scale=6.0,
        call_resolution=0.8,
        data_access_resolution=1.2,
        data_lineage_resolution=1.5,
        run_sweep=False,
        drop_callable_hubs=False,
    )

    scaled = scale_edges_for_clustering(edges, options)
    assert [edge.weight for edge in scaled] == [2.0, 3.0, 5.0]

    def fake_cluster_with_algorithm(algorithm, input_data):
        assert algorithm == "leiden_multiplex"
        assert input_data.edge_type_layer_weights == {
            "call": 4.0,
            "data_access": 5.0,
            "data_lineage": 6.0,
        }
        assert input_data.edge_type_layer_resolutions == {
            "call": 0.8,
            "data_access": 1.2,
            "data_lineage": 1.5,
        }
        assert input_data.typed_undirected_edges["call"] == {
            tuple(sorted((callable_a, callable_b))): 2.0,
        }
        assert input_data.typed_undirected_edges["data_access"] == {
            tuple(sorted((callable_a, data_a))): 3.0,
        }
        assert input_data.typed_undirected_edges["data_lineage"] == {
            tuple(sorted((data_a, data_b))): 5.0,
        }
        return {node: "C001" for node in input_data.supernodes}

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    result = cluster_structural_graph(nodes, edges, options)

    assert set(result.cluster_of) == set(nodes)


def test_multiplex_call_data_mode_combines_data_layers(monkeypatch):
    callable_a, row_a = _callable("a")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    data_b, row_data_b = _data("local_exposed:sample.b:data")
    nodes = {callable_a: row_a, data_a: row_data_a, data_b: row_data_b}
    edges = [
        _edge(callable_a, data_a, edge_type="data_access", access="read", weight=3.0),
        _edge(data_a, data_b, edge_type="data_lineage", relation="arg_to_param", weight=5.0),
    ]
    options = ClusterOptions(
        algorithm="leiden_multiplex",
        multiplex_layer_mode="call_data",
        call_weight_scale=4.0,
        data_access_weight_scale=2.0,
        data_lineage_weight_scale=3.0,
        call_resolution=0.5,
        data_access_resolution=0.75,
        run_sweep=False,
        drop_callable_hubs=False,
    )

    def fake_cluster_with_algorithm(algorithm, input_data):
        assert algorithm == "leiden_multiplex"
        assert input_data.edge_type_layer_weights == {"call": 4.0, "data": 1.0}
        assert input_data.edge_type_layer_resolutions == {
            "call": 0.5,
            "data": 0.75,
        }
        assert set(input_data.typed_undirected_edges) == {"data"}
        assert input_data.typed_undirected_edges["data"] == {
            tuple(sorted((callable_a, data_a))): 6.0,
            tuple(sorted((data_a, data_b))): 15.0,
        }
        return {node: "C001" for node in input_data.supernodes}

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    result = cluster_structural_graph(nodes, edges, options)

    assert set(result.cluster_of) == set(nodes)


def test_contracted_edges_preserve_separate_edge_type_layers():
    callable_a, _row_a = _callable("a")
    callable_b, _row_b = _callable("b")
    data_a, _row_data_a = _data("local_exposed:sample.a:data")
    data_b, _row_data_b = _data("local_exposed:sample.b:data")
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct", weight=2.0),
        _edge(callable_a, data_a, edge_type="data_access", access="read", weight=3.0),
        _edge(data_a, data_b, edge_type="data_lineage", relation="arg_to_param", weight=5.0),
    ]
    active_nodes = {callable_a, callable_b, data_a, data_b}
    supernode_of = {node: node for node in active_nodes}

    directed, undirected, typed = build_contracted_edge_views(edges, active_nodes, supernode_of)

    assert directed[(callable_a, callable_b)] == 2.0
    assert undirected[tuple(sorted((callable_a, data_a)))] == 3.0
    assert typed["call"] == {tuple(sorted((callable_a, callable_b))): 2.0}
    assert typed["data_access"] == {tuple(sorted((callable_a, data_a))): 3.0}
    assert typed["data_lineage"] == {tuple(sorted((data_a, data_b))): 5.0}


def test_weight_config_scales_are_used_and_cli_scale_flags_override(tmp_path):
    config_path = tmp_path / "weights.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "scale-test",
                "clustering": {
                    "edge_type_scales": {
                        "call": 4.0,
                        "data_access": 5.0,
                        "data_lineage": 6.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    weight_config = load_weight_config(config_path)

    args = argparse.Namespace(
        call_weight_scale=None,
        data_access_weight_scale=None,
        data_lineage_weight_scale=None,
    )
    assert _resolve_clustering_weight_scales(args, weight_config) == (4.0, 5.0, 6.0)

    args = argparse.Namespace(
        call_weight_scale=7.0,
        data_access_weight_scale=None,
        data_lineage_weight_scale=8.0,
    )
    assert _resolve_clustering_weight_scales(args, weight_config) == (7.0, 5.0, 8.0)

    callable_a, _row_a = _callable("a")
    callable_b, _row_b = _callable("b")
    data_a, _row_data_a = _data("local_exposed:sample.a:data")
    data_b, _row_data_b = _data("local_exposed:sample.b:data")
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct", weight=2.0),
        _edge(callable_a, data_a, edge_type="data_access", access="read", weight=3.0),
        _edge(data_a, data_b, edge_type="data_lineage", relation="local_assign", weight=5.0),
    ]
    options = ClusterOptions(
        call_weight_scale=4.0,
        data_access_weight_scale=5.0,
        data_lineage_weight_scale=6.0,
        weight_config=weight_config.to_dict(),
    )

    scaled = scale_edges_for_clustering(edges, options)

    assert [edge.weight for edge in scaled] == [8.0, 15.0, 30.0]
    assert options.weight_config["name"] == "scale-test"


def test_leiden_reweighted_preserves_edge_weight_and_applies_bounded_ownership_boost():
    callable_a, _row_a = _callable("a")
    callable_b, _row_b = _callable("b")
    owned_data, _row_owned_data = _data("local_exposed:sample.a:owned")
    shared_data, _row_shared_data = _data("local_exposed:sample.shared:data")
    state_data, _row_state_data = _data("class_attr_state:sample.Service:state")
    param_data, _row_param_data = _data("param:sample.consumer:state", kind="param")
    weight_config = load_weight_config("builtin:ownership_biased")
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct", weight=2.0),
        _edge(callable_a, owned_data, access="read", weight=3.0),
        _edge(callable_a, owned_data, access="write", weight=5.0),
        _edge(callable_a, shared_data, access="write", weight=5.0),
        _edge(callable_b, shared_data, access="create", weight=7.0),
        _edge(
            state_data,
            param_data,
            edge_type="data_lineage",
            relation="arg_to_param",
            weight=10.0,
        ),
    ]
    options = ClusterOptions(
        algorithm="leiden_reweighted",
        weight_config=weight_config.to_dict(),
    )

    scaled = scale_edges_for_clustering(edges, options)

    assert [edge.weight for edge in scaled] == pytest.approx(
        [
            4.0,
            3.0,
            8.0,
            6.6666667,
            9.8,
            5.0,
        ]
    )


def test_leiden_reweighted_must_links_single_writer_data_to_callable(monkeypatch):
    writer, writer_row = _callable("writer")
    reader, reader_row = _callable("reader")
    owned_data, owned_data_row = _data("local_exposed:sample.writer:owned")
    observed_data, observed_data_row = _data("local_exposed:sample.writer:observed")
    shared_data, shared_data_row = _data("local_exposed:sample.shared:data")
    nodes = {
        writer: writer_row,
        reader: reader_row,
        owned_data: owned_data_row,
        observed_data: observed_data_row,
        shared_data: shared_data_row,
    }
    edges = [
        _edge(reader, writer, edge_type="call", relation="direct"),
        _edge(writer, owned_data, access="write"),
        _edge(writer, owned_data, access="read_write"),
        _edge(writer, observed_data, access="write"),
        _edge(reader, observed_data, access="read"),
        _edge(writer, shared_data, access="write"),
        _edge(reader, shared_data, access="create"),
    ]

    def fake_cluster_with_algorithm(_algorithm, input_data):
        grouped_members = [set(members) for members in input_data.members_of.values()]
        assert any({writer, owned_data}.issubset(members) for members in grouped_members)
        assert not any({writer, observed_data}.issubset(members) for members in grouped_members)
        assert not any({writer, shared_data}.issubset(members) for members in grouped_members)

        assignments = {}
        for supernode, members in input_data.members_of.items():
            member_set = set(members)
            if writer in member_set:
                assignments[supernode] = "writer-cluster"
            elif reader in member_set or member_set & {owned_data, observed_data, shared_data}:
                assignments[supernode] = "reader-cluster"
            else:
                assignments[supernode] = supernode
        return assignments

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="leiden_reweighted",
            run_sweep=False,
            exclude_module_callables=False,
        ),
    )

    assert result.cluster_of[owned_data] == result.cluster_of[writer]
    assert result.cluster_of[observed_data] == result.cluster_of[reader]
    assert result.cluster_of[shared_data] == result.cluster_of[reader]
    owned_group = next(row for row in result.must_link_groups if owned_data in row["members"])
    assert writer in owned_group["members"]
    assert owned_group["relations"] == "single_writer"


def test_leiden_reweighted_clamps_subtype_ratios(tmp_path):
    config_path = tmp_path / "weights.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "ratio-clamp-test",
                "generation": {
                    "data_access": {"weights": {"read": 3.0}},
                    "data_lineage": {"weights": {"arg_to_param": 0.01}},
                },
                "clustering": {
                    "edge_type_scales": {
                        "data_access": 1.0,
                        "data_lineage": 1.0,
                    },
                    "reweighted": {
                        "subtype_ratio_min": 0.5,
                        "subtype_ratio_max": 1.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    weight_config = load_weight_config(config_path)
    callable_a, _row_a = _callable("a")
    data_a, _row_data_a = _data("local_exposed:sample.a:data")
    data_b, _row_data_b = _data("param:sample.b:data", kind="param")
    edges = [
        _edge(callable_a, data_a, access="read", weight=2.0),
        _edge(
            data_a,
            data_b,
            edge_type="data_lineage",
            relation="arg_to_param",
            weight=2.0,
        ),
    ]
    options = ClusterOptions(
        algorithm="leiden_reweighted",
        weight_config=weight_config.to_dict(),
    )

    scaled = scale_edges_for_clustering(edges, options)

    assert [edge.weight for edge in scaled] == pytest.approx([3.0, 1.0])


def test_isolated_nodes_and_callable_hubs_are_excluded_while_data_hubs_are_kept():
    hub, hub_row = _callable("hub")
    worker, worker_row = _callable("worker")
    isolated_callable, isolated_callable_row = _callable("isolated")
    isolated_data, isolated_data_row = _data("local_exposed:sample.make:isolated_data")
    data_hub, data_hub_row = _data(
        "local_exposed:sample.make:shared",
        kind="local_exposed; object_state",
        callable_count=10,
        access_count=100,
    )
    owned_data, owned_data_row = _data("class_attr_state:sample.Service:state", kind="class_attr_state")
    leaf_data, leaf_data_row = _data("local_exposed:sample.make:leaf")

    nodes = dict(
        [
            (hub, hub_row),
            (worker, worker_row),
            (isolated_callable, isolated_callable_row),
            (isolated_data, isolated_data_row),
            (data_hub, data_hub_row),
            (owned_data, owned_data_row),
            (leaf_data, leaf_data_row),
        ]
    )
    edges = [
        _edge(hub, data_hub, access="read"),
        _edge(hub, owned_data, access="read"),
        _edge(hub, leaf_data, access="read"),
        _edge(worker, data_hub, access="read"),
        _edge(worker, owned_data, access="read"),
    ]
    options = ClusterOptions(
        algorithm="label_propagation",
        run_sweep=False,
        hub_callable_degree_percentile=0,
        hub_callable_min_degree=3,
        hub_callable_min_in_degree=0,
        hub_callable_min_out_degree=0,
        hub_data_min_degree=99,
        hub_data_min_callable_count=10,
        hub_data_min_access_count=100,
    )

    result = cluster_structural_graph(nodes, edges, options)

    excluded = {row["node"]: row["reason"] for row in result.excluded_nodes}
    assert excluded[isolated_callable] == "isolated_callable"
    assert excluded[isolated_data] == "isolated_data"
    assert excluded[hub] == "callable_hub"
    assert hub not in result.cluster_of
    assert data_hub in result.cluster_of
    assert any(row["node"] == data_hub and row["action"] == "kept" for row in result.hub_nodes)


def test_low_in_high_out_callable_is_orchestrator_candidate_without_manual_mapping():
    orchestrator, orchestrator_row = _callable("run")
    factory, factory_row = _callable("factory")
    targets = []
    for idx, module in enumerate(("alpha", "beta", "gamma", "delta"), start=1):
        target, target_row = _callable(f"target_{idx}")
        target_row["module"] = module
        targets.append((target, target_row))
    factory_data_nodes = [
        _data(f"local_exposed:sample.factory:data_{idx}")
        for idx in range(1, 6)
    ]
    orchestrator_data_nodes = [
        _data(f"local_exposed:sample.run:data_{idx}")
        for idx in range(1, 6)
    ]
    nodes = {
        orchestrator: orchestrator_row,
        factory: factory_row,
        **dict(targets),
        **dict(factory_data_nodes),
        **dict(orchestrator_data_nodes),
    }
    edges = [
        _edge(orchestrator, target, edge_type="call", relation="direct")
        for target, _row in targets
    ]
    edges.extend(
        _edge(orchestrator, data_node, access="read")
        for data_node, _row in orchestrator_data_nodes
    )
    edges.extend(
        _edge(factory, data_node, access="create")
        for data_node, _row in factory_data_nodes
    )
    options = ClusterOptions(
        algorithm="label_propagation",
        run_sweep=False,
        callable_hub_policy="drop-all",
        hub_callable_min_degree=99,
        hub_orchestrator_min_out_degree=4,
        hub_orchestrator_min_out_call_degree=4,
        hub_orchestrator_min_target_modules=3,
        hub_orchestrator_min_target_callables=99,
        hub_orchestrator_min_target_data=4,
    )

    result = cluster_structural_graph(nodes, edges, options)

    excluded = {row["node"]: row["reason"] for row in result.excluded_nodes}
    assert excluded[orchestrator] == "callable_hub"
    assert factory in result.cluster_of
    assert any(
        row["node"] == orchestrator
        and row["action"] == "dropped"
        and row["candidate_types"] == "orchestrator"
        for row in result.hub_nodes
    )
    assert all(row["node"] != factory for row in result.hub_nodes)


def test_zero_incoming_high_out_callable_is_entrypoint_hub_node():
    entrypoint, entrypoint_row = _callable("process_all")
    factory, factory_row = _callable("factory")
    caller, caller_row = _callable("caller")
    entrypoint_data = [
        _data(f"local_exposed:sample.process_all:data_{idx}")
        for idx in range(1, 4)
    ]
    factory_data = [
        _data(f"local_exposed:sample.factory:data_{idx}")
        for idx in range(1, 4)
    ]
    nodes = {
        entrypoint: entrypoint_row,
        factory: factory_row,
        caller: caller_row,
        **dict(entrypoint_data),
        **dict(factory_data),
    }
    edges = [
        _edge(entrypoint, data_node, access="read")
        for data_node, _row in entrypoint_data
    ]
    edges.extend(
        _edge(factory, data_node, access="create")
        for data_node, _row in factory_data
    )
    edges.append(_edge(caller, factory, edge_type="call", relation="direct"))
    options = ClusterOptions(
        algorithm="label_propagation",
        run_sweep=False,
        callable_hub_policy="drop-all",
        hub_callable_min_degree=99,
        hub_entrypoint_min_out_degree=3,
    )

    result = cluster_structural_graph(nodes, edges, options)

    assert entrypoint not in result.cluster_of
    assert factory in result.cluster_of
    assert any(
        row["node"] == entrypoint
        and row["action"] == "dropped"
        and row["candidate_types"] == "entrypoint"
        and "entrypoint_fanout:in=0;out>=3" in row["reasons"]
        for row in result.hub_nodes
    )
    assert all(row["node"] != factory for row in result.hub_nodes)


def test_callable_hub_policy_can_keep_drop_all_or_drop_configured_candidates():
    drop_me, drop_me_row = _callable("drop_me")
    keep_me, keep_me_row = _callable("keep_me")
    leaf_a, leaf_a_row = _callable("leaf_a")
    leaf_b, leaf_b_row = _callable("leaf_b")
    data_a, data_a_row = _data("local_exposed:sample.drop_me:data")
    data_b, data_b_row = _data("local_exposed:sample.keep_me:data")
    nodes = {
        drop_me: drop_me_row,
        keep_me: keep_me_row,
        leaf_a: leaf_a_row,
        leaf_b: leaf_b_row,
        data_a: data_a_row,
        data_b: data_b_row,
    }
    edges = [
        _edge(drop_me, leaf_a, edge_type="call", relation="direct"),
        _edge(drop_me, data_a, access="read"),
        _edge(keep_me, leaf_b, edge_type="call", relation="direct"),
        _edge(keep_me, data_b, access="read"),
    ]

    configured = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="label_propagation",
            run_sweep=False,
            drop_callable_hubs=False,
            callable_hub_policy="drop-configured",
            callable_hub_drop=(drop_me,),
            callable_hub_keep=(keep_me,),
            hub_callable_degree_percentile=0,
            hub_callable_min_degree=2,
            hub_callable_min_in_degree=0,
            hub_callable_min_out_degree=0,
        ),
    )
    excluded = {row["node"]: row["reason"] for row in configured.excluded_nodes}
    actions = {row["node"]: row["action"] for row in configured.hub_nodes}
    assert excluded[drop_me] == "callable_hub"
    assert keep_me in configured.cluster_of
    assert actions[drop_me] == "dropped"
    assert actions[keep_me] == "kept"

    keep_all = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="label_propagation",
            run_sweep=False,
            drop_callable_hubs=True,
            callable_hub_policy="keep",
            hub_callable_degree_percentile=0,
            hub_callable_min_degree=2,
            hub_callable_min_in_degree=0,
            hub_callable_min_out_degree=0,
        ),
    )
    assert drop_me in keep_all.cluster_of
    assert keep_me in keep_all.cluster_of
    assert all(row["action"] == "kept" for row in keep_all.hub_nodes)

    drop_all_with_keep_override = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="label_propagation",
            run_sweep=False,
            callable_hub_policy="drop-all",
            callable_hub_keep=(keep_me,),
            hub_callable_degree_percentile=0,
            hub_callable_min_degree=2,
            hub_callable_min_in_degree=0,
            hub_callable_min_out_degree=0,
        ),
    )
    assert drop_me not in drop_all_with_keep_override.cluster_of
    assert keep_me in drop_all_with_keep_override.cluster_of


def test_data_nodes_orphaned_by_dropped_callable_hubs_are_excluded():
    hub, hub_row = _callable("hub")
    worker, worker_row = _callable("worker")
    orphaned_data, orphaned_data_row = _data("local_exposed:sample.hub:orphaned")
    shared_data, shared_data_row = _data("local_exposed:sample.shared:data")
    nodes = {
        hub: hub_row,
        worker: worker_row,
        orphaned_data: orphaned_data_row,
        shared_data: shared_data_row,
    }
    edges = [
        _edge(hub, orphaned_data, access="read"),
        _edge(hub, shared_data, access="read"),
        _edge(worker, shared_data, access="read"),
    ]

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="label_propagation",
            run_sweep=False,
            callable_hub_policy="drop-all",
            hub_callable_degree_percentile=0,
            hub_callable_min_degree=2,
            hub_callable_min_in_degree=0,
            hub_callable_min_out_degree=0,
            hub_data_min_degree=99,
            hub_data_min_callable_count=99,
            hub_data_min_access_count=99,
        ),
    )

    excluded = {row["node"]: row["reason"] for row in result.excluded_nodes}
    assert excluded[hub] == "callable_hub"
    assert excluded[orphaned_data] == "orphaned_by_callable_hub"
    assert orphaned_data not in result.cluster_of
    assert shared_data in result.cluster_of
    assert worker in result.cluster_of


def test_clustering_uses_structural_hub_node_files_when_available(tmp_path):
    drop_me, drop_me_row = _callable("drop_me")
    keep_me, keep_me_row = _callable("keep_me")
    data_hub, data_hub_row = _data(
        "local_exposed:sample.shared:data",
        callable_count=2,
        access_count=4,
    )
    nodes = {
        drop_me: drop_me_row,
        keep_me: keep_me_row,
        data_hub: data_hub_row,
    }
    edges = [
        _edge(drop_me, data_hub, access="read"),
        _edge(keep_me, data_hub, access="read"),
    ]
    callable_hub_nodes_path = tmp_path / "callable_hub_nodes.csv"
    callable_hub_nodes_path.write_text(
        "\n".join(
            [
                "node,node_type,label,kind,candidate_types,reasons,in_degree,out_degree,total_degree,weighted_in_degree,weighted_out_degree,weighted_degree,out_call_degree,out_data_degree,target_callable_count,target_data_count,target_module_count,target_modules,file,lineno",
                f"{drop_me},callable,drop_me,function,orchestrator,from_file,0,1,1,0.000000,1.000000,1.000000,0,1,0,1,0,,sample.py,1",
                f"{keep_me},callable,keep_me,function,degree,from_file,0,1,1,0.000000,1.000000,1.000000,0,1,0,1,0,,sample.py,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data_hub_nodes_path = tmp_path / "data_hub_nodes.csv"
    data_hub_nodes_path.write_text(
        "\n".join(
            [
                "node,node_type,label,kind,reasons,in_degree,out_degree,total_degree,weighted_in_degree,weighted_out_degree,weighted_degree,callable_count,access_count,file,lineno",
                f"{data_hub},data,data,local_exposed,from_file,2,0,2,2.000000,0.000000,2.000000,2,4,sample.py,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="label_propagation",
            run_sweep=False,
            callable_hub_policy="drop-configured",
            callable_hub_drop=(drop_me,),
            callable_hub_keep=(keep_me,),
            callable_hub_nodes_path=str(callable_hub_nodes_path),
            data_hub_nodes_path=str(data_hub_nodes_path),
            drop_data_hubs=False,
            hub_callable_min_degree=99,
            hub_data_min_degree=99,
            hub_data_min_callable_count=99,
            hub_data_min_access_count=99,
        ),
    )

    actions = {row["node"]: row["action"] for row in result.hub_nodes}
    reasons = {row["node"]: row["reasons"] for row in result.hub_nodes}
    assert drop_me not in result.cluster_of
    assert keep_me in result.cluster_of
    assert data_hub in result.cluster_of
    assert actions == {drop_me: "dropped", keep_me: "kept", data_hub: "kept"}
    assert reasons[drop_me] == "from_file"
    assert reasons[data_hub] == "from_file"


def test_must_link_keeps_producer_and_state_slot_together_but_ignores_arg_to_param():
    producer, producer_row = _callable("producer")
    consumer, consumer_row = _callable("consumer")
    local_obj, local_obj_row = _data(
        "local_exposed:sample.producer:system_particle_object_list",
        kind="local_exposed; object_state",
    )
    state_obj, state_obj_row = _data(
        "class_attr_state:sample.Service:system_particle_object_list",
        kind="class_attr_state",
    )
    param_obj, param_obj_row = _data(
        "param:sample.consumer:system_particle_object_list",
        kind="param",
    )
    nodes = dict(
        [
            (producer, producer_row),
            (consumer, consumer_row),
            (local_obj, local_obj_row),
            (state_obj, state_obj_row),
            (param_obj, param_obj_row),
        ]
    )
    edges = [
        _edge(producer, local_obj, access="create"),
        _edge(local_obj, state_obj, edge_type="data_lineage", relation="state_assign"),
        _edge(state_obj, param_obj, edge_type="data_lineage", relation="arg_to_param"),
        _edge(consumer, param_obj, access="read"),
    ]
    options = ClusterOptions(algorithm="label_propagation", run_sweep=False, drop_callable_hubs=False)

    result = cluster_structural_graph(nodes, edges, options)

    assert result.cluster_of[local_obj] == result.cluster_of[state_obj]
    must_link_members = ";".join(row["members"] for row in result.must_link_groups)
    assert local_obj in must_link_members
    assert state_obj in must_link_members
    assert param_obj not in must_link_members
    assert result.must_link_groups[0]["anchor_node"] == local_obj


def test_local_callable_must_link_pairs_use_enclosing_callable():
    outer = "callable:sample.outer"
    inner = "callable:sample.outer.<locals>.inner"
    deep = "callable:sample.outer.<locals>.inner.<locals>.deep"
    orphan = "callable:sample.missing.<locals>.orphan"
    unrelated = "callable:sample.other"
    available = {outer, inner, deep, orphan, unrelated}

    assert local_callable_parent_id(inner, available) == outer
    assert local_callable_parent_id(deep, available) == inner
    assert local_callable_parent_id(orphan, available) is None
    assert local_callable_parent_id(unrelated, available) is None
    assert set(local_callable_must_link_pairs(available)) == {
        (outer, inner, LOCAL_CALLABLE_MUST_LINK_RELATION),
        (inner, deep, LOCAL_CALLABLE_MUST_LINK_RELATION),
    }


def test_local_callable_must_link_groups_keep_callbacks_with_parent(monkeypatch):
    parent, parent_row = _callable("outer")
    callback = "callable:sample.outer.<locals>.callback"
    callback_row = {
        **parent_row,
        "id": callback,
        "label": "outer.<locals>.callback",
        "qualname": "outer.<locals>.callback",
        "display_name": "outer.<locals>.callback",
        "lineno": "4",
    }
    other, other_row = _callable("other")
    parent_data, parent_data_row = _data("local_exposed:sample.outer:state")
    callback_data, callback_data_row = _data(
        "local_exposed:sample.outer.<locals>.callback:state"
    )
    other_data, other_data_row = _data("local_exposed:sample.other:state")
    nodes = {
        parent: parent_row,
        callback: callback_row,
        other: other_row,
        parent_data: parent_data_row,
        callback_data: callback_data_row,
        other_data: other_data_row,
    }

    def fake_cluster_with_algorithm(_algorithm, input_data):
        return {
            node: f"C{idx:03d}"
            for idx, node in enumerate(sorted(input_data.supernodes), start=1)
        }

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    result = cluster_structural_graph(
        nodes,
        [
            _edge(parent, parent_data, access="create"),
            _edge(callback, callback_data, access="create"),
            _edge(other, other_data, access="create"),
        ],
        ClusterOptions(algorithm="label_propagation", run_sweep=False, drop_callable_hubs=False),
    )

    assert result.cluster_of[parent] == result.cluster_of[callback]
    assert result.cluster_of[other] != result.cluster_of[parent]
    must_link_group = next(row for row in result.must_link_groups if callback in row["members"])
    assert parent in must_link_group["members"]
    assert must_link_group["relations"] == LOCAL_CALLABLE_MUST_LINK_RELATION


def test_leiden_reweighted_uses_configured_middle_ground_must_link_policy(monkeypatch):
    local_obj, local_obj_row = _data(
        "local_exposed:sample.producer:system_particle_object_list",
        kind="local_exposed; object_state",
    )
    state_obj, state_obj_row = _data(
        "class_attr_state:sample.Service:system_particle_object_list",
        kind="class_attr_state",
    )
    local_assign_obj, local_assign_obj_row = _data(
        "local_exposed:sample.consumer:local_assign",
        kind="local_exposed; object_state",
    )
    tuple_obj, tuple_obj_row = _data(
        "local_exposed:sample.consumer:tuple_slot",
        kind="local_exposed; object_state",
    )
    return_obj, return_obj_row = _data(
        "return:sample.consumer:system_particle_object_list",
        kind="return",
    )
    arg_obj, arg_obj_row = _data(
        "param:sample.consumer:system_particle_object_list",
        kind="param",
    )
    nodes = {
        local_obj: local_obj_row,
        state_obj: state_obj_row,
        local_assign_obj: local_assign_obj_row,
        tuple_obj: tuple_obj_row,
        return_obj: return_obj_row,
        arg_obj: arg_obj_row,
    }
    edges = [
        _edge(local_obj, state_obj, edge_type="data_lineage", relation="state_assign"),
        _edge(state_obj, local_assign_obj, edge_type="data_lineage", relation="local_assign"),
        _edge(local_assign_obj, tuple_obj, edge_type="data_lineage", relation="tuple_unpack"),
        _edge(tuple_obj, return_obj, edge_type="data_lineage", relation="return_value"),
        _edge(return_obj, arg_obj, edge_type="data_lineage", relation="arg_to_param"),
    ]

    current = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(algorithm="label_propagation", run_sweep=False),
    )
    current_members = ";".join(row["members"] for row in current.must_link_groups)
    assert local_obj in current_members
    assert state_obj in current_members
    assert local_assign_obj in current_members
    assert tuple_obj in current_members
    assert return_obj in current_members
    assert arg_obj not in current_members

    def fake_cluster_with_algorithm(algorithm, input_data):
        return {
            node: f"C{idx:03d}"
            for idx, node in enumerate(sorted(input_data.supernodes), start=1)
        }

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    reweighted = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(algorithm="leiden_reweighted", run_sweep=False),
    )
    reweighted_members = ";".join(row["members"] for row in reweighted.must_link_groups)
    assert local_obj in reweighted_members
    assert state_obj in reweighted_members
    assert local_assign_obj in reweighted_members
    assert tuple_obj in reweighted_members
    assert return_obj not in reweighted_members
    assert arg_obj not in reweighted_members


def test_leiden_multiplex_uses_configured_must_link_policy(monkeypatch):
    local_obj, local_obj_row = _data(
        "local_exposed:sample.producer:system_particle_object_list",
        kind="local_exposed; object_state",
    )
    state_obj, state_obj_row = _data(
        "class_attr_state:sample.Service:system_particle_object_list",
        kind="class_attr_state",
    )
    local_assign_obj, local_assign_obj_row = _data(
        "local_exposed:sample.consumer:local_assign",
        kind="local_exposed; object_state",
    )
    nodes = {
        local_obj: local_obj_row,
        state_obj: state_obj_row,
        local_assign_obj: local_assign_obj_row,
    }
    edges = [
        _edge(local_obj, state_obj, edge_type="data_lineage", relation="state_assign"),
        _edge(state_obj, local_assign_obj, edge_type="data_lineage", relation="local_assign"),
    ]

    def fake_cluster_with_algorithm(_algorithm, input_data):
        return {
            node: f"C{idx:03d}"
            for idx, node in enumerate(sorted(input_data.supernodes), start=1)
        }

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    result = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="leiden_multiplex",
            run_sweep=False,
            weight_config={
                "clustering": {
                    "multiplex": {
                        "must_link_relations": ["state_assign"],
                    }
                }
            },
        ),
    )

    members = ";".join(row["members"] for row in result.must_link_groups)
    assert local_obj in members
    assert state_obj in members
    assert local_assign_obj not in members


def test_module_callables_are_excluded_by_default_but_can_be_kept():
    module_node, module_row = _module_callable()
    callable_a, callable_a_row = _callable("a")
    data_a, data_a_row = _data("local_exposed:sample.a:data")
    nodes = {
        module_node: module_row,
        callable_a: callable_a_row,
        data_a: data_a_row,
    }
    edges = [
        _edge(module_node, callable_a, edge_type="call", relation="direct"),
        _edge(callable_a, data_a, access="read"),
    ]

    kept = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="label_propagation",
            run_sweep=False,
            drop_callable_hubs=False,
            exclude_module_callables=False,
        ),
    )
    excluded = cluster_structural_graph(
        nodes,
        edges,
        ClusterOptions(
            algorithm="label_propagation",
            run_sweep=False,
            drop_callable_hubs=False,
            exclude_module_callables=True,
        ),
    )

    assert module_node in kept.cluster_of
    assert module_node not in excluded.cluster_of
    assert any(
        row["node"] == module_node and row["reason"] == "module_callable"
        for row in excluded.excluded_nodes
    )
    assert all(row["reason"] != "module_callable" for row in kept.excluded_nodes)


def test_semantic_cycle_detection_reports_data_callable_cycles():
    callable_a, callable_a_row = _callable("a")
    state_obj, state_obj_row = _data("class_attr_state:sample.Service:state", kind="class_attr_state")
    nodes = {callable_a: callable_a_row, state_obj: state_obj_row}
    edges = [
        _edge(callable_a, state_obj, access="create"),
        _edge(callable_a, state_obj, access="read"),
    ]
    options = ClusterOptions(algorithm="label_propagation", run_sweep=False, drop_callable_hubs=False)

    result = cluster_structural_graph(nodes, edges, options)

    node_cycles = [row for row in result.cycle_findings if row["level"] == "node"]
    assert any(callable_a in row["members"] and state_obj in row["members"] for row in node_cycles)


def test_leiden_clustering_is_deterministic_for_fixed_seed():
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    callable_a, callable_a_row = _callable("a")
    callable_b, callable_b_row = _callable("b")
    data_a, data_a_row = _data("local_exposed:sample.a:data_a")
    data_b, data_b_row = _data("local_exposed:sample.b:data_b")
    nodes = {
        callable_a: callable_a_row,
        callable_b: callable_b_row,
        data_a: data_a_row,
        data_b: data_b_row,
    }
    edges = [
        _edge(callable_a, data_a, access="create", weight=2.0),
        _edge(callable_a, callable_b, edge_type="call", relation="direct"),
        _edge(callable_b, data_b, access="create", weight=2.0),
    ]
    options = ClusterOptions(
        algorithm="leiden",
        resolution=1.0,
        seed=7,
        run_sweep=False,
        drop_callable_hubs=False,
    )

    first = cluster_structural_graph(nodes, edges, options)
    second = cluster_structural_graph(nodes, edges, options)

    assert first.cluster_of == second.cluster_of


def test_leiden_cpm_clustering_is_available_and_deterministic():
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    callable_a, callable_a_row = _callable("a")
    callable_b, callable_b_row = _callable("b")
    data_a, data_a_row = _data("local_exposed:sample.a:data_a")
    data_b, data_b_row = _data("local_exposed:sample.b:data_b")
    nodes = {
        callable_a: callable_a_row,
        callable_b: callable_b_row,
        data_a: data_a_row,
        data_b: data_b_row,
    }
    edges = [
        _edge(callable_a, data_a, access="create", weight=2.0),
        _edge(callable_a, callable_b, edge_type="call", relation="direct"),
        _edge(callable_b, data_b, access="create", weight=2.0),
    ]
    options = ClusterOptions(
        algorithm="leiden",
        leiden_quality="cpm",
        resolution=0.1,
        seed=7,
        run_sweep=False,
        drop_callable_hubs=False,
    )

    first = cluster_structural_graph(nodes, edges, options)
    second = cluster_structural_graph(nodes, edges, options)

    assert first.cluster_of == second.cluster_of
    assert first.options.leiden_quality == "cpm"


def test_leiden_multiplex_clustering_is_deterministic_for_fixed_seed():
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    callable_a, callable_a_row = _callable("a")
    callable_b, callable_b_row = _callable("b")
    data_a, data_a_row = _data("local_exposed:sample.a:data_a")
    data_b, data_b_row = _data("local_exposed:sample.b:data_b")
    nodes = {
        callable_a: callable_a_row,
        callable_b: callable_b_row,
        data_a: data_a_row,
        data_b: data_b_row,
    }
    edges = [
        _edge(callable_a, data_a, access="create", weight=2.0),
        _edge(callable_a, callable_b, edge_type="call", relation="direct"),
        _edge(data_a, data_b, edge_type="data_lineage", relation="arg_to_param"),
        _edge(callable_b, data_b, access="create", weight=2.0),
    ]
    options = ClusterOptions(
        algorithm="leiden_multiplex",
        resolution=1.0,
        seed=7,
        run_sweep=False,
        drop_callable_hubs=False,
    )

    first = cluster_structural_graph(nodes, edges, options)
    second = cluster_structural_graph(nodes, edges, options)

    assert first.cluster_of == second.cluster_of
    assert first.options.algorithm == "leiden_multiplex"


def test_sweep_range_and_quality_defaults_support_cpm_tuning():
    assert _parse_sweep_range("0.01:0.03:0.01") == (0.01, 0.02, 0.03)
    assert _parse_sweep_resolutions("0.1,0.3") == (0.1, 0.3)
    assert _parse_sweep_resolutions("0.1:0.3:0.1") == pytest.approx(
        (0.1, 0.2, 0.3)
    )
    assert _default_sweep_resolutions("cpm") != _default_sweep_resolutions("rb_configuration")
    assert _default_sweep_resolutions("cpm")[0] == 0.01
    assert _default_sweep_markov_times()[0] == 0.25
    assert _default_sweep_hac_n_clusters() == tuple(range(10, 26))


def test_sweep_stats_include_coupling_true_sm_and_newman_modularity():
    node_a, row_a = _callable("a")
    node_b, row_b = _callable("b")
    node_c, row_c = _data("local_exposed:sample.c:data_c")
    node_d, row_d = _data("local_exposed:sample.d:data_d")
    nodes = {
        node_a: row_a,
        node_b: row_b,
        node_c: row_c,
        node_d: row_d,
    }
    cluster_of = {
        node_a: "C001",
        node_b: "C001",
        node_c: "C002",
        node_d: "C002",
    }
    edges = [
        _edge(node_a, node_b, edge_type="call", relation="direct", weight=3.0),
        _edge(node_b, node_c, edge_type="data_lineage", relation="local_assign", weight=2.0),
        _edge(node_c, node_d, edge_type="data_lineage", relation="local_assign", weight=1.0),
    ]
    summary_rows = [{"cohesion": "0.5"}, {"cohesion": "0.25"}]

    stats = _cluster_stats_for_sweep(cluster_of, nodes, edges, summary_rows)

    assert stats["internal_weight"] == "4.000000"
    assert stats["external_weight"] == "2.000000"
    assert stats["coupling"] == "0.333333"
    assert stats["mean_cluster_coupling"] == "0.375000"
    assert stats["mean_cohesion"] == "0.375000"
    assert stats["true_sm"] == "0.000000"
    assert stats["newman_modularity_Q"] == "0.111111"


def test_sweep_best_selection_defaults_to_predicted_match_f1_and_allows_filters():
    rows = [
        {
            "resolution": "0.1",
            "data_hub_policy": "drop_data_hubs",
            "evaluation_predicted_match_f1": "0.40",
            "coupling": "0.30",
        },
        {
            "resolution": "0.2",
            "data_hub_policy": "drop_data_hubs",
            "evaluation_predicted_match_f1": "0.75",
            "coupling": "0.25",
        },
        {
            "resolution": "0.2",
            "data_hub_policy": "keep_data_hubs",
            "evaluation_predicted_match_f1": "0.60",
            "coupling": "0.10",
        },
    ]

    default_selection = select_sweep_best_row(
        rows,
        SweepBestSelectionOptions(metric="predicted_match_f1"),
    )

    assert default_selection.selected_index == 1
    assert default_selection.metric == "evaluation_predicted_match_f1"
    assert default_selection.reason == "selected_by_metric"

    filtered_selection = select_sweep_best_row(
        rows,
        SweepBestSelectionOptions(
            metric="coupling",
            metric_direction="min",
            resolution=0.2,
            data_hub_policy="keep_data_hubs",
        ),
    )

    assert filtered_selection.selected_index == 2
    assert filtered_selection.filtered_count == 1
    assert filtered_selection.reason == "selected_by_metric"


def test_sweep_best_selection_applies_minimum_metric_threshold():
    rows = [
        {
            "resolution": "0.1",
            "evaluation_predicted_match_f1": "0.90",
            "evaluation_best_match_macro_f1": "0.400000",
        },
        {
            "resolution": "0.2",
            "evaluation_predicted_match_f1": "0.80",
            "evaluation_best_match_macro_f1": "0.410000",
        },
    ]

    selection = select_sweep_best_row(
        rows,
        SweepBestSelectionOptions(
            metric="predicted_match_f1",
            min_metric="best_match_macro_f1",
            min_value=0.4,
        ),
    )

    assert selection.selected_index == 1
    assert selection.filtered_count == 1
    assert selection.reason == "selected_by_metric"


def test_sweep_best_selection_can_use_unique_manual_filter_without_metric():
    rows = [
        {"resolution": "0.1", "data_hub_policy": "drop_data_hubs"},
        {"resolution": "0.2", "data_hub_policy": "keep_data_hubs"},
    ]

    selection = select_sweep_best_row(
        rows,
        SweepBestSelectionOptions(
            resolution=0.2,
            data_hub_policy="keep_data_hubs",
        ),
    )

    assert selection.selected_index == 1
    assert selection.reason == "selected_by_unique_filter_without_metric"


def test_sweep_options_from_row_restores_selected_parameters():
    base_options = ClusterOptions(
        algorithm="leiden_multiplex",
        resolution=1.0,
        sweep_call_resolutions=(0.1, 0.2),
        run_sweep=True,
        drop_data_hubs=True,
    )
    row = {
        "call_resolution": "0.2",
        "data_access_resolution": "0.3",
        "data_lineage_resolution": "0.4",
        "data_hub_policy": "keep_data_hubs",
    }

    selected_options = sweep_options_from_row(base_options, row)

    assert selected_options.run_sweep is False
    assert selected_options.sweep_call_resolutions == tuple()
    assert selected_options.call_resolution == 0.2
    assert selected_options.data_access_resolution == 0.3
    assert selected_options.data_lineage_resolution == 0.4
    assert selected_options.drop_data_hubs is False


def test_materialize_sweep_best_cluster_writes_full_artifacts(monkeypatch, tmp_path):
    callable_a, row_a = _callable("a")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    nodes = {callable_a: row_a, data_a: row_data_a}
    edges = [_edge(callable_a, data_a, access="read")]
    options = ClusterOptions(
        algorithm="leiden",
        sweep_resolutions=(0.1, 0.2),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=True,
    )

    def fake_cluster_with_algorithm(_algorithm, input_data):
        return {node: "C001" for node in input_data.supernodes}

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    best_outdir = tmp_path / "sweep" / "best"
    selected_options = materialize_sweep_best_cluster(
        best_outdir,
        nodes,
        edges,
        options,
        {"resolution": "0.2", "data_hub_policy": "keep_data_hubs"},
    )

    assert selected_options.resolution == 0.2
    assert selected_options.drop_data_hubs is False
    assert (best_outdir / "cluster_assignments.csv").exists()
    payload = json.loads((best_outdir / "clusters.json").read_text(encoding="utf-8"))
    assert payload["resolution"] == 0.2
    assert payload["options"]["drop_data_hubs"] is False


def test_parameter_sweep_compares_data_hub_policy():
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    callable_a, row_a = _callable("a")
    callable_b, row_b = _callable("b")
    data_hub, row_data_hub = _data(
        "local_exposed:sample.shared:data",
        callable_count=10,
        access_count=100,
    )
    nodes = {
        callable_a: row_a,
        callable_b: row_b,
        data_hub: row_data_hub,
    }
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct"),
        _edge(callable_a, data_hub, access="read"),
        _edge(callable_b, data_hub, access="read"),
    ]
    options = ClusterOptions(
        algorithm="leiden",
        sweep_resolutions=(0.1,),
        run_sweep=True,
        drop_callable_hubs=True,
        drop_data_hubs=True,
        hub_callable_min_degree=99,
        hub_data_min_degree=99,
        hub_data_min_callable_count=2,
        hub_data_min_access_count=2,
    )

    rows = run_parameter_sweep(
        nodes,
        edges,
        options,
        manual_rows=[
            {"node": callable_a, "microservice_id": "A"},
            {"node": callable_b, "microservice_id": "A"},
            {"node": data_hub, "microservice_id": "A"},
        ],
        manual_fields=["node", "microservice_id"],
        na_labels=(),
        all_evaluation_nodes=True,
    )

    assert {row["data_hub_policy"] for row in rows} == {
        "drop_data_hubs",
        "keep_data_hubs",
    }
    assert all("callable_hub_policy" not in row for row in rows)
    assert {
        row["data_hub_policy"]: row["hubs_dropped"] for row in rows
    } == {"drop_data_hubs": 1, "keep_data_hubs": 0}
    assert all("evaluation_adjusted_rand_index" in row for row in rows)
    assert {
        row["data_hub_policy"]: row["evaluation_known_joined_rows"] for row in rows
    } == {"drop_data_hubs": 2, "keep_data_hubs": 3}


def test_parameter_sweep_evaluation_scope_is_configurable(monkeypatch):
    callable_a, row_a = _callable("a")
    state_data, row_state_data = _data(
        "class_attr_state:sample.Service:state",
        kind="class_attr_state; local_exposed",
    )
    local_data, row_local_data = _data("local_exposed:sample.a:data")
    nodes = {
        callable_a: row_a,
        state_data: row_state_data,
        local_data: row_local_data,
    }
    edges = [
        _edge(callable_a, state_data, access="read"),
        _edge(callable_a, local_data, access="read"),
    ]
    options = ClusterOptions(
        algorithm="leiden",
        sweep_resolutions=(0.1,),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=False,
        hub_data_min_degree=99,
        hub_data_min_callable_count=99,
        hub_data_min_access_count=99,
    )

    def fake_cluster_with_algorithm(_algorithm, input_data):
        return {node: "C001" for node in input_data.supernodes}

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    def manual_row(node, row):
        return {
            "Microservice_id": "service-a",
            "node": node,
            "node_type": row["node_type"],
            "label": row["label"],
            "kind": row["kind"],
            "module": row["module"],
        }

    manual_rows = [
        manual_row(callable_a, row_a),
        manual_row(state_data, row_state_data),
        manual_row(local_data, row_local_data),
    ]
    manual_fields = ["Microservice_id", "node", "node_type", "label", "kind", "module"]

    default_rows = run_parameter_sweep(
        nodes,
        edges,
        options,
        manual_rows=manual_rows,
        manual_fields=manual_fields,
        na_labels=(),
    )
    callable_only_rows = run_parameter_sweep(
        nodes,
        edges,
        options,
        manual_rows=manual_rows,
        manual_fields=manual_fields,
        na_labels=(),
        evaluation_kind_tokens=(),
    )

    assert {row["evaluation_known_joined_rows"] for row in default_rows} == {2}
    assert {row["evaluation_known_joined_rows"] for row in callable_only_rows} == {1}


def test_hac_parameter_sweep_uses_cluster_count_axis_and_evaluation(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    nodes = {}
    edges = []
    manual_rows = []
    for name in ("a", "b", "c", "d"):
        callable_node, callable_row = _callable(name)
        data_node, data_row = _data(f"local_exposed:sample.{name}:data")
        nodes[callable_node] = callable_row
        nodes[data_node] = data_row
        edges.append(_edge(callable_node, data_node, access="write", weight=1.0))
        manual_rows.append(
            {
                "Microservice_id": f"service-{name}",
                "node": callable_node,
                "node_type": "callable",
                "label": name,
                "kind": "function",
                "module": "sample",
            }
        )
    options = ClusterOptions(
        algorithm="hac_callable_projection",
        sweep_hac_n_clusters=(2, 3),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=False,
    )

    rows = run_parameter_sweep(
        nodes,
        edges,
        options,
        manual_rows=manual_rows,
        manual_fields=["Microservice_id", "node", "node_type", "label", "kind", "module"],
        node_mode="exact",
        na_labels=(),
    )

    assert [row["sweep_parameter"] for row in rows] == [
        "hac_n_clusters",
        "hac_n_clusters",
        "hac_n_clusters",
        "hac_n_clusters",
    ]
    assert [row["hac_n_clusters"] for row in rows] == ["2", "3", "2", "3"]
    assert {row["resolution"] for row in rows} == {""}
    assert {row["markov_time"] for row in rows} == {""}
    assert {row["evaluation_known_joined_rows"] for row in rows} == {4}
    assert all("evaluation_adjusted_rand_index" in row for row in rows)


def test_leiden_multiplex_sweep_uses_resolution_axis_and_quality(monkeypatch):
    callable_a, row_a = _callable("a")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    nodes = {callable_a: row_a, data_a: row_data_a}
    edges = [_edge(callable_a, data_a, access="read")]
    options = ClusterOptions(
        algorithm="leiden_multiplex",
        leiden_quality="cpm",
        sweep_resolutions=(0.1,),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=False,
    )

    def fake_cluster_with_algorithm(_algorithm, input_data):
        return {node: "C001" for node in input_data.supernodes}

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    rows = run_parameter_sweep(nodes, edges, options)

    assert [row["algorithm"] for row in rows] == ["leiden_multiplex", "leiden_multiplex"]
    assert {row["sweep_parameter"] for row in rows} == {"resolution"}
    assert {row["leiden_quality"] for row in rows} == {"cpm"}
    assert [row["resolution"] for row in rows] == ["0.1", "0.1"]
    assert [row["call_resolution"] for row in rows] == ["0.1", "0.1"]
    assert [row["data_access_resolution"] for row in rows] == ["0.1", "0.1"]
    assert [row["data_lineage_resolution"] for row in rows] == ["0.1", "0.1"]
    assert all(row["markov_time"] == "" for row in rows)


def test_leiden_multiplex_sweep_can_vary_layer_resolutions(monkeypatch):
    callable_a, row_a = _callable("a")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    nodes = {callable_a: row_a, data_a: row_data_a}
    edges = [_edge(callable_a, data_a, access="read")]
    options = ClusterOptions(
        algorithm="leiden_multiplex",
        resolution=1.0,
        sweep_call_resolutions=(0.5, 1.0),
        sweep_data_access_resolutions=(0.75,),
        sweep_data_lineage_resolutions=(1.5,),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=False,
    )
    seen = []

    def fake_cluster_with_algorithm(_algorithm, input_data):
        seen.append(input_data.edge_type_layer_resolutions)
        return {node: "C001" for node in input_data.supernodes}

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    rows = run_parameter_sweep(nodes, edges, options)

    assert [row["sweep_parameter"] for row in rows] == [
        "layer_resolution",
        "layer_resolution",
        "layer_resolution",
        "layer_resolution",
    ]
    assert [row["call_resolution"] for row in rows] == ["0.5", "1", "0.5", "1"]
    assert {row["data_access_resolution"] for row in rows} == {"0.75"}
    assert {row["data_lineage_resolution"] for row in rows} == {"1.5"}
    assert {tuple(sorted(item.items())) for item in seen} == {
        (
            ("call", 0.5),
            ("data_access", 0.75),
            ("data_lineage", 1.5),
        ),
        (
            ("call", 1.0),
            ("data_access", 0.75),
            ("data_lineage", 1.5),
        ),
    }


def test_leiden_multiplex_call_data_sweep_reports_combined_data_resolution(monkeypatch):
    callable_a, row_a = _callable("a")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    nodes = {callable_a: row_a, data_a: row_data_a}
    edges = [_edge(callable_a, data_a, access="read")]
    options = ClusterOptions(
        algorithm="leiden_multiplex",
        multiplex_layer_mode="call_data",
        resolution=1.0,
        sweep_call_resolutions=(0.5,),
        sweep_data_access_resolutions=(0.75,),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=False,
    )
    seen = []

    def fake_cluster_with_algorithm(_algorithm, input_data):
        seen.append(input_data.edge_type_layer_resolutions)
        return {node: "C001" for node in input_data.supernodes}

    monkeypatch.setattr(
        "microservice_pipeline.cluster_structural_graph.cluster_with_algorithm",
        fake_cluster_with_algorithm,
    )

    rows = run_parameter_sweep(nodes, edges, options)

    assert [row["call_resolution"] for row in rows] == ["0.5", "0.5"]
    assert [row["data_access_resolution"] for row in rows] == ["0.75", "0.75"]
    assert [row["data_lineage_resolution"] for row in rows] == ["", ""]
    assert seen == [{"call": 0.5, "data": 0.75}, {"call": 0.5, "data": 0.75}]


def test_infomap_markov_time_sweep_uses_directed_parameter_axis():
    pytest.importorskip("infomap")
    callable_a, row_a = _callable("a")
    callable_b, row_b = _callable("b")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    nodes = {
        callable_a: row_a,
        callable_b: row_b,
        data_a: row_data_a,
    }
    edges = [
        _edge(callable_a, callable_b, edge_type="call", relation="direct"),
        _edge(callable_b, data_a, access="create"),
    ]
    options = ClusterOptions(
        algorithm="infomap",
        sweep_markov_times=(0.5, 1.0),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=False,
    )

    rows = run_parameter_sweep(nodes, edges, options)

    assert [row["markov_time"] for row in rows] == ["0.5", "1", "0.5", "1"]
    assert {row["algorithm"] for row in rows} == {"infomap"}
    assert {row["sweep_parameter"] for row in rows} == {"markov_time"}
    assert all(row["resolution"] == "" for row in rows)


def test_cluster_outputs_keep_sweep_artifacts_in_separate_folder(tmp_path):
    callable_a, row_a = _callable("a")
    data_a, row_data_a = _data("local_exposed:sample.a:data")
    nodes = {callable_a: row_a, data_a: row_data_a}
    edges = [_edge(callable_a, data_a, access="read")]
    options = ClusterOptions(
        algorithm="label_propagation",
        run_sweep=False,
        drop_callable_hubs=False,
        drop_data_hubs=False,
    )
    result = cluster_structural_graph(nodes, edges, options)

    outdir = tmp_path / "clusters"
    outdir.mkdir()
    for name in ("parameter_sweep.csv", "parameter_sweep.md"):
        (outdir / name).write_text("stale", encoding="utf-8")

    write_outputs(outdir, nodes, result)

    assert (outdir / "cluster_assignments.csv").exists()
    assert not (outdir / "parameter_sweep.csv").exists()
    assert not (outdir / "parameter_sweep.md").exists()
    assert "parameter_sweep" not in (outdir / "clusters.json").read_text(encoding="utf-8")

    sweep_options = ClusterOptions(
        algorithm="leiden",
        sweep_resolutions=(0.1,),
        run_sweep=True,
        drop_callable_hubs=False,
        drop_data_hubs=False,
    )
    sweep_rows = [
        {
            "leiden_quality": "rb_configuration",
            "resolution": "0.1",
            "data_hub_policy": "drop_data_hubs",
            "nodes_clustered": 2,
            "hubs_dropped": 0,
            "num_clusters": 1,
            "mixed_clusters": 1,
            "max_cluster_size": 2,
            "median_cluster_size": 2,
            "size_distribution": "2",
            "mean_cohesion": "1.000000",
            "mean_cluster_coupling": "0.000000",
            "true_sm": "1.000000",
            "internal_weight": "1.000000",
            "external_weight": "0.000000",
            "coupling": "0.000000",
            "newman_modularity_Q": "0.000000",
        }
    ]
    sweep_outdir = tmp_path / "sweep"
    write_sweep_outputs(sweep_outdir, sweep_rows, sweep_options)

    assert (sweep_outdir / "parameter_sweep.csv").exists()
    assert (sweep_outdir / "parameter_sweep.md").exists()
    assert (sweep_outdir / "parameter_sweep.json").exists()
