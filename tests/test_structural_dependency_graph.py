import json
from pathlib import Path

import pytest

from microservice_pipeline.structural_dependency_graph.generate_structural_dependency_graph import (
    build_structural_graph,
    write_outputs,
)
from microservice_pipeline.data_access.generate_data_access_ast import (
    _edges_payload,
    _objects_payload,
    collect_data_access_from_source,
)
from microservice_pipeline.structural_dependency_graph.hub_nodes import HubDetectionOptions
from microservice_pipeline.structural_dependency_graph.weight_config import load_weight_config


STATE_CLUSTER = "class_attr_state:sample.Service:state"
PARAM_CLUSTER = "param:sample.a:config"
DICT_KEY_CLUSTER = "dict_key:sample.Service:state:mass_g"
ALIAS_OBJECT = "local_exposed:sample.a:state_alias"


def _sample_payloads():
    call_graph = {
        "nodes": [
            {
                "id": "sample.a",
                "module": "sample",
                "qualname": "a",
                "file": "sample.py",
                "lineno": 1,
                "kind": "function",
                "class_name": "",
            },
            {
                "id": "sample.b",
                "module": "sample",
                "qualname": "b",
                "file": "sample.py",
                "lineno": 20,
                "kind": "function",
                "class_name": "",
            },
        ],
        "edges": [
            {
                "caller": "sample.a",
                "callee": "sample.b",
                "file": "sample.py",
                "lineno": 2,
                "resolved": True,
                "relation": "direct",
            },
            {
                "caller": "sample.a",
                "callee": "sample.b",
                "file": "sample.py",
                "lineno": 3,
                "resolved": True,
                "relation": "direct",
            },
            {
                "caller": "sample.a",
                "callee": "sample.b",
                "file": "sample.py",
                "lineno": 4,
                "resolved": True,
                "relation": "direct",
            },
        ],
    }
    data_access = {
        "callables": call_graph["nodes"],
        "objects": [
            {
                "id": PARAM_CLUSTER,
                "kind": "param",
                "display_name": "config",
                "scope": "callable",
                "owner": "sample.a",
                "file": "sample.py",
                "lineno": 1,
                "inferred_type": "dict",
            },
            {
                "id": STATE_CLUSTER,
                "kind": "class_attr_state",
                "display_name": "Service.state",
                "scope": "class",
                "owner": "sample.Service",
                "file": "sample.py",
                "lineno": 5,
                "inferred_type": "object",
            },
            {
                "id": ALIAS_OBJECT,
                "kind": "local_exposed",
                "display_name": "state_alias",
                "scope": "callable",
                "owner": "sample.a",
                "file": "sample.py",
                "lineno": 12,
                "inferred_type": "dict",
            },
            {
                "id": DICT_KEY_CLUSTER,
                "kind": "dict_key",
                "display_name": "state.mass_g",
                "scope": "field",
                "owner": "sample.Service",
                "file": "sample.py",
                "lineno": 13,
                "inferred_type": "dict",
            },
        ],
        "edges": [
            {
                "callable": "sample.a",
                "object_id": PARAM_CLUSTER,
                "access": "read",
                "operation": "load",
                "file": "sample.py",
                "lineno": 4,
                "confidence": "high",
                "confidence_weight": 1.0,
            },
            {
                "callable": "sample.a",
                "object_id": STATE_CLUSTER,
                "access": "write",
                "operation": "assign",
                "file": "sample.py",
                "lineno": 10,
                "confidence": "high",
                "confidence_weight": 1.0,
            },
            {
                "callable": "sample.a",
                "object_id": STATE_CLUSTER,
                "access": "write",
                "operation": "assign",
                "file": "sample.py",
                "lineno": 11,
                "confidence": "high",
                "confidence_weight": 1.0,
            },
            {
                "callable": "sample.a",
                "object_id": STATE_CLUSTER,
                "access": "write",
                "operation": "assign",
                "file": "sample.py",
                "lineno": 12,
                "confidence": "high",
                "confidence_weight": 1.0,
            },
            {
                "callable": "sample.b",
                "object_id": STATE_CLUSTER,
                "access": "read",
                "operation": "load",
                "file": "sample.py",
                "lineno": 21,
                "confidence": "medium",
                "confidence_weight": 0.5,
            },
        ],
        "lineage_edges": [
            {
                "src_object_id": ALIAS_OBJECT,
                "dst_object_id": STATE_CLUSTER,
                "relation": "local_assign",
                "file": "sample.py",
                "lineno": 12,
            },
            {
                "src_object_id": STATE_CLUSTER,
                "dst_object_id": DICT_KEY_CLUSTER,
                "relation": "state_assign",
                "file": "sample.py",
                "lineno": 13,
            },
        ],
    }
    return call_graph, data_access


def _edge(graph, **criteria):
    for edge in graph["edges"]:
        if all(edge.get(key) == value for key, value in criteria.items()):
            return edge
    raise AssertionError(f"edge not found: {criteria}")


def test_callable_and_data_nodes_are_prefixed_and_use_object_identity():
    graph = build_structural_graph(*_sample_payloads())
    nodes = {node["id"]: node for node in graph["nodes"]}

    assert "callable:sample.a" in nodes
    assert "callable:sample.b" in nodes
    assert f"data:{STATE_CLUSTER}" in nodes
    assert f"data:{DICT_KEY_CLUSTER}" in nodes
    assert f"data:{PARAM_CLUSTER}" in nodes
    assert f"data:{ALIAS_OBJECT}" in nodes
    assert any(edge["dst"] == f"data:{PARAM_CLUSTER}" for edge in graph["edges"])

    state_node = nodes[f"data:{STATE_CLUSTER}"]
    assert state_node["raw_object_count"] == 1
    assert state_node["callable_count"] == 2
    assert state_node["access_count"] == 4
    assert state_node["kind"] == "class_attr_state"
    assert state_node["inferred_type"] == "object"


def test_edges_use_object_identity_and_aggregate_duplicates():
    graph = build_structural_graph(*_sample_payloads())

    call_edge = _edge(
        graph,
        src="callable:sample.a",
        dst="callable:sample.b",
        edge_type="call",
        relation="direct",
    )
    assert call_edge["evidence_count"] == 3
    assert call_edge["base_weight"] == pytest.approx(1.0 + 1.584962500721156)
    assert call_edge["weight"] == pytest.approx(1.0 + 1.584962500721156)
    assert call_edge["linenos_preview"] == "2;3;4"

    write_edge = _edge(
        graph,
        src="callable:sample.a",
        dst=f"data:{STATE_CLUSTER}",
        edge_type="data_access",
        access="write",
        operation="assign",
    )
    assert write_edge["evidence_count"] == 3
    assert write_edge["base_weight"] == pytest.approx(3.0 * (1.0 + 1.584962500721156))
    assert write_edge["weight"] == pytest.approx(1.5 * (1.0 + 1.584962500721156))
    assert write_edge["confidence_weight"] == 1.0
    assert write_edge["linenos_preview"] == "10;11;12"


def test_structural_graph_excludes_coarse_model_nodes_when_precise_paths_exist():
    objects, edges = collect_data_access_from_source(
        """
def massBalance(model):
    for p in model.system_particle_object_list:
        loss = p.RateConstants["k_fragmentation"]
    m_ss = model.R["mass_g"]
    return loss, m_ss
""",
        module="sample",
    )
    callable_rows = [
        {
            "id": "sample.massBalance",
            "module": "sample",
            "qualname": "massBalance",
            "file": "sample.py",
            "lineno": 1,
            "kind": "function",
            "class_name": "",
        }
    ]
    data_access = {
        "callables": callable_rows,
        "objects": _objects_payload(objects),
        "edges": _edges_payload(edges, objects),
        "lineage_edges": [],
    }

    graph = build_structural_graph({"nodes": callable_rows, "edges": []}, data_access)
    data_nodes = {node["id"]: node for node in graph["nodes"] if node["node_type"] == "data"}

    assert "data:param:sample.massBalance:model" not in data_nodes
    assert "data:object_state:param:sample.massBalance:model" not in data_nodes
    assert (
        "data:container_field:sample.massBalance:model.system_particle_object_list[].RateConstants:k_fragmentation"
        in data_nodes
    )
    assert "data:container_field:sample.massBalance:model.R:mass_g" in data_nodes
    assert any(
        edge["src"] == "callable:sample.massBalance"
        and edge["dst"]
        == "data:container_field:sample.massBalance:model.system_particle_object_list[].RateConstants:k_fragmentation"
        for edge in graph["edges"]
    )


def test_default_weight_config_is_recorded_in_structural_graph_payload():
    graph = build_structural_graph(*_sample_payloads())

    assert graph["weight_config"]["schema"] == "structural_weight_profile.v1"
    assert graph["weight_config"]["name"] == "default"
    assert graph["weights"]["call"]["call"] == 1.0
    assert graph["weights"]["data_access"]["write"] == 3.0
    assert graph["weights"]["data_lineage"]["state_assign"] == 2.0
    assert graph["weight_config"]["clustering"]["edge_type_scales"]["call"] == 2.0
    assert graph["hub_detection_options"]["hub_entrypoint_min_out_degree"] == 12


def test_structural_graph_writes_callable_and_data_hub_nodes(tmp_path):
    graph = build_structural_graph(
        *_sample_payloads(),
        hub_options=HubDetectionOptions(
            hub_callable_degree_percentile=0,
            hub_callable_min_degree=1,
            hub_callable_min_in_degree=0,
            hub_callable_min_out_degree=1,
            hub_entrypoint_min_out_degree=2,
            hub_data_min_degree=99,
            hub_data_min_callable_count=2,
            hub_data_min_access_count=4,
        ),
    )

    callable_hub_nodes = {row["node"]: row for row in graph["callable_hub_nodes"]}
    data_hub_nodes = {row["node"]: row for row in graph["data_hub_nodes"]}
    assert "callable:sample.a" in callable_hub_nodes
    assert callable_hub_nodes["callable:sample.a"]["candidate_types"] == "degree;entrypoint"
    assert "entrypoint_fanout:in=0;out>=2" in callable_hub_nodes["callable:sample.a"]["reasons"]
    assert f"data:{STATE_CLUSTER}" in data_hub_nodes
    assert "callable_count>=2" in data_hub_nodes[f"data:{STATE_CLUSTER}"]["reasons"]
    assert graph["stats"]["callable_hub_node_count"] == len(graph["callable_hub_nodes"])
    assert graph["stats"]["data_hub_node_count"] == len(graph["data_hub_nodes"])

    write_outputs(tmp_path, graph)

    assert (tmp_path / "callable_hub_nodes.csv").exists()
    assert (tmp_path / "data_hub_nodes.csv").exists()
    assert "callable:sample.a" in (tmp_path / "callable_hub_nodes.csv").read_text(
        encoding="utf-8"
    )
    assert f"data:{STATE_CLUSTER}" in (tmp_path / "data_hub_nodes.csv").read_text(
        encoding="utf-8"
    )


def test_ownership_biased_weight_profile_loads_without_changing_default():
    profile = load_weight_config("builtin:ownership_biased")
    default = load_weight_config()

    assert profile.name == "ownership_biased"
    assert profile.clustering_scale("call") == 2.0
    assert profile.clustering_scale("data_access") == 1.0
    assert profile.clustering_scale("data_lineage") == 1.0
    assert profile.data_access_weight("read") == 1.0
    assert profile.data_access_weight("write") == 4.0
    assert profile.data_access_weight("create") == 3.5
    assert profile.data_lineage_weight("return_value") == 0.75
    assert profile.data_lineage_weight("arg_to_param") == 0.25
    assert profile.reweighted_settings["subtype_ratio_min"] == 0.5
    assert profile.reweighted_settings["subtype_ratio_max"] == 1.5
    assert profile.reweighted_settings["single_writer_boost"] == 1.5
    assert profile.reweighted_settings["single_writer_max_weight"] == 8.0
    assert profile.reweighted_settings["must_link_relations"] == [
        "state_assign",
        "local_assign",
        "tuple_unpack",
    ]
    assert profile.multiplex_settings == {}
    assert default.name == "default"
    assert default.data_access_weight("write") == 3.0
    assert default.clustering_scale("call") == 2.0
    assert default.reweighted_settings == {}
    assert default.multiplex_settings == {}


def test_multiplex_weight_config_round_trips_optional_settings(tmp_path):
    config_path = tmp_path / "weights.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "multiplex-test",
                "clustering": {
                    "multiplex": {
                        "must_link_relations": ["state_assign"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_weight_config(config_path)

    assert profile.multiplex_settings == {"must_link_relations": ["state_assign"]}
    assert profile.to_dict()["clustering"]["multiplex"] == {
        "must_link_relations": ["state_assign"],
    }


def test_custom_weight_config_changes_generation_weights(tmp_path):
    config_path = tmp_path / "weights.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "custom-test",
                "generation": {
                    "call": {"base_weight": 4.0},
                    "data_access": {"weights": {"write": 6.0}},
                    "data_lineage": {"weights": {"state_assign": 7.0}},
                },
            }
        ),
        encoding="utf-8",
    )
    graph = build_structural_graph(
        *_sample_payloads(),
        weight_config=load_weight_config(config_path),
    )

    count_factor = 1.0 + 1.584962500721156
    call_edge = _edge(
        graph,
        src="callable:sample.a",
        dst="callable:sample.b",
        edge_type="call",
        relation="direct",
    )
    assert call_edge["base_weight"] == pytest.approx(4.0 * count_factor)
    assert call_edge["weight"] == pytest.approx(4.0 * count_factor)

    write_edge = _edge(
        graph,
        src="callable:sample.a",
        dst=f"data:{STATE_CLUSTER}",
        edge_type="data_access",
        access="write",
        operation="assign",
    )
    assert write_edge["base_weight"] == pytest.approx(6.0 * count_factor)
    assert write_edge["weight"] == pytest.approx(3.0 * count_factor)

    lineage = _edge(
        graph,
        src=f"data:{STATE_CLUSTER}",
        dst=f"data:{DICT_KEY_CLUSTER}",
        edge_type="data_lineage",
        relation="state_assign",
    )
    assert lineage["base_weight"] == 7.0
    assert lineage["weight"] == 7.0
    assert graph["weight_config"]["name"] == "custom-test"


def test_data_access_weights_apply_confidence_and_high_degree_downweighting():
    graph = build_structural_graph(*_sample_payloads())

    read_edge = _edge(
        graph,
        src="callable:sample.b",
        dst=f"data:{STATE_CLUSTER}",
        edge_type="data_access",
        access="read",
        operation="load",
    )
    assert read_edge["base_weight"] == 1.0
    assert read_edge["confidence"] == "medium"
    assert read_edge["confidence_weight"] == 0.5
    assert read_edge["weight"] == pytest.approx(0.25)


def test_lineage_edges_use_object_identity():
    graph = build_structural_graph(*_sample_payloads())

    local_assign = _edge(
        graph,
        src=f"data:{ALIAS_OBJECT}",
        dst=f"data:{STATE_CLUSTER}",
        edge_type="data_lineage",
        relation="local_assign",
    )
    assert local_assign["base_weight"] == 1.0
    assert local_assign["weight"] == 1.0

    lineage = _edge(
        graph,
        src=f"data:{STATE_CLUSTER}",
        dst=f"data:{DICT_KEY_CLUSTER}",
        edge_type="data_lineage",
        relation="state_assign",
    )
    assert lineage["base_weight"] == 2.0
    assert lineage["weight"] == 2.0


def test_call_edge_weight_is_discounted_by_confidence():
    """Confidence must reach the weight, or recording it is decoration.

    The call branch used to stamp every edge ``confidence="high"`` and then
    compute ``weight`` without ever multiplying by ``confidence_weight``, unlike
    the data-access branch. Reading the field but not applying it would leave a
    guess pulling clusters together exactly as hard as a certainty.
    """
    call_graph, data_access = _sample_payloads()
    call_graph["edges"] = [
        {
            "caller": "sample.a",
            "callee": "sample.b",
            "file": "sample.py",
            "lineno": 2,
            "resolved": True,
            "relation": "direct",
            "confidence": "medium",
        }
    ]

    graph = build_structural_graph(call_graph, data_access)
    edge = _edge(
        graph,
        src="callable:sample.a",
        dst="callable:sample.b",
        edge_type="call",
        relation="direct",
    )

    assert edge["confidence"] == "medium"
    assert edge["confidence_weight"] == pytest.approx(0.65)
    assert edge["base_weight"] == pytest.approx(1.0)
    assert edge["weight"] == pytest.approx(0.65)


def test_a_call_edge_without_confidence_keeps_its_full_weight():
    """Older artifacts have no confidence column and must not be penalised."""
    call_graph, data_access = _sample_payloads()
    call_graph["edges"] = [
        {
            "caller": "sample.a",
            "callee": "sample.b",
            "file": "sample.py",
            "lineno": 2,
            "resolved": True,
            "relation": "direct",
        }
    ]

    graph = build_structural_graph(call_graph, data_access)
    edge = _edge(
        graph,
        src="callable:sample.a",
        dst="callable:sample.b",
        edge_type="call",
        relation="direct",
    )

    assert edge["weight"] == pytest.approx(1.0)
