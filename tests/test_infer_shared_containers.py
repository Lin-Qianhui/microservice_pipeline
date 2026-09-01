from pathlib import Path

from microservice_pipeline.data_access.infer_shared_containers import (
    PROBE_RE,
    CandidateContainer,
    infer_shared_containers_from_payload,
    probe_pyright_families,
)


def _payload(objects, edges=None, lineage_edges=None):
    return {
        "callables": [],
        "objects": objects,
        "edges": edges or [],
        "lineage_edges": lineage_edges or [],
    }


def test_infer_same_name_lineage_supported_dataframe_container():
    payload = _payload(
        objects=[
            {
                "id": "local_exposed:sample.load:Results_extended",
                "kind": "local_exposed",
                "display_name": "Results_extended",
                "owner": "sample.load",
                "container": "",
                "field": "Results_extended",
                "file": "sample.py",
                "lineno": 2,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "param:sample.process:Results_extended",
                "kind": "param",
                "display_name": "Results_extended",
                "owner": "sample.process",
                "container": "",
                "field": "Results_extended",
                "file": "sample.py",
                "lineno": 5,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "local_exposed:sample.load:Results_extended",
            },
            {
                "id": "df_col:sample.load:Results_extended:Compartment",
                "kind": "df_col",
                "display_name": "Results_extended['Compartment']",
                "owner": "local_exposed:sample.load:Results_extended",
                "container": "local_exposed:sample.load:Results_extended",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 3,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.load:Results_extended:mass_g",
                "kind": "df_col",
                "display_name": "Results_extended['mass_g']",
                "owner": "local_exposed:sample.load:Results_extended",
                "container": "local_exposed:sample.load:Results_extended",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 4,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.process:Results_extended:Compartment",
                "kind": "df_col",
                "display_name": "Results_extended['Compartment']",
                "owner": "param:sample.process:Results_extended",
                "container": "param:sample.process:Results_extended",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 6,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.process:Results_extended:mass_g",
                "kind": "df_col",
                "display_name": "Results_extended['mass_g']",
                "owner": "param:sample.process:Results_extended",
                "container": "param:sample.process:Results_extended",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 7,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
        ],
        edges=[
            {"callable": "sample.load", "object_id": "df_col:sample.load:Results_extended:Compartment"},
            {"callable": "sample.load", "object_id": "df_col:sample.load:Results_extended:mass_g"},
            {"callable": "sample.process", "object_id": "df_col:sample.process:Results_extended:Compartment"},
            {"callable": "sample.process", "object_id": "df_col:sample.process:Results_extended:mass_g"},
        ],
        lineage_edges=[
            {
                "src_object_id": "local_exposed:sample.load:Results_extended",
                "dst_object_id": "param:sample.process:Results_extended",
                "relation": "arg_to_param",
                "caller": "sample.orchestrate",
                "callee": "sample.process",
                "file": "sample.py",
                "lineno": 10,
                "slot": "Results_extended",
            }
        ],
    )

    inferred_config, decisions = infer_shared_containers_from_payload(
        payload,
        [],
        pyright_families={
            "local_exposed:sample.load:Results_extended": "dataframe",
            "param:sample.process:Results_extended": "dataframe",
        },
    )

    assert inferred_config["df_col"] == {"Results_extended": "Results_extended"}
    assert all(decision.accepted for decision in decisions)


def test_generic_same_name_container_without_lineage_is_rejected():
    payload = _payload(
        objects=[
            {
                "id": "param:sample.a:data",
                "kind": "param",
                "display_name": "data",
                "owner": "sample.a",
                "container": "",
                "field": "data",
                "file": "sample.py",
                "lineno": 2,
                "inferred_type": "dict",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "param:sample.b:data",
                "kind": "param",
                "display_name": "data",
                "owner": "sample.b",
                "container": "",
                "field": "data",
                "file": "sample.py",
                "lineno": 6,
                "inferred_type": "dict",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "dict_key:sample.a:data:alpha",
                "kind": "dict_key",
                "display_name": "data['alpha']",
                "owner": "param:sample.a:data",
                "container": "param:sample.a:data",
                "field": "alpha",
                "file": "sample.py",
                "lineno": 3,
                "inferred_type": "",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "dict_key:sample.a:data:beta",
                "kind": "dict_key",
                "display_name": "data['beta']",
                "owner": "param:sample.a:data",
                "container": "param:sample.a:data",
                "field": "beta",
                "file": "sample.py",
                "lineno": 4,
                "inferred_type": "",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "dict_key:sample.b:data:alpha",
                "kind": "dict_key",
                "display_name": "data['alpha']",
                "owner": "param:sample.b:data",
                "container": "param:sample.b:data",
                "field": "alpha",
                "file": "sample.py",
                "lineno": 7,
                "inferred_type": "",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "dict_key:sample.b:data:beta",
                "kind": "dict_key",
                "display_name": "data['beta']",
                "owner": "param:sample.b:data",
                "container": "param:sample.b:data",
                "field": "beta",
                "file": "sample.py",
                "lineno": 8,
                "inferred_type": "",
                "confidence": "high",
                "alias_of": "",
            },
        ],
        edges=[
            {"callable": "sample.a", "object_id": "dict_key:sample.a:data:alpha"},
            {"callable": "sample.a", "object_id": "dict_key:sample.a:data:beta"},
            {"callable": "sample.b", "object_id": "dict_key:sample.b:data:alpha"},
            {"callable": "sample.b", "object_id": "dict_key:sample.b:data:beta"},
        ],
    )

    inferred_config, decisions = infer_shared_containers_from_payload(
        payload,
        [],
        pyright_families={
            "param:sample.a:data": "dict",
            "param:sample.b:data": "dict",
        },
    )

    assert inferred_config["dict_key"] == {}
    assert any("generic_name_without_lineage" in decision.reasons for decision in decisions)


def test_pyright_conflict_blocks_merge():
    payload = _payload(
        objects=[
            {
                "id": "param:sample.a:Results_extended",
                "kind": "param",
                "display_name": "Results_extended",
                "owner": "sample.a",
                "container": "",
                "field": "Results_extended",
                "file": "sample.py",
                "lineno": 2,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "param:sample.b:Results_extended",
                "kind": "param",
                "display_name": "Results_extended",
                "owner": "sample.b",
                "container": "",
                "field": "Results_extended",
                "file": "sample.py",
                "lineno": 6,
                "inferred_type": "dict",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.a:Results_extended:Compartment",
                "kind": "df_col",
                "display_name": "Results_extended['Compartment']",
                "owner": "param:sample.a:Results_extended",
                "container": "param:sample.a:Results_extended",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 3,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.a:Results_extended:mass_g",
                "kind": "df_col",
                "display_name": "Results_extended['mass_g']",
                "owner": "param:sample.a:Results_extended",
                "container": "param:sample.a:Results_extended",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 4,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.b:Results_extended:Compartment",
                "kind": "df_col",
                "display_name": "Results_extended['Compartment']",
                "owner": "param:sample.b:Results_extended",
                "container": "param:sample.b:Results_extended",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 7,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.b:Results_extended:mass_g",
                "kind": "df_col",
                "display_name": "Results_extended['mass_g']",
                "owner": "param:sample.b:Results_extended",
                "container": "param:sample.b:Results_extended",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 8,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
        ],
        edges=[
            {"callable": "sample.a", "object_id": "df_col:sample.a:Results_extended:Compartment"},
            {"callable": "sample.a", "object_id": "df_col:sample.a:Results_extended:mass_g"},
            {"callable": "sample.b", "object_id": "df_col:sample.b:Results_extended:Compartment"},
            {"callable": "sample.b", "object_id": "df_col:sample.b:Results_extended:mass_g"},
        ],
        lineage_edges=[
            {
                "src_object_id": "param:sample.a:Results_extended",
                "dst_object_id": "param:sample.b:Results_extended",
                "relation": "arg_to_param",
                "caller": "sample.orchestrate",
                "callee": "sample.b",
                "file": "sample.py",
                "lineno": 9,
                "slot": "Results_extended",
            }
        ],
    )

    inferred_config, decisions = infer_shared_containers_from_payload(
        payload,
        [],
        pyright_families={
            "param:sample.a:Results_extended": "dataframe",
            "param:sample.b:Results_extended": "dict",
        },
    )

    assert inferred_config["df_col"] == {}
    assert any("pyright_conflict" in decision.reasons for decision in decisions)


def test_cross_name_containers_are_not_auto_merged():
    payload = _payload(
        objects=[
            {
                "id": "param:sample.a:R",
                "kind": "param",
                "display_name": "R",
                "owner": "sample.a",
                "container": "",
                "field": "R",
                "file": "sample.py",
                "lineno": 2,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "param:sample.b:Results_extended",
                "kind": "param",
                "display_name": "Results_extended",
                "owner": "sample.b",
                "container": "",
                "field": "Results_extended",
                "file": "sample.py",
                "lineno": 6,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.a:R:Compartment",
                "kind": "df_col",
                "display_name": "R['Compartment']",
                "owner": "param:sample.a:R",
                "container": "param:sample.a:R",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 3,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.a:R:mass_g",
                "kind": "df_col",
                "display_name": "R['mass_g']",
                "owner": "param:sample.a:R",
                "container": "param:sample.a:R",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 4,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.b:Results_extended:Compartment",
                "kind": "df_col",
                "display_name": "Results_extended['Compartment']",
                "owner": "param:sample.b:Results_extended",
                "container": "param:sample.b:Results_extended",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 7,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.b:Results_extended:mass_g",
                "kind": "df_col",
                "display_name": "Results_extended['mass_g']",
                "owner": "param:sample.b:Results_extended",
                "container": "param:sample.b:Results_extended",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 8,
                "inferred_type": "dataframe_column",
                "confidence": "high",
                "alias_of": "",
            },
        ],
        edges=[],
    )

    inferred_config, _decisions = infer_shared_containers_from_payload(
        payload,
        [],
        pyright_families={
            "param:sample.a:R": "dataframe",
            "param:sample.b:Results_extended": "dataframe",
        },
    )

    assert inferred_config["df_col"] == {}


def test_unknown_pyright_family_blocks_merge():
    payload = _payload(
        objects=[
            {
                "id": "param:sample.a:Results_extended",
                "kind": "param",
                "display_name": "Results_extended",
                "owner": "sample.a",
                "container": "",
                "field": "Results_extended",
                "file": "sample.py",
                "lineno": 2,
                "inferred_type": "unknown",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "param:sample.b:Results_extended",
                "kind": "param",
                "display_name": "Results_extended",
                "owner": "sample.b",
                "container": "",
                "field": "Results_extended",
                "file": "sample.py",
                "lineno": 6,
                "inferred_type": "unknown",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.a:Results_extended:Compartment",
                "kind": "df_col",
                "display_name": "Results_extended['Compartment']",
                "owner": "param:sample.a:Results_extended",
                "container": "param:sample.a:Results_extended",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 3,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.a:Results_extended:mass_g",
                "kind": "df_col",
                "display_name": "Results_extended['mass_g']",
                "owner": "param:sample.a:Results_extended",
                "container": "param:sample.a:Results_extended",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 4,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.b:Results_extended:Compartment",
                "kind": "df_col",
                "display_name": "Results_extended['Compartment']",
                "owner": "param:sample.b:Results_extended",
                "container": "param:sample.b:Results_extended",
                "field": "Compartment",
                "file": "sample.py",
                "lineno": 7,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
            {
                "id": "df_col:sample.b:Results_extended:mass_g",
                "kind": "df_col",
                "display_name": "Results_extended['mass_g']",
                "owner": "param:sample.b:Results_extended",
                "container": "param:sample.b:Results_extended",
                "field": "mass_g",
                "file": "sample.py",
                "lineno": 8,
                "inferred_type": "dataframe",
                "confidence": "high",
                "alias_of": "",
            },
        ],
        edges=[],
        lineage_edges=[
            {
                "src_object_id": "param:sample.a:Results_extended",
                "dst_object_id": "param:sample.b:Results_extended",
                "relation": "arg_to_param",
                "caller": "sample.orchestrate",
                "callee": "sample.b",
                "file": "sample.py",
                "lineno": 9,
                "slot": "Results_extended",
            }
        ],
    )

    inferred_config, decisions = infer_shared_containers_from_payload(payload, [], pyright_families={})

    assert inferred_config["df_col"] == {}
    assert any("pyright_unknown" in decision.reasons for decision in decisions)


def test_probe_regex_extracts_revealed_types():
    output = 'x.py:12:5 - information: Type of "__msp_probe_1" is "DataFrame"\n'
    assert PROBE_RE.findall(output) == [("__msp_probe_1", "DataFrame")]


def test_probe_pyright_families_returns_the_family_mapping(tmp_path: Path):
    # ``probe_pyright_targets`` returns a report, not a mapping. This asserts the
    # unwrapping, because the caller feeds the result straight to ``.get()``.
    source_file = tmp_path / "sample.py"
    source_file.write_text("def fn(data):\n    return data\n", encoding="utf-8")

    fake_pyright = tmp_path / "fake_pyright"
    fake_pyright.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' 'sample.py:2:1 - information: Type of \"__msp_probe_1\" is \"dict[str, float]\"'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_pyright.chmod(0o755)

    candidate = CandidateContainer(
        kind="df_col",
        container_id="param:sample.fn:data",
        leaf_name="data",
        normalized_leaf="data",
        object_kind="dict_key",
        file=str(source_file),
        lineno=1,
    )

    families = probe_pyright_families(tmp_path, tmp_path, [candidate], str(fake_pyright))

    assert families == {"param:sample.fn:data": "dict"}
    # The consumer in ``infer_shared_containers_from_payload`` calls ``.get()``.
    assert families.get("param:sample.fn:data") == "dict"
