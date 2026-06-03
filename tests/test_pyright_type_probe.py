import ast
from pathlib import Path

from microservice_pipeline.data_access.generate_data_access_ast import collect_pyright_probe_targets
from microservice_pipeline.data_access.pyright_type_probe import (
    PyrightProbeTarget,
    _apply_probes_to_source,
    _callable_node_map,
    parse_pyright_probe_output,
    probe_pyright_targets,
    pyright_family_from_type_text,
)


def test_pyright_family_from_type_text_reduces_known_types():
    assert pyright_family_from_type_text("pd.DataFrame") == "dataframe"
    assert pyright_family_from_type_text("dict[str, float]") == "dict"
    assert pyright_family_from_type_text("list[ParticulatesSPM]") == "list"
    assert pyright_family_from_type_text("Path") == "path"
    assert pyright_family_from_type_text("IO[str]") == "file"
    assert pyright_family_from_type_text("Any") == "unknown"
    assert pyright_family_from_type_text("Unknown") == "unknown"
    assert pyright_family_from_type_text("ParticulatesSPM") == "object"


def test_parse_pyright_probe_output_maps_probe_ids():
    output = 'x.py:12:5 - information: Type of "__msp_probe_1" is "DataFrame"\n'
    families = parse_pyright_probe_output(output, {"__msp_probe_1": "param:sample.fn:data"})
    assert families == {"param:sample.fn:data": "dataframe"}


def test_probe_pyright_targets_raises_when_binary_missing(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    source_file = src_dir / "sample.py"
    source_file.write_text("def fn(data):\n    return data\n", encoding="utf-8")
    (tmp_path / "pyrightconfig.json").write_text('{"include": ["src"]}\n', encoding="utf-8")

    target = PyrightProbeTarget(
        target_id="param:sample.fn:data",
        expression="data",
        file=source_file,
        module="sample",
        mode="callable_entry",
        callable_id="sample.fn",
        lineno=1,
    )

    try:
        probe_pyright_targets(tmp_path, [target], "pyright-does-not-exist")
    except RuntimeError as exc:
        assert "Pyright binary not found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing Pyright binary")


def test_probe_pyright_targets_parses_fake_pyright_output(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    source_file = src_dir / "sample.py"
    source_file.write_text("def fn(data):\n    return data\n", encoding="utf-8")
    (tmp_path / "pyrightconfig.json").write_text('{"include": ["src"]}\n', encoding="utf-8")

    fake_pyright = tmp_path / "fake_pyright"
    fake_pyright.write_text(
        "#!/bin/sh\n"
        "printf '%s\n' 'sample.py:2:1 - information: Type of \"__msp_probe_1\" is \"dict[str, float]\"'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_pyright.chmod(0o755)

    target = PyrightProbeTarget(
        target_id="param:sample.fn:data",
        expression="data",
        file=source_file,
        module="sample",
        mode="callable_entry",
        callable_id="sample.fn",
        lineno=1,
    )

    families = probe_pyright_targets(tmp_path, [target], str(fake_pyright))

    assert families == {"param:sample.fn:data": "dict"}


def test_attribute_probe_uses_statement_end_for_multiline_assignment(tmp_path: Path):
    source = """\
class Compartment:
    def __init__(self):
        self.particles = {
            "free": [],
            "bound": [],
        }
"""
    source_file = tmp_path / "sample.py"
    source_file.write_text(source, encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))

    target = next(
        target
        for target in collect_pyright_probe_targets(tree, "sample", source_file)
        if target.target_id == "class_attr:sample.Compartment:particles"
    )
    modified = _apply_probes_to_source(
        source,
        [("__msp_probe_1", target)],
        _callable_node_map(tree, "sample"),
    )

    assert target.lineno == 3
    assert target.insert_lineno == 6
    assert '        }' in modified
    assert '        __msp_probe_1 = self.particles' in modified
    assert modified.index('        }') < modified.index("        __msp_probe_1 = self.particles")
