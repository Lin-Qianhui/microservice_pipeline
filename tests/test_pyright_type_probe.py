import ast
from pathlib import Path

import pytest

from microservice_pipeline.data_access.generate_data_access_ast import (
    _report_pyright_resolution,
    collect_pyright_probe_targets,
)
from microservice_pipeline.data_access.pyright_type_probe import (
    PyrightProbeReport,
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
    assert pyright_family_from_type_text("Field") == "field"
    assert pyright_family_from_type_text("climlab.domain.field.Field") == "field"
    assert pyright_family_from_type_text("numpy.ndarray") == "field"
    assert pyright_family_from_type_text("numpy.typing.NDArray[Any]") == "field"
    assert pyright_family_from_type_text("xarray.Dataset") == "xarray"
    assert pyright_family_from_type_text("xarray.DataArray") == "xarray"
    assert pyright_family_from_type_text("AttrDict") == "dict"
    assert pyright_family_from_type_text("Any") == "unknown"
    assert pyright_family_from_type_text("Unknown") == "unknown"
    assert pyright_family_from_type_text("ParticulatesSPM") == "object"


def test_pyright_field_family_does_not_match_schema_field_types():
    assert pyright_family_from_type_text("pydantic.fields.FieldInfo") == "object"
    assert pyright_family_from_type_text("dataclasses.Field[Any]") == "object"
    assert pyright_family_from_type_text("django.db.models.fields.CharField") == "object"
    assert pyright_family_from_type_text("ModelField") == "object"


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

    report = probe_pyright_targets(tmp_path, [target], str(fake_pyright))

    assert report.families == {"param:sample.fn:data": "dict"}
    assert report.probes_emitted == 1
    assert report.probes_resolved == 1


def test_probe_pyright_targets_writes_limited_config_for_top_level_package(tmp_path: Path):
    package_dir = tmp_path / "climlab"
    package_dir.mkdir()
    source_file = package_dir / "sample.py"
    source_file.write_text("def fn(data):\n    return data\n", encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "conf.py").write_text("import missing_docs_dependency\n", encoding="utf-8")

    fake_pyright = tmp_path / "fake_pyright"
    fake_pyright.write_text(
        "#!/bin/sh\n"
        "config=\"$2\"\n"
        "if [ ! -f \"$config\" ]; then echo missing config; exit 5; fi\n"
        "if grep -q 'docs' \"$config\"; then echo docs included; exit 6; fi\n"
        "if ! grep -q 'climlab/sample.py' \"$config\"; then echo sample missing; exit 7; fi\n"
        "printf '%s\n' 'climlab/sample.py:2:1 - information: Type of \"__msp_probe_1\" is \"dict[str, float]\"'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_pyright.chmod(0o755)

    target = PyrightProbeTarget(
        target_id="param:climlab.sample.fn:data",
        expression="data",
        file=source_file,
        module="climlab.sample",
        mode="callable_entry",
        callable_id="climlab.sample.fn",
        lineno=1,
    )

    report = probe_pyright_targets(tmp_path, [target], str(fake_pyright))

    assert report.families == {"param:climlab.sample.fn:data": "dict"}


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


def test_type_text_is_read_as_a_nesting_not_a_substring_soup():
    # The family wanted is the outermost constructor. Every one of these used to
    # be "unknown": the first three because the text mentioned a second family,
    # and dict[str, Any] because " any" was tested as a substring.
    assert pyright_family_from_type_text("dict[str, list[int]]") == "dict"
    assert pyright_family_from_type_text("defaultdict[str, list[int]]") == "dict"
    assert pyright_family_from_type_text("dict[str, DataFrame]") == "dict"
    assert pyright_family_from_type_text("dict[str, Any]") == "dict"
    assert pyright_family_from_type_text("dict[str, Unknown]") == "dict"
    assert pyright_family_from_type_text("dict[Unknown, Unknown]") == "dict"
    assert pyright_family_from_type_text("list[Unknown]") == "list"
    assert pyright_family_from_type_text("frozenset[Unknown]") == "set"
    assert pyright_family_from_type_text("ndarray[tuple[int], dtype[Unknown]]") == "field"


def test_uninformative_union_members_are_dropped_before_members_are_compared():
    assert pyright_family_from_type_text("Unknown | Field | None") == "field"
    assert pyright_family_from_type_text("dict[Unknown, Unknown] | dict[str, Axis]") == "dict"
    # Nothing informative left, and members that genuinely disagree, both stay
    # unknown -- a value that is a str on one branch and an array on another has
    # no one family, and saying so is the honest answer.
    assert pyright_family_from_type_text("Unknown | None") == "unknown"
    assert pyright_family_from_type_text("str | ndarray[Any]") == "unknown"


def test_a_function_is_never_a_container():
    # A probe that lands on a method reference gets a signature back. Reading the
    # names inside it is how "(name: Unknown, value: ...) -> None" was answered
    # unknown and how numpy's overloads were mistaken for lists and paths.
    assert pyright_family_from_type_text("(name: Unknown, value: int) -> None") == "object"
    assert pyright_family_from_type_text("Overload[(a: int) -> None, (a: str) -> None]") == "object"
    assert pyright_family_from_type_text("(a: Sequence[int]) -> list[int]") == "object"
    # Nor is the class object itself, or a tuple of arrays, the array.
    assert pyright_family_from_type_text("type[ndarray[tuple[Any, ...], dtype[Any]]]") == "object"
    assert pyright_family_from_type_text("tuple[ndarray[Any], ...]") == "object"


def test_pyrights_self_notation_is_still_the_class():
    assert pyright_family_from_type_text("Self@AttrDict") == "dict"


def test_a_bare_imported_name_is_qualified_through_the_module_that_wrote_it():
    # Pyright prints "Dataset" whether it is xarray's or the project's own, so
    # the module's own imports are the only thing that can tell them apart.
    assert pyright_family_from_type_text("Dataset") == "object"
    assert pyright_family_from_type_text("Dataset", {"Dataset": "xarray.Dataset"}) == "xarray"
    assert pyright_family_from_type_text("Dataset", {"Dataset": "myproject.Dataset"}) == "object"
    assert pyright_family_from_type_text("DataArray", {"DataArray": "xarray.DataArray"}) == "xarray"


def test_probe_for_a_return_expression_is_inserted_before_the_return():
    source = """\
class Grid:
    def build(self):
        return self.data['x']
"""
    source_file = Path("sample.py")
    tree = ast.parse(source, filename="sample.py")
    target = next(
        target
        for target in collect_pyright_probe_targets(tree, "sample", source_file)
        if target.target_id == "class_attr:sample.Grid:data"
    )

    assert target.mode == "before_line"
    assert target.insert_lineno == 3

    modified = _apply_probes_to_source(
        source, [("__msp_probe_1", target)], _callable_node_map(tree, "sample")
    )
    lines = modified.splitlines()

    # Placed after the return -- as it used to be -- pyright types it as
    # unreachable, answers Unknown, and reports no problem.
    assert lines[2] == "        __msp_probe_1 = self.data"
    assert lines[3] == "        reveal_type(__msp_probe_1)"
    assert lines[4] == "        return self.data['x']"
    ast.parse(modified)


def test_probe_is_dropped_rather_than_wedged_into_a_one_line_body():
    source = "class Grid:\n    def build(self):\n        if self.flag: return self.data['x']\n"
    source_file = Path("sample.py")
    tree = ast.parse(source, filename="sample.py")
    targets = {
        target.target_id: target
        for target in collect_pyright_probe_targets(tree, "sample", source_file)
    }

    assert targets["class_attr:sample.Grid:data"].mode == "before_line"
    # The condition's own statement is the ``if``, which is not terminal.
    assert targets["class_attr:sample.Grid:flag"].mode == "after_line"

    modified = _apply_probes_to_source(
        source,
        [("__msp_probe_1", targets["class_attr:sample.Grid:data"])],
        _callable_node_map(tree, "sample"),
    )

    assert modified == source
    ast.parse(modified)


def test_sandbox_holds_analyzed_files_that_carry_no_probe_target(tmp_path: Path):
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    # Pure re-exports, so never a probe target -- and every import of the package
    # goes through it.
    init_file = package_dir / "__init__.py"
    init_file.write_text("from pkg.sample import fn\n", encoding="utf-8")
    source_file = package_dir / "sample.py"
    source_file.write_text("def fn(data):\n    return data\n", encoding="utf-8")

    fake_pyright = tmp_path / "fake_pyright"
    fake_pyright.write_text(
        "#!/bin/sh\n"
        'config="$2"\n'
        'root=$(dirname "$config")\n'
        'if [ ! -f "$root/pkg/__init__.py" ]; then echo init missing; exit 5; fi\n'
        "printf '%s\n' 'pkg/sample.py:2:1 - information: Type of \"__msp_probe_1\" is \"dict[str, float]\"'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_pyright.chmod(0o755)

    target = PyrightProbeTarget(
        target_id="param:pkg.sample.fn:data",
        expression="data",
        file=source_file,
        module="pkg.sample",
        mode="callable_entry",
        callable_id="pkg.sample.fn",
        lineno=1,
    )

    report = probe_pyright_targets(
        tmp_path, [target], str(fake_pyright), support_files=[init_file, source_file]
    )

    assert report.families == {"param:pkg.sample.fn:data": "dict"}
    assert report.support_files_copied == 2


def test_a_probe_run_that_resolves_nothing_is_an_error_not_a_clean_run():
    # Every family would be "unknown", which the artifacts also use as a real
    # answer -- so without this the two are indistinguishable.
    nothing_resolved = PyrightProbeReport(
        families={}, probes_emitted=100, probes_answered=100, answers_unknown=100
    )
    with pytest.raises(RuntimeError, match="resolved none"):
        _report_pyright_resolution(nothing_resolved)

    # One resolved type is enough to say the probe ran; how many is enough is a
    # property of the analyzed project, so it is printed rather than judged.
    _report_pyright_resolution(
        PyrightProbeReport(
            families={"x": "dict"}, probes_emitted=100, probes_answered=100, answers_unknown=99
        )
    )
    _report_pyright_resolution(PyrightProbeReport(families={}))
