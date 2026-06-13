from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from microservice_pipeline.call_graph.generate_call_graph_ast import (
    iter_analysis_files_for_source_roots,
)
from microservice_pipeline.config import load_extraction_config
from microservice_pipeline.data_access.generate_data_access_ast import run_from_extraction_config
from microservice_pipeline.jsonc_config import load_jsonc


def test_extraction_config_resolves_project_relative_paths(tmp_path: Path):
    config_path = tmp_path / "analysis" / "extraction.jsonc"
    config_path.parent.mkdir()
    config_path.write_text(
        """
{
  "project_root": "..",
  "source": {
    "roots": [{"path": "src/pkg", "module_prefix": "pkg"}],
    "package_prefixes": ["pkg"],
    "entrypoints": ["tools/run.py"],
    "include_globs": ["**/*.py"],
    "exclude_globs": ["src/pkg/ignored.py"]
  },
  "call_graph": {"outdir": "artifacts/call_graph"},
  "data_access": {
    "outdir": "artifacts/data_access",
    "pyright": {"enabled": false}
  }
}
""",
        encoding="utf-8",
    )

    config = load_extraction_config(config_path)

    assert config.project_root == tmp_path
    assert config.source_roots[0].path == tmp_path / "src/pkg"
    assert config.entrypoints == (tmp_path / "tools/run.py",)
    assert config.call_graph.outdir == tmp_path / "artifacts/call_graph"
    assert config.data_access.pyright.enabled is False


def test_source_selection_supports_excludes_and_multiple_roots(tmp_path: Path):
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    src_a.mkdir()
    src_b.mkdir()
    (src_a / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (src_a / "skip.py").write_text("def skip():\n    pass\n", encoding="utf-8")
    (src_b / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")

    config_path = tmp_path / "extraction.jsonc"
    config_path.write_text(
        """
{
  "project_root": ".",
  "source": {
    "roots": [
      {"path": "src_a", "module_prefix": "pkg_a"},
      {"path": "src_b", "module_prefix": "pkg_b"}
    ],
    "include_globs": ["**/*.py"],
    "exclude_globs": ["src_a/skip.py"]
  }
}
""",
        encoding="utf-8",
    )
    config = load_extraction_config(config_path)

    analysis_files = iter_analysis_files_for_source_roots(
        config.source_roots,
        project_root=config.project_root,
        include_globs=config.include_globs,
        exclude_globs=config.exclude_globs,
    )

    assert {file.module for file in analysis_files} == {"pkg_a.a", "pkg_b.b"}


def test_data_access_config_can_run_without_pyright(tmp_path: Path):
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "module.py").write_text(
        "def read(config):\n    return config['solver']\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "extraction.jsonc"
    config_path.write_text(
        """
{
  "project_root": ".",
  "source": {
    "roots": [{"path": "src/pkg", "module_prefix": "pkg"}],
    "package_prefixes": ["pkg"]
  },
  "data_access": {
    "outdir": "artifacts/data_access",
    "pyright": {"enabled": false}
  }
}
""",
        encoding="utf-8",
    )

    callable_count, object_count, edge_count = run_from_extraction_config(
        load_extraction_config(config_path)
    )

    assert callable_count >= 1
    assert object_count >= 1
    assert edge_count >= 1
    assert (tmp_path / "artifacts/data_access/data_access.json").is_file()


def test_data_access_config_missing_pyright_falls_back_to_unknown(tmp_path: Path):
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "module.py").write_text(
        "def read(config):\n    return config['solver']\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "extraction.jsonc"
    config_path.write_text(
        """
{
  "project_root": ".",
  "source": {
    "roots": [{"path": "src/pkg", "module_prefix": "pkg"}],
    "package_prefixes": ["pkg"]
  },
  "data_access": {
    "outdir": "artifacts/data_access",
    "pyright": {
      "enabled": true,
      "bin": "definitely-missing-pyright",
      "fail_policy": "fallback_unknown"
    }
  }
}
""",
        encoding="utf-8",
    )

    _callable_count, _object_count, edge_count = run_from_extraction_config(
        load_extraction_config(config_path)
    )

    assert edge_count >= 1


def test_packaged_templates_are_generic():
    template_dir = (
        Path(__file__).resolve().parents[1]
        / "src/microservice_pipeline/configs/templates"
    )
    template_paths = sorted(template_dir.iterdir())
    templates = {
        path.name: load_jsonc(path)
        for path in template_paths
        if path.suffix in {".json", ".jsonc"}
    }
    text = "\n".join(path.read_text(encoding="utf-8") for path in template_paths)
    manual_mapping = (template_dir / "manual_mapping.csv").read_text(encoding="utf-8")
    structural_graph = templates["structural_graph.jsonc"]
    structural_clustering = templates["structural_clustering.jsonc"]
    evaluation = templates["evaluation.jsonc"]

    assert "src/utopia" not in text
    assert "artifacts/manual_results_mapping" not in text
    assert "UTOPIA" not in text
    assert {path.name for path in template_paths} == {
        "evaluation.jsonc",
        "extraction.jsonc",
        "manual_mapping.csv",
        "notebook_task_analysis.jsonc",
        "shared_containers.jsonc",
        "structural_clustering.jsonc",
        "structural_graph.jsonc",
    }
    assert manual_mapping == "microservice_id,node,node_type,label,kind,module\n"
    assert structural_graph["paths"]["data_access"] == "artifacts/data_access_inferred/data_access.json"
    assert structural_graph["weighting"]["weight_config"] == "builtin:default"
    assert structural_clustering["weighting"]["weight_config"] == "builtin:default"
    assert structural_clustering["paths"]["outdir"] == "artifacts/structural_microservice_candidates_{algorithm}"
    assert structural_clustering["paths"]["sweep_outdir"] == "artifacts/structural_microservice_candidates_{algorithm}_sweep"
    assert structural_clustering["sweep_best"]["outdir"] == (
        "artifacts/structural_microservice_candidates_{algorithm}_sweep/best"
    )
    assert evaluation["paths"]["manual"] == "configs/microservice_pipeline/manual_mapping.csv"
    assert set(structural_clustering["hub_policy"]) >= {
        "hub_callable_degree_percentile",
        "hub_callable_min_degree",
        "hub_callable_min_in_degree",
        "hub_callable_min_out_degree",
        "hub_entrypoint_min_out_degree",
        "hub_orchestrator_max_in_degree",
        "hub_orchestrator_min_out_degree",
        "hub_orchestrator_min_out_call_degree",
        "hub_orchestrator_min_target_modules",
        "hub_orchestrator_min_target_callables",
        "hub_orchestrator_min_target_data",
        "hub_orchestrator_min_data_to_call_ratio",
        "hub_data_min_degree",
        "hub_data_min_callable_count",
        "hub_data_min_access_count",
    }


def test_package_readme_is_standalone():
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    text = readme_path.read_text(encoding="utf-8").lower()

    assert "utopia" not in text
    assert "no_database_project" not in text
    assert "no_database_quickstart" not in text
    assert "pip install -e ." in text
    assert "pip install -e ./microservice_pipeline" not in text
    assert "artifacts/structural_microservice_candidates/cluster_assignments.csv" in text
    assert "microservice_id,node,node_type,label,kind,module" in text


def test_packaged_cli_help():
    package_src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(package_src)}

    cli = subprocess.run(
        [sys.executable, "-m", "microservice_pipeline.cli.main", "call-graph", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert cli.returncode == 0, cli.stderr
    assert "--config" in cli.stdout
