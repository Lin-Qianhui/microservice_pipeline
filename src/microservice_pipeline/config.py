"""Config models and path helpers for packaged pipeline commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from microservice_pipeline.jsonc_config import load_jsonc


@dataclass(frozen=True)
class SourceRootConfig:
    path: Path
    module_prefix: Optional[str] = None


@dataclass(frozen=True)
class PyrightConfig:
    enabled: bool = True
    bin: str = "pyright"
    fail_policy: str = "fallback_unknown"


@dataclass(frozen=True)
class CallGraphExtractionConfig:
    include_external: bool = False
    outdir: Path = Path("artifacts/call_graph")


@dataclass(frozen=True)
class DataAccessExtractionConfig:
    outdir: Path = Path("artifacts/data_access")
    pyright: PyrightConfig = PyrightConfig()
    shared_containers_config: Optional[Path] = None
    inferred_shared_containers_config: Path = Path("artifacts/inferred_shared_containers.json")
    inferred_shared_containers_report: Path = Path("artifacts/inferred_shared_containers.md")


@dataclass(frozen=True)
class ExtractionConfig:
    project_root: Path
    source_roots: Tuple[SourceRootConfig, ...]
    package_prefixes: Tuple[str, ...] = ()
    entrypoints: Tuple[Path, ...] = ()
    include_globs: Tuple[str, ...] = ()
    exclude_globs: Tuple[str, ...] = ()
    call_graph: CallGraphExtractionConfig = CallGraphExtractionConfig()
    data_access: DataAccessExtractionConfig = DataAccessExtractionConfig()

    @property
    def package_prefix(self) -> Optional[str]:
        return self.package_prefixes[0] if self.package_prefixes else None


def resolve_project_path(project_root: Path, value: str | Path | None) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_extraction_config(path: Path | str) -> ExtractionConfig:
    config_path = Path(path).resolve()
    payload = load_jsonc(config_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Extraction config must be a JSON object: {config_path}")

    base_dir = config_path.parent
    raw_project_root = payload.get("project_root", ".")
    project_root = Path(raw_project_root)
    if not project_root.is_absolute():
        project_root = (base_dir / project_root).resolve()
    else:
        project_root = project_root.resolve()

    source = _mapping(payload.get("source"), "source")
    source_roots = _source_roots(source, project_root)
    if not source_roots:
        raise ValueError(f"Extraction config source.roots must contain at least one root: {config_path}")

    call_graph = _call_graph_config(_mapping(payload.get("call_graph"), "call_graph"), project_root)
    data_access = _data_access_config(_mapping(payload.get("data_access"), "data_access"), project_root)

    return ExtractionConfig(
        project_root=project_root,
        source_roots=tuple(source_roots),
        package_prefixes=tuple(_string_list(source.get("package_prefixes", source.get("package", [])))),
        entrypoints=_path_tuple(project_root, source.get("entrypoints", source.get("entrypoint", []))),
        include_globs=tuple(_string_list(source.get("include_globs", source.get("include", [])))),
        exclude_globs=tuple(_string_list(source.get("exclude_globs", source.get("exclude", [])))),
        call_graph=call_graph,
        data_access=data_access,
    )


def _source_roots(source: Mapping[str, Any], project_root: Path) -> list[SourceRootConfig]:
    roots_value = source.get("roots")
    if roots_value is None and source.get("root") is not None:
        roots_value = [{"path": source.get("root"), "module_prefix": source.get("module_prefix")}]
    if roots_value is None and source.get("source_root") is not None:
        roots_value = [{"path": source.get("source_root"), "module_prefix": source.get("module_prefix")}]

    roots: list[SourceRootConfig] = []
    if isinstance(roots_value, Sequence) and not isinstance(roots_value, (str, bytes)):
        for item in roots_value:
            if isinstance(item, Mapping):
                raw_path = item.get("path") or item.get("root") or item.get("source_root")
                module_prefix = _optional_text(item.get("module_prefix"))
            else:
                raw_path = item
                module_prefix = None
            path = resolve_project_path(project_root, str(raw_path) if raw_path is not None else None)
            if path is not None:
                roots.append(SourceRootConfig(path=path, module_prefix=module_prefix))
    return roots


def _path_tuple(project_root: Path, value: Any) -> Tuple[Path, ...]:
    paths: list[Path] = []
    for item in _string_list(value):
        path = resolve_project_path(project_root, item)
        if path is not None:
            paths.append(path)
    return tuple(paths)


def _call_graph_config(payload: Mapping[str, Any], project_root: Path) -> CallGraphExtractionConfig:
    return CallGraphExtractionConfig(
        include_external=_bool(payload.get("include_external"), False),
        outdir=resolve_project_path(project_root, payload.get("outdir")) or (project_root / "artifacts/call_graph"),
    )


def _data_access_config(payload: Mapping[str, Any], project_root: Path) -> DataAccessExtractionConfig:
    pyright = _mapping(payload.get("pyright"), "data_access.pyright")
    fail_policy = str(pyright.get("fail_policy") or "fallback_unknown")
    if fail_policy not in {"fallback_unknown", "error"}:
        raise ValueError("data_access.pyright.fail_policy must be 'fallback_unknown' or 'error'")
    return DataAccessExtractionConfig(
        outdir=resolve_project_path(project_root, payload.get("outdir")) or (project_root / "artifacts/data_access"),
        pyright=PyrightConfig(
            enabled=_bool(pyright.get("enabled"), True),
            bin=str(pyright.get("bin") or "pyright"),
            fail_policy=fail_policy,
        ),
        shared_containers_config=resolve_project_path(project_root, payload.get("shared_containers_config")),
        inferred_shared_containers_config=(
            resolve_project_path(project_root, payload.get("inferred_shared_containers_config"))
            or (project_root / "artifacts/inferred_shared_containers.json")
        ),
        inferred_shared_containers_report=(
            resolve_project_path(project_root, payload.get("inferred_shared_containers_report"))
            or (project_root / "artifacts/inferred_shared_containers.md")
        ),
    )


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section must be an object: {key}")
    return value


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None and item != ""]
    return [str(value)]


def _optional_text(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
