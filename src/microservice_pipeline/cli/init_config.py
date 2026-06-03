"""Copy editable starter configs into an analyzed project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from microservice_pipeline.package_resources import CONFIG_TEMPLATE_NAMES, config_template_resource


def _render_template(name: str, *, project_root: Path, outdir: Path) -> str:
    text = config_template_resource(name).read_text(encoding="utf-8")
    if name != "extraction.jsonc":
        return text
    relative_project_root = Path(os.path.relpath(project_root, outdir)).as_posix()
    return text.replace(
        '"project_root": "."',
        f'"project_root": {json.dumps(relative_project_root)}',
        1,
    )


def copy_starter_configs(
    *,
    project_root: Path,
    outdir: Path,
    force: bool = False,
) -> tuple[Path, ...]:
    project_root = project_root.resolve()
    outdir = outdir if outdir.is_absolute() else project_root / outdir
    outdir = outdir.resolve()
    targets = tuple(outdir / name for name in CONFIG_TEMPLATE_NAMES)
    existing = tuple(path for path in targets if path.exists())
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing config files: {joined}. Pass --force to overwrite.")

    outdir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        target.write_text(
            _render_template(target.name, project_root=project_root, outdir=outdir),
            encoding="utf-8",
        )
    return targets


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", type=Path, help="Analyzed project root")
    parser.add_argument(
        "--outdir",
        default=Path("configs/microservice_pipeline"),
        type=Path,
        help="Config output directory, relative to --project-root unless absolute",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing starter config files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    targets = copy_starter_configs(
        project_root=args.project_root,
        outdir=args.outdir,
        force=args.force,
    )
    print("Starter configs written:")
    for target in targets:
        print(f"  - {target}")


if __name__ == "__main__":
    main()

