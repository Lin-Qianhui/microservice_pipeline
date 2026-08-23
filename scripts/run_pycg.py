"""Run PyCG over the same file scope the pipeline's call-graph extraction uses.

PyCG is unmaintained and does not import cleanly on modern Python, so it lives in
its own virtualenv rather than the project one. Create that env once:

    /path/to/python3.11 -m venv .pycg-venv
    .pycg-venv/bin/pip install "pycg==0.0.7" "setuptools<81"

then run this script with that interpreter, pointing it at the same extraction
config the analyzed project uses:

    .pycg-venv/bin/python scripts/run_pycg.py \
        --config /path/to/project/configs/extraction.jsonc \
        -o /path/to/project/artifacts/call_graph/pycg_call_graph.json

Two constraints are baked in and both matter:

- Python 3.11 exactly. PyCG 0.0.7 still uses `ast.Str`/`ast.Num`, removed in 3.12.
- setuptools<81, because PyCG imports `pkg_resources`, dropped in setuptools 81.

The file scope is read from the extraction config rather than restated here, so
the two graphs stay comparable when the config's excludes change.
"""

from __future__ import annotations

# Must precede PyCG. PyCG installs a custom sys.path_hooks entry and clears the
# importer cache (pycg/machinery/imports.py). CPython lazily imports unicodedata
# when compiling non-ASCII identifiers, and that lazy import gets intercepted by
# PyCG's hook, yielding a module without `normalize`. Importing it here puts the
# real module in sys.modules before the hook exists.
#
# Any project with a non-ASCII identifier anywhere in scope trips this -- UTOPIA
# does so via the `φ1_*` names in results_processing/emission_fractions_calculation.py.
import unicodedata  # noqa: F401

import argparse
import fnmatch
import json
import os
import runpy
import sys
from pathlib import Path


def strip_jsonc(text: str) -> str:
    """Strip // and /* */ comments and trailing commas, respecting string literals."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            i = text.find("\n", i)
            if i == -1:
                break
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        out.append(ch)
        i += 1

    # Trailing commas, now that comments are gone.
    cleaned = "".join(out)
    result: list[str] = []
    in_str = False
    for idx, ch in enumerate(cleaned):
        if in_str:
            result.append(ch)
            if ch == '"' and cleaned[idx - 1] != "\\":
                in_str = False
            continue
        if ch == '"':
            in_str = True
            result.append(ch)
            continue
        if ch == ",":
            rest = cleaned[idx + 1 :].lstrip()
            if rest[:1] in ("}", "]"):
                continue
        result.append(ch)
    return "".join(result)


def matches_any(rel_paths: list[str], globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(p, g) for p in rel_paths for g in globs)


def collect_files(config: dict, config_path: Path) -> tuple[list[Path], Path]:
    """Return (files to analyze, the --package root to pass to PyCG)."""
    project_root = (config_path.parent / config.get("project_root", ".")).resolve()
    source = config.get("source", {})
    include = source.get("include_globs", ["**/*.py"])
    exclude = source.get("exclude_globs", [])

    files: list[Path] = []
    package_roots: list[Path] = []

    for root in source.get("roots", []):
        root_path = (project_root / root["path"]).resolve()
        package_roots.append(root_path)
        for path in sorted(root_path.rglob("*.py")):
            # Globs in the config may be written relative to either the source
            # root or the project root, so test both spellings.
            rels = [
                path.relative_to(root_path).as_posix(),
                path.relative_to(project_root).as_posix(),
            ]
            if not matches_any(rels, include):
                continue
            if matches_any(rels, exclude):
                continue
            files.append(path)

    for entry in source.get("entrypoints", []):
        path = (project_root / entry).resolve()
        if path.is_file():
            files.append(path)

    # PyCG derives module names by stripping the --package prefix. Passing the
    # *parent* of the scanned root ("src", not "src/utopia") makes them come out
    # as `utopia.foo.bar`, matching the pipeline's node ids.
    package = package_roots[0].parent if package_roots else project_root
    return files, package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=-1,
        help="PyCG fixpoint iteration cap; -1 runs to fixpoint",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 11):
        sys.exit(
            f"PyCG 0.0.7 requires Python 3.11 (uses ast.Str/ast.Num); "
            f"this interpreter is {sys.version.split()[0]}"
        )

    config_path = args.config.resolve()
    config = json.loads(strip_jsonc(config_path.read_text(encoding="utf-8")))
    files, package = collect_files(config, config_path)
    if not files:
        sys.exit("no source files matched the config's scope")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"analyzing {len(files)} files, --package {package}", file=sys.stderr)

    # PyCG resolves relative imports against the cwd, so anchor it at the root
    # that contains the package.
    os.chdir(package.parent)
    sys.argv = [
        "pycg",
        "--package",
        str(package),
        "--max-iter",
        str(args.max_iter),
        "-o",
        str(args.output.resolve()),
        *[str(f) for f in files],
    ]
    runpy.run_module("pycg", run_name="__main__")


if __name__ == "__main__":
    main()
