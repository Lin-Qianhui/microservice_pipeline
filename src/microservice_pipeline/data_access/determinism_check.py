"""Prove the data-access artifacts are a function of the input, not of file order.

The extractor walks one file at a time and threads eight mutable dictionaries
through every pass, so a fact learned from file A can win over a better fact from
file B for no reason other than that A was reached first. Nothing in the artifacts
says which happened, and an artifact diff therefore cannot tell a real change from
a reshuffle -- which is what blocks any later step that verifies itself by diffing
artifacts.

This module is the instrument for that. It runs the whole stage twice on the same
project, once in discovery order and once with the file list shuffled, and
compares the artifacts byte for byte.

Two properties, both learned from the access oracle of Step 1a:

*It reports named differences, not a verdict.* A boolean that can only be true or
false hides its own bugs. When a run differs, the exact rows are printed so they
can be taken to the source, which is how the oracle's own three defects were
caught.

*The call graph is checked too, on request.* Data access consumes ``callable_map``,
``registration_rules`` and ``project_index`` from the call-graph passes, so a
difference there would look exactly like a difference here. ``--check-inputs``
compares those three before the artifacts, so the two causes cannot be confused.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from microservice_pipeline.call_graph.discovery import (
        iter_analysis_files_for_source_roots,
    )
    from microservice_pipeline.call_graph.generate_call_graph_ast import analyze_analysis_files
    from microservice_pipeline.call_graph.models import AnalysisFile
    from microservice_pipeline.config import ExtractionConfig, load_extraction_config
    from microservice_pipeline.data_access.generate_data_access_ast import (
        run_from_extraction_config,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from call_graph.discovery import (  # type: ignore
        iter_analysis_files_for_source_roots,
    )
    from call_graph.generate_call_graph_ast import analyze_analysis_files  # type: ignore
    from call_graph.models import AnalysisFile  # type: ignore
    from config import ExtractionConfig, load_extraction_config  # type: ignore
    from data_access.generate_data_access_ast import (  # type: ignore
        run_from_extraction_config,
    )


def artifact_names(*dirs: Path) -> List[str]:
    """Every file either run wrote.

    Discovered rather than hardcoded so a new artifact is compared the day it
    appears. A hardcoded list that silently skips a file is the failure mode this
    whole check exists to rule out.
    """
    names = set()
    for directory in dirs:
        if directory.exists():
            names.update(path.name for path in directory.iterdir() if path.is_file())
    return sorted(names)


# --------------------------------------------------------------------------
# Running the stage
# --------------------------------------------------------------------------


def run_stage(
    config: ExtractionConfig,
    analysis_files: Sequence[AnalysisFile],
    outdir: Path,
) -> None:
    """Run the real production entry point, with the file list we were given.

    Deliberately not a local copy of what ``run_from_extraction_config`` does.
    A copy would drift from it -- and a determinism check that has quietly
    stopped exercising the real pipeline still reports success, which is exactly
    the class of silent failure this check exists to rule out. It is also the
    defect ``code_review.md`` §5 catalogues: this package reimplementing
    something it could call.

    Note that the call-graph passes are re-run per ordering, inside that
    function, rather than computed once and shared. That is the point: shuffling
    has to shuffle the whole pipeline's view of the project, or an
    order-dependence originating upstream would be invisible here.
    """
    run_from_extraction_config(config, outdir=outdir, analysis_files=analysis_files)


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------


def _normalize(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _normalize(item) for key, item in sorted(asdict(value).items())}
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def compare_call_graph_inputs(
    config: ExtractionConfig,
    baseline_files: Sequence[AnalysisFile],
    shuffled_files: Sequence[AnalysisFile],
) -> List[str]:
    """Differences in the three call-graph structures data access consumes."""
    first = analyze_analysis_files(
        baseline_files, summary_packages=config.call_graph.summary_packages
    )
    second = analyze_analysis_files(
        shuffled_files, summary_packages=config.call_graph.summary_packages
    )
    checks: Tuple[Tuple[str, object, object], ...] = (
        ("callable_map", first.project_nodes(), second.project_nodes()),
        ("registration_rules", first.registration_rules, second.registration_rules),
        (
            "project_index.known_classes",
            first.project_index.known_classes,
            second.project_index.known_classes,
        ),
        (
            "project_index.callable_ids",
            first.project_index.callable_ids,
            second.project_index.callable_ids,
        ),
        (
            "project_index.callable_aliases",
            first.project_index.callable_aliases,
            second.project_index.callable_aliases,
        ),
        (
            "project_index.module_map",
            first.project_index.module_map,
            second.project_index.module_map,
        ),
        (
            "project_index.static_method_ids",
            first.project_index.static_method_ids,
            second.project_index.static_method_ids,
        ),
    )
    return [
        name
        for name, left, right in checks
        if _normalize(left) != _normalize(right)
    ]


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _row_key(header: str, line: str) -> str:
    """The first field of a CSV row, which is the closest thing to an ID."""
    return line.split(",", 1)[0]


def diff_artifact(
    baseline_dir: Path, shuffled_dir: Path, name: str, limit: int
) -> Optional[Dict[str, object]]:
    """Compare one artifact, and describe *how* it differs rather than *that* it does.

    Line-set differences and same-line-different-content are reported apart,
    because they mean different things: a row present in one run and absent in
    the other is a fact that was found or lost, while a row that moved position
    is an ordering defect in the writer.
    """
    left = _read_lines(baseline_dir / name)
    right = _read_lines(shuffled_dir / name)
    if left == right:
        return None

    left_set, right_set = set(left), set(right)
    only_left = sorted(left_set - right_set)
    only_right = sorted(right_set - left_set)
    reordered = not only_left and not only_right

    first_line = next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )
    return {
        "artifact": name,
        "baseline_lines": len(left),
        "shuffled_lines": len(right),
        "only_in_baseline": only_left[:limit],
        "only_in_shuffled": only_right[:limit],
        "only_in_baseline_total": len(only_left),
        "only_in_shuffled_total": len(only_right),
        "reordered_only": reordered,
        "first_differing_line": first_line + 1,
        "first_baseline": left[first_line] if first_line < len(left) else "",
        "first_shuffled": right[first_line] if first_line < len(right) else "",
    }


def _print_artifact_diff(report: Dict[str, object], limit: int) -> None:
    print(f"\n  {report['artifact']}  *** DIFFERS ***")
    print(
        f"    rows: {report['baseline_lines']} baseline / {report['shuffled_lines']} shuffled"
    )
    if report["reordered_only"]:
        print("    every row is present in both runs -- this is an ordering difference only")
    else:
        print(
            f"    rows only in baseline: {report['only_in_baseline_total']}  "
            f"only in shuffled: {report['only_in_shuffled_total']}"
        )
    print(f"    first difference at line {report['first_differing_line']}:")
    print(f"      baseline: {report['first_baseline']}")
    print(f"      shuffled: {report['first_shuffled']}")
    for line in list(report["only_in_baseline"])[:limit]:
        print(f"    only in baseline: {line}")
    for line in list(report["only_in_shuffled"])[:limit]:
        print(f"    only in shuffled: {line}")


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


def check_determinism(
    config: ExtractionConfig,
    analysis_files: Sequence[AnalysisFile],
    *,
    seeds: Sequence[int] = (1234,),
    check_inputs: bool = False,
    limit: int = 5,
    workdir: Optional[Path] = None,
) -> int:
    """Return the number of seeds whose artifacts differed from the baseline."""
    analysis_files = list(analysis_files)
    print(f"analysis files: {len(analysis_files)}")

    temp_root = Path(tempfile.mkdtemp(prefix="da-determinism-")) if workdir is None else workdir
    temp_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    try:
        baseline_dir = temp_root / "baseline"
        print("\nrunning the stage in discovery order ...")
        run_stage(config, analysis_files, baseline_dir)

        for seed in seeds:
            shuffled = list(analysis_files)
            random.Random(seed).shuffle(shuffled)
            if [f.path for f in shuffled] == [f.path for f in analysis_files]:
                print(f"\nseed {seed}: shuffle was a no-op, skipping")
                continue

            print(f"\n--- seed {seed} ---")
            if check_inputs:
                moved = compare_call_graph_inputs(config, analysis_files, shuffled)
                if moved:
                    print(
                        "  call-graph inputs MOVED: "
                        + ", ".join(moved)
                        + "\n  Differences below may originate upstream of data_access/."
                    )
                else:
                    print("  call-graph inputs identical (callable_map, registration_rules, project_index)")

            shuffled_dir = temp_root / f"seed-{seed}"
            run_stage(config, shuffled, shuffled_dir)

            reports = [
                report
                for report in (
                    diff_artifact(baseline_dir, shuffled_dir, name, limit)
                    for name in artifact_names(baseline_dir, shuffled_dir)
                )
                if report is not None
            ]
            if not reports:
                print("  all artifacts byte-identical")
                continue
            failures += 1
            for report in reports:
                _print_artifact_diff(report, limit)
    finally:
        if workdir is None:
            shutil.rmtree(temp_root, ignore_errors=True)

    print(
        f"\nverdict: {'DETERMINISTIC' if failures == 0 else f'{failures} of {len(seeds)} shuffles produced different artifacts'}"
    )
    return failures


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check that data-access artifacts do not depend on file processing order",
    )
    parser.add_argument("--config", default=None, type=Path, help="Extraction JSON/JSONC config")
    parser.add_argument("--root", default=None, type=Path, help="Python source root (when no config is given)")
    parser.add_argument("--module-prefix", default=None, help="Prefix for discovered module names")
    parser.add_argument("--seed", action="append", default=[], type=int, help="Shuffle seed; repeatable")
    parser.add_argument("--repeat", default=1, type=int, help="Number of seeds to try when none are given")
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Also verify the call-graph structures data access consumes are order-stable",
    )
    parser.add_argument("--limit", default=5, type=int, help="Sample rows to print per artifact")
    parser.add_argument(
        "--keep",
        default=None,
        type=Path,
        help="Keep both artifact sets in this directory instead of a temp dir",
    )
    args = parser.parse_args(argv)

    if args.config:
        config = load_extraction_config(args.config)
        analysis_files = list(
            iter_analysis_files_for_source_roots(
                config.source_roots,
                entrypoints=config.entrypoints,
                project_root=config.project_root,
                include_globs=config.include_globs,
                exclude_globs=config.exclude_globs,
            )
        )
    elif args.root:
        raise SystemExit(
            "--root is not supported yet: this check runs the config path, which is the "
            "one that threads registration rules and the project index. Pass --config."
        )
    else:
        raise SystemExit("--config is required")

    seeds = args.seed or [1234 + index for index in range(max(1, args.repeat))]
    failures = check_determinism(
        config,
        analysis_files,
        seeds=seeds,
        check_inputs=args.check_inputs,
        limit=args.limit,
        workdir=args.keep,
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
