#!/usr/bin/env python3
"""Regenerate a manual mapping CSV from a module -> microservice rule table.

A manual mapping joins to cluster assignments on the exact ``node`` string, so a
pipeline change that alters data-object identity silently drops labelled rows out
of the metrics. Re-labelling by hand after every such change does not scale.

When a project's hand-built labels turn out to be a function of the owning Python
module -- one microservice per module, no exceptions -- the mapping can instead be
regenerated from a small rule table that is versioned alongside the config. This
script does that, and ``--verify-against`` proves the rule reproduces the original
human adjudication before anything is overwritten.

Deriving the rule table from an existing mapping:

    regenerate_manual_mapping.py derive-rules \\
        --manual configs/microservice_pipeline/manual_mapping_labeled.csv \\
        --source-root . \\
        --out configs/microservice_pipeline/manual_mapping_rules.csv

Regenerating, checking the rule against the mapping it came from first:

    regenerate_manual_mapping.py generate \\
        --clusters artifacts/structural_microservice_candidates_leiden/cluster_assignments.csv \\
        --rules configs/microservice_pipeline/manual_mapping_rules.csv \\
        --source-root . \\
        --verify-against configs/microservice_pipeline/manual_mapping_labeled.csv \\
        --out configs/microservice_pipeline/manual_mapping_labeled.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Column order of the generated mapping. ``microservice_name`` is carried for
# human review; the evaluator only needs ``microservice_id`` and ``node``.
OUTPUT_FIELDS = [
    "microservice_id",
    "microservice_name",
    "node",
    "node_type",
    "kind",
    "label",
    "module",
]

RULE_FIELDS = ["module", "microservice_id", "microservice_name"]

# Data kinds worth adjudicating. This is the union of the scopes the evaluator
# actually reads: ``evaluation.jsonc`` (callable + class_attr_state) and the
# sweep in ``structural_clustering.jsonc``. Kinds outside it -- param, dict_key,
# container_field, object_state, file -- never reach a metric under those configs,
# so labelling them would be busywork that also inflates coverage numbers.
DEFAULT_DATA_KINDS = ("class_attr_state", "module_global", "local_exposed")

DOTTED_PATH_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def read_csv(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def discover_modules(source_root: Path) -> Set[str]:
    """Every importable module and package under ``source_root``."""
    modules: Set[str] = set()
    for path in source_root.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(source_root).parts):
            continue
        relative = path.relative_to(source_root).with_suffix("")
        parts = list(relative.parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts:
            modules.add(".".join(parts))
    return modules


def resolve_module(node: str, modules: Set[str]) -> str:
    """The module that owns ``node``.

    Node ids embed a dotted path that runs past the module into the class and
    attribute (``data:class_attr_state:pkg.mod.Class:attr``), and some kinds embed
    more than one. Take the longest prefix of any embedded path that is a real
    module, which stops at the module boundary without needing to know how each
    kind is spelled.
    """
    best = ""
    for candidate in DOTTED_PATH_RE.findall(node):
        parts = candidate.split(".")
        for size in range(len(parts), 0, -1):
            prefix = ".".join(parts[:size])
            if prefix in modules:
                if len(prefix) > len(best):
                    best = prefix
                break
    return best


def load_rules(path: Path) -> Dict[str, Tuple[str, str]]:
    rows, fields = read_csv(path)
    missing = [name for name in RULE_FIELDS if name not in fields]
    if missing:
        raise SystemExit(f"{path}: rule table is missing column(s): {', '.join(missing)}")

    rules: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        module = (row.get("module") or "").strip()
        if not module:
            continue
        if module in rules:
            raise SystemExit(f"{path}: duplicate rule for module {module!r}")
        rules[module] = (
            (row.get("microservice_id") or "").strip(),
            (row.get("microservice_name") or "").strip(),
        )
    if not rules:
        raise SystemExit(f"{path}: rule table has no rows")
    return rules


def in_scope(row: dict, data_kinds: Sequence[str]) -> bool:
    node_type = (row.get("node_type") or "").strip().lower()
    if node_type == "callable":
        return True
    kinds = {token.strip() for token in (row.get("kind") or "").split(";") if token.strip()}
    return bool(kinds & set(data_kinds))


def derive_rules(args: argparse.Namespace) -> int:
    manual_rows, _ = read_csv(args.manual)
    modules = discover_modules(args.source_root)

    observed: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    names: Dict[str, str] = {}
    unresolved: List[str] = []

    for row in manual_rows:
        node = (row.get("node") or "").strip()
        if not node:
            continue
        module = resolve_module(node, modules)
        if not module:
            unresolved.append(node)
            continue
        label = (row.get("microservice_id") or "").strip()
        observed[module][label] += 1
        names.setdefault(label, (row.get("microservice_name") or "").strip())

    conflicts = {module: counts for module, counts in observed.items() if len(counts) > 1}
    if conflicts:
        print(
            f"{len(conflicts)} module(s) carry more than one label, so the mapping is "
            "not a pure function of the module and cannot be regenerated this way:",
            file=sys.stderr,
        )
        for module, counts in sorted(conflicts.items()):
            breakdown = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
            print(f"  {module}: {breakdown}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RULE_FIELDS)
        writer.writeheader()
        for module in sorted(observed):
            label = next(iter(observed[module]))
            writer.writerow(
                {
                    "module": module,
                    "microservice_id": label,
                    "microservice_name": names.get(label, ""),
                }
            )

    print(f"Derived {len(observed)} module rule(s) from {len(manual_rows)} labelled row(s).")
    print(f"Rule table written to {args.out}")
    if unresolved:
        print(f"{len(unresolved)} node(s) had no resolvable module and were skipped:")
        for node in unresolved[:10]:
            print(f"  {node}")
    return 0


def build_rows(
    cluster_rows: Sequence[dict],
    rules: Dict[str, Tuple[str, str]],
    modules: Set[str],
    data_kinds: Sequence[str],
) -> Tuple[List[dict], List[Tuple[str, str]]]:
    generated: List[dict] = []
    unmatched: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    for row in cluster_rows:
        node = (row.get("node") or "").strip()
        if not node or not in_scope(row, data_kinds):
            continue
        if node in seen:
            raise SystemExit(
                f"Duplicate node in cluster assignments: {node!r}. The evaluator "
                "rejects duplicate node keys, so the mapping cannot be built."
            )
        seen.add(node)

        module = resolve_module(node, modules)
        if module not in rules:
            unmatched.append((node, module or "<unresolved>"))
            continue

        microservice_id, microservice_name = rules[module]
        generated.append(
            {
                "microservice_id": microservice_id,
                "microservice_name": microservice_name,
                "node": node,
                "node_type": row.get("node_type", ""),
                "kind": row.get("kind", ""),
                "label": row.get("label", ""),
                "module": row.get("module", ""),
            }
        )

    return generated, unmatched


def verify(generated: Sequence[dict], reference_path: Path, modules: Set[str], rules) -> int:
    """Replay the rule against an existing mapping and report disagreements.

    This is the gate that proves a regenerated mapping preserves the original
    human adjudication: every node present in both must carry the same label.
    """
    reference_rows, _ = read_csv(reference_path)
    reference = {
        (row.get("node") or "").strip(): (row.get("microservice_id") or "").strip()
        for row in reference_rows
        if (row.get("node") or "").strip()
    }
    produced = {row["node"]: row["microservice_id"] for row in generated}

    shared = sorted(set(reference) & set(produced))
    disagreements = [(node, reference[node], produced[node]) for node in shared if reference[node] != produced[node]]

    print(f"Replay check against {reference_path}:")
    print(f"  nodes in both: {len(shared)}")
    print(f"  disagreements: {len(disagreements)}")
    if disagreements:
        for node, expected, actual in disagreements[:20]:
            print(f"    {node}\n      reference={expected!r} regenerated={actual!r}")
        if len(disagreements) > 20:
            print(f"    ... and {len(disagreements) - 20} more")
        return 1
    return 0


def generate(args: argparse.Namespace) -> int:
    cluster_rows, cluster_fields = read_csv(args.clusters)
    if "node" not in cluster_fields:
        raise SystemExit(f"{args.clusters}: no 'node' column")

    rules = load_rules(args.rules)
    modules = discover_modules(args.source_root)
    data_kinds = [kind.strip() for kind in args.data_kinds.split(",") if kind.strip()]

    generated, unmatched = build_rows(cluster_rows, rules, modules, data_kinds)

    if unmatched:
        # A module with no rule is a taxonomy decision the rule table has not
        # recorded yet. Emitting NA would bury it, so refuse instead.
        print(
            f"{len(unmatched)} in-scope node(s) belong to a module with no rule. "
            f"Add the module(s) to {args.rules} and re-run:",
            file=sys.stderr,
        )
        by_module: Dict[str, int] = defaultdict(int)
        for _node, module in unmatched:
            by_module[module] += 1
        for module, count in sorted(by_module.items(), key=lambda item: -item[1]):
            print(f"  {count:5d}  {module}", file=sys.stderr)
        for node, module in unmatched[:10]:
            print(f"    e.g. {node}  ->  {module}", file=sys.stderr)
        return 1

    if args.verify_against and verify(generated, args.verify_against, modules, rules) != 0:
        print(
            "Rule table does not reproduce the reference mapping; refusing to write.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(generated)

    counts: Dict[str, int] = defaultdict(int)
    for row in generated:
        counts[row["microservice_id"]] += 1

    print(f"Wrote {len(generated)} row(s) to {args.out}")
    print(f"  from {len(cluster_rows)} cluster row(s), scope: callable + {', '.join(data_kinds)}")
    for microservice_id, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {count:5d}  {microservice_id}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    derive = sub.add_parser("derive-rules", help="Extract a module rule table from an existing mapping.")
    derive.add_argument("--manual", type=Path, required=True)
    derive.add_argument("--source-root", type=Path, required=True, help="Directory the modules live under.")
    derive.add_argument("--out", type=Path, required=True)
    derive.set_defaults(func=derive_rules)

    gen = sub.add_parser("generate", help="Regenerate a mapping from a rule table.")
    gen.add_argument("--clusters", type=Path, required=True, help="cluster_assignments.csv; the authoritative node source.")
    gen.add_argument("--rules", type=Path, required=True)
    gen.add_argument("--source-root", type=Path, required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument(
        "--verify-against",
        type=Path,
        help="Existing mapping to replay the rule against; refuses to write on any disagreement.",
    )
    gen.add_argument("--data-kinds", default=",".join(DEFAULT_DATA_KINDS))
    gen.set_defaults(func=generate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
