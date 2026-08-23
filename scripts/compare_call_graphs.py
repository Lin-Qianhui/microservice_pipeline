"""Compare the pipeline's call graph against PyCG's, as three buckets.

    python3 scripts/compare_call_graphs.py \
        --ours <artifacts>/call_graph/edges.csv \
        --pycg <artifacts>/call_graph/pycg_call_graph.json \
        --outdir <artifacts>/call_graph_comparison

Neither graph is ground truth, so this reports agreement and two disagreement
sets rather than a precision/recall score against a reference:

  both       - edges present in each graph
  pycg_only  - candidate recall gaps in our resolver
  ours_only  - mostly relations PyCG does not model (see BEYOND_PYCG)

Six normalizations are applied before the sets are compared. Each one exists
because the two tools spell the same fact differently; without them the overlap
reads as near-zero for purely cosmetic reasons.

 1. `import` edges dropped -- the import statement itself, which PyCG models as
    a name binding rather than an edge. `imported` edges are KEPT: despite the
    name they are calls to imported functions, and dropping them hides ~30 real
    agreements.
 2. `.<module>` suffix stripped. We key module-level code as `mod.<module>`,
    PyCG as `mod`.
 3. `.<locals>` segments stripped. We spell nested functions
    `outer.<locals>.inner`, PyCG spells them `outer.inner`.
 4. Leading dots stripped. Entry points outside the --package root come back
    from PyCG as `...docs.run_utopia_example`.
 5. Scope filtered to first-party names on both sides, since PyCG also emits
    `<builtin>.*` and third-party targets that our config excludes by design.
    The first-party namespaces are read off our own `nodes.csv` rather than
    configured, so this works on any analyzed project without being told.
 6. Re-exported names rewritten to the definition site. A package `__init__.py`
    doing `from .core import get_n_faces` gives one function two spellings; we
    always emit the definition, PyCG emits whichever the call site named. See
    build_alias_map -- this one reads the analyzed source, the other five are
    string rules.

Stdlib only; runs under any Python 3.9+.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

# Rewrites one graph's spelling of a node onto the shared one.
Canon = Callable[[str], str]

# Depth cap when following a re-export chain to the definition. Also the cycle
# guard: `a.py` and `b.py` importing each other's names is legal Python.
MAX_ALIAS_HOPS = 10

# `import` is the import statement itself, not a call: every such edge targets a
# `mod.<module>` node. PyCG models imports as name bindings rather than edges, so
# keeping these would count guaranteed disagreements.
#
# `imported` is NOT in this set despite the name -- it marks a *call* to a name
# that was imported (e.g. `solver_SS(self)` at utopia.py:336), carries a real
# call-site lineno, and never targets a `.<module>` node. PyCG emits these too.
NON_CALL_RELATIONS = {"import"}

# Our relation kinds that model dispatch PyCG does not attempt. Edges carrying
# these are expected in ours_only and are reported separately so they do not
# read as resolver bugs.
#
# `inferred_type` is deliberately NOT here: PyCG's whole design is
# inter-procedural points-to analysis, so it competes with us there and those
# disagreements are genuine.
#
# `virtual_override` is NOT here either, and the temptation to add it is the
# reason this note exists. It looks like a mechanism PyCG lacks -- we fan an
# edge out to every subclass override of `self.hook()`, which is not something
# a points-to analysis does. But PyCG reaches the same edges from the other
# direction, by resolving the receiver to the concrete class that flowed in: on
# Parcels it found all 10 of ours. Excusing the relation here would hide the one
# we most need to see, since `virtual_override` is our deliberate
# over-approximation (36/60 confirmed at fan-out >=3 against runtime ground
# truth -- see call_graph/ground_truth_and_roadmap.md). Its false positives
# belong in ours_only_contested where someone reads them.
#
# call_graph/graph_comparison.py defines similar-looking sets
# (UNOBSERVABLE_RELATIONS, SYNTHETIC_RELATIONS). Do not import them. Those mean
# "the runtime tracer cannot see this", which is a different claim from "PyCG
# does not model this"; the two overlap today only by coincidence, and coupling
# them would make a change to either silently corrupt the other.
BEYOND_PYCG = {
    # Dispatch we resolve through mechanisms PyCG has no model for.
    "dynamic_getattr",
    "super_method",
    # Synthetic: asserts a parent/child coupling, not an interpreter dispatch.
    "registered_invoke",
    # Implicit invocations. PyCG records explicit call expressions only, so it
    # never emits an edge for attribute access or an operator dunder.
    "property_getter",
    "dunder_call",
    "dunder_contains",
    "dunder_delitem",
    "dunder_getitem",
    "dunder_iter",
    "dunder_operator",
    "dunder_setitem",
}

# Relations PyCG genuinely competes on. Anything in edges.csv that appears in
# neither this set nor the two above is unclassified, and the run says so rather
# than letting it drift silently into ours_only_contested.
#
# `inferred_callable` is the callable-value counterpart of `inferred_type` -- a
# call through an expression that evaluates to a function rather than an
# instance (`handler = helper; handler()`). Tracking a function through an
# assignment is ordinary points-to work, so PyCG competes here too.
COMPARABLE_RELATIONS = {
    "class_method",
    "constructor",
    "direct",
    "imported",
    "inferred_callable",
    "inferred_type",
    "self_method",
    "virtual_override",
}


def normalize(name: str) -> str:
    """Map either tool's spelling of a node onto a common form."""
    name = name.lstrip(".")
    if name.endswith(".<module>"):
        name = name[: -len(".<module>")]
    return ".".join(part for part in name.split(".") if part != "<locals>")


def load_ours(
    path: Path, canon: Canon = lambda name: name
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]], int]:
    """Return (call edges, relation kinds per edge, count of non-call rows dropped)."""
    edges: set[tuple[str, str]] = set()
    relations: dict[tuple[str, str], set[str]] = {}
    dropped = 0
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relation = row["relation"]
            if relation in NON_CALL_RELATIONS:
                dropped += 1
                continue
            edge = (canon(normalize(row["caller"])), canon(normalize(row["callee"])))
            edges.add(edge)
            relations.setdefault(edge, set()).add(relation)
    return edges, relations, dropped


def load_nodes(nodes_path: Path) -> tuple[dict[str, str], set[str]]:
    """Read our own node table: ({module: defining file}, {node id}).

    Both outputs describe only files the pipeline actually analyzed, which is
    what makes them safe to reason from -- the module map bounds the source
    build_alias_map parses, and the id set is the authority on which dotted
    names denote a real definition.
    """
    module_files: dict[str, str] = {}
    known: set[str] = set()
    with nodes_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            module = (row.get("module") or "").strip()
            path = (row.get("file") or "").strip()
            node_id = (row.get("id") or "").strip()
            if module and path:
                module_files.setdefault(module, path)
            if node_id:
                known.add(normalize(node_id))
    return module_files, known


def derive_prefixes(module_files: dict[str, str]) -> tuple[str, ...]:
    """Read the first-party namespaces off our own node table.

    The top-level segment of every node's module is first-party by definition:
    the pipeline only creates nodes for files it analyzed. Deriving them beats
    reading `source.package_prefixes` from the config, which omits the namespace
    of any entrypoint living outside the source roots (a `docs/run_example.py`
    entrypoint yields `docs.*` nodes but is not a configured package prefix).
    """
    return tuple(sorted({module.split(".", 1)[0] for module in module_files}))


def load_pycg(path: Path, canon: Canon = lambda name: name) -> set[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        (canon(normalize(caller)), canon(normalize(callee)))
        for caller, callees in raw.items()
        for callee in callees
    }


def module_level_imports(tree: ast.Module) -> list:
    """Import statements that bind a name in the module's own namespace.

    Descends into `if` and `try` -- TYPE_CHECKING guards and optional-dependency
    fallbacks both re-export from inside them -- but not into functions or
    classes, whose imports bind locally and can never be spelled `module.name`
    from outside.
    """
    found = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif isinstance(node, (ast.If, ast.Try)):
            stack.extend(node.body)
            stack.extend(node.orelse)
            stack.extend(getattr(node, "handlers", ()))
            stack.extend(getattr(node, "finalbody", ()))
        elif isinstance(node, ast.ExceptHandler):
            stack.extend(node.body)
    return found


def absolute_module(current: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    """Resolve an ImportFrom's target to an absolute dotted module name.

    `level` counts leading dots. One dot means "this package", which is the
    module itself when it is an `__init__.py` and its parent otherwise; each
    further dot climbs one more.
    """
    if not node.level:
        return node.module
    base = current.split(".") if is_package else current.split(".")[:-1]
    climb = node.level - 1
    if climb:
        if climb > len(base):
            return None
        base = base[:-climb]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base) or None


def build_alias_map(module_files: dict[str, str], known: set[str]) -> dict[str, str]:
    """Map re-exported names onto the definition site both graphs can share.

    `parcels/_sgrid/__init__.py` doing `from .core import get_n_faces` makes
    `parcels._sgrid.get_n_faces` and `parcels._sgrid.core.get_n_faces` two
    spellings of one function. We always emit the definition site; PyCG emits
    whichever spelling the call site used. Left alone, one agreed-on edge lands
    in `ours_only_contested` *and* `pycg_only`, reading as a disagreement in
    both directions at once -- the same failure mode as dropping `imported`,
    and just as invisible.

    Two rails stop this from inventing agreements. A binding is kept only if it
    resolves to a name in `known`, and never if its own spelling is already
    there, so a rewrite can only ever turn a name that denotes nothing into the
    definition it aliases -- never one real node into a different real node.
    Star imports contribute only names that exactly one source module defines;
    ambiguous ones are dropped rather than guessed.

    Unlike the other five normalizations this reads the analyzed source, since
    nothing in either graph records that the two names are the same function.
    Files that have moved or fail to parse are skipped: a missing alias leaves
    the old double-counting, which is the pre-existing behavior.
    """
    direct: dict[str, str] = {}
    star_sources: dict[str, list[str]] = {}

    for module, path in module_files.items():
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
            continue
        is_package = Path(path).name == "__init__.py"
        for node in module_level_imports(tree):
            if isinstance(node, ast.Import):
                # Only the aliased form binds a name we could ever see as a
                # callee prefix; plain `import a.b` binds the package `a`.
                for alias in node.names:
                    if alias.asname:
                        direct[f"{module}.{alias.asname}"] = alias.name
                continue
            source = absolute_module(module, is_package, node)
            if not source:
                continue
            for alias in node.names:
                if alias.name == "*":
                    star_sources.setdefault(module, []).append(source)
                else:
                    direct[f"{module}.{alias.asname or alias.name}"] = f"{source}.{alias.name}"

    if star_sources:
        defined: dict[str, list[str]] = {}
        for node_id in known:
            head, _, leaf = node_id.rpartition(".")
            if head:
                defined.setdefault(head, []).append(leaf)
        for module, sources in star_sources.items():
            offers = Counter(leaf for source in sources for leaf in defined.get(source, ()))
            for source in sources:
                for leaf in defined.get(source, ()):
                    if offers[leaf] == 1:
                        # setdefault: an explicit import of the same name wins.
                        direct.setdefault(f"{module}.{leaf}", f"{source}.{leaf}")

    # Re-exports chain -- `parcels` re-exports from `parcels._sgrid`, which
    # re-exports from `parcels._sgrid.core` -- so follow each binding to its end.
    resolved: dict[str, str] = {}
    for name in direct:
        target = name
        for _ in range(MAX_ALIAS_HOPS):
            step = direct.get(target)
            if step is None or step == target:
                break
            target = step
        else:
            continue  # cycle or a chain past any plausible depth; leave it alone
        if target != name and target in known and name not in known:
            resolved[name] = target
    return resolved


def make_canonicalizer(aliases: dict[str, str]) -> tuple[Canon, set[str]]:
    """Return (rewriter, set of aliases it actually used).

    Rewrites the longest aliased prefix, so a re-exported class carries its
    members with it: `pkg.Meta.method` -> `pkg.core.Meta.method`.
    """
    used: set[str] = set()

    def canon(name: str) -> str:
        parts = name.split(".")
        for cut in range(len(parts), 0, -1):
            key = ".".join(parts[:cut])
            target = aliases.get(key)
            if target:
                used.add(key)
                return ".".join([target] + parts[cut:])
        return name

    return (canon if aliases else (lambda name: name)), used


def first_party(
    edges: set[tuple[str, str]],
    prefixes: tuple[str, ...],
    drop: tuple[str, ...] = (),
) -> set[tuple[str, str]]:
    """Keep edges whose endpoints both sit inside the analyzed namespaces."""
    return {
        (caller, callee)
        for caller, callee in edges
        if caller.startswith(prefixes)
        and callee.startswith(prefixes)
        and not (drop and (caller.startswith(drop) or callee.startswith(drop)))
    }


def write_bucket(path: Path, edges: set[tuple[str, str]], relations: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["caller", "callee", "our_relations"])
        for caller, callee in sorted(edges):
            kinds = ";".join(sorted(relations.get((caller, callee), ())))
            writer.writerow([caller, callee, kinds])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--pycg", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--nodes",
        default=None,
        type=Path,
        help="nodes.csv, used to derive first-party namespaces (default: beside --ours)",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Override the first-party namespaces derived from nodes.csv; repeatable",
    )
    parser.add_argument(
        "--drop-module",
        action="append",
        default=None,
        metavar="DOTTED",
        help=(
            "Drop edges touching this module from both sides; repeatable. "
            "Needed when the two graphs disagree on file scope"
        ),
    )
    parser.add_argument(
        "--no-alias-resolution",
        action="store_true",
        help=(
            "Skip normalization 6: leave re-exported names spelled as each tool "
            "found them. Restores the pre-alias-map buckets, for comparing "
            "against an older run"
        ),
    )
    args = parser.parse_args()
    drop = tuple(args.drop_module or ())

    # nodes.csv drives both the namespace filter and the re-export alias map, so
    # read it whenever it exists -- --prefix overrides the former, not the latter.
    nodes_path = args.nodes or args.ours.parent / "nodes.csv"
    module_files: dict[str, str] = {}
    known: set[str] = set()
    if nodes_path.exists():
        module_files, known = load_nodes(nodes_path)
    elif not args.prefix:
        raise SystemExit(
            f"Cannot derive first-party namespaces: {nodes_path} not found. "
            f"Pass --nodes, or --prefix to set them explicitly."
        )

    if args.prefix:
        prefixes = tuple(args.prefix)
    else:
        prefixes = derive_prefixes(module_files)
        if not prefixes:
            raise SystemExit(f"No modules found in {nodes_path}")

    aliases = {} if args.no_alias_resolution else build_alias_map(module_files, known)
    canon, aliases_used = make_canonicalizer(aliases)

    ours_all, relations, import_rows = load_ours(args.ours, canon)
    pycg_all = load_pycg(args.pycg, canon)
    ours = first_party(ours_all, prefixes, drop)
    pycg = first_party(pycg_all, prefixes, drop)

    both = ours & pycg
    ours_only = ours - pycg
    pycg_only = pycg - ours

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_bucket(args.outdir / "both.csv", both, relations)
    write_bucket(args.outdir / "ours_only.csv", ours_only, relations)
    write_bucket(args.outdir / "pycg_only.csv", pycg_only, relations)

    # Every rewrite that moved an edge, written out so the normalization can be
    # audited rather than taken on faith -- it is the only one that silently
    # changes which bucket an edge lands in.
    with (args.outdir / "reexport_aliases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alias", "definition"])
        writer.writerows(sorted((name, aliases[name]) for name in aliases_used))

    # Split ours_only by whether PyCG could have found the edge at all.
    expected = {e for e in ours_only if relations.get(e, set()) & BEYOND_PYCG}
    contested = ours_only - expected

    # A relation the extractor gained but nobody classified here would land in
    # ours_only_contested and read as a resolver defect. Say so instead.
    seen_relations = {r for kinds in relations.values() for r in kinds}
    unclassified = sorted(seen_relations - BEYOND_PYCG - COMPARABLE_RELATIONS)

    summary = {
        "first_party_prefixes": list(prefixes),
        "unclassified_relations": unclassified,
        "reexport_aliases_found": len(aliases),
        "reexport_aliases_applied": len(aliases_used),
        "ours_call_edges": len(ours),
        "ours_non_call_rows_dropped": import_rows,
        "ours_edges_outside_scope": len(ours_all) - len(ours),
        "pycg_edges_first_party": len(pycg),
        "pycg_edges_total": len(pycg_all),
        "both": len(both),
        "ours_only": len(ours_only),
        "ours_only_beyond_pycg": len(expected),
        "ours_only_contested": len(contested),
        "pycg_only": len(pycg_only),
        "jaccard": round(len(both) / len(ours | pycg), 4) if (ours | pycg) else 0.0,
        "ours_only_by_relation": dict(
            Counter(r for e in ours_only for r in relations.get(e, ())).most_common()
        ),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    scalars = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
    width = max(len(k) for k in scalars)
    print(f"first-party namespaces: {', '.join(prefixes)}")
    for key, value in scalars.items():
        print(f"{key:<{width}}  {value}")

    print("\nours_only by relation:")
    for relation, count in summary["ours_only_by_relation"].items():
        marker = "  (not modeled by PyCG)" if relation in BEYOND_PYCG else ""
        print(f"  {relation:<18} {count}{marker}")

    if unclassified:
        print(
            "\nWARNING: unclassified relations in edges.csv: "
            + ", ".join(unclassified)
            + "\n  These fall into ours_only_contested and will read as resolver"
            "\n  defects. Add each to BEYOND_PYCG or COMPARABLE_RELATIONS."
        )
    print(
        f"\nwrote both.csv, ours_only.csv, pycg_only.csv, reexport_aliases.csv, "
        f"summary.json to {args.outdir}"
    )


if __name__ == "__main__":
    main()
