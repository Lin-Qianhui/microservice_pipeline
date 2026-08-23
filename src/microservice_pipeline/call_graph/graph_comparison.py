"""Diff the static call graph against a runtime trace and report the gaps.

The two graphs are not symmetric and must not be scored as if they were. The
static graph covers the whole program; the trace covers only what the drivers
executed. That asymmetry decides how each direction is read:

**Recall is the real measurement.** An edge the interpreter dispatched but the
analyzer never inferred is a genuine static gap, full stop. These are bucketed
by callee class so "which pattern costs the most edges" falls straight out of
the report rather than having to be eyeballed.

**Precision is a weak signal and is deliberately hedged.** A static edge missing
from the trace usually means the branch never ran, not that the edge is wrong.
It is therefore scoped to callers the trace actually entered and reported as
*unconfirmed*, never as *false*.

Three classes of static edge are excluded from both directions, and counting
them would manufacture fake gaps. They are kept apart because they are excluded
for genuinely different reasons:

* **unresolved edges** -- their callee is a descriptive name, not a project
  callable, so there is no ID for the trace to match against.
* **unobservable relations** -- the interpreter really does make these calls,
  but it makes them somewhere ``sys.monitoring`` cannot see. Module bodies are
  executed by the import machinery in C; property access and every operator
  dunder dispatch through C-level descriptor and type slots. Verified directly:
  ``c.p``, ``c[0]``, ``c + c`` and ``c()`` produce zero ``CALL`` events, while an
  ordinary ``c.method()`` on the same object produces one.
* **synthetic relations** -- ``registered_invoke`` is not a dispatch the
  interpreter ever performs. It states that a parent and a registered child are
  coupled, which at runtime is mediated by a shared base-class driver. It is
  unconfirmable *by construction*, which is a different fact from "the tracer
  cannot see it" and is reported under its own heading rather than folded in.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from microservice_pipeline.artifact_io import write_json, write_markdown
    from microservice_pipeline.call_graph.models import NO_SOURCE_SITE
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import write_json, write_markdown  # type: ignore
    from models import NO_SOURCE_SITE  # type: ignore


# Relations the runtime tracer structurally cannot observe. See module docstring.
# These calls do happen; they happen in C, where no CALL event is emitted.
UNOBSERVABLE_RELATIONS = frozenset(
    {
        "import",
        "property_getter",
        "dunder_getitem",
        "dunder_setitem",
        "dunder_delitem",
        "dunder_contains",
        "dunder_iter",
        "dunder_operator",
        "dunder_call",
    }
)

# Relations that deliberately depart from the interpreter's call sequence, so a
# trace can never confirm them however wide its coverage gets. Kept apart from
# UNOBSERVABLE_RELATIONS because the reason differs: those are a limitation of
# the instrument, these are a property of the model. See ground_truth_and_roadmap
# section 2 -- the 21 registered_invoke edges encode a parent/child tree that the
# runtime expresses as a 20-edge hub off a single base-class dispatcher.
SYNTHETIC_RELATIONS = frozenset({"registered_invoke"})

# Every relation excluded from scoring, for the one place that needs the union.
UNSCORED_RELATIONS = UNOBSERVABLE_RELATIONS | SYNTHETIC_RELATIONS


@dataclass(frozen=True)
class StaticEdge:
    caller: str
    callee: str
    relation: str
    resolved: bool
    # Defaulted because the site is optional information: an edge with no call
    # expression behind it (``registered_invoke``) has none, and older artifacts
    # predate the columns.
    lineno: int = NO_SOURCE_SITE
    col_offset: int = NO_SOURCE_SITE

    @property
    def site(self) -> Optional[Tuple[str, int, int]]:
        """The call site this edge claims, or ``None`` if it claims none."""
        if self.lineno == NO_SOURCE_SITE or self.col_offset == NO_SOURCE_SITE:
            return None
        return (self.caller, self.lineno, self.col_offset)


@dataclass
class ComparisonReport:
    static_edge_count: int = 0
    dynamic_edge_count: int = 0
    comparable_static_count: int = 0
    unobservable_static_count: int = 0
    synthetic_static_count: int = 0

    matched: Set[Tuple[str, str]] = field(default_factory=set)
    missing_from_static: Set[Tuple[str, str]] = field(default_factory=set)
    unconfirmed_static: Set[Tuple[str, str]] = field(default_factory=set)
    # Static edges whose *site* ran and dispatched elsewhere. Unlike
    # ``unconfirmed_static`` these are not untaken branches: the call happened
    # and went somewhere else, so the edge is an invention.
    falsified_static: Set[Tuple[str, str]] = field(default_factory=set)
    falsified_by_relation: Counter = field(default_factory=Counter)
    sited_static_count: int = 0
    matchable_site_count: int = 0

    executed_callables: Set[str] = field(default_factory=set)
    recall_by_relation: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    missing_by_callee_class: Counter = field(default_factory=Counter)
    missing_by_caller_class: Counter = field(default_factory=Counter)
    driver_errors: List[str] = field(default_factory=list)
    external_calls: int = 0

    @property
    def recall(self) -> float:
        observed = len(self.matched) + len(self.missing_from_static)
        return len(self.matched) / observed if observed else 0.0

    @property
    def scoped_precision(self) -> float:
        total = len(self.matched) + len(self.unconfirmed_static)
        return len(self.matched) / total if total else 0.0

    @property
    def site_match_rate(self) -> float:
        """Share of sited static edges whose site the trace actually entered.

        The denominator of any falsification claim. A low rate does not mean the
        graph is right -- it means the instrument could not look, and the
        falsified count below should not be read as if it had.
        """
        if not self.sited_static_count:
            return 0.0
        return self.matchable_site_count / self.sited_static_count


def load_static_edges(path: Path) -> List[StaticEdge]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            StaticEdge(
                caller=row["caller"],
                callee=row["callee"],
                relation=row["relation"],
                resolved=row["resolved"] in {"1", "True", "true"},
                lineno=_int_or_missing(row.get("lineno")),
                col_offset=_int_or_missing(row.get("col_offset")),
            )
            for row in csv.DictReader(handle)
        ]


def _int_or_missing(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return NO_SOURCE_SITE


def load_observed_sites(path: Path) -> Dict[Tuple[str, int, int], Set[str]]:
    """Read ``dynamic_sites.csv`` into ``site -> callees dispatched there``."""
    sites: Dict[Tuple[str, int, int], Set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["caller"],
                _int_or_missing(row.get("lineno")),
                _int_or_missing(row.get("col_offset")),
            )
            sites.setdefault(key, set()).add(row["callee"])
    return sites


def load_dynamic_edges(path: Path) -> List[Tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [(row["caller"], row["callee"]) for row in csv.DictReader(handle)]


def _owner(callable_id: str) -> str:
    """Best-effort class (or module) that owns a callable ID.

    IDs look like ``pkg.mod.Class.method``. Nothing in the ID marks where the
    module stops and the class starts, so this uses the standard convention that
    class names are capitalized and falls back to the module otherwise. It only
    feeds report grouping, so an occasional miss costs nothing.
    """
    parts = callable_id.split(".")
    if len(parts) < 2:
        return callable_id
    for index in range(len(parts) - 2, -1, -1):
        if parts[index][:1].isupper():
            return ".".join(parts[: index + 1])
    return ".".join(parts[:-1])


def compare(
    static_edges: Sequence[StaticEdge],
    dynamic_edges: Iterable[Tuple[str, str]],
    executed_callables: Set[str],
    *,
    observed_sites: Optional[Mapping[Tuple[str, int, int], Set[str]]] = None,
    disabled_positions: Optional[Set[Tuple[str, int, int]]] = None,
    driver_errors: Sequence[str] = (),
    external_calls: int = 0,
) -> ComparisonReport:
    dynamic_pairs: Set[Tuple[str, str]] = set(dynamic_edges)

    comparable = [
        edge
        for edge in static_edges
        if edge.resolved and edge.relation not in UNSCORED_RELATIONS
    ]
    static_pairs: Set[Tuple[str, str]] = {(edge.caller, edge.callee) for edge in comparable}

    # Reported rather than silently dropped. Excluding these is what keeps the
    # unconfirmed count meaningful, so the size of the exclusion has to be
    # visible -- on climlab it is roughly two thirds of what used to be counted.
    unobservable_pairs = {
        (edge.caller, edge.callee)
        for edge in static_edges
        if edge.resolved and edge.relation in UNOBSERVABLE_RELATIONS
    }
    synthetic_pairs = {
        (edge.caller, edge.callee)
        for edge in static_edges
        if edge.resolved and edge.relation in SYNTHETIC_RELATIONS
    }

    # Relation is per static edge, but comparison is per (caller, callee) pair;
    # a pair can carry several relations, so keep them all for the breakdown.
    relations_by_pair: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for edge in comparable:
        relations_by_pair[(edge.caller, edge.callee)].add(edge.relation)

    report = ComparisonReport(
        static_edge_count=len(static_edges),
        dynamic_edge_count=len(dynamic_pairs),
        comparable_static_count=len(static_pairs),
        # A pair carrying both a scored and an unscored relation is already
        # counted as comparable; subtract it so the three buckets partition.
        unobservable_static_count=len(unobservable_pairs - static_pairs),
        synthetic_static_count=len(synthetic_pairs - static_pairs),
        executed_callables=executed_callables,
        driver_errors=list(driver_errors),
        external_calls=external_calls,
    )

    report.matched = dynamic_pairs & static_pairs
    report.missing_from_static = dynamic_pairs - static_pairs

    # Only callers the trace entered can say anything about a static edge that
    # did not appear. Everything else was simply never exercised.
    report.unconfirmed_static = {
        pair
        for pair in static_pairs - dynamic_pairs
        if pair[0] in executed_callables
    }

    # Falsification. An unconfirmed edge is usually an untaken branch, which is
    # why this module refuses to call one false. But when the trace saw the very
    # call site the edge claims, and that site dispatched to other callees and
    # never to this one, "untaken branch" is ruled out: the call ran and went
    # somewhere else. That is an invention, and it is the only direction in which
    # this comparison can say so.
    #
    # Without it the instrument is asymmetric by construction -- misses are
    # measured, inventions are not -- and over-approximation looks free.
    if observed_sites:
        retired = disabled_positions or set()
        sited = [edge for edge in comparable if edge.site is not None]
        report.sited_static_count = len(sited)
        for edge in sited:
            witnessed = observed_sites.get(edge.site)
            if witnessed is None:
                continue  # the site never ran; says nothing either way
            if edge.site in retired:
                # The tracer stopped watching this site once it went
                # ``disable_after`` hits without a new callee. Its callee set is
                # therefore truncated, and a callee absent from it may simply
                # have arrived on a later iteration. Concluding "invented" here
                # would turn a cost-saving heuristic into a false accusation.
                continue
            report.matchable_site_count += 1
            if edge.callee not in witnessed:
                report.falsified_static.add((edge.caller, edge.callee))
                report.falsified_by_relation[edge.relation] += 1

        # A pair confirmed from some other site is not an invention, whatever a
        # single site says: the same callee is often reached from several places.
        report.falsified_static -= dynamic_pairs

    hit_by_relation: Counter = Counter()
    total_by_relation: Counter = Counter()
    for pair in static_pairs:
        for relation in relations_by_pair[pair]:
            total_by_relation[relation] += 1
            if pair in dynamic_pairs:
                hit_by_relation[relation] += 1
    report.recall_by_relation = {
        relation: (hit_by_relation[relation], total_by_relation[relation])
        for relation in sorted(total_by_relation)
    }

    for caller, callee in report.missing_from_static:
        report.missing_by_callee_class[_owner(callee)] += 1
        report.missing_by_caller_class[_owner(caller)] += 1

    return report


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: ComparisonReport, *, top_n: int = 25) -> List[str]:
    lines: List[str] = [
        "# Static vs runtime call graph",
        "",
        "Runtime edges come from executing the project under `sys.monitoring`.",
        "The trace is a lower bound: it only sees what the drivers exercised.",
        "",
        "## Totals",
        "",
        f"- Static edges: {report.static_edge_count} "
        f"({report.comparable_static_count} comparable, after excluding the three "
        f"classes below and unresolved callees)",
        f"- Excluded, unobservable: {report.unobservable_static_count} — property and"
        " operator dispatch happens in C, so no `CALL` event ever fires",
        f"- Excluded, synthetic: {report.synthetic_static_count} — `registered_invoke`"
        " deliberately departs from the runtime call sequence and can never be confirmed",
        f"- Runtime edges observed: {report.dynamic_edge_count}",
        f"- Callables entered at runtime: {len(report.executed_callables)}",
        f"- Calls to code outside the source roots: {report.external_calls}",
        "",
        "## Recall — the headline number",
        "",
        "Of the edges the interpreter really dispatched, how many did the static",
        "analyzer find? Every miss here is a genuine gap.",
        "",
        f"**Recall: {_percent(report.recall)}** "
        f"({len(report.matched)} found / {len(report.matched) + len(report.missing_from_static)} executed)",
        "",
    ]

    if report.missing_by_callee_class:
        lines += [
            "### Missing edges grouped by callee owner",
            "",
            "| Callee owner | Missing edges |",
            "| --- | ---: |",
        ]
        for owner, count in report.missing_by_callee_class.most_common(top_n):
            lines.append(f"| `{owner}` | {count} |")
        lines.append("")

    if report.missing_by_caller_class:
        lines += [
            "### Missing edges grouped by caller owner",
            "",
            "| Caller owner | Missing edges |",
            "| --- | ---: |",
        ]
        for owner, count in report.missing_by_caller_class.most_common(top_n):
            lines.append(f"| `{owner}` | {count} |")
        lines.append("")

    lines += [
        "## Per-relation confirmation",
        "",
        "How much of each static relation the run actually confirmed. A low",
        "score means the relation was not exercised, **not** that it is wrong.",
        "",
        "| Relation | Confirmed | Static edges | Confirmed % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for relation, (hit, total) in sorted(
        report.recall_by_relation.items(), key=lambda item: -item[1][1]
    ):
        share = _percent(hit / total) if total else "-"
        lines.append(f"| `{relation}` | {hit} | {total} | {share} |")
    lines.append("")

    lines += [
        "## Unconfirmed static edges (weak signal)",
        "",
        "Static edges whose caller *did* run but which never dispatched. Most are",
        "untaken branches rather than errors — treat as leads, not defects.",
        "",
        f"- Unconfirmed: {len(report.unconfirmed_static)}",
        f"- Scoped agreement: {_percent(report.scoped_precision)}",
        "",
    ]

    if report.sited_static_count:
        lines += [
            "## Falsified static edges (strong signal)",
            "",
            "Edges whose exact call site the trace watched dispatching, where the",
            "claimed callee never appeared. Unlike the section above these are not",
            "untaken branches: the call ran and went elsewhere. This is the only",
            "direction in which the runtime can say a static edge is *wrong* rather",
            "than merely unconfirmed.",
            "",
            "Sites the tracer retired early are excluded — their callee set is",
            "truncated by the cost-saving disable rule, so a callee's absence there",
            "proves nothing.",
            "",
            f"- **Falsified: {len(report.falsified_static)}**",
            f"- Site coverage: {_percent(report.site_match_rate)} "
            f"({report.matchable_site_count} of {report.sited_static_count} sited "
            f"static edges had a usable site)",
            "",
        ]
        if report.falsified_by_relation:
            lines += [
                "| Relation | Falsified |",
                "| --- | ---: |",
            ]
            for relation, count in report.falsified_by_relation.most_common():
                lines.append(f"| `{relation}` | {count} |")
            lines.append("")
        lines += [
            "A relation that over-approximates *by design* is expected here and is",
            "not a defect, provided it is labelled and weighted accordingly. One",
            "that claims certainty is a defect.",
            "",
        ]

    if report.driver_errors:
        lines += ["## Driver problems", ""]
        lines += [f"- {message}" for message in report.driver_errors]
        lines.append("")

    return lines


def write_comparison(outdir: Path, report: ComparisonReport, *, top_n: int = 25) -> None:
    write_markdown(outdir / "graph_comparison.md", render_markdown(report, top_n=top_n), trailing_newline=True)
    write_json(
        outdir / "graph_comparison.json",
        {
            "static_edge_count": report.static_edge_count,
            "dynamic_edge_count": report.dynamic_edge_count,
            "comparable_static_count": report.comparable_static_count,
            "unobservable_static_count": report.unobservable_static_count,
            "synthetic_static_count": report.synthetic_static_count,
            "matched": len(report.matched),
            "missing_from_static": len(report.missing_from_static),
            "unconfirmed_static": len(report.unconfirmed_static),
            "falsified_static": len(report.falsified_static),
            "falsified_by_relation": dict(sorted(report.falsified_by_relation.items())),
            "falsified_edges": [
                {"caller": caller, "callee": callee}
                for caller, callee in sorted(report.falsified_static)
            ],
            "site_match_rate": report.site_match_rate,
            "sited_static_count": report.sited_static_count,
            "matchable_site_count": report.matchable_site_count,
            "recall": report.recall,
            "scoped_precision": report.scoped_precision,
            "external_calls": report.external_calls,
            "recall_by_relation": {
                relation: {"confirmed": hit, "total": total}
                for relation, (hit, total) in report.recall_by_relation.items()
            },
            "missing_by_callee_class": dict(report.missing_by_callee_class.most_common()),
            "missing_by_caller_class": dict(report.missing_by_caller_class.most_common()),
            "missing_edges": [
                {"caller": caller, "callee": callee}
                for caller, callee in sorted(report.missing_from_static)
            ],
            "driver_errors": report.driver_errors,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import json

    try:
        from microservice_pipeline.config import load_extraction_config
    except ImportError:  # pragma: no cover
        from config import load_extraction_config  # type: ignore

    parser = argparse.ArgumentParser(
        description="Compare the static call graph against a runtime trace",
    )
    parser.add_argument("--config", default=None, type=Path, help="Extraction JSON/JSONC config")
    parser.add_argument(
        "--artifacts",
        default=None,
        type=Path,
        help="Directory holding edges.csv and dynamic_edges.csv (when no config is given)",
    )
    parser.add_argument("--outdir", default=None, type=Path, help="Where to write the report")
    parser.add_argument("--top", default=25, type=int, help="Rows per grouping table")
    args = parser.parse_args(argv)

    if args.config:
        artifacts = load_extraction_config(args.config).call_graph.outdir.resolve()
    elif args.artifacts:
        artifacts = args.artifacts.resolve()
    else:
        raise SystemExit("--config or --artifacts is required")

    static_path = artifacts / "edges.csv"
    dynamic_path = artifacts / "dynamic_edges.csv"
    if not static_path.exists():
        raise SystemExit(f"Missing {static_path}; run 'call-graph' first")
    if not dynamic_path.exists():
        raise SystemExit(f"Missing {dynamic_path}; run 'trace-runtime' first")

    meta_path = artifacts / "dynamic_trace.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    # Optional: an older trace has no site file, and the comparison still works
    # -- it simply cannot falsify anything, which the report says out loud.
    sites_path = artifacts / "dynamic_sites.csv"
    observed_sites = load_observed_sites(sites_path) if sites_path.exists() else None
    disabled_positions = {
        (str(caller), int(line), int(col))
        for caller, line, col in meta.get("disabled_positions", [])
    }

    report = compare(
        load_static_edges(static_path),
        load_dynamic_edges(dynamic_path),
        set(meta.get("executed_callables", [])),
        observed_sites=observed_sites,
        disabled_positions=disabled_positions,
        driver_errors=meta.get("driver_errors", []),
        external_calls=int(meta.get("external_calls", 0)),
    )

    outdir = (args.outdir or artifacts).resolve()
    write_comparison(outdir, report, top_n=args.top)

    print(f"Comparison written to {outdir}")
    print(f"Runtime edges observed: {report.dynamic_edge_count}")
    print(f"Recall: {_percent(report.recall)} "
          f"({len(report.matched)} found / {len(report.matched) + len(report.missing_from_static)} executed)")
    print(f"Missing from static graph: {len(report.missing_from_static)}")
    print(f"Unconfirmed static edges (weak signal): {len(report.unconfirmed_static)}")
    if report.sited_static_count:
        print(
            f"Falsified static edges (site ran, dispatched elsewhere): "
            f"{len(report.falsified_static)} "
            f"[site coverage {_percent(report.site_match_rate)}]"
        )


if __name__ == "__main__":
    main()
