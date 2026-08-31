"""Score the static alias and lineage claims against observed object identity.

``access_comparison`` answers *which accesses happened*. This module answers the
other half of ``code_review.md`` section 6: **are two static object IDs the same
runtime object?** That is the claim ``alias_of`` and the lineage graph make, and
until now nothing had ever checked it.

**Why this had to wait for Step 2.** What it scores is precisely what sections
1.3 and 1.4 made a function of file processing order. Step 0 measured the cost
directly: a change that added twelve objects and removed nothing still moved 22
derived ``alias_of`` / ``access_path`` fields. A baseline taken against that would
not have been reproducible, so it would not have been a baseline. Step 2 closed
it; ``check-data-access-determinism`` is what keeps it closed.

**The one way this instrument is weaker than its sibling, stated up front.**
``access_comparison`` can *prove* a static claim wrong: the complete set of
access instructions in a code object is fixed at compile time, so a claimed
attribute that appears in no instruction is false no matter how little the
drivers ran. Identity has no such property. Two sites that never held the same
object may be a false alias -- or may be two code paths the drivers never
exercised together. So the verdicts here are:

===============  ==========================================================
confirmed        the two sites were seen holding one object -- strong
contradicted     both sites were observed holding recorded objects, and
                 never once the same one -- **suggestive, and coverage
                 dependent**; every one must be read against the source
                 before it is believed
unobserved       one side never ran, or never held anything recordable --
                 says nothing at all
===============  ==========================================================

Section 7 asks for "a recorded count of static alias claims the trace
contradicts". That count is a **lower bound carrying the caveat above**, and this
module prints the claims by name and location so the caveat can be acted on
rather than merely noted. Step 1a's first run accused 89 claims and was wrong
about all 89; hand-checking is what caught it, and this instrument has less
protection against that failure, not more.

**Recall is the direction that finds unlisted defects.** Every runtime identity
class of two or more sites is an aliasing the extractor either connected or did
not. The unconnected ones are the gaps, and unlike the contradictions they are
sound: the interpreter really did put one object in both places.

**Section 1.7 is a value question, not an identity question.** ``file:`` objects
are keyed on the source text of the argument, so ``load_users(path)`` and
``load_invoices(path)`` share the node ``file:path``. Whether that is real
coupling turns on the *path string*, not on object identity -- two equal paths
name one file whether or not they are one ``str``. So the file check reads the
value channel instead, and asks whether the callables sharing a ``file:`` node
were ever observed holding a common path.

**What is excluded, and why.** Same discipline as ``access_comparison``: every
exclusion is counted and printed, in both directions, because an exclusion nobody
can see is a way to choose the denominator quietly.

* Object kinds with no single frame-visible site -- ``class_state`` (a whole-class
  rollup, not one value), ``df_col`` / ``dict_key`` / ``container_field`` (a
  subscript result, which lives only on the stack), ``file``, ``unknown``.
* ``object_state`` paths deeper than one attribute -- ``a.b.c`` has an
  intermediate receiver that is never in the frame.
* Lineage edges whose endpoint is a ``return:`` slot -- a pseudo-node, not an
  object, and never registered in the object table.
* Sites the drivers never reached, and sites that only ever held excluded value
  types. Both are reported as ``unobserved`` rather than folded into either
  direction.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from microservice_pipeline.artifact_io import write_json, write_markdown
    from microservice_pipeline.data_access.dynamic_access_trace import (
        TIER_ATTR,
        TIER_NAME,
        IdentityKey,
        IdentitySite,
        ValueKey,
    )
    from microservice_pipeline.data_access.models import IDENTITY_RELATIONS
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import write_json, write_markdown  # type: ignore
    from data_access.dynamic_access_trace import (  # type: ignore
        TIER_ATTR,
        TIER_NAME,
        IdentityKey,
        IdentitySite,
        ValueKey,
    )
    from data_access.models import IDENTITY_RELATIONS  # type: ignore


# Verdicts on one static identity claim.
CONFIRMED = "confirmed"
CONTRADICTED = "contradicted"
UNOBSERVED = "unobserved"

# ``IDENTITY_RELATIONS`` -- the lineage relations that assert *this is the same
# object over there* -- comes from ``models`` rather than being re-listed here.
# It is a property of the artifact schema, and both the thing that writes the
# claims and the thing that scores them have to mean the same set by it. A
# relation absent from that set is counted as excluded with its name printed,
# which is how ``derived_from`` stays out of the score.

# Object kinds whose runtime value is reachable from a frame. Everything else is
# excluded with a named reason -- see the module docstring.
MAPPABLE_KINDS = frozenset({"param", "local_exposed", "class_attr_state", "object_state", "module_global"})

RECEIVER_NAMES = ("self", "cls")


# -- mapping static objects onto observation sites ------------------------


@dataclass
class SiteIndex:
    """Which frame sites each static object ID predicts a value at.

    A ``param`` or ``local_exposed`` predicts exactly one. A ``class_attr_state``
    predicts one per method of its class, because the attribute is the same
    object seen from every method. A ``module_global`` predicts one per callable
    in its module, for the same reason.
    """

    sites: Dict[str, Set[IdentitySite]] = field(default_factory=dict)
    objects_by_site: Dict[IdentitySite, Set[str]] = field(default_factory=lambda: defaultdict(set))
    excluded: Counter = field(default_factory=Counter)

    def for_object(self, object_id: str) -> Set[IdentitySite]:
        return self.sites.get(object_id, set())


def _split_scoped_id(object_id: str, prefix: str) -> Tuple[str, str]:
    """``param:pkg.mod.f:name`` -> ``('pkg.mod.f', 'name')``.

    ``rpartition`` and not ``split``: a callable ID contains colons nowhere but a
    dotted name contains dots everywhere, and the name is the last field.
    """
    rest = object_id[len(prefix) :]
    callable_id, _, name = rest.rpartition(":")
    return callable_id, name


def build_site_index(
    objects: Sequence[Mapping[str, str]],
    callables: Sequence[Mapping[str, object]],
) -> SiteIndex:
    index = SiteIndex()

    methods_by_class: Dict[str, List[str]] = defaultdict(list)
    callables_by_module: Dict[str, List[str]] = defaultdict(list)
    for entry in callables:
        callable_id = str(entry.get("id", ""))
        module = str(entry.get("module", ""))
        class_name = entry.get("class_name")
        if module:
            callables_by_module[module].append(callable_id)
        if class_name:
            methods_by_class[f"{module}.{class_name}"].append(callable_id)

    for row in objects:
        object_id = row["id"]
        kind = row["kind"]
        sites: Set[IdentitySite] = set()

        if kind not in MAPPABLE_KINDS:
            index.excluded[f"object kind {kind}"] += 1
            continue

        if kind == "param" and object_id.startswith("param:"):
            callable_id, name = _split_scoped_id(object_id, "param:")
            if callable_id and name:
                sites = {(callable_id, TIER_NAME, name)}
        elif kind == "local_exposed" and object_id.startswith("local_exposed:"):
            callable_id, name = _split_scoped_id(object_id, "local_exposed:")
            if callable_id and name:
                sites = {(callable_id, TIER_NAME, name)}
        elif kind == "class_attr_state" and object_id.startswith("class_attr_state:"):
            owner, _, attr = object_id[len("class_attr_state:") :].rpartition(":")
            if owner and attr:
                sites = {
                    (method, TIER_ATTR, f"{receiver}.{attr}")
                    for method in methods_by_class.get(owner, ())
                    for receiver in RECEIVER_NAMES
                }
        elif kind == "module_global" and object_id.startswith("module_global:"):
            module, _, name = object_id[len("module_global:") :].rpartition(".")
            if module and name:
                sites = {
                    (callable_id, TIER_NAME, name)
                    for callable_id in callables_by_module.get(module, ())
                }
        elif kind == "object_state":
            sites = _object_state_sites(row, index)

        if not sites:
            index.excluded[f"no site derivable for {kind}"] += 1
            continue

        index.sites[object_id] = sites
        for site in sites:
            index.objects_by_site[site].add(object_id)

    return index


def _object_state_sites(row: Mapping[str, str], index: SiteIndex) -> Set[IdentitySite]:
    """Sites for ``object_state:{root}:{path}``, when the path is one attribute deep.

    ``model.R`` is reachable: ``model`` is in the frame and ``R`` comes out of its
    ``__dict__``. ``model.R.units`` is not -- the receiver ``model.R`` exists only
    on the operand stack -- so it is excluded rather than approximated by its
    prefix, which would score the wrong object.
    """
    access_path = row.get("access_path", "")
    owner = row.get("owner", "") or row.get("container", "")
    segments = access_path.split(".")
    if len(segments) != 2 or not all(segments):
        index.excluded["object_state path not one attribute deep"] += 1
        return set()

    receiver, attr = segments
    if owner.startswith("param:") or owner.startswith("local_exposed:"):
        prefix = "param:" if owner.startswith("param:") else "local_exposed:"
        callable_id, name = _split_scoped_id(owner, prefix)
        if callable_id and name == receiver:
            return {(callable_id, TIER_ATTR, access_path)}
    index.excluded["object_state root is not a frame-visible name"] += 1
    return set()


# -- the runtime side -----------------------------------------------------


@dataclass
class RuntimeIdentity:
    """What the trace saw, indexed both ways."""

    tokens_by_site: Dict[IdentitySite, Set[int]] = field(default_factory=lambda: defaultdict(set))
    sites_by_token: Dict[int, Set[IdentitySite]] = field(default_factory=lambda: defaultdict(set))
    sites_seen: Set[IdentitySite] = field(default_factory=set)

    def tokens_for(self, sites: Set[IdentitySite]) -> Set[int]:
        found: Set[int] = set()
        for site in sites:
            found |= self.tokens_by_site.get(site, set())
        return found


def build_runtime_identity(
    identities: Set[IdentityKey], sites_seen: Set[IdentitySite]
) -> RuntimeIdentity:
    runtime = RuntimeIdentity(sites_seen=set(sites_seen))
    for callable_id, tier, path, token in identities:
        site = (callable_id, tier, path)
        runtime.tokens_by_site[site].add(token)
        runtime.sites_by_token[token].add(site)
        runtime.sites_seen.add(site)
    return runtime


# -- scoring --------------------------------------------------------------


@dataclass(frozen=True)
class IdentityClaim:
    """One static assertion that two object IDs are the same runtime object."""

    src: str
    dst: str
    source: str  # "alias_of" or a lineage relation name
    file: str = ""
    lineno: int = 0


@dataclass
class ClaimVerdict:
    claim: IdentityClaim
    verdict: str
    src_sites: int = 0
    dst_sites: int = 0
    shared_tokens: int = 0


@dataclass
class IdentityComparisonReport:
    # -- claim direction (precision) --
    verdicts: Counter = field(default_factory=Counter)
    verdicts_by_source: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    contradicted: List[ClaimVerdict] = field(default_factory=list)
    claim_total: int = 0
    claims_excluded: Counter = field(default_factory=Counter)

    # -- recall direction --
    runtime_classes: int = 0
    runtime_classes_connected: int = 0
    runtime_classes_split: int = 0
    runtime_classes_unmapped: int = 0
    split_examples: List[Tuple[int, List[List[str]]]] = field(default_factory=list)

    # -- section 1.7 --
    file_nodes_multi_callable: int = 0
    file_nodes_disagreeing: int = 0
    file_nodes_unobserved: int = 0
    file_disagreements: List[Tuple[str, List[Tuple[str, List[str]]]]] = field(default_factory=list)

    # -- provenance --
    site_exclusions: Counter = field(default_factory=Counter)
    trace_exclusions: Dict[str, int] = field(default_factory=dict)
    objects_total: int = 0
    objects_mapped: int = 0
    sites_predicted: int = 0
    sites_observed: int = 0
    identity_observations: int = 0
    objects_retained: int = 0
    retention_cap_hit: bool = False
    driver_errors: List[str] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return self.verdicts[CONFIRMED] + self.verdicts[CONTRADICTED]

    @property
    def precision(self) -> float:
        return self.verdicts[CONFIRMED] / self.decided if self.decided else 0.0

    @property
    def recall(self) -> float:
        scored = self.runtime_classes_connected + self.runtime_classes_split
        return self.runtime_classes_connected / scored if scored else 0.0


def _score_claim(
    claim: IdentityClaim, site_index: SiteIndex, runtime: RuntimeIdentity
) -> ClaimVerdict:
    src_sites = site_index.for_object(claim.src)
    dst_sites = site_index.for_object(claim.dst)
    src_tokens = runtime.tokens_for(src_sites)
    dst_tokens = runtime.tokens_for(dst_sites)

    shared = src_tokens & dst_tokens
    if shared:
        verdict = CONFIRMED
    elif src_tokens and dst_tokens:
        # Both ends really held objects, and never the same one. Suggestive of a
        # false alias -- but the drivers may simply never have run the two
        # together, which is why this is hand-checked and not trusted.
        verdict = CONTRADICTED
    else:
        verdict = UNOBSERVED
    return ClaimVerdict(
        claim=claim,
        verdict=verdict,
        src_sites=len(src_sites),
        dst_sites=len(dst_sites),
        shared_tokens=len(shared),
    )


class _Components:
    """Union-find over object IDs, for "does the static graph connect these at all".

    Recall asks a reachability question, not a "did it say exactly this edge"
    question: an alias the extractor recorded as a two-hop lineage path is still
    an alias it found. Components answer that in one pass.
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self._parent.setdefault(item, item)
        while parent != item:
            item, parent = parent, self._parent.setdefault(parent, parent)
        return item

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[left_root] = right_root


def compare(
    objects: Sequence[Mapping[str, str]],
    callables: Sequence[Mapping[str, object]],
    lineage_edges: Sequence[Mapping[str, object]],
    access_edges: Sequence[Mapping[str, str]],
    identities: Set[IdentityKey],
    values: Set[ValueKey],
    sites_seen: Set[IdentitySite],
    *,
    trace_meta: Optional[Mapping[str, object]] = None,
    top_n: int = 25,
) -> IdentityComparisonReport:
    trace_meta = trace_meta or {}
    report = IdentityComparisonReport()
    report.objects_total = len(objects)
    report.identity_observations = len(identities)
    report.objects_retained = int(trace_meta.get("objects_retained", 0) or 0)
    report.retention_cap_hit = bool(trace_meta.get("retention_cap_hit", False))
    report.trace_exclusions = dict(trace_meta.get("exclusions", {}) or {})
    report.driver_errors = list(trace_meta.get("driver_errors", []) or [])

    site_index = build_site_index(objects, callables)
    report.site_exclusions = site_index.excluded
    report.objects_mapped = len(site_index.sites)
    report.sites_predicted = len(site_index.objects_by_site)

    runtime = build_runtime_identity(identities, sites_seen)
    report.sites_observed = len(runtime.tokens_by_site)

    # -- direction one: every static identity claim -----------------------

    claims: List[IdentityClaim] = []
    for row in objects:
        alias_of = row.get("alias_of", "")
        if alias_of:
            claims.append(IdentityClaim(src=row["id"], dst=alias_of, source="alias_of"))
    for edge in lineage_edges:
        relation = str(edge.get("relation", ""))
        if relation not in IDENTITY_RELATIONS:
            report.claims_excluded[f"relation {relation} is not an identity claim"] += 1
            continue
        claims.append(
            IdentityClaim(
                src=str(edge.get("src_object_id", "")),
                dst=str(edge.get("dst_object_id", "")),
                source=relation,
                file=str(edge.get("file", "")),
                lineno=int(edge.get("lineno", 0) or 0),
            )
        )

    report.claim_total = len(claims)
    known_objects = {row["id"] for row in objects}
    for claim in claims:
        if claim.src not in known_objects or claim.dst not in known_objects:
            # Overwhelmingly ``return:{callable}`` slots, which are lineage
            # pseudo-nodes rather than data objects.
            report.claims_excluded["endpoint is not a registered object"] += 1
            continue
        if claim.src not in site_index.sites or claim.dst not in site_index.sites:
            report.claims_excluded["endpoint has no frame-visible site"] += 1
            continue
        scored = _score_claim(claim, site_index, runtime)
        report.verdicts[scored.verdict] += 1
        report.verdicts_by_source[claim.source][scored.verdict] += 1
        if scored.verdict == CONTRADICTED:
            report.contradicted.append(scored)

    report.contradicted.sort(key=lambda item: (item.claim.source, item.claim.src, item.claim.dst))

    # -- direction two: every runtime identity class ----------------------

    components = _Components()
    for row in objects:
        components.find(row["id"])
        if row.get("alias_of"):
            components.union(row["id"], row["alias_of"])
    for edge in lineage_edges:
        if str(edge.get("relation", "")) not in IDENTITY_RELATIONS:
            continue
        src, dst = str(edge.get("src_object_id", "")), str(edge.get("dst_object_id", ""))
        if src in known_objects and dst in known_objects:
            components.union(src, dst)

    split_details: List[Tuple[int, List[List[str]]]] = []
    for token, token_sites in runtime.sites_by_token.items():
        if len(token_sites) < 2:
            continue
        report.runtime_classes += 1
        groups: Dict[str, Set[str]] = defaultdict(set)
        for site in token_sites:
            for object_id in site_index.objects_by_site.get(site, ()):
                groups[components.find(object_id)].add(object_id)
        if not groups:
            # The interpreter saw an aliasing between places the extractor models
            # no object for at all. Not a wrong answer -- an absent one.
            report.runtime_classes_unmapped += 1
        elif len(groups) == 1:
            report.runtime_classes_connected += 1
        else:
            report.runtime_classes_split += 1
            split_details.append((len(token_sites), [sorted(group) for group in groups.values()]))

    split_details.sort(key=lambda item: (-len(item[1]), -item[0]))
    report.split_examples = split_details[:top_n]

    # -- section 1.7: do callables sharing a file node share a path? ------

    report_file_rows = _compare_file_nodes(objects, access_edges, values, top_n=top_n)
    (
        report.file_nodes_multi_callable,
        report.file_nodes_disagreeing,
        report.file_nodes_unobserved,
        report.file_disagreements,
    ) = report_file_rows

    return report


def _compare_file_nodes(
    objects: Sequence[Mapping[str, str]],
    access_edges: Sequence[Mapping[str, str]],
    values: Set[ValueKey],
    *,
    top_n: int,
) -> Tuple[int, int, int, List[Tuple[str, List[Tuple[str, List[str]]]]]]:
    """Which ``file:`` nodes fuse callables that never saw a common path.

    Deliberately asked of *every* multi-callable file node rather than only the
    ones whose ID came from a non-literal expression. Telling a literal path from
    an expression by looking at the ID is guesswork -- ``data/users.csv`` parses
    as Python -- and it is not needed: a genuinely shared literal produces
    agreeing values and drops out of the answer on its own.
    """
    file_objects = {row["id"]: row for row in objects if row["kind"] == "file"}
    callables_by_file: Dict[str, Set[str]] = defaultdict(set)
    for edge in access_edges:
        object_id = edge.get("object_id", "")
        if object_id in file_objects:
            callables_by_file[object_id].add(edge.get("callable", ""))

    values_by_site: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for callable_id, path, value in values:
        values_by_site[(callable_id, path)].add(value)

    multi = 0
    disagreeing = 0
    unobserved = 0
    details: List[Tuple[str, List[Tuple[str, List[str]]]]] = []
    for object_id, callers in sorted(callables_by_file.items()):
        if len(callers) < 2:
            continue
        multi += 1
        expression = file_objects[object_id].get("access_path", "") or file_objects[object_id].get(
            "display_name", ""
        )
        observed: List[Tuple[str, List[str]]] = []
        for caller in sorted(callers):
            seen = values_by_site.get((caller, expression), set())
            if seen:
                observed.append((caller, sorted(seen)))
        if len(observed) < 2:
            unobserved += 1
            continue
        common = set(observed[0][1])
        for _caller, seen in observed[1:]:
            common &= set(seen)
        if not common:
            disagreeing += 1
            details.append((object_id, observed))

    details.sort(key=lambda item: -len(item[1]))
    return multi, disagreeing, unobserved, details[:top_n]


# -- artifacts ------------------------------------------------------------


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ratio(hit: int, total: int) -> str:
    return f"{hit}/{total} ({_percent(hit / total)})" if total else f"{hit}/0 (n/a)"


def render_markdown(report: IdentityComparisonReport, *, top_n: int = 25) -> List[str]:
    lines = [
        "# Object-identity comparison",
        "",
        "Static `alias_of` and lineage claims, scored against the objects the",
        "interpreter actually put at each site. This is the second half of",
        "`code_review.md` section 6 — Step 1b.",
        "",
    ]

    if report.retention_cap_hit:
        lines += [
            "> **INCOMPLETE — the object retention cap was hit.** Some observations were",
            "> dropped to keep memory bounded, so every number below is computed on a",
            "> truncated set. Re-run with a larger `--identity-cap` before recording a",
            "> baseline.",
            "",
        ]

    lines += [
        "## Read this before the numbers",
        "",
        "`contradicted` is **not** proof. Unlike `access_comparison`, which can refute a",
        "claim from bytecode alone, this instrument only knows what the drivers ran. Two",
        "sites that never held the same object may be a false alias, or may be two paths",
        "that were never exercised together. Every contradiction is listed by name below",
        "so it can be read against the source; the count is a lower bound with that",
        "caveat, not a defect tally.",
        "",
        "Recall is the sound direction: an object the interpreter really did put in two",
        "places is an aliasing that exists, whether or not the extractor found it.",
        "",
        "## Coverage",
        "",
        "| | |",
        "| --- | ---: |",
        f"| static objects | {report.objects_total} |",
        f"| — with a frame-visible site | {_ratio(report.objects_mapped, report.objects_total)} |",
        f"| sites predicted by the static side | {report.sites_predicted} |",
        f"| sites the trace actually observed | {report.sites_observed} |",
        f"| identity observations | {report.identity_observations} |",
        f"| distinct objects retained | {report.objects_retained} |",
        "",
        "## Precision — static claims, scored",
        "",
        "| | |",
        "| --- | ---: |",
        f"| identity claims made | {report.claim_total} |",
        f"| — scored | {report.decided + report.verdicts[UNOBSERVED]} |",
        f"| **confirmed** | **{report.verdicts[CONFIRMED]}** |",
        f"| **contradicted** | **{report.verdicts[CONTRADICTED]}** |",
        f"| unobserved | {report.verdicts[UNOBSERVED]} |",
        f"| **precision** (confirmed / decided) | **{_percent(report.precision)}** |",
        "",
        "### By claim source",
        "",
        "| source | confirmed | contradicted | unobserved |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source in sorted(report.verdicts_by_source):
        counts = report.verdicts_by_source[source]
        lines.append(
            f"| `{source}` | {counts[CONFIRMED]} | {counts[CONTRADICTED]} | {counts[UNOBSERVED]} |"
        )

    lines += [
        "",
        "## Recall — runtime identity classes",
        "",
        "One class is one object the interpreter put at two or more sites.",
        "",
        "| | |",
        "| --- | ---: |",
        f"| classes observed | {report.runtime_classes} |",
        f"| — connected by the static graph | {report.runtime_classes_connected} |",
        f"| — split across components (missed) | {report.runtime_classes_split} |",
        f"| — no static object at any site | {report.runtime_classes_unmapped} |",
        f"| **recall** (connected / scored) | **{_percent(report.recall)}** |",
        "",
    ]

    if report.split_examples:
        lines += [
            "### Largest missed aliasings",
            "",
            "The interpreter put one object at all of these, and the static graph has them",
            "in separate components.",
            "",
        ]
        for site_count, groups in report.split_examples[:top_n]:
            lines.append(f"- **{len(groups)} components across {site_count} sites**")
            for group in groups[:6]:
                lines.append(f"  - {', '.join(f'`{name}`' for name in group[:4])}")
        lines.append("")

    lines += [
        "## Section 1.7 — do shared `file:` nodes mean shared files?",
        "",
        "| | |",
        "| --- | ---: |",
        f"| `file:` nodes touched by two or more callables | {report.file_nodes_multi_callable} |",
        f"| — where no two callables saw a common path | **{report.file_nodes_disagreeing}** |",
        f"| — where fewer than two callables were observed | {report.file_nodes_unobserved} |",
        "",
    ]
    if report.file_disagreements:
        lines += ["### Manufactured file coupling", ""]
        for object_id, observed in report.file_disagreements[:top_n]:
            lines.append(f"- `{object_id}`")
            for caller, seen in observed[:6]:
                lines.append(f"  - `{caller}` → {', '.join(repr(value) for value in seen[:3])}")
        lines.append("")

    if report.contradicted:
        lines += [
            "## Contradicted claims — hand-check every one",
            "",
            "| source | claim | src sites | dst sites | where |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for scored in report.contradicted[:top_n]:
            claim = scored.claim
            where = f"`{claim.file}:{claim.lineno}`" if claim.file else "—"
            lines.append(
                f"| `{claim.source}` | `{claim.src}` → `{claim.dst}` "
                f"| {scored.src_sites} | {scored.dst_sites} | {where} |"
            )
        if len(report.contradicted) > top_n:
            lines.append(f"| … | {len(report.contradicted) - top_n} more | | | |")
        lines.append("")

    lines += ["## What was excluded", "", "### Static objects with no frame-visible site", ""]
    if report.site_exclusions:
        lines += ["| reason | objects |", "| --- | ---: |"]
        for reason, count in report.site_exclusions.most_common():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("None.")
    lines += ["", "### Claims not scored", ""]
    if report.claims_excluded:
        lines += ["| reason | claims |", "| --- | ---: |"]
        for reason, count in report.claims_excluded.most_common():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("None.")
    lines += ["", "### Observations the tracer could not record", ""]
    if report.trace_exclusions:
        lines += ["| reason | count |", "| --- | ---: |"]
        for reason, count in sorted(report.trace_exclusions.items()):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("None.")
    lines.append("")

    if report.driver_errors:
        lines += ["## Driver problems", ""] + [f"- {message}" for message in report.driver_errors] + [""]

    return lines


def write_comparison(outdir: Path, report: IdentityComparisonReport, *, top_n: int = 25) -> None:
    write_markdown(
        outdir / "identity_comparison.md",
        render_markdown(report, top_n=top_n),
        trailing_newline=True,
    )
    write_json(
        outdir / "identity_comparison.json",
        {
            "objects_total": report.objects_total,
            "objects_mapped": report.objects_mapped,
            "sites_predicted": report.sites_predicted,
            "sites_observed": report.sites_observed,
            "identity_observations": report.identity_observations,
            "objects_retained": report.objects_retained,
            "retention_cap_hit": report.retention_cap_hit,
            "claim_total": report.claim_total,
            "verdicts": dict(sorted(report.verdicts.items())),
            "verdicts_by_source": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(report.verdicts_by_source.items())
            },
            "precision": report.precision,
            "recall": report.recall,
            "runtime_classes": report.runtime_classes,
            "runtime_classes_connected": report.runtime_classes_connected,
            "runtime_classes_split": report.runtime_classes_split,
            "runtime_classes_unmapped": report.runtime_classes_unmapped,
            "split_examples": [
                {"sites": site_count, "components": groups}
                for site_count, groups in report.split_examples
            ],
            "file_nodes_multi_callable": report.file_nodes_multi_callable,
            "file_nodes_disagreeing": report.file_nodes_disagreeing,
            "file_nodes_unobserved": report.file_nodes_unobserved,
            "file_disagreements": [
                {"object_id": object_id, "observed": [{"callable": c, "values": v} for c, v in seen]}
                for object_id, seen in report.file_disagreements
            ],
            "contradicted_claims": [
                {
                    "source": scored.claim.source,
                    "src": scored.claim.src,
                    "dst": scored.claim.dst,
                    "file": scored.claim.file,
                    "lineno": scored.claim.lineno,
                    "src_sites": scored.src_sites,
                    "dst_sites": scored.dst_sites,
                }
                for scored in report.contradicted
            ],
            "site_exclusions": dict(report.site_exclusions.most_common()),
            "claims_excluded": dict(report.claims_excluded.most_common()),
            "trace_exclusions": dict(sorted(report.trace_exclusions.items())),
            "driver_errors": report.driver_errors,
        },
    )


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    try:
        from microservice_pipeline.config import load_extraction_config
        from microservice_pipeline.data_access.dynamic_access_trace import (
            load_observed_identities,
            load_observed_values,
        )
    except ImportError:  # pragma: no cover
        from config import load_extraction_config  # type: ignore
        from data_access.dynamic_access_trace import (  # type: ignore
            load_observed_identities,
            load_observed_values,
        )

    parser = argparse.ArgumentParser(
        description="Score static alias and lineage claims against observed object identity",
    )
    parser.add_argument("--config", default=None, type=Path, help="Extraction JSON/JSONC config")
    parser.add_argument(
        "--artifacts",
        default=None,
        type=Path,
        help="Directory holding data_access.json and dynamic_identity.csv",
    )
    parser.add_argument("--outdir", default=None, type=Path, help="Where to write the report")
    parser.add_argument("--top", default=25, type=int, help="Rows per grouping table")
    args = parser.parse_args(argv)

    if args.config:
        config = load_extraction_config(args.config)
        artifacts = (args.artifacts or config.data_access.outdir).resolve()
    elif args.artifacts:
        artifacts = args.artifacts.resolve()
    else:
        raise SystemExit("--config or --artifacts is required")

    static_path = artifacts / "data_access.json"
    identity_path = artifacts / "dynamic_identity.csv"
    values_path = artifacts / "dynamic_identity_values.csv"
    meta_path = artifacts / "dynamic_identity.json"
    if not static_path.exists():
        raise SystemExit(f"Missing {static_path}; run 'data-access' first")
    if not identity_path.exists():
        raise SystemExit(f"Missing {identity_path}; run 'trace-data-access --identity' first")

    payload = json.loads(static_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    sites_seen = {tuple(site) for site in meta.get("identity_sites_seen", [])}

    report = compare(
        payload.get("objects", []),
        payload.get("callables", []),
        payload.get("lineage_edges", []),
        _load_csv(artifacts / "access_edges.csv") if (artifacts / "access_edges.csv").exists() else [],
        load_observed_identities(identity_path),
        load_observed_values(values_path) if values_path.exists() else set(),
        sites_seen,  # type: ignore[arg-type]
        trace_meta=meta,
        top_n=args.top,
    )

    outdir = (args.outdir or artifacts).resolve()
    write_comparison(outdir, report, top_n=args.top)

    print(f"Identity comparison written to {outdir}")
    print(f"Objects with a frame-visible site: {_ratio(report.objects_mapped, report.objects_total)}")
    print(f"Identity claims: {report.claim_total}")
    print(f"  confirmed:    {report.verdicts[CONFIRMED]}")
    print(f"  contradicted: {report.verdicts[CONTRADICTED]}  (hand-check every one)")
    print(f"  unobserved:   {report.verdicts[UNOBSERVED]}")
    print(f"Alias precision: {_percent(report.precision)}")
    print(
        f"Alias recall: {_percent(report.recall)} "
        f"({report.runtime_classes_connected} connected, {report.runtime_classes_split} split)"
    )
    print(
        f"Section 1.7: {report.file_nodes_disagreeing} of {report.file_nodes_multi_callable} "
        "shared file nodes join callables that never saw a common path"
    )
    if report.retention_cap_hit:
        print("WARNING: retention cap hit -- these numbers are computed on a truncated set")


if __name__ == "__main__":  # pragma: no cover
    main()
