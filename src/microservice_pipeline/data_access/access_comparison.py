"""Score the static data-access artifacts against a runtime trace.

This is the instrument ``code_review.md`` section 6 asks for: until it exists,
every finding in that document is a claim about *mechanism* and none is a claim
about *impact*. It is the data-access sibling of ``call_graph/graph_comparison``
and keeps that module's discipline about asymmetry, but it can say one thing the
call-graph comparison structurally cannot.

**Why this comparison has a third bucket.** A callee is a runtime value, so a
static call edge the trace did not see is only ever *unconfirmed* -- the branch
may simply not have run. An *access* is different: the complete set of attribute,
subscript and name instructions a code object can ever execute is fixed at
compile time and can be read straight out of its bytecode. So a static edge
claiming that ``EBM._compute`` reads ``self.spacing``, where no ``LOAD_ATTR
spacing`` exists anywhere in that code object, is **wrong** -- and it is wrong
regardless of how little the drivers covered. Four buckets, not three:

===============  ==========================================================
matched          the instruction exists and the interpreter executed it
unexecuted       it exists, the callable ran, that offset did not -- weak,
                 an untaken branch, *not* a defect
not exercised    it exists but the callable never ran -- says nothing
falsified        no such instruction exists in the code object at all --
                 **strong, and independent of coverage**
===============  ==========================================================

plus **missing**: observed at runtime, claimed by no static edge. That is the
recall number, and the direction that finds defects nobody listed.

**What is excluded before scoring, and why.** Section 6 requires that relations
the instrument cannot express be excluded rather than counted as gaps. Here that
means:

* *exposure* operations (``return``, ``passed_arg``, ``passed_kwarg``,
  ``escape_assign``) -- these are not memory operations at all. They record that
  a local escaped its scope, which no single instruction corresponds to. The
  ``SYNTHETIC_RELATIONS`` analogue.
* *file* objects -- identity is established at a ``CALL`` (``open``,
  ``pd.read_csv``), not at an access instruction. The ``UNOBSERVABLE_RELATIONS``
  analogue, and reachable by the call tracer instead.
* *method loads* on the observed side -- ``x.f()`` emits ``LOAD_ATTR f``, but the
  static side models a read of the *receiver*, never of the method name.
  Comparing them would report every method call in the project as a missed
  attribute access.
* ``super().x`` -- the receiver is a dispatch proxy, not an object, and
  ``_resolve_attribute`` returns nothing for a call receiver. Zero static rows on
  climlab mention ``super(``, so every one of these would be a guaranteed miss.
* *ordinary locals* on the observed side -- whether a local is "exposed" is a
  static judgment (``LocalBinding.exposed``) the interpreter has no view of.
* ``self`` and ``cls`` -- every method body emits ``LOAD_FAST self``, but the
  extractor deliberately excludes them from ``callable_params`` and models
  ``self.x`` as class state instead. They are the receiver, not data.
* *globals that are not module-level data* -- ``LOAD_GLOBAL`` fires for ``len``,
  ``range`` and every imported symbol. The static side registers a
  ``module_global`` only for a name assigned at module scope, so the observed
  side is filtered through ``collect_module_globals`` -- the extractor's own
  definition, imported rather than restated.

Every exclusion is counted and printed. An exclusion nobody can see is a way for
the denominator to be quietly chosen, which is the failure section 6 names when
it says *a number that cannot fail is not a measurement*.

**Literal and computed keys are scored separately**, never folded together. The
static side names a key by its literal; the runtime side sees a value. ``d['k']``
is comparable and ``d[k]`` is not, and merging them would let a computed-key miss
look like a literal-key miss -- hiding which of sections 1.1 and 1.7 is expensive,
which is the question the instrument was built to answer.
"""

from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from microservice_pipeline.artifact_io import write_json, write_markdown
    from microservice_pipeline.call_graph.dynamic_trace import callable_id_for_qualname
    from microservice_pipeline.call_graph.models import AnalysisFile
    from microservice_pipeline.data_access.dynamic_access_trace import (
        ACCESS_READ,
        ACCESS_WRITE,
        COMPUTED_KEY,
        ROLE_ATTRIBUTE,
        ROLE_COMPUTED,
        ROLE_GLOBAL,
        ROLE_LITERAL,
        ROLE_PARAM,
        TIER_ATTR,
        TIER_KEY,
        TIER_NAME,
        AccessKey,
        decode_access_instructions,
        iter_code_objects,
    )
    from microservice_pipeline.data_access.generate_data_access_ast import collect_module_globals
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import write_json, write_markdown  # type: ignore
    from call_graph.dynamic_trace import callable_id_for_qualname  # type: ignore
    from call_graph.models import AnalysisFile  # type: ignore
    from data_access.dynamic_access_trace import (  # type: ignore
        ACCESS_READ,
        ACCESS_WRITE,
        COMPUTED_KEY,
        ROLE_ATTRIBUTE,
        ROLE_COMPUTED,
        ROLE_GLOBAL,
        ROLE_LITERAL,
        ROLE_PARAM,
        TIER_ATTR,
        TIER_KEY,
        TIER_NAME,
        AccessKey,
        decode_access_instructions,
        iter_code_objects,
    )
    from data_access.generate_data_access_ast import collect_module_globals  # type: ignore


# Static ``operation`` values that record a local *escaping* rather than a memory
# access. No instruction corresponds to them. See the module docstring.
EXPOSURE_OPERATIONS = frozenset({"return", "passed_arg", "passed_kwarg", "escape_assign"})

# xarray label selection (``ds.sel(time=...)``) is modelled as a keyed access but
# compiles to a call with a keyword argument -- there is no subscript instruction
# to match it against.
LABELED_ACCESS_SUFFIX = ":labeled_access"

# Observed (tier, role) pairs that may enter the recall denominator. Globals are
# filtered further, against the extractor's own ``collect_module_globals``.
COMPARABLE_ROLES = frozenset(
    {
        (TIER_ATTR, ROLE_ATTRIBUTE),
        (TIER_KEY, ROLE_LITERAL),
        (TIER_KEY, ROLE_COMPUTED),
        (TIER_NAME, ROLE_PARAM),
        (TIER_NAME, ROLE_GLOBAL),
    }
)

# The receiver. ``generate_data_access_ast`` strips these from ``callable_params``
# (line 1858) precisely because ``self.x`` is modelled as class state rather than
# as a read of ``self``, so scoring them would report every method in the project
# as a missed parameter access.
RECEIVER_NAMES = frozenset({"self", "cls"})

# Why a static row was not scored. Each is reported with its count.
EXCLUDED_EXPOSURE = "exposure"
EXCLUDED_FILE = "file"
EXCLUDED_UNOBSERVABLE = "unobservable"
UNDERIVABLE = "underivable"

# Verdicts on a scored static row.
MATCHED = "matched"
UNEXECUTED = "unexecuted"
NOT_EXERCISED = "not_exercised"
FALSIFIED = "falsified"
NO_BYTECODE = "no_bytecode"


# -- the static bytecode index -------------------------------------------


@dataclass
class BytecodeIndex:
    """What every callable in the project is *capable* of accessing.

    Built by compiling each analysed file and decoding its code objects. No
    execution, so this is complete: absence from it is proof, which is what makes
    the ``falsified`` verdict independent of driver coverage.
    """

    accesses: Dict[str, Set[Tuple[str, str, str]]] = field(default_factory=dict)
    module_by_callable: Dict[str, str] = field(default_factory=dict)
    module_globals: Dict[str, Set[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def can_access(self, callable_id: str, tier: str, name: str, accesses: Set[str]) -> bool:
        known = self.accesses.get(callable_id)
        if known is None:
            return False
        return any((tier, name, access) in known for access in accesses)

    def is_module_global(self, callable_id: str, name: str) -> bool:
        module = self.module_by_callable.get(callable_id)
        if module is None:
            return False
        return name in self.module_globals.get(module, frozenset())


def build_bytecode_index(analysis_files: Sequence[AnalysisFile]) -> BytecodeIndex:
    """Compile every analysed file and record what each callable can touch.

    Per-file failures are collected rather than raised. Section 1.12 is still
    open -- one unparseable vendored file currently kills a whole data-access run
    -- and adding another uncaught ``compile`` site would be reintroducing the
    defect in the module built to measure it.
    """
    index = BytecodeIndex()
    for analysis_file in analysis_files:
        path = analysis_file.path
        module = analysis_file.module
        try:
            # ``read_bytes`` rather than ``read_text`` so a PEP-263 coding cookie
            # is honoured -- the other half of section 1.12.
            code = compile(path.read_bytes(), str(path), "exec")
        except (SyntaxError, ValueError, OSError, UnicodeDecodeError) as exc:
            index.errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        for nested in iter_code_objects(code):
            callable_id = callable_id_for_qualname(module, nested.co_qualname)
            bucket = index.accesses.setdefault(callable_id, set())
            for access in decode_access_instructions(nested):
                bucket.add((access.tier, access.name, access.access))
            index.module_by_callable.setdefault(callable_id, module)

        try:
            index.module_globals[module] = collect_module_globals(ast.parse(path.read_bytes()))
        except (SyntaxError, ValueError, OSError) as exc:  # pragma: no cover - compile already succeeded
            index.errors.append(f"{path}: module globals: {type(exc).__name__}: {exc}")

    return index


# -- deriving the static side's triple ------------------------------------


@dataclass(frozen=True)
class StaticClaim:
    """One row of ``callable_data_access.csv``, reduced to a comparable form."""

    callable: str
    tier: str
    name: str
    accesses: frozenset
    object_kind: str
    object_id: str
    operation: str
    lineno: int
    evidence: str

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.callable, self.tier, self.name)


def _accesses_for(access: str) -> frozenset:
    """The instruction accesses one static ``access`` label may correspond to."""
    if access == ACCESS_READ:
        return frozenset({ACCESS_READ})
    if access in {ACCESS_WRITE, "create"}:
        return frozenset({ACCESS_WRITE})
    # ``read_write`` covers mutation and augmented assignment; either instruction
    # confirms it, and only the absence of both falsifies it.
    return frozenset({ACCESS_READ, ACCESS_WRITE})


def _classify_expression(node: ast.AST) -> Optional[Tuple[str, str]]:
    """Reduce an access expression to ``(tier, name)``."""
    if isinstance(node, ast.Attribute):
        return (TIER_ATTR, node.attr)
    if isinstance(node, ast.Subscript):
        return (TIER_KEY, _slice_name(node.slice))
    if isinstance(node, ast.Name):
        return (TIER_NAME, node.id)
    return None


def _slice_name(node: ast.AST) -> str:
    """The literal a subscript names, or ``COMPUTED_KEY``.

    Only the string case is a key both sides can name. ``df.loc[:, 'col']``
    parses as a tuple slice and contributes its string element, matching how the
    compiler folds it into one constant.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Tuple):
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                return element.value
    return COMPUTED_KEY


def derive_static_claim(row: Mapping[str, str]) -> Tuple[Optional[StaticClaim], str]:
    """Turn one artifact row into a claim, or say why it cannot be scored.

    ``evidence`` is the authority, not ``field`` or ``access_path``. It is an
    unparsed copy of the *access expression itself* and is recorded per edge,
    whereas ``access_path`` belongs to the *object* and is whatever path first
    registered it -- verified stale in practice: a ``class_state`` edge whose
    evidence is ``self.state`` carries ``access_path='self.time_type'``. Using
    the object's path would compare the wrong attribute and read as a defect in
    the extractor rather than in this module.
    """
    operation = row.get("operation", "")
    object_kind = row.get("object_kind", "")

    if object_kind == "file" or row.get("object_id", "").startswith("file:"):
        return (None, EXCLUDED_FILE)
    if operation in EXPOSURE_OPERATIONS:
        return (None, EXCLUDED_EXPOSURE)
    if operation.endswith(LABELED_ACCESS_SUFFIX):
        return (None, EXCLUDED_UNOBSERVABLE)

    evidence = (row.get("evidence") or "").strip()
    classified: Optional[Tuple[str, str]] = None
    if evidence:
        try:
            node: ast.AST = ast.parse(evidence, mode="eval").body
        except (SyntaxError, ValueError):
            node = ast.Pass()
        if operation.startswith("method:"):
            # The edge is about the receiver, not the method: ``self.state.items()``
            # is recorded as a read of ``self.state``.
            if isinstance(node, ast.Call):
                node = node.func
            if isinstance(node, ast.Attribute):
                node = node.value
        classified = _classify_expression(node)

    if classified is None:
        # Fall back to the object's own field name. ``class_state`` rolls its
        # attributes up to the class and leaves ``field`` empty, so this cannot
        # be the primary path -- but for every other kind it is populated.
        name = (row.get("field") or "").strip()
        if not name:
            return (None, UNDERIVABLE)
        tier = TIER_KEY if object_kind in {"dict_key", "df_col", "container_field"} else TIER_ATTR
        classified = (tier, name)

    tier, name = classified
    try:
        lineno = int(row.get("lineno") or 0)
    except (TypeError, ValueError):
        lineno = 0

    return (
        StaticClaim(
            callable=row.get("callable", ""),
            tier=tier,
            name=name,
            accesses=_accesses_for(row.get("access", "")),
            object_kind=object_kind,
            object_id=row.get("object_id", ""),
            operation=operation,
            lineno=lineno,
            evidence=evidence,
        ),
        "",
    )


def load_static_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# -- the comparison --------------------------------------------------------


@dataclass
class AccessComparisonReport:
    static_row_count: int = 0
    scored_row_count: int = 0
    excluded: Counter = field(default_factory=Counter)

    # Static -> runtime. Keyed by (callable, tier, name); a verdict per claim.
    verdicts: Counter = field(default_factory=Counter)
    verdict_by_kind: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    falsified_examples: List[StaticClaim] = field(default_factory=list)

    # Runtime -> static. The recall direction.
    observed_total: int = 0
    observed_comparable: int = 0
    # Observations kept out of the recall denominator, by (tier, role). Printed
    # for the same reason the static exclusions are: an exclusion nobody can see
    # is a denominator chosen quietly.
    observed_excluded: Counter = field(default_factory=Counter)
    observed_matched: Set[Tuple[str, str, str, str]] = field(default_factory=set)
    observed_missing: Set[Tuple[str, str, str, str]] = field(default_factory=set)
    missing_by_tier: Counter = field(default_factory=Counter)
    missing_by_owner: Counter = field(default_factory=Counter)
    # Which *names* the misses are. On climlab this is what shows that the
    # attribute gap is largely reads through imported modules (``np.newaxis``)
    # rather than project data -- a composition worth seeing rather than
    # quietly excluding, since excluding it would be tuning the denominator.
    missing_by_name: Counter = field(default_factory=Counter)
    # Accesses observed in a callable the static artifacts never mention at all.
    # Which callables these are is a property of the analysed project, not a
    # given: on climlab they are module and class bodies, and *none* is a
    # lambda or a generator expression, because climlab contains no lambdas.
    # An earlier version of this comment asserted the opposite.
    missing_in_unmodelled_callable: Counter = field(default_factory=Counter)

    # Literal and computed keys, never folded together. See the module docstring.
    literal_key_recall: Tuple[int, int] = (0, 0)
    computed_key_recall: Tuple[int, int] = (0, 0)

    executed_callables: int = 0
    unknown_kind_rows: int = 0
    index_errors: List[str] = field(default_factory=list)
    driver_errors: List[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        total = len(self.observed_matched) + len(self.observed_missing)
        return len(self.observed_matched) / total if total else 0.0

    @property
    def confirmed_share(self) -> float:
        """Share of scored static claims the run confirmed. A weak number."""
        return self.verdicts[MATCHED] / self.scored_row_count if self.scored_row_count else 0.0

    @property
    def falsified_share(self) -> float:
        return self.verdicts[FALSIFIED] / self.scored_row_count if self.scored_row_count else 0.0


def _owner(callable_id: str) -> str:
    """Best-effort class (or module) owning a callable ID; report grouping only."""
    parts = callable_id.split(".")
    if len(parts) < 2:
        return callable_id
    for index in range(len(parts) - 2, -1, -1):
        if parts[index][:1].isupper():
            return ".".join(parts[: index + 1])
    return ".".join(parts[:-1])


def _is_comparable_observation(
    key: AccessKey, index: BytecodeIndex
) -> bool:
    callable_id, tier, name, _access, role = key
    if (tier, role) not in COMPARABLE_ROLES:
        return False
    if tier == TIER_NAME:
        if name in RECEIVER_NAMES:
            return False
        if role == ROLE_GLOBAL:
            return index.is_module_global(callable_id, name)
    return True


def compare(
    static_rows: Sequence[Mapping[str, str]],
    observed: Set[AccessKey],
    executed_callables: Set[str],
    index: BytecodeIndex,
    *,
    driver_errors: Sequence[str] = (),
    max_examples: int = 200,
) -> AccessComparisonReport:
    report = AccessComparisonReport(
        static_row_count=len(static_rows),
        observed_total=len(observed),
        executed_callables=len(executed_callables),
        index_errors=list(index.errors),
        driver_errors=list(driver_errors),
    )

    observed_triples: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
    for callable_id, tier, name, access, _role in observed:
        observed_triples[callable_id].add((tier, name, access))

    claims: List[StaticClaim] = []
    for row in static_rows:
        if row.get("object_kind") == "unknown":
            report.unknown_kind_rows += 1
        claim, reason = derive_static_claim(row)
        if claim is None:
            report.excluded[reason] += 1
            continue
        claims.append(claim)

    report.scored_row_count = len(claims)

    # -- static -> runtime -------------------------------------------------
    for claim in claims:
        seen = observed_triples.get(claim.callable, frozenset())
        if any((claim.tier, claim.name, access) in seen for access in claim.accesses):
            verdict = MATCHED
        elif claim.callable not in index.accesses:
            # The static artifacts name a callable no compiled code object
            # produced. Almost always an ID-convention mismatch rather than a
            # finding, which is why it is a bucket of its own rather than a
            # falsification.
            verdict = NO_BYTECODE
        elif index.can_access(claim.callable, claim.tier, claim.name, set(claim.accesses)):
            verdict = UNEXECUTED if claim.callable in executed_callables else NOT_EXERCISED
        else:
            verdict = FALSIFIED
            if len(report.falsified_examples) < max_examples:
                report.falsified_examples.append(claim)
        report.verdicts[verdict] += 1
        report.verdict_by_kind[claim.object_kind or "(none)"][verdict] += 1

    # -- runtime -> static, the recall direction ---------------------------
    claimed: Set[Tuple[str, str, str, str]] = set()
    for claim in claims:
        for access in claim.accesses:
            claimed.add((claim.callable, claim.tier, claim.name, access))

    literal_hit = literal_total = computed_hit = computed_total = 0
    for key in observed:
        callable_id, tier, name, access, role = key
        if not _is_comparable_observation(key, index):
            label = f"{tier}/{role}"
            if tier == TIER_NAME and name in RECEIVER_NAMES:
                label = "name/receiver"
            elif tier == TIER_NAME and role == ROLE_GLOBAL:
                # A global-namespace name that is not module-level data: a
                # builtin, an imported symbol, a function, or a class-body
                # binding. The extractor registers none of these as objects.
                label = "name/global (not module data)"
            report.observed_excluded[label] += 1
            continue
        report.observed_comparable += 1
        target = (callable_id, tier, name, access)
        found = target in claimed
        if found:
            report.observed_matched.add(target)
        else:
            report.observed_missing.add(target)
            report.missing_by_tier[f"{tier}/{role}"] += 1
            report.missing_by_owner[_owner(callable_id)] += 1
            report.missing_by_name[f"{tier}/{name}"] += 1
            if callable_id not in {claim.callable for claim in claims}:
                report.missing_in_unmodelled_callable[_owner(callable_id)] += 1

        if tier == TIER_KEY:
            if role == ROLE_LITERAL:
                literal_total += 1
                literal_hit += int(found)
            else:
                computed_total += 1
                computed_hit += int(found)

    report.literal_key_recall = (literal_hit, literal_total)
    report.computed_key_recall = (computed_hit, computed_total)
    return report


# -- report ----------------------------------------------------------------


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ratio(hit: int, total: int) -> str:
    return f"{hit}/{total} ({_percent(hit / total)})" if total else "0/0 (-)"


def render_markdown(report: AccessComparisonReport, *, top_n: int = 25) -> List[str]:
    lines: List[str] = [
        "# Static vs runtime data access",
        "",
        "Runtime accesses come from executing the project under `sys.monitoring`",
        "and recording which attribute, key and name instructions actually ran.",
        "The trace is a lower bound: it only sees what the drivers exercised.",
        "",
        "The unit compared is `(callable, tier, name, read/write)`. Source",
        "positions are deliberately not part of it — see `access_comparison.py`.",
        "",
        "## Totals",
        "",
        f"- Static access edges: {report.static_row_count} ({report.scored_row_count} scored)",
        f"- Runtime accesses observed: {report.observed_total} "
        f"({report.observed_comparable} comparable)",
        f"- Callables entered at runtime: {report.executed_callables}",
        f"- Static rows with `kind == unknown` (review §4.6 counter): {report.unknown_kind_rows}",
        "",
        "### Excluded before scoring",
        "",
        "Counting these would manufacture gaps. Each is excluded for a stated",
        "reason, and the count is printed so the denominator cannot be chosen",
        "quietly.",
        "",
        "| Excluded | Rows | Why |",
        "| --- | ---: | --- |",
        f"| exposure (`return`, `passed_arg`, …) | {report.excluded[EXCLUDED_EXPOSURE]} | "
        "not memory operations — a local escaping its scope |",
        f"| file objects | {report.excluded[EXCLUDED_FILE]} | identity established at a `CALL`, not an access |",
        f"| labeled access (`ds.sel(...)`) | {report.excluded[EXCLUDED_UNOBSERVABLE]} | "
        "compiles to a keyword call, not a subscript |",
        f"| underivable | {report.excluded[UNDERIVABLE]} | "
        "neither `evidence` nor `field` yielded a name — **a defect in this module, not the extractor** |",
        "",
        "And on the observed side — accesses the interpreter really performed",
        "that are kept out of the recall denominator, because the extractor",
        "structurally does not model them:",
        "",
        "| Observed but not scored | Count |",
        "| --- | ---: |",
    ]
    for label, count in report.observed_excluded.most_common():
        lines.append(f"| `{label}` | {count} |")
    lines += [
        "",
        "## Recall — the headline number",
        "",
        "Of the accesses the interpreter really performed, how many did the",
        "extractor find? Every miss here is a genuine gap.",
        "",
        f"**Recall: {_percent(report.recall)}** "
        f"({len(report.observed_matched)} found / "
        f"{len(report.observed_matched) + len(report.observed_missing)} observed)",
        "",
        "### Keys: literal and computed, scored apart",
        "",
        "The static side names a key by its literal; the runtime side sees a",
        "value. Folding these together would hide which one is expensive.",
        "",
        "| Key kind | Recall |",
        "| --- | --- |",
        f"| literal (`d['k']`) | {_ratio(*report.literal_key_recall)} |",
        f"| computed (`d[k]`) | {_ratio(*report.computed_key_recall)} |",
        "",
    ]

    if report.missing_by_tier:
        lines += ["### Missing accesses by tier", "", "| Tier / role | Missing |", "| --- | ---: |"]
        for label, count in report.missing_by_tier.most_common():
            lines.append(f"| `{label}` | {count} |")
        lines.append("")

    if report.missing_by_name:
        lines += [
            "### Missing accesses by name",
            "",
            "What the gap is actually made of. Reads through an imported module",
            "(`np.newaxis`, `np.ones_like`) are not project data and the",
            "extractor registers no object for them — but they are **not**",
            "excluded here, because deciding which names count would be tuning",
            "the denominator. Read this table before reading the recall figure.",
            "",
            "| Tier / name | Missing |",
            "| --- | ---: |",
        ]
        for label, count in report.missing_by_name.most_common(top_n):
            lines.append(f"| `{label}` | {count} |")
        lines.append("")

    if report.missing_by_owner:
        lines += ["### Missing accesses by owner", "", "| Owner | Missing |", "| --- | ---: |"]
        for owner, count in report.missing_by_owner.most_common(top_n):
            lines.append(f"| `{owner}` | {count} |")
        lines.append("")

    if report.missing_in_unmodelled_callable:
        lines += [
            "### Missing in callables the extractor models no accesses for",
            "",
            "A callable that ran and touched data, for which `access_edges.csv`",
            "has no row at all. Which callables these are is a property of",
            "the analysed project: lambdas and generator expressions can land",
            "here, but so can module and class bodies, and on climlab they are",
            "entirely the latter. Read the owners below rather than assuming.",
            "",
            "| Owner | Missing |",
            "| --- | ---: |",
        ]
        for owner, count in report.missing_in_unmodelled_callable.most_common(top_n):
            lines.append(f"| `{owner}` | {count} |")
        lines.append("")

    lines += [
        "## Verdicts on static claims",
        "",
        "`falsified` is the strong signal and is unique to this comparison: the",
        "full set of access instructions in a code object is fixed at compile",
        "time, so a claimed access that appears nowhere in the bytecode is wrong",
        "**however little the drivers covered**. `unexecuted` and `not exercised`",
        "are coverage artifacts and are not defects.",
        "",
        "| Verdict | Claims | Share |",
        "| --- | ---: | ---: |",
    ]
    for verdict in (MATCHED, UNEXECUTED, NOT_EXERCISED, FALSIFIED, NO_BYTECODE):
        count = report.verdicts[verdict]
        share = _percent(count / report.scored_row_count) if report.scored_row_count else "-"
        lines.append(f"| `{verdict}` | {count} | {share} |")
    lines.append("")

    if report.verdict_by_kind:
        lines += [
            "### By object kind",
            "",
            "This is the breakdown the review asks for: it says *which* object",
            "kinds the extractor gets right, and therefore which findings are",
            "expensive.",
            "",
            "| Object kind | Scored | matched | unexecuted | not exercised | falsified | no bytecode |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for kind in sorted(report.verdict_by_kind, key=lambda k: -sum(report.verdict_by_kind[k].values())):
            counts = report.verdict_by_kind[kind]
            total = sum(counts.values())
            lines.append(
                f"| `{kind}` | {total} | {counts[MATCHED]} | {counts[UNEXECUTED]} | "
                f"{counts[NOT_EXERCISED]} | {counts[FALSIFIED]} | {counts[NO_BYTECODE]} |"
            )
        lines.append("")

    if report.falsified_examples:
        lines += [
            "### Falsified claims (sample)",
            "",
            "| Callable | Claimed | Evidence | Object kind | Line |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for claim in report.falsified_examples[:top_n]:
            evidence = claim.evidence.replace("|", "\\|")[:60]
            lines.append(
                f"| `{claim.callable}` | `{claim.tier}/{claim.name}` | `{evidence}` | "
                f"{claim.object_kind} | {claim.lineno} |"
            )
        lines.append("")

    if report.index_errors:
        lines += ["## Files that could not be compiled", ""]
        lines += [f"- {message}" for message in report.index_errors[:top_n]]
        lines.append("")

    if report.driver_errors:
        lines += ["## Driver problems", ""]
        lines += [f"- {message}" for message in report.driver_errors]
        lines.append("")

    return lines


def write_comparison(outdir: Path, report: AccessComparisonReport, *, top_n: int = 25) -> None:
    write_markdown(
        outdir / "access_comparison.md",
        render_markdown(report, top_n=top_n),
        trailing_newline=True,
    )
    write_json(
        outdir / "access_comparison.json",
        {
            "static_row_count": report.static_row_count,
            "scored_row_count": report.scored_row_count,
            "excluded": dict(sorted(report.excluded.items())),
            "observed_total": report.observed_total,
            "observed_comparable": report.observed_comparable,
            "observed_excluded": dict(report.observed_excluded.most_common()),
            "executed_callables": report.executed_callables,
            "unknown_kind_rows": report.unknown_kind_rows,
            "recall": report.recall,
            "matched": len(report.observed_matched),
            "missing": len(report.observed_missing),
            "literal_key_recall": {
                "found": report.literal_key_recall[0],
                "observed": report.literal_key_recall[1],
            },
            "computed_key_recall": {
                "found": report.computed_key_recall[0],
                "observed": report.computed_key_recall[1],
            },
            "verdicts": dict(sorted(report.verdicts.items())),
            "verdict_by_kind": {
                kind: dict(sorted(counts.items()))
                for kind, counts in sorted(report.verdict_by_kind.items())
            },
            "missing_by_tier": dict(report.missing_by_tier.most_common()),
            "missing_by_name": dict(report.missing_by_name.most_common()),
            "missing_by_owner": dict(report.missing_by_owner.most_common()),
            "missing_in_unmodelled_callable": dict(report.missing_in_unmodelled_callable.most_common()),
            "missing_accesses": [
                {"callable": callable_id, "tier": tier, "name": name, "access": access}
                for callable_id, tier, name, access in sorted(report.observed_missing)
            ],
            "falsified_claims": [
                {
                    "callable": claim.callable,
                    "tier": claim.tier,
                    "name": claim.name,
                    "object_id": claim.object_id,
                    "object_kind": claim.object_kind,
                    "operation": claim.operation,
                    "evidence": claim.evidence,
                    "lineno": claim.lineno,
                }
                for claim in report.falsified_examples
            ],
            "index_errors": report.index_errors,
            "driver_errors": report.driver_errors,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    import json

    try:
        from microservice_pipeline.call_graph.discovery import (
            iter_analysis_files,
            iter_analysis_files_for_source_roots,
        )
        from microservice_pipeline.config import load_extraction_config
        from microservice_pipeline.data_access.dynamic_access_trace import load_observed_accesses
    except ImportError:  # pragma: no cover
        from call_graph.discovery import iter_analysis_files, iter_analysis_files_for_source_roots  # type: ignore
        from config import load_extraction_config  # type: ignore
        from data_access.dynamic_access_trace import load_observed_accesses  # type: ignore

    parser = argparse.ArgumentParser(
        description="Compare the static data-access artifacts against a runtime trace",
    )
    parser.add_argument("--config", default=None, type=Path, help="Extraction JSON/JSONC config")
    parser.add_argument("--root", default=None, type=Path, help="Python source root (when no config is given)")
    parser.add_argument("--module-prefix", default=None, help="Prefix for discovered module names")
    parser.add_argument(
        "--artifacts",
        default=None,
        type=Path,
        help="Directory holding callable_data_access.csv and dynamic_access.csv",
    )
    parser.add_argument("--outdir", default=None, type=Path, help="Where to write the report")
    parser.add_argument("--top", default=25, type=int, help="Rows per grouping table")
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
        artifacts = (args.artifacts or config.data_access.outdir).resolve()
    elif args.root and args.artifacts:
        analysis_files = list(iter_analysis_files(args.root.resolve(), module_prefix=args.module_prefix))
        artifacts = args.artifacts.resolve()
    else:
        raise SystemExit("--config, or both --root and --artifacts, is required")

    static_path = artifacts / "callable_data_access.csv"
    dynamic_path = artifacts / "dynamic_access.csv"
    if not static_path.exists():
        raise SystemExit(f"Missing {static_path}; run 'data-access' first")
    if not dynamic_path.exists():
        raise SystemExit(f"Missing {dynamic_path}; run 'trace-data-access' first")

    meta_path = artifacts / "dynamic_access.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    index = build_bytecode_index(analysis_files)
    report = compare(
        load_static_rows(static_path),
        load_observed_accesses(dynamic_path),
        set(meta.get("executed_callables", [])),
        index,
        driver_errors=meta.get("driver_errors", []),
    )

    outdir = (args.outdir or artifacts).resolve()
    write_comparison(outdir, report, top_n=args.top)

    print(f"Comparison written to {outdir}")
    print(f"Static rows: {report.static_row_count} ({report.scored_row_count} scored)")
    print(f"Runtime accesses: {report.observed_total} ({report.observed_comparable} comparable)")
    print(
        f"Recall: {_percent(report.recall)} "
        f"({len(report.observed_matched)} found / "
        f"{len(report.observed_matched) + len(report.observed_missing)} observed)"
    )
    print(f"Confirmed static claims: {_percent(report.confirmed_share)}")
    print(f"Falsified static claims: {report.verdicts[FALSIFIED]} ({_percent(report.falsified_share)})")
    if report.verdicts[NO_BYTECODE]:
        print(
            f"WARNING: {report.verdicts[NO_BYTECODE]} claims name a callable with no compiled "
            "code object -- check the ID conventions agree"
        )
    if report.excluded[UNDERIVABLE]:
        print(f"WARNING: {report.excluded[UNDERIVABLE]} rows yielded no name; the evidence parser is incomplete")


if __name__ == "__main__":
    main()
