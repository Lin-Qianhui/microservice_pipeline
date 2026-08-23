# Validating the call graph against PyCG

Cross-checks the call graph produced by `microservice-pipeline call-graph` against
[PyCG](https://github.com/vitsalis/PyCG), an independent static call-graph
extractor. The two tools use different techniques — ours is an AST pass with
targeted type inference, PyCG runs an inter-procedural points-to fixpoint — so
edges they agree on are unlikely to be shared mistakes.

**Neither graph is ground truth.** This produces three buckets, not an accuracy
score. Disagreements are triaged by hand; in the worked example below, every one
turned out to be PyCG's error rather than ours.

This is the weaker of the two available checks. `trace-runtime` + `compare-graphs`
(see [../README.md](../README.md), "measure the call graph against runtime ground
truth") observe what the interpreter actually dispatched, which *is* ground truth
for the paths the drivers cover. Reach for PyCG when the analyzed project cannot
be run, or as a second opinion on the edges the static pass did produce.

| Script | Runs under | Purpose |
| --- | --- | --- |
| `run_pycg.py` | `.pycg-venv` (3.11) | Runs PyCG over the extraction config's file scope |
| `compare_call_graphs.py` | any Python 3.9+, stdlib only | Normalizes both graphs and emits the buckets |

---

## One-time setup

PyCG is unmaintained (last release 2022) and cannot be installed into `.venv`.
It lives in its own environment:

```bash
/path/to/python3.11 -m venv .pycg-venv
.pycg-venv/bin/pip install "pycg==0.0.7" "setuptools<81"
```

`.pycg-venv` is gitignored. Every pin here is load-bearing:

- **Python 3.11 exactly.** PyCG 0.0.7 uses `ast.Str` / `ast.Num`, removed in 3.12.
  `run_pycg.py` refuses to start on anything else.
- **pycg 0.0.7, not 0.0.8.** The 0.0.8 wheel ships its package directory as
  `PyCG/` while every internal import says `pycg`. It fails at import on any
  case-sensitive path and is unusable as published.
- **setuptools<81.** PyCG imports `pkg_resources`, which setuptools 81 removed.
- **Not in `.venv`.** Pinning `setuptools<81` in the project environment to
  satisfy an unmaintained analysis tool is a dependency conflict with real
  blast radius. Keep it isolated.

---

## Running it

All three steps take the analyzed project's extraction config. Set it once:

```bash
CFG=/path/to/project/configs/extraction.jsonc
ART=/path/to/project/artifacts       # call_graph.outdir's parent, per that config
```

### 1. Generate our graph

```bash
microservice-pipeline call-graph --config $CFG
```

Writes `nodes.csv`, `edges.csv`, `call_graph.json`, and `call_graph_health.json`
to the config's `call_graph.outdir`.

### 2. Run PyCG over the same scope

```bash
.pycg-venv/bin/python scripts/run_pycg.py \
    --config $CFG \
    -o $ART/call_graph/pycg_call_graph.json
```

The script reads the *same* extraction config rather than restating the file
list, so the two graphs stay comparable when the config's globs change. It
prints the file count and resolved `--package` root; confirm the count matches
the number of distinct files in `nodes.csv`:

```bash
tail -n +2 $ART/call_graph/nodes.csv | cut -d, -f4 | sort -u | wc -l
```

A mismatch means the scopes have diverged and the comparison is invalid — see
[Scope alignment](#scope-alignment).

### 3. Compare

```bash
python3 scripts/compare_call_graphs.py \
    --ours $ART/call_graph/edges.csv \
    --pycg $ART/call_graph/pycg_call_graph.json \
    --outdir $ART/call_graph_comparison
```

The first-party namespaces are read off `nodes.csv` (found beside `--ours`, or
passed with `--nodes`) — no configuration, and it picks up entrypoint namespaces
that live outside the source roots. `--prefix` overrides if needed.

Writes `both.csv`, `ours_only.csv`, `pycg_only.csv`, `reexport_aliases.csv`, and
`summary.json`.

---

## Reading the output

| Bucket | Meaning | How to triage |
| --- | --- | --- |
| `both` | Both tools found the edge | Baseline confidence; no action |
| `pycg_only` | PyCG found it, we did not | **Read every row.** Either a recall gap in our resolver or a PyCG false positive |
| `ours_only` | We found it, PyCG did not | Split automatically — see below |

`ours_only` is split by whether PyCG models the mechanism at all:

- **`ours_only_beyond_pycg`** — edges whose relation is in `BEYOND_PYCG`:
  `dynamic_getattr`, `super_method`, `registered_invoke`, `property_getter`, and
  the seven `dunder_*` relations. PyCG records explicit call expressions only, so
  it never emits an edge for attribute access or operator dispatch. Expected, not
  defects.
- **`ours_only_contested`** — everything else. PyCG could plausibly have found
  these, so each one is either our false positive or its false negative. **Read
  every row.**

`inferred_type` is deliberately *not* in `BEYOND_PYCG`. Points-to analysis is
PyCG's whole design, so it competes with us there and those disagreements are
real. The same goes for `inferred_callable`, its callable-value counterpart.

**`virtual_override` is not in `BEYOND_PYCG` either**, though it reads like it
should be — fanning an edge out to every subclass override of `self.hook()` is
not something a points-to analysis does. PyCG reaches those edges from the other
direction, by resolving the receiver to the concrete class that flowed in; on
Parcels it found all 10 of ours. Since `virtual_override` is our deliberate
over-approximation (36/60 confirmed at fan-out ≥3 against runtime ground truth,
per `call_graph/ground_truth_and_roadmap.md`), excusing it here would hide the
relation most likely to be wrong. Its false positives belong in
`ours_only_contested`.

If the extractor gains a relation that neither `BEYOND_PYCG` nor
`COMPARABLE_RELATIONS` names, the run prints an **unclassified relations**
warning. Fix it by classifying the relation — left alone it lands in
`ours_only_contested` and reads as a resolver defect.

`jaccard` is `both / (ours ∪ pycg)`. Useful only as a regression signal across
runs; the absolute value is not meaningful, since a large share of `ours_only`
is by design.

### Normalizations

Six spelling differences are reconciled before the sets are compared. Without
them the overlap reads as near-zero for cosmetic reasons. All are in
`compare_call_graphs.py`.

| # | Difference | Handling |
| --- | --- | --- |
| 1 | `import` relation | Dropped. The import statement itself; always targets a `mod.<module>` node. PyCG models imports as name bindings, not edges |
| 2 | `mod.<module>` vs `mod` | Suffix stripped |
| 3 | `outer.<locals>.inner` vs `outer.inner` | `<locals>` segments stripped |
| 4 | `...docs.run_example` | Leading dots stripped — PyCG's spelling for entry points outside `--package` |
| 5 | `<builtin>.*`, `numpy.*`, … | Filtered to the first-party namespaces derived from `nodes.csv` |
| 6 | `pkg.f` vs `pkg.core.f` | Re-exported names rewritten to the definition site — see below |

#### Re-export aliases

A package `__init__.py` doing `from .core import get_n_faces` gives one function
two names. We always emit the definition site, `pkg.core.get_n_faces`; PyCG
emits whichever spelling the call site used, usually `pkg.get_n_faces`. One
agreed-on edge then lands in `ours_only_contested` **and** `pycg_only` — it
reads as a resolver false positive and a recall gap simultaneously, and the two
rows sit in different files where nobody lines them up. This is the same failure
mode as dropping the `imported` relation, and it inflates both disagreement
buckets rather than just one.

Unlike the other five, this normalization reads the analyzed source: nothing in
either graph records that the two names are the same function. Module paths come
from `nodes.csv`, so the parse is bounded to files the pipeline already analyzed.
Chains are followed (`pkg` → `pkg.sub` → `pkg.sub.impl`) and `import *` is
resolved against the definitions the source module actually has.

Two rails keep it from manufacturing agreement. A binding is kept only if it
resolves to a real node in `nodes.csv`, and never if its own spelling is already
one — so a rewrite can only turn a name denoting *nothing* into the definition it
aliases, never one real node into a different real node. Names offered by two
star imports are dropped rather than guessed, as are import cycles.

Every rewrite that moved an edge is written to `reexport_aliases.csv` for audit;
`summary.json` carries `reexport_aliases_found` and `reexport_aliases_applied`.
`--no-alias-resolution` turns it off to reproduce a pre-existing run's buckets.

> On Parcels this found 88 aliases, applied 2, and moved 3 edges: `both` 457 →
> 460, `pycg_only` 5 → 2, `ours_only_contested` 34 → 31. The 3 were genuine
> agreements on `parcels._sgrid.{get_n_faces,_attach_sgrid_metadata}`, hidden
> because `_sgrid/__init__.py` re-exports them from `.core`.

**The `imported` relation is deliberately kept.** Despite the name it is *not* an
import binding — it marks a *call* to a name that was imported, carries a real
call-site lineno, and never targets a `.<module>` node. Only `import` is the
statement itself, and every such edge targets `mod.<module>`; that clean split is
what makes the rule safe rather than a judgment call.

Dropping `imported` hides real agreements and manufactures the same number of
phantom `pycg_only` rows — in the worked example below, 26 of each. This was the
single most expensive mistake in building the comparison.

### Scope alignment

Both tools must see the same files. `run_pycg.py` derives its scope from the
extraction config, so the two agree as long as the pipeline honors its own
globs. Watch `ours_edges_outside_scope` in `summary.json` — anything above zero
means our graph contains edges PyCG never had a chance to see.

If the scopes do diverge, `--drop-module DOTTED` removes edges touching a module
from both sides. Treat it as a stopgap that masks a real problem, not a setting:

```bash
python3 scripts/compare_call_graphs.py ... --drop-module pkg.module.name
```

> A bug in `iter_analysis_files_for_source_roots` previously defeated all
> include/exclude globs whenever `entrypoints` was non-empty: the second pass
> that adds entry points re-walked the source root without filters, re-admitting
> every excluded file. Fixed in `call_graph/discovery.py` and guarded by
> `tests/test_call_graph_ast.py::test_exclude_globs_survive_entrypoints`. If
> excluded files reappear in `nodes.csv`, check that both `iter_analysis_files`
> calls in that function still forward `include_globs` and `exclude_globs`.

---

## Known PyCG failure modes

All four were hit on a single ~30-file project. Recognize them before filing a
`pycg_only` row as our gap:

- **Stdlib mangled into the first-party namespace.** `from pathlib import Path`
  inside a module that is later star-imported yields callees like
  `pkg.preprocessing.objects_generation.Path.resolve` and
  `...objects_generation.os.path.join`. These carry a first-party prefix, so the
  namespace filter cannot catch them.
- **Receiver types lost across `copy.deepcopy`.** `comp.assign_box(self)`
  resolves to `objects_generation.copy.deepcopy.assign_box` — the receiver typed
  as `deepcopy`'s return rather than the class. Our graph resolved the same site
  correctly.
- **Method/function shadowing.** A method calling a module-level function of the
  same name becomes a self-loop.
- **Crash on non-ASCII identifiers.** PyCG installs a custom `sys.path_hooks`
  entry and clears the importer cache. CPython lazily imports `unicodedata` when
  compiling non-ASCII identifiers, and that import gets intercepted, yielding a
  module with no `normalize`. `run_pycg.py` pre-imports `unicodedata` before PyCG
  loads. **Do not remove that import**; it looks unused and is not.

- **Run-to-run nondeterminism.** Three consecutive runs over unchanged Parcels
  source produced 10025, 11098, and 8230 total edges. The fixpoint does not
  converge identically each time. All three produced *identical* first-party
  numbers (462 edges, `both` 457, `pycg_only` 5), because the variance is
  confined to the `<builtin>.*` and third-party edges normalization 5 discards.
  So the comparison is reproducible, but `pycg_edges_total` is not — do not read
  a swing in it as a change in either tool.

PyCG may also hang or exhaust the recursion limit on larger inputs. A crash is
not evidence about the analyzed code.

