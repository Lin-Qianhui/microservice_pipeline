# Modularising `collectors.py`

## Why

`collectors.py` was 2,493 lines, and 2,380 of those were a single class:
`CallCollector(ast.NodeVisitor)`, 131 methods. The rest of the package is already
well-factored — 17 modules, none over 730 lines, import graph strictly acyclic.
This one file was the outlier, and it is the file that changes most often, so
every change carried a 2,500-line blast radius.

An earlier split (`436b049`) peeled off `dunders.py`, `project_index.py`,
`type_env.py`, and `summary_collectors.py`, taking the file from 2,108 → 1,565
lines. `71971fe` (the registration mechanism) then added ~1,100 lines back.

## What this does *not* buy

Be clear-eyed about the trade. **Mixins reduce coupling by exactly zero.** Every
method still reads and writes one shared `self`. `_resolve_callees` ↔
`_infer_class_types_from_call` remain mutually recursive. No dependency is
narrowed by one byte.

What the split buys is **navigation**: 15 files of ≤400 lines instead of one of
2,493, a reviewable blast radius per change, and a place to hang per-concern
module docstrings. What it costs is a ~15% line-count increase from module
headers and repeated import blocks, and degraded IDE "go to definition" across
mixin boundaries (see *Static analysis* below).

That is a navigation change, not an architecture change. The architecture change
— the CallSite IR — is deferred, for the reason `code_review.md:76-89` gives:
resolving `build().submit()` needs the inferred type of `build()`, and inferring
that type needs to resolve `build`. Splitting resolution from inference into
separate *objects* produces an import cycle, not a separation of concerns.
Splitting them into mixins over one `self` sidesteps the cycle entirely, because
there is still only one object.

## Why mixins rather than free functions

The obvious alternative — `def resolve_callees(c, node)` in a plain module —
sounds better than it is. The first parameter is typed `CallCollector`, i.e. *the
entire object*: the dependency is made syntactically explicit but not narrowed.
It does not buy unit-testability either, because the barrier to unit-testing
resolution is the state (15 constructor params, 8 collections aliased by
reference, a `ProjectIndex`, a `TypeEnv` with a pushed scope), not the binding.
And it cannot be applied uniformly: `visit_*` must stay methods for
`ast.NodeVisitor` dispatch, and `summary_collectors` *overrides* `visit_Call`, so
the result is two idioms in one class.

## Layout

```
call_graph/collector/
```

| File | Responsibility |
|---|---|
| `__init__.py` | Re-export `CallCollector`; the package's only public name. |
| `constants.py` | Source-level name tables: partials, element-key suffix, transparent/container annotation names, `SELF_NAMES`. |
| `shapes.py` | `self`-free AST shape predicates and result de-duplication. |
| `health.py` | Bucket a call site by callee shape and charge it to a `CallGraphHealth`. |
| `state.py` | `CollectorState(ast.NodeVisitor)` — every attribute, the class-level config flags, return-link scoping, and the `TypeEnv`/`ProjectIndex` facades. |
| `expressions.py` | Infer and record class types of name/attribute expressions, incl. property getters. |
| `origins.py` | Which *value* a name holds; turn attribute stores into escape/registry evidence. |
| `edges.py` | The single `edges.append` chokepoint, dunder/membership emission, operator and call visitors. |
| `registration_edges.py` | Re-project framework registration rules into parent-hook → child-hook edges. |
| `scopes.py` | Enter module/class/function/lambda scopes, resolve imports, read annotations, seed parameters. |
| `statements.py` | Assignment target typing, branch-merging control flow, container-mutation tracking. |
| `callables.py` | Which *callables* an expression may hold — higher-order and dispatch-table calls. |
| `inference.py` | Infer class and container-element types from values, calls, and tuple slots. |
| `resolution.py` | Turn a callee expression into `(callee_id, relation, resolved)` candidates. |
| `collector.py` | Compose the mixins. The MRO is declared here and nowhere else. |

## Status (2026-08-06)

**Landed.** `collectors.py` (2,493 lines, one 2,380-line class) is now
`collector/`, 15 files, largest 379 lines. Every step was verified by 81→87
passing unit tests *and* a byte-identical four-artifact diff against a climlab
baseline; the graph is unchanged at 1,144 edges throughout.

| File | Lines | | File | Lines |
|---|--:|---|---|--:|
| `__init__.py` | 14 | | `origins.py` | 215 |
| `constants.py` | 43 | | `state.py` | 231 |
| `health.py` | 68 | | `inference.py` | 245 |
| `collector.py` | 95 | | `edges.py` | 283 |
| `shapes.py` | 97 | | `statements.py` | 315 |
| `callables.py` | 162 | | `scopes.py` | 372 |
| `expressions.py` | 167 | | `resolution.py` | 379 |
| `registration_edges.py` | 168 | | **total** | **2,854** |

**On the line count.** 2,854 vs 2,473 after dead-code removal: **+381 lines, and
no code was added.** The delta is 15 module docstrings, 14 repeated import
blocks, and the class headers. Anyone reading `git diff --stat` will see the
package grow; it did not.

### What was done

1. **Dead code removed** (~20 lines): `_resolve_super_method_targets`,
   `_resolve_name_from_module_exports`, `_infer_class_from_call` — all private,
   all with zero call sites in `src/` and `tests/`. A comment claiming the
   `TypeEnv` facade had "several hundred call sites" was corrected; the real
   number was 81.
2. **Pure extractions**, as free functions rather than mixins: 8 predicates into
   `shapes.py`, the health accounting into `health.py` (re-united with the
   `callee_shape` it feeds), and the name tables into `constants.py`. `SELF_NAMES`
   was introduced because the literal `{"self", "cls"}` appeared 11 times across
   five clusters that no longer sit on one screen.
3. **`CollectorState`** took the constructor, all attributes, the class-level
   flags, and the return-link scoping.
4. **Nine mixins peeled**, one file per commit-sized step, each a pure textual
   move.
5. **The delegating facade was deleted** — 26 pure pass-throughs onto `TypeEnv`
   and `ProjectIndex`, 76 in-package call sites plus 2 external ones
   (`passes.py`, `summary_collectors.py`) rewritten to `self.types.X` /
   `self.project_index.X`. `_unique` folded into the module-level `unique`.
6. **Six guard tests added** (`tests/test_call_graph_ast.py`) pinning the three
   invariants above, the 19 `visit_*` methods, the reachability of
   `_visit_call_children`, and `records_registry_facts` being a *class*
   attribute.

### One deliberate deviation from the plan

The plan said delete **all 28** shims with "no partial facade". Implementation
found the 28 are not one kind of thing: 26 are pure pass-throughs, and **2 are
adapters** — `_resolve_class_reference_name` and `_resolve_star_import_targets`
bind `self.module` / `self.module_index` into a whole-project lookup. Those were
kept. Inlining them would copy that state-threading to each of their 7 call
sites, which is the opposite of the readability the deletion was for.

This is not the partial facade the plan warned against. The rule is now crisp
and complete: **a method on the collector always does something.** Nothing
merely forwards. The two survivors are commented in `state.py` to say why.

### Left open

Nothing from the plan's scope is unfinished. The deferred list below is
unchanged and none of it was attempted — the split is behaviour-preserving by
construction, and every item there is a behavioural change.

Two smaller things noticed in passing and deliberately not fixed here:

- `passes.py` imports `typing.Set` and `typing.Tuple` unused. Pre-existing,
  confirmed against `HEAD`; unrelated to this work.
- The `try/except ImportError` dual-import idiom now lives in exactly one file
  (`scopes.py`) instead of being package-wide dead weight in the collector. It
  is commented as dead insurance there. The 17-file audit is still open.

## Two hard rules

1. **Every mixin's base list is exactly `(CollectorState,)`.** A mixin must never
   inherit another mixin. This is what keeps the MRO a single diamond that C3 can
   linearise trivially.
2. **`CollectorState` is always last** in `CallCollector`'s bases. Putting it
   first makes C3 fail outright.

A third invariant makes the mixin *order* semantically inert: **no method name is
defined in two mixins.** Because nothing is overridden, C3 only has to linearise
— it never has to arbitrate. The order in `collector.py` is therefore chosen as a
reading gradient (most-derived semantics first, most-primitive last) so the file
doubles as a table of contents.

All three invariants are pinned by a guard test in `tests/test_call_graph_ast.py`
(`test_collector_mixins_*`), which also asserts that the set of `visit_*` methods
on `CallCollector` is exactly the 19 that existed before the split. That last
assertion is the regression test for "a visitor got lost or shadowed during a
move", which would otherwise surface only as a quiet edge-count drop.

## Three cluster-boundary corrections

Grouping by line range misfiles three methods; they are placed by *what they do*:

- **`_replay_var_sources`** sat among the callable-id inference methods but
  returns class/element types, and both its callers are in class-type inference.
  → `inference.py`.
- **`_lambda_id`** sat among the same, but reads `enclosing_function` /
  `current_class` / `module` to compute a scope-qualified name — a sibling of
  `_resolve_enclosing_local_callable`. → `scopes.py`.
- **`current_class_id`** sat among the origin trackers but is a state accessor
  with 12 call sites across five clusters. → `state.py`.

## Verification protocol

Every step was verified two ways before the next was started.

```bash
# unit gate
python -m pytest tests/test_call_graph_ast.py -q          # 81 passed (82 after the guard test)

# integration gate: byte-identical artifacts on a real project
find src -name __pycache__ -prune -exec rm -rf {} +
python -m microservice_pipeline.cli.main call-graph \
  --config ../climlab/configs/microservice_pipeline/extraction.jsonc \
  --outdir "$SCRATCH/cg_stepN"
diff -r "$SCRATCH/cg_base1" "$SCRATCH/cg_stepN"
```

Two details that matter:

- **Compare all four artifacts**, not just `edges.csv`. `call_graph_health.json`
  is the only one that catches a `note_call_health` regression, since health
  counters do not affect the edge list at all — which is exactly the risk the
  `health.py` extraction introduces.
- **Prove determinism first.** The whole strategy assumes the output is a
  deterministic function of the input. It is (ordered traversal, `sorted()` at
  every fan-out, deterministic file order from `iter_analysis_files`), but
  running the baseline twice and diffing costs one run and converts every later
  `diff -r` from "probably fine" into evidence. Baseline: 1,144 edges.

## Known traps

**`records_registry_facts` must stay a *class* attribute** on `CollectorState`.
It is overridden `= True` by `TypeSummaryCollector`. "Tidying" it into
`self.records_registry_facts = False` inside `__init__` would set an *instance*
attribute that shadows the subclass's class-level `True`, silently disabling all
registry-fact collection — zero unit-test failures, and a quiet loss of
`registered_invoke` edges. This is the single best reason to run the artifact
diff on every step, not just at the end.

**`_add_registration_edges` transiently rebinds `self.current_callable`** and
restores it in a `finally`. Dropping the `finally` corrupts the `caller` field of
every edge emitted after the first registration in that file.

**`_visit_call_children` has no in-file caller.** Its only caller is
`summary_collectors.ReturnSummaryCollector.visit_Call`. It looks like dead code
and is not; deleting it breaks at *runtime*, not import time. `collector.py`'s
docstring lists the full private surface `summary_collectors` reaches into, so
the contract is written down somewhere.

**Static analysis degrades.** Mixin bodies call `self._resolve_callees`,
`self._add_edge` etc. that live in sibling mixins; Pyright flags those as
unresolved and "go to definition" stops working across them — in exactly the
resolution code that most needs navigation. Nothing breaks (there is no
type-checker gate). The mitigation is a **"Requires from siblings:"** list at the
end of each mixin's module docstring: zero runtime cost, and it makes the
coupling that mixins hide at least legible. Stub methods raising
`NotImplementedError` on `CollectorState` were considered and rejected — a real
40-line lie for a fake IDE win.

## Deferred

1. **The CallSite IR** (`code_review.md` item (c)) — pass 1 emits
   `(caller, receiver_expr, name, syntactic_kind, loc)`, pass 2 resolves it. The
   only real unblocker for separating resolution from inference. This split is
   not a step toward it, but not an obstacle either: `resolution.py` and
   `inference.py` are the two files it would sit between.
2. **The sink refactor** (`code_review.md` item (b)) — replacing
   inheritance-as-configuration with one visitor feeding `EdgeSink` / `ReturnSink`
   / `ParamSink`. Should ride along with the IR. This split deliberately
   *preserves* the private subclass contract rather than narrowing it, because
   narrowing it is a behavioural change.
3. **`AnalysisContext`** (`code_review.md` item (a)) — the duplicated
   `build_X` / `build_X_from_analysis_files` pairs and the 15-parameter
   constructor. Lives in `generate_call_graph_ast.py`, not here.
4. **Package-wide `try/except ImportError` audit.** The dual-import idiom appears
   in 17 files to let a module run as a loose script. A file inside a sub-package
   cannot meaningfully do that, so the copy in `scopes.py` is now dead insurance.
   Kept verbatim for behaviour preservation.
5. **The one-table dunder collapse** (`op → (forward, reverse, inplace)`), still
   open from the previous split.
