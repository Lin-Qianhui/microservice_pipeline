# Step 0 — Adopting the fixes `call_graph/` already made

This is the first step of the plan in [`code_review.md`](code_review.md) §7. It closes
findings §5.1, §5.2 and §5.3.

---

## 1. What this step was about

`data_access/` and `call_graph/` both have to answer the same question in places: *given
the way a call or a class is written in the source, what does it actually refer to?*

`call_graph/` answers it with a shared object called `ProjectIndex`, built once per run.
`data_access/` answers it with its own private copies of the same logic. Those copies were
written earlier and never updated, so when `call_graph/` fixed bugs in its version, the
copies here kept the bugs.

This step deletes that gap. It does not invent anything: every fix below is code that was
already written, already tested, and already running on the other side of the package
boundary — and in two of the three cases, `data_access/` was **already being handed the
object that carries the fix** and simply was not using it.

The three problems all failed the same way: **they made the analysis output nothing rather
than output something wrong.** A missing object or a missing edge produces no warning, no
error and no diagnostic. It just quietly is not there. That is the worst kind of defect to
leave in place while building a measuring instrument (Step 1), because the instrument would
have measured a number that was wrong for reasons nobody could see.

---

## 2. The three problems

### 2.1 Calls through a package's re-exported names resolved to nothing (§5.1)

**What the code did.** When a call was written through a module alias, the code guessed the
target by pasting the alias's meaning onto the attribute name:

```python
if "." in call_name:
    prefix, _, suffix = call_name.partition(".")
    if prefix in self.module_imports:
        candidates.append(f"{self.module_imports[prefix]}.{suffix}")
```

**Why that was wrong.** The way a call is *spelled* is not always where the function *lives*.
A package `__init__.py` commonly re-exports things:

```python
# pkg/__init__.py
from .core import build_index
```

Now `import pkg as p` followed by `p.build_index(rows)` spells the target `pkg.build_index`.
But no function is defined at `pkg.build_index` — it is defined at `pkg.core.build_index`
and merely *visible* under the shorter name. Only the defining path is a key in the
callable map, so the lookup found nothing and the call was treated as external.

**What that cost.** Not just an edge. Two things downstream depend on resolving the call:

- `_return_ref_from_call` returns `None`, so the value coming out of the call has no type
  and no object — so the variable it is assigned to is never tracked as a container. **The
  object disappears, along with every access edge that would have pointed at it.**
- `_record_confirmed_param_lineage` finds no candidate, so no `arg_to_param` lineage is
  recorded, so `_apply_confirmed_param_aliases` has nothing to work with.

**Concretely**, on a three-file fixture:

```python
# pkg/core.py                          # consumer.py
def build_index(rows):                 import pkg as p
    index = {}
    for row in rows:                   def lookup(rows):
        index[row] = 1                     index = p.build_index(rows)
    return index                           return index['alpha']
```

| | before | after |
| --- | --- | --- |
| `local_exposed:consumer.lookup:index` | *not produced at all* | produced, `inferred_type=dict` |
| `dict_key:...build_index:index:alpha` | *not produced at all* | produced; read edge attributed |
| `arg_to_param` lineage into `build_index` | none | recorded |

**What changed.** A new helper `DataAccessCollector._canonical_callable_id` asks
`project_index.canonical_callable_id` for the defining path and keeps it only when it is a
real known callable. `_append_candidate` adds it to the candidate list. Applied in
`_candidate_callable_ids_for_call` (the three branches the call-graph fix covers) and in
`_dynamic_getattr_callable_ids`. This mirrors
`call_graph/collector/resolution.py::_resolve_module_alias_target`.

### 2.2 Class references through re-exports resolved to nothing, which silently switched registration lineage off (§5.2)

**What the code did.** `_resolve_class_reference_name` was a local reimplementation of
`ProjectIndex.resolve_class_reference_name`. Side by side:

| | `ProjectIndex` | the local copy |
| --- | --- | --- |
| where classes come from | the project's class index | a flat set derived from the callable map |
| re-export aliases | canonicalized | **not handled** |
| star imports (`from x import *`) | followed | **not handled** |
| classes with no methods | indexed | **invisible** |
| imports inside `try:` / functions | seen | **invisible** (only top-level ones) |
| result ordering | sorted | unordered set |

**Why that mattered so much.** Registration lineage — the analysis that says "this object
handed its state to that object" — is gated like this:

```python
parent_types = sorted(self._registration_parent_types(node))
if len(parent_types) != 1:
    return                      # give up
...
child_types = sorted(self._class_types_from_expr(child_expr))
if len(child_types) != 1:
    continue                    # give up
```

When class resolution returns an **empty** set, `len(...)` is `0`, the gate declines, and
**nothing is recorded**. No edge, no warning, no diagnostic — indistinguishable from "there
was genuinely nothing here".

climlab reaches its classes through package `__init__` re-exports throughout. So on the
codebase this project is built around, this analysis was mostly switched off, and nothing
said so.

**What changed.** `_resolve_class_reference_name` now returns the **union** of two
resolvers: the original local one (renamed `_local_class_reference_matches`, body
unchanged) and `ProjectIndex.resolve_class_reference_name`.

> **Why a union rather than a replacement**, which is what §5.2 literally proposed:
> `ProjectIndex` is built from the call graph's class index, which has no notion of an
> *attrdict* class — a mapping class that exposes its keys as attributes. The local
> resolver is the only thing that resolves those. Replacing it outright would have lost
> them, and because the gates are `len(...) != 1`, a loss produces no edge and no warning:
> exactly the failure mode being fixed. The union keeps every resolution that worked before
> and adds the ones that did not.

Both halves are guarded, so when there is no `ProjectIndex` (the `collect_data_access_from_source`
path, which most unit tests use) behaviour is byte-for-byte what it was.

### 2.3 Every file was parsed four times, using a cache that was being handed over and thrown away (§5.3)

**What the code did.** `run_from_extraction_config` runs the call-graph passes first and
gets back a `CallGraphAnalysis`. That object's **first field** is `cache: ParsedFileCache`,
and its docstring says in as many words that this package is the reason it exists. By the
time it is returned, the cache holds every file, parsed once, with parent links already
attached.

`run_from_extraction_config` took three things off that result — the callable map, the
registration rules, the project index — and **never touched `.cache`**. It then re-parsed
every file from disk, four separate times.

**Why that is worse than an ordinary missed optimization.** This was not "a useful utility
exists and is unused". It was "a fully populated value is already in hand and is being
dropped on the floor."

**Measured**, on a 3-file package (the same measurement §3.3 reports):

```
before:                            {'__init__.py': 4, 'a.py': 4, 'b.py': 4}
after,  no cache passed in:        {'__init__.py': 1, 'a.py': 1, 'b.py': 1}
after,  analysis.cache passed in:  {}          <- zero; every tree was already there
```

Four is the *floor*: it is one parse for attrdict detection, one per fix-point pass, and
one for the final pass. `MAX_RETURN_SUMMARY_PASSES` is 8, so the worst case was ten.

**What changed.** An optional `cache: Optional[ParsedFileCache] = None` parameter on
`collect_data_access_from_analysis_files`, `collect_data_access`,
`collect_pyright_families_from_analysis_files` and
`collect_attrdict_classes_from_analysis_files`; all four `parse_python_file` calls became
`cache.get(...)`. When no cache is passed in, one is created locally — so **even callers
that thread nothing drop from four parses to one**. `run_from_extraction_config` and the
`main() --root` path now pass `analysis.cache`, which takes it to zero.

This was safe to do because nothing in `data_access/` modifies an AST node — the collector
is a read-only `ast.NodeVisitor`, and the Pyright probe works on source text rather than on
the tree — so one shared tree can serve every pass.

---

## 3. The numbers

Measured on climlab (72 files), before = commit `333704a`, after = this step. Same config,
same command, artifacts written to a scratch directory so climlab's own were untouched:

```
microservice-pipeline data-access \
  --config <climlab>/configs/microservice_pipeline/extraction.jsonc \
  --outdir <scratch>/da-{before,after}
```

### The acceptance number

§5.2 predicted registration lineage would rise, and named it the test of whether the low
climlab count was really explained by the `state=` keyword gate of §2.7:

| | before | after |
| --- | --- | --- |
| **registration lineage edges** | **3** | **13** (+10) |

**The hypothesis survived, and decisively.** The ten new edges are climlab's actual physics
compositions — the model classes and the subprocesses they own:

```
EBM                     <- MeridionalHeatDiffusion, SimpleAbsorbedShortwave, AplusBT
EBM_seasonal            <- SimpleAbsorbedShortwave
GreyRadiationModel      <- GreyGas, GreyGasSW
BandRCModel             <- FourBandLW, ThreeBandSW, ManabeWaterVapor
RadiativeConvectiveModel<- ConvectiveAdjustment
```

None of these were being recorded before. So §2.7's `state=` gate is **not** the whole
explanation for the low count — class-reference resolution was the larger cause. §2.7
remains open and is still worth doing (Step 7), but it is no longer the prime suspect.

### Everything else

| metric | before | after | change |
| --- | --- | --- | --- |
| data objects | 1858 | 1870 | +12 |
| access edges | 4221 | 4239 | +18 |
| lineage edges (all) | 1154 | 1175 | +21 |
| — `state_assign` | 107 | 117 | +10 |
| — `arg_to_param` | 352 | 357 | +5 |
| — `local_assign` | 450 | 454 | +4 |
| — `return_value` | 82 | 84 | +2 |
| objects with `kind == unknown` (§4.6 counter) | 4 | 4 | **0** |
| objects with `inferred_type == unknown` | 757 | 757 | **0** |
| objects / edges / lineage edges **lost** | — | — | **0** |
| parses per file | 4 | 1 (0 with the shared cache) | −75% / −100% |
| data-access stage, best of 3 (Pyright off) | 1.087 s | 0.914 s | −16% |

**Nothing was lost.** Not one object, access edge or lineage edge that existed before is
absent afterwards. That is by design, not luck — see §4.1 below.

Newly visible objects include climlab's central shared container, the `state` dictionary
built by `climlab.domain.initial.column_state`, and its `Ts` / `Tatm` keys. Those had been
invisible to this analysis.

### One honest caveat

22 fields changed on objects that exist in both runs (out of 1858). No object changed its
`kind` or its `inferred_type`. The changes are:

- **18 `alias_of` values re-pointed.** New lineage edges give `_lineage_roots` more graph to
  walk, so some parameters now trace to a different root — generally a caller one level
  further out, which is more correct. One alias was erased, which is what
  `_apply_lineage_aliases` does when it finds more than one root.
- **4 `access_path` values changed** on classes that newly became registration children.

Neither is caused by this step: they are **§1.3** (`_lineage_roots` caches answers computed
inside a cycle, so results depend on which node was queried first) and **§1.4** (cross-file
object merges keep only `confidence` and discard every later refinement, so the *first file
processed* wins). Both are order-dependent by nature. This step did not create them; it
changed the inputs enough to move them.

This is precisely why the review orders **Step 2 — Determinism and ID consistency** before
any work that gets judged by comparing artifacts: until §1.3 and §1.4 are fixed, an artifact
diff cannot reliably tell a real change from a reshuffle.

---

## 4. What deliberately did *not* change

### 4.1 Candidates are appended, never substituted

`_return_ref_from_call` takes the **first** candidate that matches a map. So the canonical
form is added *after* the raw spelling, never in place of it. Any call that resolved before
still resolves the same way; the canonical entry only fires where nothing resolved at all.

That is what makes this change strictly additive, and it is why the "lost = 0" row above is
a design property rather than a happy accident. It also makes the before/after diff
unambiguous: every difference is something that used to be missing.

### 4.2 §5.2 is a union, not a replacement

Covered in §2.2. The short version: `ProjectIndex` does not know about attrdict classes, and
a silent loss here looks exactly like the bug being fixed.

### 4.3 `known_ids` is still rebuilt at every call site

§3.1 notes that `_candidate_callable_ids_for_call` rebuilds a four-way union of whole-corpus
sets for **every call node it visits**, and that hoisting it looks free while editing this
function.

**It is not free, and it was left alone on purpose.** `return_summaries` and
`return_tuple_summaries` are filled in *by the collector during its own traversal*. Caching
the union once per collector would freeze a snapshot that the traversal then keeps changing,
so results would shift mid-walk. Doing it correctly means hoisting to a frozen per-run index
— which is Step 6's job, alongside §3.2.

### 4.4 `attach_parents` still runs once per pass

`collect_data_access_from_tree` still calls `attach_parents(tree)` on entry. `ParsedFileCache`
has already done it, so this is a redundant recursive walk of the whole tree. It was kept
because it is harmless (the operation is idempotent) and because callers that hand in a
freshly parsed tree still need it. The remaining cost belongs to Step 6.

### 4.5 Scope

§5.4 (the `resolvable_callable_ids` filter), §5.5 (lambda callables), §5.6 (the diamond
hierarchy fixture) and §5.7 (the disconnected confidence weights) are **not** in this step.
They are Steps 4 and 5 in the plan, and none of them is a live defect today.

---

## 5. Tests

Five tests added to `tests/test_data_access_ast.py`. They use the `_collect_with_registration`
helper, which is the only path in the test suite that builds a real `ProjectIndex` — the
`collect_data_access_from_source` path that most tests use cannot accept one, which is §4.1's
"the tests exercise a different analysis from production", still open.

| test | covers |
| --- | --- |
| `test_module_alias_reexport_call_still_tracks_the_returned_container` | §5.1 — the object exists and keeps its family |
| `test_module_alias_reexport_call_records_arg_to_param_lineage` | §5.1 — the lineage edge |
| `test_registration_child_reached_through_reexport_records_state_lineage` | §5.2 — the climlab shape |
| `test_attrdict_class_resolution_survives_the_project_index_union` | §5.2 — guard: the union cost nothing |
| `test_data_access_parses_each_file_once` | §5.3 — one parse, or zero with a shared cache |

The first four were run against the pre-change source to confirm they fail there. Four fail,
one passes — the attrdict test passes on both, which is correct: it is a no-regression guard,
not a demonstration of new behaviour.

Full suite: 287 passed. One pre-existing unrelated failure in
`tests/test_cluster_structural_graph.py` (an argparse-state leak between tests) fails
identically before and after.

---

## 6. What this unblocks

- **Step 1 (the oracle, §6)** now gets to measure a resolver that is not silently dropping
  edges. Had it been built first, its baseline would have been depressed by causes invisible
  to it, and the recall numbers would have been blamed on the wrong things.
- **Step 6 (performance)** is largely pre-paid: parses per file are already at one, or zero
  when the cache is threaded. What is left is §3.1 and §3.2, the per-call-site set rebuilds.
- **Step 8 (structure)** now has two working precedents for its actual end goal, which is
  not merely splitting this package up but having it **call** `ProjectIndex` instead of
  paraphrasing it. Two of the three paraphrases are now gone.

It also changed the plan's shape in three places, all recorded in `code_review.md`:

1. **Step 1 split into 1a and 1b** with Step 2 between them (§6.1). The access oracle is
   ready today; the object-identity oracle scores `alias_of` and the lineage graph, which
   the 22 moved fields in §3 above show is order-dependent until Step 2 lands. Execution
   order is now `0 → 1a → 2 → 1b → 3 → 4 → 5 → 7 → 8`.
2. **Step 6 left the sequence.** Its stated acceptance criterion is already met here. The
   remainder (§3.1) profiles at 26% of the stage but saves ~0.2s on climlab, so it gates
   nothing; it is best folded into Step 2, which needs the same frozen index for §1.2.
3. **Step 8 is blocked on Step 2**, not merely on the oracle — its verification method is
   byte-identical artifact diff, and §3's caveat above is a demonstration that this does
   not currently work.

The thing this step demonstrates, and the reason the review put it first: three of the four
regressions in §5 existed because this package reimplemented resolution that `call_graph`
already owned. Sharing the real thing was cheap, lost nothing, and moved the number the
review cared about by a factor of four.

---

*Written 2026-08-23, against commit `333704a`. Line numbers in `code_review.md` will drift;
the findings it names will not.*
