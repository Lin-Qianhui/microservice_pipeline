# Step 4b — The rest of the small correctness fixes

This is the seventh step of the plan in [`code_review.md`](../code_review.md) §7, after
Step 0, Step 1a, Step 2, Step 1b, Step 4a and Step 3. It finishes Step 4: the nine
findings §7 lists, plus §5.5, plus §1.16, which had been written down but never given a
step.

Every one of these is a small, local mistake. None of them needed new machinery. What
they have in common is that each one quietly puts a fact in the wrong place, and the
wrong place is then read by the clustering this whole pipeline exists to produce.

---

## 1. What was wrong, one line each

| | what the extractor did | why that is wrong |
| --- | --- | --- |
| §1.14 | credited a function with reading the values in its own default arguments | Python works those out **before** the function exists, so somebody else read them |
| §1.6 | decided what a call builds by looking at how its name **ends** | `ax.set(...)` is not building a set, and `node.list()` is not building a list |
| §1.16 | only modelled a variable if it recognised the right-hand side **by name** | the type checker had already said `ds = Dataset()` was a data container, and nobody asked it |
| §1.8 | walked the inside of a `class` statement as if it were part of the file around it | class-level names then leaked out, two classes overwrote each other, and everything the class body did was filed under the file |
| §1.9 | let a `lambda`'s own parameters mean the surrounding function's variables | `lambda data: data['a']` was recorded as reading an outer `data` it never touches |
| §5.5 | never treated a `lambda` as a piece of code in its own right | the call graph does, so the two halves of the pipeline disagreed about who touched the data |
| §1.10 | ignored `.index` and `.columns` on **everything** | that rule is about tables; on any other object those are ordinary attribute names |
| §1.13 | recorded one method call as two separate accesses | one thing happened, so there should be one record |
| §1.12 | stopped the entire run if a single file would not parse | one bad file in a project should cost you that file, not the analysis |
| §1.7 | named a file after the **text** used to open it | every `open(path)` in the project became the same file |
| §1.11 | let one `getattr(module, name)(...)` claim a link to every function in that module | a claim about everything is a claim about nothing, and here it also **deleted** good claims |

## 2. Why the last two matter more than they look

§1.7 and §1.11 are the two that invent connections rather than lose them, and an invented
connection is the expensive kind. The output of this stage is fed to a clusterer whose
job is to find parts of the program that could be separated. Two functions joined by a
mistake cannot be separated by any amount of downstream cleverness — the evidence says
they belong together.

§1.7 turned out to fire on climlab, which the previous step had concluded it did not. §7
of the plan.

## 3. Where the plan was wrong, and how we found out

Three things in the plan did not survive contact with the code. All three were found by
checking rather than by reading, and all three changed what got done.

### 3.1 The review's own suggested fix for §1.6 would not have worked

§1.6 says the correct test already exists in `rules.py` and "the fix is to use it
everywhere." It is not, quite. That helper asks whether the call name **ends with a whole
dotted piece**:

```
"ax.set"  ends with ".set"   ->  match
```

which is exactly the thing §1.6 is complaining about. It is the right test for a library
function, because those are reached through a module name — `pd.read_csv`, and it
correctly turns down a project's own `loader.my_read_csv`. It is the wrong test for a
**builtin**, because a builtin has no module name in front of it: `set(...)` is the
builtin and `ax.set(...)` is somebody's method that happens to share the name.

So there are two tests now, not one. Builtins (`dict`, `list`, `set`, `open`) must match
exactly; library functions keep the dotted test.

### 3.2 §1.14 could not be finished without §1.8, and the record of it was wrong

Step 1a found §1.14 by catching a real example: `RRTMG_SW.__init__` claimed to read a
module-level number called `nbndsw`, and the running program showed no such read. Step
1a's write-up says the real reader is the **module**.

It is not. The running program says the reader is the **class body**:

```
climlab.radiation.rrtm.rrtmg_sw.RRTMG_SW   name  nbndsw  read   line 95
```

`nbndsw` is used in a default argument of a method, and a method's defaults are worked
out while the `class` statement is running — inside the class body, which Python treats
as a piece of code with the class's own name.

This matters because it means moving the read out of `__init__` was not enough. Done on
its own it moved the read from one wrong place to another, and the check that watches for
this stayed at 1:

| | claims the running program contradicts |
| --- | ---: |
| before | 1 |
| after §1.14 alone | 1 — the same claim, now blamed on the module |
| after §1.8 as well | **0** |

### 3.3 §1.8 moves towards a trap §5.4 describes, so its guard was brought forward

When the extractor sees `SlabOcean()` it builds a list of names the call might mean and
takes **the first one it recognises**. The class comes before the class's constructor in
that list:

```
SlabOcean()  ->  1. sample.SlabOcean            <- the class body
                 2. sample.SlabOcean.__init__   <- the constructor
```

Today only the constructor is ever recognised, so the first entry is skipped and the
right answer wins. But `sample.SlabOcean` is the name of the **class body**, and §1.8's
whole point is to make class bodies things the extractor knows about. The closer that
name gets to being recognisable, the closer `SlabOcean()` gets to resolving to the class
body — and since the search stops at the first match, whatever the constructor was known
to return is simply never consulted. Nothing errors; the information just goes missing.

The call graph already had the answer — a filter that keeps class bodies out of the list
of things a call may resolve to — and this package had never used it. It was scheduled
for Step 5, and was brought forward and landed **on its own, first**: the artifacts came
out **byte-identical**, which is the evidence that it was doing nothing yet.

**One correction to how this was first written up here.** The guard was described as a
*precondition* for §1.8. It is not, quite. §1.8 as landed does not put class bodies into
the maps the constructor lookup consults, and removing the guard afterwards still gives
byte-identical artifacts on climlab — it is a guard against a further change, exactly as
§5.4 says ("one `_enter_callable` change away"), not a fix for something §1.8 broke.

Checking that claim honestly is what found a **real gap in the guard as first written**.
It was applied where a candidate is added from `callable_map`, which only covers a call
spelled with its full dotted path. A bare `SlabOcean()` reaches the list by a different
route and was not covered — that is, the guard would have missed the common case. It now
filters the assembled list once, at the end, the way the call graph filters the universe
rather than each lookup. There is a test pinning it, which there had not been.

## 4. The instrument was wrong again, and this time it was the strong verdict

The access oracle has two kinds of bad news. "Not found" is weak: it only means the test
runs never went there. "**Contradicted**" is supposed to be strong — the full list of
things a piece of code can possibly touch is fixed when Python compiles it, so a claim
that appears nowhere in that list is wrong no matter what the tests covered.

After §1.16 the count went from 0 to 2. Both were hand-checked, as the rules require, and
both were the instrument's fault:

```python
longorbit['long_peri'] += 180.       # long.py:39
tendencies['Tatm'] *= 0.             # turbulent.py:192
```

Both are `x['key'] += ...`. The instrument found a key by looking at the single
instruction before the subscript, on the stated grounds that "the key is always the
immediately preceding instruction." For `+=` it is not:

```
LOAD_CONST  'k'
COPY                  <- and the key is now two instructions back
COPY
BINARY_OP   []
...
SWAP                  <- and for the store it is far further back still
SWAP
STORE_SUBSCR
```

So the key was recorded as "computed", the literal never entered the list of things the
code can touch, and a perfectly true static claim was reported as **provably wrong**.

Two fixes, both in the instrument: step over the shuffling instructions when looking back
for the key, and — because the store half of `+=` is out of reach even then — give it the
key that the load half of the *same source expression* already resolved. Both halves come
from one piece of source and carry its position, so this is not a guess.

Re-measuring the untouched, pre-Step-4b artifacts with the corrected instrument moves the
baseline slightly, and every number in §6 is against that corrected baseline rather than
the published one:

| | published | corrected |
| --- | ---: | ---: |
| accesses the program performed that are comparable | 2,317 | 2,318 |
| recall | 69.0% | 68.9% |
| contradicted | 1 | 1 |

This is the third time an instrument in this revision has been wrong in a new way, and
the third time hand-checking the named claims is what caught it.

## 5. What climlab cannot show

Five of these fixes change nothing on climlab, because climlab does not contain the thing
they fix. That is worth stating plainly rather than reporting five zeroes:

| fix | why climlab cannot judge it | checked how |
| --- | --- | --- |
| §1.6 | its only loose matches are genuine `pd.read_csv` calls, which still match | every `.set`/`.list`/`.dict`/`.open` call in the source listed |
| §1.10 | it never writes `.index` or `.columns` — not once | searched the whole package |
| §1.9, §5.5 | **it contains no lambdas at all** | searched the whole package |
| §1.8's class-state half | it has no class-level dictionaries or lists | every class body's assignments listed |
| §1.12 | every file parses | the run reports no skips |

These are judged by their fixture tests instead, in the same way §1.7's fix and Step 3's
bare-`Dataset` fix were.

### The lambda claim in the oracle's own report is wrong for this project

The access report has a table of accesses in code the extractor has no rows for at all,
under the heading that lambdas and generator expressions "land here by construction."
Counted directly against the artifacts:

| accesses in code with no static row | 1,903 |
| --- | ---: |
| in lambdas, generator expressions or comprehensions | **0** |
| in **class bodies** | 1,196 |
| in module bodies | 707 |

So the recall the plan hoped §5.5 would recover was never §5.5's to recover — it was
§1.8's. Most of it is still out of reach for a different reason: the class bodies climlab
actually runs are almost entirely `def` statements and constants, which this extractor
does not model as data. After §1.8 that table falls from 76 to 75 and is now entirely
module bodies.

## 6. The numbers

Measured on climlab (72 files), same drivers as every previous step. The baseline was
re-run first and reproduced the published artifacts **byte-identically** before anything
was changed.

### Slice by slice

| after | objects | access edges | recall | contradicted |
| --- | ---: | ---: | ---: | ---: |
| baseline (corrected instrument) | 1,934 | 4,239 | 68.9% | 1 |
| §1.14 defaults | 1,934 | 4,239 | 68.9% | 1 |
| §5.4 guard *(alone, byte-identical)* | 1,934 | 4,239 | 68.9% | 1 |
| §1.8 class bodies | 1,934 | 4,239 | 69.0% | **0** |
| §1.6 anchored names *(byte-identical)* | 1,934 | 4,239 | 69.0% | 0 |
| §1.16 probe families | 2,010 | 4,401 | **69.5%** | 0 |
| §1.13 one call, one access | 2,010 | **4,381** | 69.5% | 0 |
| §1.12 bad files *(byte-identical)* | 2,010 | 4,381 | 69.5% | 0 |
| §1.9 + §5.5 lambdas *(byte-identical)* | 2,010 | 4,381 | 69.5% | 0 |
| §1.7 + §1.11 false coupling | **2,011** | 4,381 | 69.5% | 0 |

### Overall

| | before | after |
| --- | ---: | ---: |
| objects | 1,934 | 2,011 |
| access edges | 4,239 | 4,381 |
| lineage edges | 1,171 | 1,249 |
| objects with an `alias_of` | 434 | 447 |
| placeholder `unknown` objects (§4.6) | 0 | 0 |
| **access recall** | 68.9% | **69.5%** |
| static claims confirmed | 75.2% | 75.2% |
| **claims the program contradicts** | **1** | **0** |
| identity: alias precision | 93.1% | **94.0%** |
| identity: alias recall | 64.2% | **64.9%** |
| identity: contradicted claims | 20 | 20 |
| tests | 345 | **380** |

Nothing was traded for anything. Both oracles moved the same way, which is unusual in
this revision — Step 4a bought precision with recall and said so.

### The three that moved, and what moved

**§1.16 is the whole of the recall gain.** 76 new objects and 162 new access edges: local
variables the type checker had always known were containers, which the extractor threw
away because it only recognised right-hand sides by name.

**§1.13 removed 20 duplicate rows.** All 20 were checked: each is a plain `load` that has
a `method:...:receiver` row on the same object at the same line, which is one method call
recorded twice. Recall does not move, and cannot: the oracle's unit already treats the
pair as one. What moves is the weight the structural graph puts on those callables.

**§1.7 split one shared file node into two.** climlab's `file:path` was joining
`solar.orbital.long._get_Laskar_data` and `solar.orbital.table._get_Berger_data` — two
functions in two modules, each opening a different remote data file, both of which
happened to call their variable `path`:

```python
# long.py:25    path = remote_path                        (Laskar orbital data)
# table.py:10   path = _datapath_http + 'orbital/orbit91' (Berger orbit91 data)
```

The identity oracle's own §1.7 line now reads `0 of 0 shared file nodes` where it read
`0 of 1` before: there is no longer any file node joining two callables.

## 7. Correction to Step 1b

Step 1b recorded that **§1.7 does not fire on climlab** — "only one `file:` node is
touched by more than one callable and its callables never disagreed on the path."

The first half is right and the second half is the wrong test. The two callables sharing
that node never disagreed *about the value of the variable named `path`* at any moment
the tracer looked, because they are in different modules and never run at the same time.
They are still two different files. The defect fires here; the instrument could not see
it because it was asking whether a value ever differed rather than whether two unrelated
functions had been given one node.

That is not a fault in the instrument so much as a limit of what a value comparison can
settle, and it is worth recording because §1.7's fix was scheduled to be judged
"elsewhere, or by `evaluate`". Part of it can be judged here after all.

## 8. What was deliberately not done

- **§1.15** — the finding that a plain number is modelled as something that could hold
  fields. Still nobody's step. It is the one change in this document that would *lower*
  recall on purpose, and the review is right that it needs its own argument and its own
  measurement. §1.16's gate deliberately excludes it, and there is a test pinning that.
- **Decorators, base classes and annotations.** A `class` statement evaluates its bases
  and its methods' decorators, and a `def` evaluates its decorators and annotations. None
  of these has ever been walked here, and §1.14 and §1.8 both invited widening the walk
  to include them. Left alone: it would add access edges that nothing in this step
  measures, and it belongs with `class_body_expressions`, which the call graph already
  has.
- **A derived cap for §1.11.** The fan-out limit is a **chosen** number, 8. It is named
  in `rules.py` with that said out loud, and it joins §4.5's list of thresholds that
  ought to be derived. climlab has no dynamic dispatch of this shape, so there was
  nothing here to derive it from.
- **Tuple returns position by position**, still, from Step 4a §5.

## 9. What changed in the code

| file | what |
| --- | --- |
| `data_access/generate_data_access_ast.py` | all eleven findings: defaults moved before the callable is entered; class bodies get a scope, a callable and class-state assignments; lambdas get a callable and a scope; the two name-matching tests; the probe-family fallback; the `index`/`columns` gate; the receiver-read suppression; callable-scoped file IDs; the `getattr` cap and its `derived_from` marking; the §5.4 class-body filter |
| `data_access/rules.py` | `_builtin_call_matches` beside `_call_name_matches`, the constructor name sets, `DATAFRAME_STRUCTURE_ATTRS`, `MAX_DYNAMIC_GETATTR_TARGETS` |
| `data_access/dynamic_access_trace.py` | the augmented-subscript key fix — §4 above |
| `call_graph/ast_utils.py` | `parse_python_file` reads bytes so a coding cookie works; `partition_parseable`, shared by both packages |
| `call_graph/generate_call_graph_ast.py` | drops unparseable files once, up front, and reports them |
| `tests/test_data_access_ast.py` | 24 new tests; 6 existing ones updated to the new file IDs and the new `getattr` relation |

### Why the bad-file fix touches the call graph

§1.12 says a bad file kills the data-access run, and it does — but not from inside this
package. The call-graph analysis runs first and parses everything, so it raises before
data access starts. Fixing only this package would have produced a fix that cannot be
reached from the real entry point. The filter is therefore shared, applied once at the
top of each stage, and the call-graph artifacts for climlab come out **byte-identical**.

Doing it once at the top, rather than catching at each parse site, is the point: every
later pass then sees the same set of files by construction. Catching per site would let a
file be present for one pass and missing from another, which is the shape of the bug
Step 2 found in the return fixpoint.

## 10. Everything else held still

| | |
| --- | --- |
| `check-data-access-determinism`, five shuffled seeds, `--check-inputs` | byte-identical, **DETERMINISTIC** |
| call-graph `nodes.csv` / `edges.csv` on climlab | **byte-identical** |
| static claims confirmed | 75.2%, unchanged |
| placeholder `unknown` objects | 0, unchanged |
| test suite | 380 passing |

The one pre-existing failure
(`test_cluster_structural_graph.py::test_parse_args_uses_structural_config_and_cli_overrides`)
fails identically on a clean tree and is untouched by this step.

## 11. Corrections to `code_review.md`

- **§1.6's proposed fix is wrong** for the builtin half. §3.1 above.
- **§1.14's fix is not complete without §1.8**, and Step 1a's record of which code reads
  `nbndsw` names the module where the running program names the class body. §3.2.
- **§5.4's guard is a precondition for §1.8**, not a Step 5 nicety. §3.3.
- **§7's Step 4 list is missing §1.16**, which Step 3 added to §1 and never scheduled.
- **§1.13 was missing from the previous two steps' hand-off lists** even though §7 names
  it; following either list would have skipped it.
- **§5.5 cannot be measured on climlab**, and the access report's claim that the
  unmodelled-callable bucket is lambdas "by construction" is wrong for this project. §5.

## 12. What is next

Step 4 is finished. Remaining in §7:

- **Step 5** — cross-stage consistency: §5.7 (the confidence weights knob is not
  connected to data-access edges), plus §5.6's diamond-hierarchy fixture. §5.4 is already
  done, brought forward by this step.
- **Step 7** — framework independence: §2.7, the `state=` keyword.
- **Step 8** — the structural split, §4.1–§4.3.

And two findings with no step: **§1.15**, and **§4.4**, still the one item in the review
that nobody has scheduled.
