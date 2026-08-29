# Step 2 — Making the answer depend on the code, not on the reading order

This is the third step of the plan in [`code_review.md`](../code_review.md) §7. It closes
findings §1.2, §1.3, §1.4 and §1.5, and folds in §3.1, §3.2 and §4.6.

---

## 1. What this step was about

The extractor reads one file at a time. While it reads, it fills in eight shared notebooks —
what each function returns, which objects exist, what type each one is — and each file it reads
can add to them or overwrite them.

That is fine as long as the *last* answer is the *best* answer. It was not. In several places
the rule was "whatever we learned first, keep", and "first" means "whichever file happened to be
read first". So the same project, analysed twice, could produce two different answers.

Step 0 measured this without meaning to. It made a change that added 12 objects and removed
nothing at all — and **22 fields on unrelated objects still moved**. Nobody had touched those
objects. They moved because the new information shifted the reading order's effect.

Two things were blocked by that, and both are now unblocked:

- **Comparing artifacts could not be trusted.** If you change the code and the output changes,
  you cannot tell whether your change did it or whether files were simply read in a different
  order. Step 8 (splitting the package up) has no other way to check itself, so it could not
  start.
- **The next measuring instrument had nothing stable to measure.** Step 1b scores whether two
  object IDs are really the same runtime object. That is exactly what was moving.

**The goal, stated as a test:** run the whole thing twice with the file list shuffled, and get
byte-for-byte identical output.

---

## 2. The check comes first

Nothing in the repo could answer that question, so the first thing built was the check itself,
**before any fix** — the same order Step 1a used, and for the same reason. A fix that is not
measured is a guess.

`microservice-pipeline check-data-access-determinism --config <config>`

It runs the whole stage twice — once in normal order, once with the file list shuffled — and
compares every file it writes. `--repeat 5` tries five different shufflings.

Two design choices, both borrowed from Step 1a:

**It says what differs, not just that something does.** When two runs disagree it prints the
actual rows: which ones appear in one run and not the other, and which line first differs. A
check that only says "different" cannot be taken to the source and argued with. Every finding in
§4 below came from reading those printed rows.

**It compares whatever was written, not a list of filenames.** The files to compare are
discovered from the two output folders. A hardcoded list that quietly skips a new artifact is the
exact failure this whole exercise exists to rule out.

### First run, before any fix

```
verdict: 1 of 1 shuffles produced different artifacts

data_objects.csv    1870 rows -> 1868 rows
access_edges.csv    4239 rows -> 4233 rows
```

Two objects and six access edges **existed or did not exist depending on the order the files were
read in**, and several hundred fields disagreed.

---

## 3. Ruling out the other suspect first

Before blaming this package, one thing had to be checked: data access is handed three things by
the call-graph stage (the callable list, the registration rules, the project index). Shuffling
the file list shuffles that stage too. If *its* answers move, the difference would look identical
and the search would be in the wrong package — and `call_graph/code_review.md` records a
non-determinism finding of exactly this kind.

Checked directly, on climlab, before touching anything:

| what data access is handed | identical under shuffle? |
| --- | --- |
| callable map | yes |
| registration rules | yes |
| project index (classes, callable IDs, aliases, module map, static methods) | yes |

Also identical, though data access does not use them: the return summaries and all four type
summaries. So the call-graph stage is order-stable, and **every difference found afterwards was
this package's own.** The check keeps this test as `--check-inputs` so the question never has to
be re-asked from scratch.

*(One thing noticed in passing and left alone: the call graph's own type-summary loop hits its
iteration limit on climlab and says so. Its results are still order-stable, so it does not affect
anything here, but it belongs on that package's list.)*

---

## 4. The problems

### 4.1 One class could get two different identities (§1.2)

**What the code did.** Some classes are big enough that the extractor models their state as one
object *per attribute* instead of one object for the whole class. That decision was made by
looking at one file at a time, so it only ever knew about classes written in *that* file.

**Why that was wrong.** The decision is then consulted for classes defined somewhere else. When a
parent class registers a child — `EBM.add_subprocess('LW', AplusBT(...))` — the code asks "is
`AplusBT` one of the split ones?" while reading `ebm.py`. But `AplusBT` is written in
`aplusbt.py`, so the answer while reading `ebm.py` was always "no", and while reading
`aplusbt.py` it was "yes".

**What that cost.** Two names for one thing. On climlab, before this fix:

| class | how many objects carried its state |
| --- | --- |
| `AplusBT` | **2** — `class_attr_state:…AplusBT:state` and `class_state:…AplusBT` |
| `StepFunctionAlbedo` | **2** |
| `GreyGas`, `ConvectiveAdjustment`, `MeridionalHeatDiffusion` | 1, but the wrong one |

Splitting one class's state across two nodes is worse than describing it imprecisely: the
clustering step downstream has no reason to put them back together, so one component looks like
two.

**What changed.** The decision is now made **once, for the whole project**, before any file is
read, and handed to every pass. After the fix each of those classes has exactly one node.

**One extra hole closed while in there.** The scan only looked at classes written at the top level
of a file. But a class written inside a function, or inside another class, is treated as an
ordinary class everywhere else in the extractor — so it could be asked about and would always get
"no". The scan now looks everywhere in the file.

### 4.2 An answer worked out inside a loop was saved as if it were final (§1.3)

**What the code did.** To find where a piece of data originally came from, the code walks
backwards through "this came from that" links. Data can flow in a circle — two functions passing
a value to each other — so there is a guard: if the walk arrives somewhere it is already standing,
it stops and reports "nothing found here", which is correct *for that particular walk*.

The problem is what happened next. That "nothing found" answer travelled back up to the caller,
the caller finished its own answer using it, and **that answer was saved in a cache and reused
forever**.

**Why that was wrong.** "Nothing found here" was only true because of where the walk started. Save
it, and every later question gets an answer shaped by a walk that has nothing to do with it. Which
answer you get depends on which question was asked first — and the questions were asked in the
order the objects had been created, which is the order the files were read.

**What that cost.** Reproduced exactly:

```
links: R -> A, A -> B, B -> A (a circle), B -> C

before:  where did C come from, asked first        = ['B', 'R']
         where did C come from, after asking A     = ['A']        <- different answer
after:   both                                      = ['B', 'R']
```

This decides whether two objects are recorded as the same thing. A one-answer result becomes an
alias; a two-answer result **erases** an alias that was already there. Step 0's 18 shifted
`alias_of` values were this.

**What changed.** The walk now reports whether it had to use the circle guard. If it did, the
answer is used but **not saved**, so the next question works it out fresh. On data with no circles
in it — almost everything — nothing is saved differently and nothing is slower.

**The same fix was applied twice.** This exact function is copy-pasted into
`infer_shared_containers.py` with the same bug (§4.3 of the review names this: fixing one copy and
not the other is how the two drift apart). Both are fixed. They are still two copies — merging
them is Step 8's job, and doing it here would have mixed a rearrangement into a diff that had to
stay readable.

### 4.3 Everything learned about an object after the first file was thrown away (§1.4)

**What the code did.** When the extractor records an access to something it has not seen yet, it
creates a placeholder marked `unknown` and moves on, expecting a later pass to fill it in. Inside
one file that worked: there was careful code to upgrade the placeholder, fill in a missing line
number, take a type it did not have.

Across files, none of that ran. The whole merge was one line:

```python
objects[object_id].confidence = _confidence_max(...)
```

Only the confidence. Everything else — the real kind arriving to replace `unknown`, a known type
arriving to replace "no idea", a path arriving where there was none — was discarded.

**Why that was wrong.** Whatever the first file said, stood. And "first" is just reading order.

**What that cost.** On climlab, four objects were stuck at `kind = unknown` forever. Not because
nothing knew what they were — something did, in another file, and was ignored.

**What changed.** The in-file merge was pulled out into one function that takes two objects, and
the cross-file loop now calls the same function. Beyond reuse, every rule in it was rewritten to
be **order-blind**: merging A into B and merging B into A now give the same result.

| field | before | after |
| --- | --- | --- |
| kind and its description | first one wins | a real kind beats a placeholder; two real ones settle on the value itself |
| confidence | highest | unchanged — already order-blind |
| inferred type | first known one wins | the known one; two known ones settle on the value |
| file and line | first non-zero wins | the earliest of the two |
| alias, access path | first non-empty wins | the non-empty one; two settle on the value |

Objects stuck at `kind = unknown`: **4 → 0.**

Where two files genuinely disagree, the tie is settled by comparing the values themselves. Which
one wins is arbitrary — but it is **the same one every run**, which is the whole point. Where this
shows up is `access_path` on class-state objects, and Step 1a already recorded that this field is
meaningless there anyway (a `class_state` row whose evidence is `self.state` carried
`access_path = 'self.time_type'`). It is now meaningless *and stable*.

### 4.4 The repeat-until-settled loop could stop too early (§1.5)

**What the code did.** Some facts need several passes to travel — A returns what B returns, B
returns what C returns. So the extractor repeats the whole pass until nothing changes, up to 8
times. To decide "nothing changed" it took a summary of the facts before and after and compared
them.

**Why that was wrong.** There were two summary functions and they wrote down **different things**.
One recorded 5 of the 6 fields, the other recorded 4. Neither recorded all of them. So a pass
that changed only one of the missing fields produced an identical summary, and the loop concluded
it had finished when it had not.

The missing field is not decorative: `access_path` is what object IDs for nested fields are built
from.

**Also**, neither loop recorded whether it *settled* or merely *ran out of turns*. The call-graph
side reached the conclusion, after both of its loops ran out on matplotlib, that running out is a
failure and should say so. Here there was not even a note.

**What changed.** Both summaries are now generated from the object's own field list, so every
field is compared and any field added later is included without anyone remembering. Both loops now
say so when they run out of turns, using the same wording as the call-graph side.

On climlab the loop settles well inside the limit, so nothing is being missed there.

---

## 5. What the check found that nobody had listed

Two of the six problems fixed in this step are not in `code_review.md`. Both were found by reading
the rows the check printed. This is the second time in a row that building the instrument found
more than the review had — the same thing happened in Step 1a.

### 5.1 The settling loop was running a *weaker* analysis than the real pass

This one caused the last two objects to appear and disappear, and it was the hardest to see.

The repeat-until-settled loop and the final pass that actually produces the output were being
given **different information**. The final pass gets the project index; the loop did not. With the
project index in hand the final pass can resolve calls the loop could not — so the final pass was
still *learning new facts while it ran*, and handing them only to the files it happened to reach
afterwards.

So the loop settling meant nothing. It had settled on a smaller question.

Concretely, `local_exposed:climlab.model.column.GreyRadiationModel.__init__:state` and
`local_exposed:climlab.model.ebm.EBM.__init__:state` existed only when `column.py` and `ebm.py`
were read *after* the file that taught the extractor what `column_state()` returns.

**What changed:** the loop is now given the same information as the final pass, so by the time the
final pass runs there is genuinely nothing left to learn.

### 5.2 Two output files could not be sorted into a stable order

The output is sorted before it is written, so reading order should not survive. But two of the
sort keys did not use every column, and Python's sort keeps tied rows in the order they arrived —
which is file order.

- `data_access.json`'s lineage edges sorted on 5 of their 8 columns. Two edges from different
  files agreeing on those 5 would swap.
- `callable_data_access.csv` sorted on a *different* key from `access_edges.csv`, which holds the
  same rows.

Both keys now use every column. This one is not an analysis bug at all; it would have produced a
non-empty diff with nothing behind it, and cost an afternoon to explain.

---

## 6. Two speed problems fixed in the same place (§3.1, §3.2)

Folded in here deliberately. Both are rearrangements that cannot change the output — and the
byte-identical check is precisely the proof of that, so this was the cheapest moment to do them.

**§3.1.** For every function call in the project, the code built four whole-project sets and glued
them together, just to ask "is this name a known function?". Step 0 left this alone because two of
those four collections are still being written while the walk is happening, so a saved copy would
go stale.

The review proposed freezing them into one index, which would bring the staleness back. It is not
needed: the set was only ever used to ask *is this in there*, so the fix is to ask the four
collections directly. Four lookups instead of four whole-project set builds, always current,
impossible to go stale.

**§3.2.** The project's class list was rebuilt from scratch inside every file's collector — once
per file per pass. Unlike the above it never changes during a run, so it is now built once and
handed in.

| | before | after |
| --- | ---: | ---: |
| data-access stage on climlab, best of 3, Pyright off | 0.904 s | **0.431 s** |

That is a 52% cut even though the settling loop now does *more* work per pass (§5.1). Step 6 of the
plan is now fully discharged.

---

## 7. The numbers

climlab, 72 files. Before = commit `d89133e` (post-Step-1a), after = this step. Same config, same
command, artifacts written to a scratch directory.

### The acceptance gate

```
microservice-pipeline check-data-access-determinism --config <climlab>/…/extraction.jsonc --repeat 5
```

| | before | after |
| --- | --- | --- |
| shuffled runs produce identical artifacts | **no** | **yes, on all 5 shufflings** |

### Everything else

| metric | before | after | change |
| --- | ---: | ---: | --- |
| data objects | 1870 | 1866 | −4, see below |
| access edges | 4239 | 4239 | **0** |
| lineage edges | 1175 | 1175 | **0** |
| — every relation kind individually | | | **0** |
| objects with `kind == unknown` (§4.6's counter) | 4 | **0** | −4 |
| objects with `inferred_type == unknown` | 757 | 759 | +2 |
| access edges naming a non-existent object | 0 | 0 | 0 |
| stage time, best of 3, Pyright off | 0.904 s | 0.431 s | −52% |

**The −4 objects are the §1.2 defect being removed, not information being lost.** Five
`class_state:…` nodes disappeared and one `class_attr_state:…:state` node appeared:

| class | before | after |
| --- | --- | --- |
| `AplusBT` | `class_attr_state:…:state` **and** `class_state:…` | `class_attr_state:…:state` |
| `StepFunctionAlbedo` | both | `class_attr_state:…:state` |
| `GreyGas` | `class_state:…` | `class_attr_state:…:state` |
| `ConvectiveAdjustment`, `MeridionalHeatDiffusion` | `class_state:…` | merged into their existing attribute node |

Two names for one class's state became one name. Every lineage edge still points at an object
that exists, and no access edge lost its target.

### Fields that moved on objects present in both runs

72 in total, out of 1866 objects. Every one is a rule that used to depend on reading order:

| field | moved | why |
| --- | ---: | --- |
| `access_path` | 36 | two files disagreed; now settled on content (§4.3) |
| `lineno` | 8 | earliest site instead of first-seen |
| `kind`, `display_name`, `scope`, `owner`, `field` | 4 each | placeholders finally upgraded across files (§4.3) |
| `inferred_type` | 4 | same |
| `file` | 2 | earliest site |
| `container`, `structural_role` | 1 each | same |

### The Step 1a oracle: unchanged

Re-run against the same trace, before and after:

| | before | after |
| --- | --- | --- |
| recall | 69.0% (1598 / 2317) | **69.0% (1598 / 2317)** |
| confirmed static claims | 75.2% | **75.2%** |
| falsified static claims | 1 | **1** |

Identical, which is the right answer and a good sign about the instrument. Step 1a §6.1 predicted
exactly this: what the access oracle scores is a `(callable, name, read/write)` triple that never
mentions an object ID, so a step that changes object identity should not move it. It did not.

*(Note for future runs: the tracer must be started from the analysed project's own directory. Run
from elsewhere, its pytest driver fails with a usage error and silently drops from 428 traced
functions to 256 — the recall number then looks like a regression and is not one.)*

---

## 8. What this instrument proves, and what it cannot

Stated plainly, because the review's rule is that blind spots are written down rather than
discovered later.

- **It proves the output does not depend on file order.** That is all. It is not a correctness
  check: two runs can agree perfectly on a wrong answer, and this would call that a pass.
- **It only shuffles file order.** Other orderings — the order of items within a file, the order
  of dictionary keys — are not varied. Those are already fixed by how the walk works, but this
  does not test them.
- **It needs the call-graph stage to be order-stable**, which §3 checks separately and which
  `--check-inputs` re-checks on demand. Without that, a failure would point at the wrong package.
- **Five shufflings is not a proof.** It is five samples. A dependency that only shows on a
  particular pairing of files could survive.
- **It cannot say the fields that moved moved to better values.** Where two files disagreed, the
  tie is now settled the same way every time; that makes it reproducible, not correct. Whether
  `access_path` on a rolled-up class object should say anything at all is a separate question, and
  Step 1a already noted that field is not meaningful there.

---

## 9. Is the check worth keeping once the revision is over?

Asked while this step was being written down, and worth answering here so it does not have to be
re-argued later. **Yes — keep it.**

**It guards an invariant, not a milestone.** "The output depends only on the input" is a property
of the extractor. It has no finish line. It is one line of new code away from breaking at any
moment: a set iterated where a sorted list was meant, a "keep whichever we saw first" rule, a new
output column left out of a sort key. None of those produce an error. They produce a *different
answer*, which is the whole reason this step existed.

**The remaining steps are the most likely moments to break it.** Step 8 takes a 1,700-line class
apart and checks itself by comparing artifacts byte for byte — which only works while this
property holds. Retiring the check when Step 8 finishes would mean deleting the thing that made
Step 8 checkable on the day it stops being watched. Steps 3, 4 and 7 all add inference that
combines facts from more than one file, which is precisely where §4.3's problem came from.

**The unit test is not a substitute for it.** `test_artifacts_do_not_depend_on_file_processing_order`
runs on a four-file fixture, and it does catch all six defects found here. But a fixture only
contains the shapes somebody thought to write down, and the two defects in §5 were found on
climlab, not on a fixture. The command is what makes checking a real codebase one line.

**What would make retiring it right:** if determinism ever became cheap to assert structurally —
every merge rule provably order-blind, every fan-out sorted where it is built — the check would be
a slow restatement of something the code already guarantees. That is not close today.

### The one real problem with it, found by asking this question

The first version of the check contained its own copy of what `run_from_extraction_config` does,
because it needed to feed in a shuffled file list. That is the defect §5 of the review catalogues —
this package reimplementing something it could call — and it is worse here than usual: **a
determinism check that has quietly drifted from the real pipeline still reports success.** It
would have failed silently, in the one piece of code written to make silent failure impossible.

Fixed by giving `run_from_extraction_config` an optional `analysis_files` argument and having the
check call it. There is now no second copy to drift.

---

## 10. What deliberately did not change

**The two copies of the reaching-roots walk are still two copies.** Both are fixed identically.
Merging them is Step 8, and doing it here would have put a rearrangement inside a diff whose job
was to stay readable.

**Nothing was done about the 37 lineage endpoints naming objects that do not exist.** They are the
same 37 before and after, so this step neither caused nor fixed them. They belong with §4.6's
argument that these counts are best read as a defect counter.

**The thresholds that decide whether a class is "big enough to split" are still chosen, not
derived** (§4.5). This step made the decision consistent across files; it did not ask whether 4,
3 and 2 are the right numbers.

**The tests still exercise a different analysis from production** (§4.1). Most tests go through
the single-file path, which cannot accept a project index. The new tests that need one use
`_collect_with_registration`, the one helper that builds a real one.

**No attempt was made to reduce the `inferred_type == unknown` count.** That is Step 3, and it now
has a stable baseline to move: 759.

---

## 11. Tests

Eleven added to `tests/test_data_access_ast.py`. **All eleven were run against the pre-change code
and all eleven fail there**, which is the only thing that makes a determinism test worth having —
one that passes before the fix is testing nothing.

| test | what breaks silently without it |
| --- | --- |
| `test_split_class_owner_is_the_same_answer_from_every_file` | one class, two object identities |
| `test_split_class_owners_include_nested_classes` | a nested class can be asked about but never qualifies |
| `test_lineage_roots_do_not_cache_an_answer_computed_inside_a_cycle` | the answer depends on which question came first |
| `test_shared_container_lineage_roots_have_the_same_cycle_fix` | the second copy of that function drifting away from the first |
| `test_cross_file_object_merge_keeps_the_better_description` | refinements from later files discarded |
| `test_object_merge_is_commutative` | a merge rule added later that is not order-blind |
| `test_return_summary_snapshot_sees_every_field` | the settling loop stopping early |
| `test_return_summary_tie_breaks_on_content_not_arrival_order` | §5's unlisted tie-break |
| `test_lineage_edge_ordering_uses_every_column` | tied output rows keeping file order |
| `test_artifacts_do_not_depend_on_file_processing_order` | the gate itself, on a fixture that hits all four findings |
| `test_fixpoint_reports_when_the_cap_stopped_it` | running out of turns looking like success |

Full suite: **319 passed**. The one failure,
`tests/test_cluster_structural_graph.py::test_parse_args_uses_structural_config_and_cli_overrides`,
is the argparse-state leak recorded in
[`step0_adopt_call_graph_fixes.md`](step0_adopt_call_graph_fixes.md) §5 and
[`step1a_access_oracle.md`](step1a_access_oracle.md) §7, and fails identically before and after.

---

## 12. What changed in the code

### New

| file | what |
| --- | --- |
| `data_access/determinism_check.py` | the gate. CLI: `check-data-access-determinism` |

### Edited

- `data_access/generate_data_access_ast.py` — project-wide split-owner and class-list passes; the
  cycle-aware roots walk; `merge_data_object` used in both merge sites; field-complete settling
  summaries with a content tie-break; both loops report running out of turns; membership tests
  replacing whole-project set builds; the settling loop given the same inputs as the final pass.
  `run_from_extraction_config` gained an optional `analysis_files` argument, which is the only
  thing the determinism check needs in order to call it rather than copy it (§9).
- `data_access/infer_shared_containers.py` — the same cycle fix in the duplicated walk.
- `data_access/outputs.py` — both sort keys now use every column.
- `cli/main.py` — one command registered.

### Not changed, deliberately

- **No new artifact columns.** The kind-conflict count is a warning, not a column. Adding a column
  to prove a step that verifies itself by comparing columns would have been circular.
- **No `ProjectIndex`-style frozen index for §3.1.** The review proposed one; it would reintroduce
  the staleness that made Step 0 defer this. Membership tests are equivalent and cannot go stale.
- **No change to `MAX_RETURN_SUMMARY_PASSES`.** climlab settles inside it. Now that running out
  says so, raising it is an evidence-driven decision rather than a guess.

---

## 13. What this unblocks

- **Step 1b (the object-identity oracle)** was blocked on this and is now clear. What it scores —
  `alias_of` and the lineage graph — no longer moves when the file order does, so a baseline taken
  now is reproducible.
- **Step 8 (structure)** was blocked on this and is now clear. Its verification method is
  byte-identical artifact comparison, which did not work before. `check-data-access-determinism`
  can be re-run after every slice of the split.
- **Step 6 (performance)** is fully discharged. Parses per file went to one, or zero, in Step 0;
  the two remaining rebuilds are gone here, and the stage is 52% faster.
- **Step 3 (container families)** has a stable number to move: 759 objects with an unknown type.

The thing this step demonstrates is the same thing Step 1a demonstrated, which is starting to look
like the rule for this project rather than a coincidence: **build the check before the fix, and
make it print what it found rather than a verdict.** Two of the six defects closed here are in no
section of the review. Both were found by reading rows the check printed and taking them to the
source — not by reasoning about the code, which had already been done carefully and had missed
them both.

---

*Written 2026-08-29, against commit `d89133e`. Line numbers in `code_review.md` will drift; the
findings it names will not.*
