# Step 1a — Building the access oracle

This is the second step of the plan in [`code_review.md`](../code_review.md) §7. It builds
the first half of the instrument §6 asks for.

---

## 1. What this step was about

Every finding in `code_review.md` says *this code does the wrong thing*. Not one says
*and here is what that costs*. There was no way to tell, because `data_access/` had no
way to check its own answers against reality.

The call graph had the same problem, and the thing that fixed it was not any individual
bug fix — it was `dynamic_trace.py`. Once there was an instrument that could say which
gaps were expensive, recall went from 70.8% to 99.7%. The review quotes its own two
hardest-won lessons:

> An instrument that can only find one kind of error will only find one kind of error.
>
> A number that cannot fail is not a measurement.

This step builds the equivalent for data access.

### The idea in one paragraph

`generate_data_access_ast.py` reads source code and *infers* which callable touches which
data. This step gets the same facts the opposite way: **run the project and watch what it
actually does**. Python 3.12+ can tell a monitoring tool about every instruction the
interpreter executes. Reading data is a specific set of instructions — fetch an attribute,
look up a dictionary key, load a variable — so watching for those and noting where they
happened gives a list of accesses that really occurred. Compare that list with
`access_edges.csv`, and for the first time the extractor's claims can be checked.

### Why 1a and not all of §6

§6 describes two instruments. §6.1 (added after Step 0) split them, and this step is only
the first:

- **1a, this step** — *which accesses happened*. Its unit is read off the access site
  itself and never involves object identity, so §1.2/§1.3/§1.4 (the order-dependence
  findings) cannot move it.
- **1b, later** — *whether two static object IDs are the same runtime object*. That scores
  `alias_of` and the lineage graph, which Step 0 showed move 22 fields under a change that
  added nothing. A baseline taken there today would not be reproducible, so it waits for
  Step 2.

---

## 2. How the instrument works

Three pieces, each of which can be read on its own.

### 2.1 The tracer — `dynamic_access_trace.py`

Runs the project (climlab's own test suite and three of its notebooks, already configured
for the call-graph tracer) and records which access instructions executed.

The naive version of this does not finish. `INSTRUCTION` monitoring fires on *every*
instruction in *every* piece of code, including all of numpy, pytest and the standard
library. So the tracer does two things:

- **It only watches our code.** It subscribes to "a function started running", and the
  first time a function from an analysed module starts, it switches instruction-watching on
  for *that one function*. Everything else is switched off permanently on first sight.
- **It stops watching each spot after the first hit.** The question is only "did this
  access ever happen", never "how often". So the cost is the number of distinct
  instructions executed, not the number of executions.

The result on climlab: **9 seconds**, 787 functions watched, 428 entered.

Each observation is recorded as four things plus a label:

```
callable                                    tier   name       read/write   role
climlab.model.ebm.EBM._compute              attr   heat_rate  read         attribute
climlab.process.process.Process.__init__    key    Ts         write        literal
climlab.domain.field.Field.__new__          name   domain     read         param
```

`tier` says what kind of access it is — an attribute, a container key, or a plain variable.
`role` says what kind of thing was touched. The tracer **does not decide** what is worth
scoring; it records everything and labels it. That decision is policy and lives in the
comparison, next to the static definitions it has to stay consistent with.

### 2.2 The bytecode index — the part that makes falsification possible

Before any comparison, every analysed file is compiled and every function's instructions
are read out. That gives, for each callable, **the complete set of accesses it could ever
perform**. No execution involved, so this list is not a sample — it is exhaustive.

This is what makes the instrument sharper than its call-graph sibling, and the reason is
worth stating precisely:

> For a **call**, the destination is a runtime value. The static analyzer says `f` calls
> `g`; if the trace never saw it, that may just mean the branch did not run. The comparison
> can only ever say "unconfirmed".
>
> For an **access**, the attribute name is fixed when the file is compiled. If the static
> analyzer says `EBM._compute` reads `self.spacing`, and no instruction anywhere in that
> function fetches an attribute called `spacing`, the claim is **wrong** — and it is wrong
> no matter how little the drivers covered.

### 2.3 The comparison — `access_comparison.py`

Reads `callable_data_access.csv`, turns each row into the same
`(callable, tier, name, read/write)` shape, and sorts every claim into one of four
verdicts:

| verdict | meaning | how strong |
| --- | --- | --- |
| **matched** | the instruction exists and the interpreter ran it | confirmed |
| **unexecuted** | it exists, the function ran, that spot did not | weak — an untaken branch, **not a defect** |
| **not exercised** | it exists but the function never ran | says nothing |
| **falsified** | no such instruction exists in the function at all | **strong, and independent of coverage** |

And in the other direction, **missing**: an access the interpreter really performed that no
static edge claims. That is the recall number, and it is the direction that finds problems
nobody had listed.

### 2.4 How a static row is turned into a comparable claim

The `evidence` column is used, not `access_path`. `evidence` is a copy of the *access
expression itself* and is recorded per edge; `access_path` belongs to the *object* and is
whatever path first registered it. They disagree in practice — a `class_state` edge whose
evidence is `self.state` carries `access_path = 'self.time_type'`, because class attributes
are rolled up to the class. Using the object's path would compare the wrong attribute and
blame the extractor for this module's mistake.

---

## 3. Three corrections to §6

§6 was written before anyone tried to build this. Two of its statements do not survive
contact with the interpreter, and a third gap was found by hand-checking the first results.

### 3.1 `BINARY_SUBSCR` does not exist on Python 3.14

§6 names four instructions: `LOAD_ATTR`, `STORE_ATTR`, `BINARY_SUBSCR`, `STORE_SUBSCR`.
This repo runs Python 3.14.5, where `BINARY_SUBSCR` was folded into the general `BINARY_OP`
instruction under an argument code. `d['lit']` compiles to:

```
LOAD_CONST 'lit'
BINARY_OP  26        # 26 == NB_SUBSCR
```

Every instruction name is now looked up by name at import and silently skipped if this
interpreter does not have it, so a future rename degrades what the instrument can see
rather than crashing it.

### 3.2 Those four instructions cover about a third of the artifact

Measured on climlab's 4,221 access edges before this step:

| static operation | edges | which instruction |
| --- | ---: | --- |
| `load` | 1,892 | `LOAD_FAST` / `LOAD_NAME` — **not** in §6's list |
| `attribute_load` | 1,244 | `LOAD_ATTR` |
| `assign` | 765 | `STORE_ATTR`, `STORE_SUBSCR`, or `STORE_FAST` |
| `subscript_load` | 133 | `BINARY_OP` (subscript) |
| everything else | 187 | |

`load` alone is 45% of the artifact, and 1,409 of those are on `param` — **the single
largest object kind**. Restricting the instrument to §6's four instructions would have left
the biggest category with no number at all, and would have made this an instrument that can
only find one kind of error.

So a third **name tier** was added: variables, parameters and module-level globals. The
share of the artifact that can be scored went from about 36% to **98%** (4,156 of 4,239
rows).

### 3.3 Two whole instruction families were missing — found by checking the results by hand

The first climlab run reported **89 falsified claims**. The plan called for reading the
top entries in the source rather than trusting the number, and that is what happened. All
89 were the instrument's fault:

- **`LOAD_NAME` / `STORE_NAME`** (2,284 instructions on climlab). Code at module level and
  inside a class body addresses its variables *by name*, not by the `LOAD_GLOBAL`
  instruction used inside functions. Nothing at module scope was being decoded at all, so
  every module-level global read looked invented — `box`, `x` and `y` in `stommelbox.py`
  among the first found.
- **`STORE_FAST_STORE_FAST`** (117 instructions). Tuple unpacking stores two variables per
  instruction. All 34 targets of
  `(ncol, nlay, icld, ...) = self._prepare_lw_arguments()` in `RRTMG_LW` were accused of
  not existing.

A systematic sweep of every instruction in climlab that touches a name, attribute or
subscript then found the rest: `LOAD_SUPER_ATTR`, `BINARY_SLICE`, `STORE_SLICE`,
`STORE_FAST_MAYBE_NULL`.

**After the fixes, falsified went from 89 to 1.** Each gap now has a regression test naming
the climlab case that found it.

> This is the single most useful thing that happened in this step, and it is an argument for
> the review's own rule about instruments. A recall number alone would have looked
> plausible and been wrong. What caught it was that the instrument produces *named,
> checkable claims* — "this exact attribute does not exist in this exact function" — which
> can be taken to the source and refuted. A score that could only go up or down would have
> hidden all of it.

---

## 4. The numbers

Measured on climlab (72 files), against the artifacts this pipeline produces today
(post-Step 0: 1,870 objects, 4,239 access edges). Drivers: climlab's own `fast` test suite
plus three EBM notebooks — the same ones the call-graph tracer uses.

```
microservice-pipeline data-access         --config <climlab>/configs/.../extraction.jsonc --outdir <scratch>
microservice-pipeline trace-data-access   --config ... --outdir <scratch>
microservice-pipeline compare-data-access --config ... --artifacts <scratch>
```

### The acceptance numbers

| | |
| --- | ---: |
| static access edges | 4,239 |
| — scored | **4,156** (98.0%) |
| runtime accesses observed | 6,759 |
| — comparable | 2,317 |
| callables entered at runtime | 428 |
| **Recall** — of accesses that really happened, how many the extractor found | **69.0%** (1,598 / 2,317) |
| **Confirmed** — of scored static claims, how many the run confirmed | **75.2%** (3,124 / 4,156) |
| **Falsified** — claims the bytecode proves wrong | **1** |
| trace wall time | 9 s |

Note that 69.0% is close to where the call graph started (70.8%), and for a comparable
reason: an instrument arriving at a package that has never been measured finds roughly a
third of its work undone.

### By object kind — the breakdown the review asked for

| Object kind | Scored | matched | unexecuted | not exercised | falsified |
| --- | ---: | ---: | ---: | ---: | ---: |
| `param` | 1,409 | 1,056 | 5 | 348 | 0 |
| `class_attr_state` | 793 | 659 | 86 | 48 | 0 |
| `class_state` | 703 | 460 | 55 | 188 | 0 |
| `local_exposed` | 495 | 363 | 103 | 29 | 0 |
| `object_state` | 257 | 224 | 20 | 13 | 0 |
| `container_field` | 195 | 167 | 12 | 16 | 0 |
| `unknown` | 113 | 106 | 3 | 4 | 0 |
| `module_global` | 106 | 53 | 22 | 30 | 1 |
| `dict_key` | 85 | 36 | 8 | 41 | 0 |

Read the `not exercised` column as coverage, not as quality: `dict_key` looks weak at 36/85
mostly because 41 of its claims are in code the drivers never ran.

### Keys: literal and computed, scored apart

§6 requires this and the result shows why.

| Key kind | Recall |
| --- | --- |
| literal (`d['k']`) | **147/166 (88.6%)** |
| computed (`d[k]`) | **2/117 (1.7%)** |

The extractor is good at keys it can read in the source and essentially blind to keys
computed at runtime. Folding these together would produce a single meaningless ~50% and
hide both facts.

### Where the misses are

| Tier | Missing |
| --- | ---: |
| `attr/attribute` | 474 |
| `key/computed` | 115 |
| `name/global` | 79 |
| `name/param` | 32 |
| `key/literal` | 19 |

### What is excluded, and why

Nothing is dropped silently. Both directions print their exclusions.

**Static rows not scored — 83 of 4,239:**

| | rows | why |
| --- | ---: | --- |
| `return`, `passed_arg`, `passed_kwarg`, `escape_assign` | 77 | not memory operations at all — they record a local *escaping its scope*, which no instruction corresponds to |
| `file` objects | 6 | file identity is established at a function call (`open`, `pd.read_csv`), not at an access |
| underivable | **0** | rows this module could not parse. Zero is the good answer; a large number here would be a defect in the comparison, not in the extractor |

**Observed accesses not scored — 4,442 of 6,759:**

| | count | why |
| --- | ---: | --- |
| `name/global (not module data)` | 2,187 | `len`, `range`, imported names, functions, classes. The extractor registers a `module_global` only for a name *assigned* at module level, and this filter uses the extractor's own `collect_module_globals` rather than a second opinion |
| `name/local` | 1,684 | whether a local counts is `LocalBinding.exposed`, a static judgment the interpreter cannot see |
| `attr/method` | 224 | `x.f()` fetches an attribute called `f`, but the extractor models a read of the *receiver*. Scoring these would report every method call in the project as a missed access |
| `name/receiver` | 211 | every method body loads `self`; the extractor deliberately strips `self`/`cls` from its parameters |
| `name/deref` | 73 | closure variables, which the extractor has no model for |
| `attr/super` | 63 | `super().x` — the receiver is a dispatch proxy, not an object. Zero static rows on climlab mention `super(`, so scoring these would be 63 guaranteed misses |

Each exclusion removes something the extractor *structurally does not model*, never
something it merely got wrong. The counts are printed so the denominator cannot be chosen
quietly.

---

## 5. What the instrument found

### 5.1 One real defect, previously unlisted: default arguments are attributed to the wrong callable

The single surviving falsified claim:

```
callable : climlab.radiation.rrtm.rrtmg_sw.RRTMG_SW.__init__
claims   : reads the module global `nbndsw`
evidence : nbndsw          (rrtmg_sw.py:95)
```

Line 95 is inside the `__init__` *signature*:

```python
def __init__(self, ...,
             bndsolvar = np.ones(nbndsw),   # <- line 95
             **kwargs):
```

Python evaluates default arguments **once, in the enclosing scope, when the `def`
statement runs** — so `nbndsw` is read by the module body, not by `__init__`. The
extractor visits defaults deliberately (`generate_data_access_ast.py`, *"Defaults can read
module globals or outer-scope data"*) but does so *after* switching `current_callable` to
the function being defined, so the access is filed against the callee.

This is in no section of `code_review.md`. It is exactly what §6 predicted an instrument
would do: *the instrument found defects nobody had listed.*

### 5.2 §5.5 is now measured, not argued

84 observed accesses are in callables `access_edges.csv` has no row for at all. Lambdas and
generator expressions are separate code objects at runtime, keyed the way
`call_graph.definitions.visit_Lambda` keys them — and §5.5 records that `data_access` enters
no callable for them. They are reported under their own heading rather than mixed into the
general miss count.

### 5.3 Which findings are now measurable

| finding | what the instrument now says |
| --- | --- |
| §1.1 container families | per-kind confirmation is the before/after for Step 3 |
| §1.6, §1.10, §1.13 | falsified count is the gate — currently 1, so any regression is loud |
| §5.5 / §1.9 lambdas | the 84 unmodelled-callable accesses above |
| §4.6 `unknown` counter | 114 rows, printed in the report; 106 of them confirm |
| computed keys | 1.7% — quantified for the first time |

---

## 6. What this instrument deliberately cannot see

Stated plainly, because the review's rule is that an instrument's blind spots must be
written down rather than discovered later.

- **It cannot judge object identity.** It says an access to `state['Ts']` happened in a
  given callable. It says nothing about whether the extractor's `local_exposed:f:df` and
  `param:g:frame` are the same runtime object. That is `alias_of` and the lineage graph, and
  it is **Step 1b**, still blocked on Step 2.
- **It is a lower bound.** An access it never saw is not thereby false — that is what the
  `unexecuted` and `not exercised` buckets exist to say. Only `falsified` is coverage-free.
- **The name it compares is the name in the source, not the object behind it.** For a
  computed key (`d[k]`) it can only say "a subscript happened here", never which key.
- **Modules already imported before the trace starts are missed**, since their bodies never
  run again. Same limitation as the call tracer.
- **The recall figure is a floor, and its composition matters more than its value.** 474 of
  the 719 misses are attribute reads, and reading the by-name table shows most of them are
  reads *through imported modules* — `np.newaxis` (21), `np.ones_like` (21), `np.array`
  (15) — which are not project data and which the extractor never intended to model. Those
  are **not** excluded, because deciding which names count would be tuning the denominator
  to taste. The report prints the composition instead so the next step can decide with the
  evidence in front of it. Genuine gaps are in the same table: `attr/domain` (20),
  `attr/axes` (11), and the physical constants `attr/g`, `attr/cp`, `attr/Rd`, `attr/S0`
  read from `climlab.utils.constants` — cross-module global reads the extractor does not
  model.

---

## 7. What changed in the code

### New

| file | what |
| --- | --- |
| `data_access/dynamic_access_trace.py` | the tracer and the bytecode decoder. CLI: `trace-data-access` |
| `data_access/access_comparison.py` | the bytecode index, the claim derivation, the scoring, the report. CLI: `compare-data-access` |
| `tests/test_dynamic_access_trace.py` | 21 tests |

### Edited

- `call_graph/dynamic_trace.py` — the file→module and code→callable-ID rule was pulled out
  into a shared `CodeIdResolver` and `callable_id_for_qualname`. Both tracers now use one
  implementation. This was not tidiness: if the two conventions ever drift, the comparison
  matches nothing and reports a recall of zero, which looks exactly like a recall of zero.
  §2.5 records three copies of the *static* callable-ID rule for the same reason; there was
  no case for opening a fourth front. No behaviour change — `tests/test_dynamic_trace.py`
  passes unaltered.
- `cli/main.py` — two commands registered.

### Not changed, deliberately

- **No `col_offset` on `AccessEdge`.** The plan considered adding one to match call-graph
  edges. It is not needed: matching on `(callable, tier, name, read/write)` without a
  position is what makes the falsified verdict coverage-independent, and a position would
  buy per-occurrence precision at that cost.
- **No second trace configuration block.** Which drivers exercise a project is a property of
  the project, not of which analysis is being scored, so this reads `call_graph.trace` —
  where climlab's test arguments and notebooks are already configured.
- **No extractor fixes.** Not one line of `generate_data_access_ast.py` changed. The
  numbers above are a baseline for the current code, and mixing a fix into the step that
  builds the measuring device would leave nothing to measure it against.

### Artifacts added

Written to the data-access output directory:

- `dynamic_access.csv` — one row per observed access
- `dynamic_access.json` — callables entered, driver problems
- `access_comparison.md` / `.json` — the report

### Tests

21 new, in the style of `test_dynamic_trace.py`: each asserts a property that would
otherwise fail *silently*, not an aggregate score.

| test | guards |
| --- | --- |
| callable IDs match the static artifacts | if they drift, every number is zero and nothing says so |
| literal vs computed keys distinguished | §6's key-identity rule |
| folded constant tuple slice (`df.loc[:, 'col']`) | the pandas case |
| `self.n += 1` is both read and write | augmented assignment |
| method loads tagged apart from data attributes | else every method call is a "miss" |
| parameters tagged apart from ordinary locals | the scoping rule |
| a lambda is its own callable | §5.5 |
| comprehensions stay with the enclosing callable | PEP 709; catches a future CPython change |
| `LOAD_NAME` at module level | §3.3's first gap |
| tuple unpacking records every target | §3.3's second gap |
| `super().x` gets its own role | §3.3's third gap |
| `evidence` beats `access_path` when they disagree | the derivation rule |
| an unrun method is *not exercised*, never *falsified* | the instrument must not cry wolf |
| an invented claim is falsified with no coverage | the strong signal actually fires |
| exclusions are counted, not dropped | a number that cannot fail is not a measurement |

Full suite: **308 passed**. The one failure,
`tests/test_cluster_structural_graph.py::test_parse_args_uses_structural_config_and_cli_overrides`,
is the pre-existing argparse-state leak recorded in
[`step0_adopt_call_graph_fixes.md`](step0_adopt_call_graph_fixes.md) and fails identically
in isolation, before and after this step.

---

## 8. What this unblocks

- **Step 3 (container families)** now has a before/after. Its acceptance criterion — "the
  share of objects with `inferred_type == unknown` falls, and per-kind recall rises with it"
  — is a table this report prints.
- **Step 4 (cheap correctness)** has a gate. Falsified is at 1; §1.6, §1.10 and §1.13 all
  predict claims the bytecode can refute, so a regression is now loud rather than invisible.
- **Step 2 (determinism)** is unaffected and still next. Nothing here depends on it, which
  is precisely why §6.1 put 1a before it.
- **Step 1b** remains blocked on Step 2, unchanged.

One thing this step did *not* settle: whether the extractor's misses matter for clustering,
which is the question `evaluate` answers and this instrument does not. Recall is a measure
of completeness, not of usefulness. §1.7 and §1.11 — the two findings that manufacture
*false* coupling — still have to be judged with `evaluate`, exactly as §7 Step 4 says.

The thing this step demonstrates: the review was right that the instrument comes first, and
right about why. Building it turned up a defect in the extractor that no section of the
review lists — and, before that, three defects in the instrument itself, which only showed
up because it makes named claims that can be taken to the source and refuted.

---

*Written 2026-08-24. Baseline measured against the tree at that date, post-Step 0. Line
numbers in `code_review.md` will drift; the findings it names will not.*
