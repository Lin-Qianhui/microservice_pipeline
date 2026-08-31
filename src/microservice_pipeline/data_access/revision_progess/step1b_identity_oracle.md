# Step 1b — Checking whether two names really point at the same thing

This is the fourth step of the plan in [`code_review.md`](../code_review.md) §7, after
Step 0, Step 1a and Step 2. It builds the second half of the instrument §6 asks for.

---

## 1. What this step was about

The extractor does not only record *which* data each function touches. It also records
claims that two different names are **the same piece of data**:

- the `alias_of` column in `data_objects.csv` — "this object is really that object";
- the lineage graph in `data_access.json` — "the value here flowed to there".

These claims matter more than they look. Everything downstream clusters the extractor's
output, and two names the extractor says are one thing get merged into one node. If that
merge is wrong, two unrelated parts of the program get glued together and the clustering
inherits the mistake.

Nothing had ever checked these claims. Step 1a checked *accesses* and deliberately did not
touch identity, because at the time the identity answers moved around depending on the
order files happened to be read. Step 2 fixed that. So this step could finally ask the
question:

> When the extractor says two names hold the same object, does the program agree?

### How you can even find out

Run the program and look. Python can tell a monitoring tool about every instruction it
executes. When the program is about to read a variable, the tool can look inside the
running function and see what that variable currently holds. Do that at both ends of a
claim, and if the same object turns up in both places, the claim is confirmed.

### Why this had to wait for Step 2

A measurement is only worth writing down if running it again gives the same answer. Before
Step 2, the very things this step measures — `alias_of` and the lineage graph — changed
when the file order changed. Step 0 showed it concretely: a change that added twelve
objects and removed nothing still moved 22 of these fields. A baseline taken then would
have been a baseline of nothing.

---

## 2. How the instrument works

It reuses the Step 1a tracer rather than starting again — same runner, same drivers, same
naming of functions. What is new is a second channel switched on with `--identity`.

### 2.1 Reading the value out of the running function

§7 describes this step as "`id()` at the observed access" — take the identity of the object
at the moment it is touched. That turns out not to be literally possible. When the
monitoring tool is told "instruction number 42 is about to run", it is told the function
and the position, and nothing else. The value the instruction is about to work on is held
in a place Python code cannot reach.

There is a way round it. From inside the notification, the tool can ask for the *function
call currently running* and read its local variables. So:

- **Plain variables** — parameters, locals, module-level names — are read directly.
- **`self.something`** — read `self` from the function's variables, then look the attribute
  up in the object's own storage.

That covers the four kinds of object the extractor actually assigns `alias_of` to.

### 2.2 Never ask the object for the attribute

To read `self.rows`, the natural thing is `getattr(self, "rows")`. The instrument must not
do that. In Python an attribute read can be a piece of code that runs — a "property" — and
running it would mean the act of measuring changes the program being measured. So the
instrument looks in the object's own storage directly, which cannot run anything. When an
attribute is not there (it is computed, or the class stores things unusually), the
instrument records nothing and counts the miss.

There is a test for this: the test fixture has a property that raises an error if it is
ever executed.

### 2.3 A written value cannot be read yet

The notification arrives *before* the instruction runs. So at `made = given`, the moment
the instrument is told about the store, `made` does not exist yet. Identity is therefore
recorded when a name is **read**, and a name that has just been written is picked up the
next time it is used. Names that are only ever written and never read are counted as
unobservable.

### 2.4 Object identity is not the same as an object's address

Python identifies an object by its address in memory. Addresses get **reused**: free one
object and the next one may land in the same place. If the instrument recorded raw
addresses, two unrelated objects that happened to occupy the same address at different
times would look like one — which is precisely the false merge this step exists to detect.
The instrument would report its own bug as a finding.

The fix is to keep one reference to every object it records, which stops the address being
handed out again. The neat alternative — a "weak" reference that does not keep the object
alive — cannot be used, because Python does not allow weak references to plain lists and
dictionaries, and those are exactly the containers this step is about.

Keeping objects alive costs memory, so there is a limit. If the limit is ever reached, the
run **says so** and the report marks every number as incomplete, rather than quietly
computing an answer from a truncated set. On climlab the limit is 500,000 and the run used
93,981, so it was never close.

Values whose identity is meaningless are skipped before any of this: numbers, strings,
`True`/`False`, `None`, tuples. Python reuses these deliberately — every `0` in the program
is often literally the same `0` — so counting them would alias together every function in
the project that mentions a small number.

### 2.5 The file question needs values, not identity

§7 also asks this step to judge §1.7, which is about `file:` nodes. The extractor names a
file object after the *text* of the argument, so `load_users(path)` and
`load_invoices(path)` both produce a node called `file:path` and the graph reads that as
two functions sharing a file.

Object identity cannot answer this. Two files are the same file when their *paths are
equal*, whether or not the two path strings are the same object in memory. So the identity
channel records a second thing: the **string values** seen at each site. The question then
becomes "did the functions sharing this file node ever see the same path?"

---

## 3. Six corrections to §7's wording

§7 was written before anyone tried to build this.

| § 7 says | what is actually true |
| --- | --- |
| "`id()` at the observed access" | the value at the instruction is unreachable; it has to be read out of the running function instead (§2.1) |
| — | reading an attribute must not go through `getattr`, or the instrument runs the program's own code (§2.2) |
| — | a value being written cannot be read yet, because the notification comes first (§2.3) |
| — | a raw address is not an identity, and the obvious fix does not work on lists and dicts (§2.4) |
| this step judges §1.7 | §1.7 is a question about path *values*; identity cannot answer it at all (§2.5) |
| "a recorded count of static alias claims the trace contradicts" | that count is **not** proof, and is far more fragile than Step 1a's equivalent — see §4 |

### The sixth is the important one

Step 1a could **prove** a claim wrong. The full list of attributes a function can ever touch
is fixed when the file is compiled, so a claim naming an attribute that appears nowhere in
the function is false no matter how little of the program was run.

Identity has no such backstop. If two names were never seen holding the same object, that
might mean the extractor is wrong — or it might mean the two pieces of code never ran close
enough together for the instrument to catch them. There is no compile-time fact to fall
back on.

So this instrument reports three verdicts, not two:

| verdict | meaning | how much it is worth |
| --- | --- | --- |
| **confirmed** | both names were seen holding one object | strong |
| **contradicted** | both names were seen holding objects, never the same one | a lead, **not proof** |
| **unobserved** | one side never ran, or never held anything recordable | nothing |

And in the other direction, **recall**: the program put one object in several places, and
the question is whether the extractor connected them. That direction *is* sound — an
aliasing the interpreter performed really happened.

---

## 4. The instrument was wrong the first time, again

Step 1a's first run accused 89 claims and every one was the instrument's fault. The same
thing happened here, and it is worth writing down because the mechanism is different and
it would happen to anyone building this next.

### What went wrong

For an *access* site, watching once is enough: an instruction that fetches `spacing` will
always fetch `spacing`. So the Step 1a tracer stops watching each spot after the first hit,
which is what makes it fast.

Identity is not like that. The same line inside a loop holds a **different object every
time round**. Watching once means seeing the first object and no others. Two names that
really do hold the same object can easily be sampled at two different moments, holding two
different objects — and be reported as contradicting each other.

Carrying the Step 1a habit over, the first version watched each spot 8 times. It reported
**152 contradictions**. Raising the budget did this:

| executions watched per spot | confirmed | contradicted | precision | trace time |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 167 | **152** | 52.4% | 10 s |
| 16 | 187 | 143 | 56.7% | 10 s |
| 32 | 224 | 106 | 67.9% | 10 s |
| 64 | 274 | 56 | 83.0% | 13 s |
| 128 | 278 | 52 | 84.2% | 12 s |
| **512** | **279** | **51** | **84.5%** | **16 s** |
| 4096 | 279 | 51 | 84.5% | 35 s |

**Two thirds of the original 152 were the instrument running out of patience, not the
extractor being wrong.** The answer stops moving at 512 — 4096 gives an identical result —
so 512 is the default, and the number is written down as *calibrated* rather than chosen.

### What was added so this cannot hide again

The run now reports how many spots used up their whole budget: on climlab, 3,312 of 17,086.
That is the signal that says whether sampling is still shaping the answer. A project where
the default is too small will say so instead of quietly reporting inflated contradictions.

There is also a test that pins the property directly: a loop that builds a new list each
pass must yield more than one object when the budget allows it, and exactly one when the
budget is one.

---

## 5. The numbers

Measured on climlab (72 files), against today's artifacts: 1,866 objects, 4,239 access
edges, 1,175 lineage edges. Same drivers as Step 1a — climlab's own `fast` test suite plus
three EBM notebooks.

```
microservice-pipeline data-access                     --config <climlab>/configs/.../extraction.jsonc --outdir <scratch>
microservice-pipeline trace-data-access --identity    --config ... --outdir <scratch>
microservice-pipeline compare-data-access-identity    --config ... --artifacts <scratch>
```

### Coverage

| | |
| --- | ---: |
| static objects | 1,866 |
| — that name a place the instrument can look | **1,425 (76.4%)** |
| places the instrument watched | 3,477 |
| identity observations recorded | 148,511 |
| distinct objects kept alive | 93,981 of a 500,000 limit — **limit not reached** |
| trace wall time | 16 s |

### Precision — the extractor's claims, scored

| | |
| --- | ---: |
| identity claims made (`alias_of` + lineage) | 1,597 |
| — scored | 1,020 |
| **confirmed** | **279** |
| **contradicted** | **51** |
| unobserved | 690 |
| **precision** | **84.5%** |

| claim source | confirmed | contradicted | unobserved |
| --- | ---: | ---: | ---: |
| `arg_to_param` — an argument becomes a parameter | 106 | 6 | 186 |
| `local_assign` — a value is assigned to a local | 78 | **31** | 151 |
| `alias_of` — the derived alias column | 51 | 10 | 299 |
| `tuple_unpack` | 36 | 1 | 9 |
| `state_assign` | 8 | 3 | 45 |

Read this as: **passing an argument is the claim the extractor gets right, and assigning
the result of a call is the claim it gets wrong.** §6 says the point of an instrument is to
say which problem is expensive; this is that answer.

### Recall — aliasings the program performed

| | |
| --- | ---: |
| objects seen in two or more places | 31,027 |
| — connected by the static graph | 15,310 |
| — **split across separate parts of the graph (missed)** | **7,915** |
| — in places the extractor models no object for | 7,802 |
| **recall** | **65.9%** |

A third of the aliasings that really happen are invisible to the extractor.

### What is excluded, and why

Nothing is dropped silently.

**Objects with no place to look — 441 of 1,866:**

| | count | why |
| --- | ---: | --- |
| `object_state` with no reachable root | 180 | the object it hangs off is not a plain name in the function |
| `object_state` more than one attribute deep | 154 | `a.b.c` — the middle value never exists in a variable, only in transit |
| `container_field` | 141 | the value comes out of a lookup, which is also only ever in transit |
| `dict_key` | 69 | same |
| `class_state` | 46 | a whole-class rollup, not one value |
| `file` | 5 | handled by the value channel instead (§2.5) |

**Claims not scored — 577 of 1,597:**

| | count | why |
| --- | ---: | --- |
| one end is a `return:` marker | 361 | a bookkeeping node for "the value a function returns", never a real object |
| one end has no place to look | 216 | the object is in the table above |

**Observations the tracer could not record:** 234,210 skipped as value types (numbers,
strings, `None`, tuples), 59,960 attributes not stored on the instance, 30,702 names not
bound at that moment.

---

## 6. What the instrument found

### 6.1 The main finding: "made from" is being recorded as "is"

Nearly all 51 surviving contradictions are one problem, and it is in no section of
`code_review.md`.

When a value is produced by a call and assigned to a name, the extractor frequently records
that the new name **is** one of the call's arguments. It is not; it is a new object made
from that argument. Hand-checked examples, all confirmed against the source:

```python
# climlab/process/process.py:555
dic = self.state.copy()          # a copy is a new dictionary, not self.state

# climlab/process/time_dependent_process.py:404
self.timeave = self.state.copy() # so timeave is not state

# climlab/radiation/cam3.py:135 and ~18 more lines like it
Tatm = self._climlab_to_cam3(self.Tatm)   # returns a re-shaped new array

# climlab/utils/thermo.py, via emanuel_convection.py:207
QS = qsat(T, P)                  # a computation, not T
```

**Why it happens — two separate causes, one cheap and one less so.**

**(a) `.copy()` is written down as an alias, on purpose.** In `_infer_type_from_value`, the
branch for a call contains:

```python
if isinstance(value.func, ast.Attribute) and value.func.attr == "copy":
    base = self._resolve_expr(value.func.value)
    if base:
        return base.inferred_type or FAMILY_UNKNOWN, base.object_id, "medium"
```

That middle return value is the `alias_of`. So `x.copy()` is recorded as being `x` — the one
call in Python whose entire purpose is to *not* be the same object. Four lines, and it
accounts for 5 of the 51 contradictions.

**(b) One return path speaks for the whole function.** The extractor keeps a single summary
per function saying what it returns, picking the "best" one when there are several return
statements. `_climlab_to_cam3` begins:

```python
def _climlab_to_cam3(self, field):
    if np.isscalar(field):
        return field       # <-- this branch returns the argument
    ...                    # every other branch builds a new array
```

That one branch makes the whole function "returns its own argument", and every one of the
eighteen call sites inherits the claim. This is §4.4 — no modelling of branches — turning
into *wrong* identity rather than merely coarse identity, and it is the largest single
source of false merges in the artifact.

**Neither fix needs §4.4's join lattice.** "Does every return statement return this
parameter?" is answerable by listing a function's `Return` nodes. That matters, because §4.4
is the one finding in the review with no step assigned to it, and this defect is not blocked
behind it.

The same shape explains the cross-class case: `Iceline(Tf=Tf, state=self.state, ...)` makes
`StepFunctionAlbedo.state` and `Iceline.state` one object in the artifact, when at runtime
the constructor builds a new one.

### 6.2 A mis-resolved call target

```python
# climlab/utils/attrdict/mixins.py:167
super(MutableAttr, self).__setattr__(key, value)
```

The extractor resolved this to `MutableAttr._setattr` and recorded the value flowing into
that method's parameter. It does not: it goes to the built-in `object.__setattr__`. This is
the same family §5.6 puts `resolve_method_targets` on notice for, now with an example.

### 6.3 Three contradictions that are the instrument's limitation, not the extractor's

```python
# climlab/domain/domain.py:533
    latax = lat                       # a real alias, on this branch
else:
    latax = Axis(axis_type='lat', points=lat)   # the branch the drivers took
```

The claim recorded at line 533 is correct. The drivers never took that branch, so the
instrument saw `latax` and `lat` holding different objects and called it a contradiction.
The cause is that a site is identified by *function and name* with no line number, so a
claim tied to one branch is tested against observations from all of them. This is the
concrete form of the caveat in §3, and it is why the hand-check is mandatory.

### 6.4 The biggest missed aliasings

The recall direction found large groups the extractor keeps apart. The worst: **one `Tatm`
array held at 23 places, which the static graph has in 10 unconnected pieces**, including

```
class_attr_state:...rrtmg_lw.RRTMG_LW:Tatm
class_attr_state:...simplified_betts_miller.SimplifiedBettsMiller:Tatm
param:...time_dependent_process.TimeDependentProcess.set_state:value
param:...utils.thermo.clausius_clapeyron:T
```

This is climlab's central design — subprocesses share one state array — and the extractor
models each subprocess's view of it as a separate, unconnected node. Anything clustering
this output cannot know they are one array.

### 6.5 §1.7 does not fire on climlab — a real negative result

climlab has 5 `file:` objects, and only **one** is touched by more than one function. The
functions sharing it were never observed with different paths, so there is **no manufactured
file coupling to find here**.

That is not evidence that §1.7 is wrong. §1.7 is confirmed by reading the code, and the
smoke fixture in the tests reproduces it exactly: two functions taking a parameter called
`path` collapse onto one node, and the instrument reports the disagreement. climlab simply
does not open files from parameters. Judging §1.7's fix will need a project that does, or
the `evaluate` route §7 already prescribes for it.

---

## 7. What this instrument deliberately cannot see

- **Anything whose value is only in transit.** Results of lookups (`d[k]`), attributes
  reached through other attributes (`a.b.c`), intermediate expressions. This is why a
  quarter of objects have no place to look.
- **Which branch a claim came from.** Sites carry no line number, so a claim that is true on
  an unexercised branch can be reported as contradicted (§6.3).
- **Anything the drivers did not run.** 690 of 1,020 scored claims are unobserved.
- **Whether a contradiction is the extractor's fault.** It cannot; only a person reading the
  source can. That is why contradictions are printed by name.
- **Objects it chose not to keep alive.** Numbers, strings and tuples are excluded by design.
- **Modules imported before the trace starts** — the same limitation Step 1a has.

---

## 8. The hand-check

§7's acceptance asks for a count of contradicted claims. All 51 were read against the
source before the count was recorded. The result:

| | count |
| --- | ---: |
| a value made by a call, claimed to *be* an argument — cause (b), the return summary | ~41 |
| the same, via the hard-coded `.copy()` branch — cause (a) | 5 |
| a mis-resolved call target (§6.2) | 1 |
| true on a branch the drivers never took — instrument limitation (§6.3) | 3 |
| dispatched to a subclass override the static side did not name | 1 |

Every one is an extractor claim the program refuses, except the three in §6.3, which are
claims the drivers never got to test.

**No line of `generate_data_access_ast.py` was changed in this step.** The findings above
are recorded, not fixed; §6.1 belongs with Step 4's correctness work and §6.2 with Step 5's
cross-stage guards.

---

## 9. What changed in the code

### New

| file | what |
| --- | --- |
| `data_access/identity_comparison.py` | the comparison and its report. CLI: `compare-data-access-identity` |
| `tests/test_identity_comparison.py` | 19 tests, all on properties that would otherwise fail silently |

### Edited

| file | what |
| --- | --- |
| `data_access/dynamic_access_trace.py` | the `--identity` channel: `ObjectTokens`, reading values out of the running function, the sample budget, the string-value channel, and three new artifacts |
| `cli/main.py` | registers `compare-data-access-identity` |

### Artifacts added

`dynamic_identity.csv`, `dynamic_identity_values.csv`, `dynamic_identity.json`,
`identity_comparison.md`, `identity_comparison.json`.

The identity run writes **only** these. It deliberately does not overwrite
`dynamic_access.csv`, because it samples each spot hundreds of times where the access run
samples once — the two are not interchangeable, and merging them would let someone compare
numbers that were not measured the same way.

### Verified unchanged

- `check-data-access-determinism` on climlab: **byte-identical**, still passing.
- The Step 1a numbers, re-run: 4,239 static rows (4,156 scored), 2,317 comparable, recall
  **69.0%**, confirmed **75.2%**, falsified **1** — exactly as recorded in Step 1a.
- Full test suite: 338 passing. The one failure,
  `test_cluster_structural_graph.py::test_parse_args_uses_structural_config_and_cli_overrides`,
  fails identically on a clean checkout and is unrelated to this step.

---

## 10. What this unblocks

- **Step 3 (container families)** and **Step 4 (cheap correctness)** were waiting only on
  Step 1a and can now be judged on both axes.
- **§6.1 is new work for Step 4**, and it is bigger than several items already listed there:
  it is the main source of false merges in the artifact today, and false merges are the
  thing §1.7 was ranked above its apparent size for. It was in **no** section of the review
  beforehand — §4.4 records its cause, but as a limitation rather than a bug, and §4.4 has no
  step of its own. Cause (a), the `.copy()` branch, is four lines and is the cheapest fix in
  Step 4; do it first, and the oracle will confirm it directly.
- **§6.2 gives §5.6 a concrete example** rather than a prediction.
- **§1.7's fix now has a way to be judged**, but not on climlab — see §6.5.
- The recall figure (65.9%) is the number Step 3 and Step 4 should move, and the largest
  missed aliasings in §6.4 say where to push.

### One note for whoever runs this next

If `identity_offsets_at_cap` is a large fraction of `identity_offsets_sampled` on your
project, raise `--identity-samples` and re-check before believing any contradiction count.
The calibration in §4 was done on climlab and does not automatically transfer.
