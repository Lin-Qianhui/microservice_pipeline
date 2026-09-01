# Step 3 — Working out what kind of container something is

This is the sixth step of the plan in [`code_review.md`](../code_review.md) §7, after
Step 0, Step 1a, Step 2, Step 1b and Step 4a. It covers §1.1, §2.1, §2.2 and §2.4, and
it removes §2.6 along the way.

---

## 1. What the "family" is, and why it matters

When the extractor sees `x['temperature']`, it has to decide what `x` is. A dictionary?
A table with columns? Something else entirely? The answer is called the **family**, and
it decides how the access is recorded:

| what `x` is | how `x['temperature']` is filed |
| --- | --- |
| a dictionary | a **dictionary key** named `temperature` |
| a table | a **table column** named `temperature` |
| anything else, or not known | an anonymous **container field** |

The third row is the fallback, and it is where everything the extractor cannot work out
ends up. Two functions that both reach into the same dictionary should be recorded as
touching the same named key; if the family is unknown they are recorded as touching two
anonymous somethings instead.

The extractor works the family out by running Pyright — a type checker — over a copy of
the project with small probes inserted, and reading back what Pyright says each
expression's type is. There were four separate faults in that chain, and every one of
them ended in the same place: the family came back "unknown", which is also a legitimate
answer, so nothing anywhere reported a problem.

## 2. What was wrong, measured before anything was changed

Pyright answered 3,183 of the probes on climlab. **2,220 of those answers — 69.7% —
were thrown away as "unknown."** Splitting them apart:

- **1,535 were answers where Pyright itself had nothing to say.** It printed the literal
  word `Unknown`. That is almost always a failed import inside the probe copy, not a
  fact about the code.
- **685 were real answers that the code then discarded**, including 158 that plainly
  said `dict`.

And a further 155 probes got no answer at all.

### 2.1 The reading of Pyright's answers (§1.1)

Pyright describes a type as a piece of text such as `dict[str, list[int]]`. The old code
searched that whole text for tell-tale substrings anywhere inside it, collected every
family it spotted, and then **refused to answer if it had spotted more than one**. A
dictionary of lists mentions both "dict" and "list", so it was neither.

A second rule looked for the word "any" anywhere in the text and gave up if it found it.
That made `dict[str, Any]` an unknown, even though a dictionary of anything is still a
dictionary. It was also sensitive to spacing in a way nobody could have intended:

```
dict[str, Any]   ->  unknown     (there is a space before "Any")
list[Any]        ->  list        (there is not)
```

The shape of the text is a nesting, and the question is what the *outermost* thing is.
Reading it that way is the whole fix.

### 2.2 The copy of the project the probes run in (§2.1)

The probe run builds a temporary copy of the project. It copied three directories by
name — `src`, `scripts` and `tests` — and climlab has none of them; its code lives in a
directory called `climlab`. **So it copied nothing.** What survived was only the files
that happened to carry a probe of their own, written in one at a time.

Eight files therefore never reached the copy at all, and all eight are the same kind of
file: the `__init__.py` of a package, which usually contains nothing but re-exports and
so never carries a probe. Those are precisely the files every import inside the project
goes *through*. With them missing, imports failed, and the setting that would have
reported the failure (`reportMissingImports`) is deliberately switched off.

### 2.3 The environment the probes run in (§2.2)

The generated Pyright settings named no interpreter, no virtual environment and no
import path. Third-party types — numpy, xarray, pandas, which are most of what this
analysis wants to read — resolved only because they happened to be installed in whatever
Python was running the pipeline. That worked on this machine by luck.

### 2.4 Where the probes were put (§2.4)

A probe is an extra line inserted near the expression being asked about. It was always
inserted *after* the statement containing the expression. For a `return` or a `raise`,
nothing after the statement ever runs:

```python
        return self.data['x']
        __msp_probe_7 = self.data      # nothing reaches this line
        reveal_type(__msp_probe_7)
```

**153 probes on climlab landed in code that cannot run** — 134 after a `return`, 19
after a `raise`. Pyright does not answer at all for unreachable code, which is where the
155 unanswered probes came from.

### 2.5 A project's name written into shared code (§2.6)

The classifier contained the line `or "climlab" in lowered`, deciding whether a type
called `Field` was a data container. A project name hardcoded into a general-purpose
tool. This belongs to Step 7 in the plan, but it sat inside the exact function this step
rewrites, so leaving it would have meant measuring the rewrite against
climlab-special-cased behaviour.

## 3. The fixes, landed and measured one at a time

Step 4a's lesson was that pieces landed together cannot be told apart afterwards. Each
of these was landed on its own and the full measurement re-run.

### Slice 0 — delete the project name

Deleting `or "climlab" in lowered` produced **byte-identical artifacts**. It was dead
code: the other three tests in the same condition already recognised climlab's `Field`.
Worth knowing rather than assuming, and it means the rest of this step was measured on
code with no project name in it.

### Slice 1 — copy the whole project into the probe copy, and name its environment

Instead of three directory names, the probe copy now receives **every file the stage
analyses**, and only the ones carrying probes are rewritten. The settings also gained an
import path, a virtual environment when the project keeps one beside its source, and
otherwise the interpreter running the pipeline named explicitly.

Measured separately, because two things changed at once:

| | answers Pyright could type |
| --- | ---: |
| before slice 1 | 1,648 |
| with the missing files added | **1,761** |
| with the environment settings added, files still missing | 1,648 |
| both | 1,761 |

**The eight missing `__init__.py` files were the entire effect: +113 answers.** The
environment settings changed nothing on climlab, because the temporary copy already
serves as the import root and the Python on the path already had numpy. They are
insurance for project layouts climlab does not have, and the honest thing is to say so
rather than let them share credit.

### Slice 2 — read the type text as a nesting

The classifier now splits the text on `|` into the alternatives it lists, drops the ones
that say nothing (`Unknown`, `Any`, `None`, `Never`), takes the outermost name of each
of the rest, and answers only if they all agree.

```
dict[str, list[int]]                     -> dict     (was unknown)
dict[str, Unknown]                       -> dict     (was unknown)
Unknown | Field | None                   -> field    (was unknown)
str | ndarray[Any]                       -> unknown  (unchanged, and correct)
```

It also now recognises when Pyright has described a **function** rather than a value —
`Overload[...]`, or anything of the shape `(a: int) -> None`. A function is never a
container, and reading the type names inside its signature is how numpy's `moveaxis` was
being recorded as a list and `os.path.join` as a path.

### Slice 3 — work out what a bare name refers to

Pyright prints a class by the plain name the source used. `climlab/domain/xarray.py`
contains `from xarray import Dataset`, so its probes come back saying `Dataset`, and the
old code only recognised the spelled-out `xarray.Dataset`. The extractor already builds,
for other purposes, a list of what each module imported and from where; that list is now
handed to the classifier, so a bare name is resolved to where it came from before being
judged.

```
"Dataset" in a module that did `from xarray import Dataset`  -> xarray
"Dataset" in a module that defines its own                   -> object
```

**On climlab this changed two answers and no rows in the artifact.** The two are local
variables in `climlab/domain/xarray.py` that the extractor does not turn into objects at
all, for reasons unrelated to their family (§7 below). The fix is real and the unit tests
hold it; climlab simply does not exercise it anywhere that reaches the output. That is
the same situation §1.7 is in.

### Slice 4 — put probes in front of statements nothing follows

When the statement containing the expression is a `return`, `raise`, `break` or
`continue`, the probe now goes in front of it. One guard: if the statement shares its
line with something else — `if x: return y` — the probe is dropped rather than inserted
somewhere that would not parse.

Probes answered went **3,183 → 3,318**, and answers Pyright could type **1,761 → 1,857**.

## 4. The numbers

Measured on climlab (72 files), same drivers and same runtime trace as Steps 1a, 1b and
4a. The baseline was re-run first and reproduced Step 4a exactly — 1,868 objects, 4,239
access edges, recall 69.0%, alias precision 93.6%, 17 contradictions — before anything
was changed.

### What Pyright was able to tell us

| | before | after |
| --- | ---: | ---: |
| probes emitted | 3,338 | 3,338 |
| probes answered | 3,183 | **3,318** |
| answers Pyright could type | 1,648 | **1,857** |
| answers that were the bare word `Unknown` | 1,535 | 1,461 |

And what those answers were turned into:

| family | before | after |
| --- | ---: | ---: |
| unknown | 2,220 | **1,637** |
| object | 779 | 1,167 |
| dict | 34 | **263** |
| field | 114 | **216** |
| list | 34 | 30 |
| set | 0 | 2 |
| xarray | 0 | 2 |
| dataframe | 1 | 1 |
| path | 1 | 0 |

### The artifact

| | baseline | slice 0 | slice 1 | slices 2+3 | slice 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| objects | 1,868 | 1,868 | 1,850 | 1,935 | **1,934** |
| access edges | 4,239 | 4,239 | 4,239 | 4,239 | **4,239** |
| lineage edges | 1,169 | 1,169 | 1,169 | 1,171 | 1,171 |
| **objects with an unknown family** | **759 (40.6%)** | 759 | 745 | 558 | **556 (28.7%)** |
| dict | 110 | 110 | 124 | 294 | **296** |
| field | 8 | 8 | 8 | 15 | **15** |
| anonymous `container_field` objects | 141 | 141 | 125 | 28 | **25** |
| named `dict_key` objects | 69 | 69 | 85 | 184 | **187** |

Access edges did not move once, at any slice. That is the check that this step changed
how accesses are *described* without adding or removing any.

### Whether the claims are true

| | baseline | after |
| --- | ---: | ---: |
| Step 1a — access recall | 69.0% | **69.0%** |
| Step 1a — confirmed static claims | 75.2% | **75.2%** |
| Step 1a — falsified static claims | 1 | **1** |
| Step 1b — objects with a readable site | 76.3% | **80.4%** |
| Step 1b — confirmed identity claims | 250 | **271** |
| Step 1b — contradicted | 17 | **20** |
| Step 1b — alias precision | 93.6% | 93.1% |
| Step 1b — alias recall | 64.3% | 64.2% |
| determinism, five shuffled seeds | byte-identical | **byte-identical** |
| test suite | 345 passing | **354 passing**, 9 of them new |

The one pre-existing failure (`test_cluster_structural_graph.py::
test_parse_args_uses_structural_config_and_cli_overrides`) fails identically without
these changes.

### Access recall did not move at all, and that is the interesting result

`code_review.md` §7 expected "the share of objects with `inferred_type == unknown` falls,
**and per-kind recall rises with it**." The first half happened; the second did not, and
the reason is worth writing down.

What the access oracle scores is a triple: which function, which attribute or key, and
what kind of access. **It never mentions which object was touched.** So changing what an
object is called, or which bucket it is filed under, cannot change that score — the same
property that kept the number still through Step 2 and Step 4a.

What *did* change is where the confirmed accesses are filed:

| filed as | confirmed accesses, before | after |
| --- | ---: | ---: |
| named dictionary key | 36 | **226** |
| anonymous container field | 167 | **29** |
| class attribute | 653 | 844 |
| whole-class state | 460 | 259 |
| everything else | 1,808 | 1,766 |
| **total** | **3,124** | **3,124** |

The total is identical to the row. **The same accesses, correctly described.** Six times
as many are now recorded as a named key in a dictionary rather than as an anonymous
something, which is exactly what the family is for — and it is invisible to a score that
was designed not to depend on it.

## 5. The hand-check

Three identity claims that the running program contradicts are new, so all twenty were
read against the source before the number was recorded, as both instruments require.

The seventeen from Step 4a are unchanged and unchanged in kind. The three new ones are:

```python
# climlab/model/ebm.py, lines 265, 267, 273
lw   = AplusBT(state=self.state, ...)
alb  = albedo.StepFunctionAlbedo(state=self.state, ...)
diff = MeridionalHeatDiffusion(state=self.state, ...)
```

All three are the same claim: that the sub-process's state **is** the model's state. It
is not — the constructor builds a new dictionary from it. This is exactly the defect
Step 4a found once already, as `Iceline(state=self.state)`, and assigned to Step 7,
whose job is to derive the shared-state rule from evidence instead of matching on the
word `state`.

They are **not new defects; they are the existing one becoming visible.** Before this
step, `EBM`'s state was modelled as one node for the whole class, so the claim had
nowhere to land. Now that the families are known, `EBM` is modelled with a separate node
per attribute, `class_attr_state:...EBM:state` exists, and the claim can be scored — and
is refuted. Checked directly: that node is absent from the baseline artifact and present
afterwards.

So of the twenty, **none is this step's defect**: seventeen are Step 4a's already
hand-checked survivors, and three are Step 7's, newly countable.

### The reclassification was also checked by hand, and that found a mistake

626 of the 3,183 answers are classified differently by the new code than the old. Rather
than trust the totals, each group was read:

| | count | verdict |
| --- | ---: | --- |
| unknown → dict / field / list / set | 331 | the fix |
| unknown → object | 270 | function types, mostly numpy's overloads |
| list / field / path / dict → object | 23 | function signatures, and `type[...]` and `tuple[...]` wrappers — **all wrong before** |
| object → xarray | 2 | slice 3 |

The 23 look like losses in the totals and are not: `type[ndarray[...]]` is the array
*class*, not an array, and `tuple[ndarray[...], ...]` is a tuple of arrays.

One of them was a genuine mistake. Pyright writes a method's own class as `Self@AttrDict`,
and the new code read the head of that as `Self@AttrDict` rather than as `AttrDict`, so
an AttrDict stopped being recognised as a dictionary. Fixed, and the fix has a test. It
would not have shown up in any total; only reading the list found it.

## 6. What was deliberately left alone

- **Scalars.** 450 of the probes come back `float`, `int`, `bool` or `str`, and all of
  them are filed as `object`, which the extractor still treats as something that could
  hold fields. A number is not a container. Fixing this would *remove* objects and
  access edges at the same moment this step's fix was creating them, and the access
  recall figure would have moved for two reasons at once with no way to separate them.
  It is written up as a new finding in `code_review.md` §1.15.
- **Unions whose members genuinely disagree.** 91 answers on climlab are of the form
  `generic[Any] | complex | str | bytes | memoryview[int] | ndarray[...] | Unknown`.
  These stay `unknown`. A value that is a string on one branch and an array on another
  has no one family, and guessing would be the same mistake in a new place.
- **§2.7, the `state=` keyword gate.** Still Step 7's, and now with three more measured
  consequences.
- **Choosing a threshold for "enough probes resolved".** The check added below refuses a
  run only when *nothing at all* resolved. Where to draw a line short of that is a
  property of the project being analysed, so the counts are printed and not judged — the
  "derive thresholds, do not choose them" rule from the roadmap.

## 7. What this does not fix

- **A probe can know a local variable's family and the extractor still not model it.**
  This is why slice 3 moved nothing: `ds = Dataset()` is now known to be an xarray
  dataset, but `ds` never becomes an object, because whether a local is worth modelling
  is decided from the assignment's own shape and not from what Pyright knows about it.
  Recorded in `code_review.md` §1.16.
- **The 1,461 answers that are still the bare word `Unknown`.** Some are genuinely
  untyped code; how many is not known. The counts are now printed at every run, so this
  can be watched rather than guessed at.
- **20 probes that still get no answer**, which are the `if x: return y` cases slice 4
  deliberately drops.

## 8. Corrections to `code_review.md`

- **§7's Step 3 acceptance criterion is half wrong.** "Per-kind recall rises with it"
  cannot happen: the access oracle's unit does not mention the object, by design. What
  container families change is which kind of object the confirmed accesses are filed
  under, and that has to be read as a redistribution, not as a gain. §4 above.
- **§2.1's account of the sandbox is right about the cause and understates the size of
  the hole.** It says a module with no probe target "never reaches the sandbox", which is
  exactly what happens; what it does not say is that the missing files are almost
  entirely package `__init__.py` files, so the damage is concentrated on precisely the
  files that imports pass through.
- **§2.2 is right in mechanism but did not fire on climlab.** Adding the environment
  settings changed nothing measurable, because the temporary copy is already the import
  root and the ambient interpreter already had numpy. The finding stands as a
  correctness point — nothing should depend on that luck — but it is not a source of
  lost families here.
- **§1.1's own example table can be read as understating the problem.** It lists cases
  that return `unknown`; the more useful framing is that the two rules are independent
  and either alone is enough to refuse, so the better a project's annotations are, the
  more of them are dropped.

## 9. What changed in the code

### Edited

| file | what |
| --- | --- |
| `data_access/pyright_type_probe.py` | the classifier rewritten around the outermost name and top-level unions (`_split_top_level_union`, `_type_head`, `_is_callable_type_text`, `FAMILY_BY_TYPE_HEAD`); the project name removed; the sandbox copies the analysed files; `extraPaths`/`venvPath`/`--pythonpath`; `PyrightProbeReport` with the resolution counts; `before_line` probe placement |
| `data_access/generate_data_access_ast.py` | `probe_insert_position` replaces `probe_insert_lineno` and picks the placement; the import map is built per module and passed down; the import roots are derived per file; `_report_pyright_resolution` prints the counts and refuses a run that resolved nothing |
| `data_access/outputs.py` | the report now states the share of objects whose family is unknown, which is this step's acceptance criterion and previously had nowhere to be read |
| `tests/test_pyright_type_probe.py` | 9 tests |

### The nine tests

Each pins something that would otherwise fail silently: a nesting is read by its
outermost name; a union drops its uninformative members and refuses when the rest
disagree; a function is never a container, and neither is a class object or a tuple of
arrays; `Self@AttrDict` is still an AttrDict; a bare imported name is xarray's only in a
module that imported it from xarray; a probe for an expression in a `return` goes in
front of it; a probe that would land in the middle of `if x: return y` is dropped; a
package `__init__.py` with no probe of its own still reaches the sandbox; and a run that
resolves nothing raises instead of reporting unknown families as though they were an
answer.

### Verified unchanged

- `check-data-access-determinism` on climlab: byte-identical over five seeds.
- Step 1a re-scored at every slice: recall 69.0%, confirmed 75.2%, falsified 1.
- Access edges: 4,239 at every slice.
- Slice 0 alone: byte-identical artifacts.

## 10. What is next

Step 4's remaining items, unchanged: §1.6 (a method call whose name ends in `.set` is
treated as building a set), §1.7 (files keyed on the text of the expression that opened
them), §1.8 (class bodies have no scope), §1.9 (a lambda's own parameters resolve to the
enclosing function's variables), §1.10 (`self.index` and `self.columns` are invisible on
every object), §1.11 (a dynamic `getattr` claims a link to every function in the module),
§1.12 (one unparseable file stops the whole run), §1.14 (default arguments credited to
the function instead of to the module that evaluates them), and §5.5 (lambdas are
functions in the call graph and nothing here).

§1.6 is worth doing early for a reason this step supplies: it is the *other* place a
family is decided, from the shape of the call rather than from Pyright, and it is now
the less careful of the two.
