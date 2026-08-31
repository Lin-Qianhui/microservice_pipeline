# Step 4a — Telling "made from" apart from "is"

This is the fifth step of the plan in [`code_review.md`](../code_review.md) §7, after
Step 0, Step 1a, Step 2 and Step 1b. It is the first item of Step 4, taken ahead of
Step 3 on purpose — the reason is in §2.

---

## 1. What was wrong

The extractor does not only record which data each function touches. It also records
that two different names hold **the same piece of data**. Those claims live in the
`alias_of` column and in the lineage graph.

They were often wrong in one particular way. When a value was produced by a call and
given a name, the extractor frequently wrote down that the new name **is** one of that
call's arguments. It is not. It is a new thing, built out of that argument.

Three lines from climlab, all checked against the source:

```python
dic = self.state.copy()                    # a copy is a new dictionary
self.timeave = self.state.copy()           # so timeave is not state
Tatm = self._climlab_to_cam3(self.Tatm)    # a reshaped new array
```

In every case the extractor said the thing on the left was the thing on the right.

### Where the claim is written, and why the third example is the worst

There are two places the extractor can say "these two are one thing", and they are not
equally visible.

- The **`alias_of` column** on an object row: "this object *is* that object."
- A **lineage edge** carrying one of the identity relations: the same claim, as an edge.

The first two examples used the column, and read exactly as you would fear:

```
local_exposed:...Process.to_xarray:dic     alias_of = class_attr_state:...Process:state
class_attr_state:...TimeDependentProcess:timeave
                                           alias_of = class_attr_state:...:state
```

So `dic` *is* the process's live state — even though the next line is
`dic.update(self.timeave)`, which does not touch the real state — and the class's
time-averaged history *is* the class's live state, when keeping those two apart is
roughly the point of the class.

The third example did something worse, through the edges instead. The extractor models
a function's parameter as **one node for the whole program**, not one per call. So when
it decided that `_climlab_to_cam3` hands back its own parameter, every call site said
"what I got back is that one node". In `CAM3._prepare_arguments` that is 18 call sites:

```
                          self.Tatm ─┐
                          self.Ts ───┤
                                     ▼
             ┌───────  param:...CAM3._climlab_to_cam3:field  ───────┐
             │           one node, shared by all 18 calls            │
        ┌────┴───┬────────┬────────┬──── … ────┬─────────────────────┘
        ▼        ▼        ▼        ▼           ▼
      Tatm      Ts     coszen    eccf     …14 more…      r_ice
```

Temperature, surface temperature, solar zenith angle, albedos, pressures, humidity,
ozone, cloud fraction, cloud water, droplet radii — eighteen unrelated quantities, each
recorded as being the same object as the others, because they were each passed through
the same reshaping helper.

Step 1b built the instrument that could see this. It runs the program and looks at what
each name actually holds. Its verdict on climlab: of the extractor's identity claims it
could check, **51 were contradicted by the running program**, and about 46 of those were
this one mistake.

## 2. Why this mattered more than it looks, and why it went first

A wrong "same thing" is not just a slightly-too-heavy line in the graph. Two of the
relations that carry these claims — `local_assign` and `state_assign` — are read
downstream as **must-link** constraints: the two nodes are put in the same cluster
whatever the rest of the evidence says
([`cluster_structural_graph.py`](../../cluster_structural_graph.py),
[`leiden_reweighted.py`](../../structural_dependency_graph/clustering/leiden_reweighted.py)).
So each false claim was gluing two unrelated parts of the program together, by force,
in the output this whole pipeline exists to produce. That is not recorded anywhere in
`code_review.md`; it was found by reading the clustering code while planning this step.

Concretely, `build_must_link_groups` runs a union-find over those relations and replaces
each connected group with a **single node** before clustering begins. Not a heavier
edge — one node, which the clusterer cannot take apart because it never sees the pieces.
Counting those groups on the artifacts before and after:

| | before | after |
| --- | ---: | ---: |
| forced-merge groups | 164 | 175 |
| **largest group** | **72 nodes** | **46 nodes** |
| nodes swallowed into a group | 582 | 533 |

The 72-node group spanned **four modules** — `rrtmg_lw`, `rrtmg_sw`,
`emanuel_convection` and `utils` — fusing ozone, CO₂, methane, cloud water, temperature,
pressure and both wind components, all because they passed through `_climlab_to_rrtm`.
No clustering of that input could have separated those subsystems; they were not
separate nodes to separate. The largest group left afterwards is 46 and sits entirely
inside one module.

### One thing the older behaviour got right, for the wrong reason

Not every false claim reached the `alias_of` column, and it is worth knowing why,
because it shaped the fix.

A final pass fills the column by following the "came from" arrows backwards until they
stop. The stopping points are **roots**. One root means an unambiguous answer, which is
recorded; more than one root means "this could be several things", and the pass refuses
and leaves the column empty.

Walking back from the local `Tatm` above: it came from `param:field`, and `param:field`
came from whatever was passed in. Only two of those eighteen arguments were plain enough
to name a modelled object — `self.Tatm` and `self.Ts`; the rest are expressions such as
`self.coszen * np.ones_like(self.Ts)`. So the walk ends at **two** roots, and the column
is left empty.

That is the right answer for the wrong reason. The rule was not saying "a reshaped array
is not the array it came from" — it has no notion of that. It was saying "I cannot tell
whether this came from `self.Tatm` or `self.Ts`". Had the helper been called with a
single resolvable argument, there would have been one root and the column would have
been filled in confidently and wrongly. And the protection never extended past the
column: all eighteen edges were written regardless, and the edges are what the
clustering unions.

This is why §4's replacement rule requires the two readings of the graph to *agree*
rather than simply ignoring `derived_from` edges. Ignoring them would have removed the
ambiguity that was accidentally protecting cases like this one, turning empty columns
into confident wrong answers.

The plan's order was `3 → 4`, so this step happens out of order. That was deliberate,
for a reason that only holds right now: **Step 3 would have spoiled the measurement.**
Step 3 changes how container families are decided, which changes the names objects are
filed under, which is exactly what the Step 1b baseline is made of. Doing Step 3 first
would have moved the identity numbers for reasons that had nothing to do with false
merges, and this fix could no longer have been credited with anything. Done now, it is
judged against a baseline recorded two days earlier and reproduced exactly, to the last
number, before a line was changed.

## 3. The two causes

**(a) A copy was written down as an alias, on purpose.** The code had a rule saying
"`x.copy()` gives you `x`". A copy is the one call in the language whose entire purpose
is to give you something that is *not* `x`. Small, and unambiguous.

**(b) One way out of a function spoke for all of them.** The extractor keeps a single
note per function saying what that function returns. `_climlab_to_cam3` begins:

```python
def _climlab_to_cam3(self, field):
    if np.isscalar(field):
        return field       # this one branch returns the argument
    ...                    # every other branch builds a new array
```

That one branch made the whole function "returns its own argument", and all eighteen of
its call sites inherited the claim. This is the larger of the two.

## 4. The one idea behind the fix

A lineage edge used to mean only one thing: *these two names are the same object*. Now
it can mean either of two things, and it says which:

| kind | meaning | may create an alias? | may force a cluster merge? |
| --- | --- | --- | --- |
| the six existing relations | the two names hold the same object | yes | yes |
| the new `derived_from` | this value was **made from** that one | no | no |

Everything else follows from that one distinction.

**The edge is kept, not deleted.** This is the part that is easy to get wrong. Inside
the extractor, whether a local variable becomes a real object at all was decided by
whether it had an alias. Simply removing the false alias would therefore have deleted
the object too, and every access recorded on it — so the Step 1a recall figure would
have fallen while the Step 1b precision figure rose, and neither number would have been
readable against the other. Keeping a `derived_from` edge keeps the object, keeps its
accesses, and keeps the fact that the data flowed, while withholding only the claim that
the two are one thing. The numbers in §6 show this worked: not one access edge moved.

**Deciding whether a function returns one thing.** Answered from the shape of the code
alone, with no analysis: every `return` in it carries a value, they are all the same
expression, and control cannot fall off the end. Generators are excluded — calling one
gives you a generator, never the value in its `return`.

Reading the *text* rather than working out what each `return` resolves to was a choice.
It is settled the first time the file is read, so it needs no place in the repeated
passes the extractor makes (where an earlier revision of this package learned the hard
way that a fact left out of the "have we finished?" test makes finishing meaningless).
It cannot flip back and forth. And when it is wrong it is wrong in the safe direction:
`return d` and `return self.data` naming one object read as two, which gives up a true
claim instead of asserting a false one.

**Falling off the end counts as a second way out.** `code_review.md` §7 says listing a
function's `return` statements is enough to tell "returns its argument sometimes" from
"returns its argument". That is not quite true:

```python
def maybe(field):
    if field:
        return field
    # and here it returns None
```

There is one `return`, and it returns the argument, and the function still does not
return the argument on every path. So the check also asks whether the body always
returns.

**One more rule, or the whole fix would have been silently undone.** After the code is
read, a separate pass rebuilds `alias_of` by following the lineage graph back to a
single source. Left alone it would have followed the new `derived_from` edges straight
back to the source and put every merge back. But simply *ignoring* those edges would
have been wrong in the opposite direction: they also carry the ambiguity that makes an
alias unsafe, and hiding them would have handed out aliases the old, blunter rule
refused. So the pass now reads the graph twice — once with everything, once with
identity edges only — and records an alias only when both readings agree on the same
single source. That can only ever take an alias away, never invent one, which is what
makes the direction of every number below unambiguous.

The same filter was applied to the shared-container inference, which walks the same
graph with its own copy of the same code. `code_review.md` §4.3 predicted that the two
copies would drift; they had already drifted once, over a caching bug in Step 2.

## 5. What was deliberately given up

The check is conservative, and the cost is measurable. **25 claims the program had
confirmed were withdrawn along with the wrong ones**, and 22 of those 25 are a single
function, `_climlab_to_rrtm`:

```python
def _climlab_to_rrtm(field, ...):
    try:
        field = field[..., ::-1]      # the parameter is given a new value here
    except:
        if np.isscalar(field):
            return field              # one way out
    ...
    modfield = field                  # ... on one branch out of two
    return modfield                   # the other way out
```

Two ways out, textually different, so the check says no. On the runs that were watched,
the second one really did hand back the object then held by `field`, so the instrument
had confirmed those claims. Withdrawing 22 correct claims to remove 27 wrong ones from
*the same function* is the trade this step accepts on purpose: a wrong merge is a hard
constraint on the output, and a missing one is only a missing line.

A smaller loss worth naming: `_standardize_inputs` in `insolation.py` returns a
seven-item tuple from two different branches, and six of the seven items are the same in
both. Because the check compares whole `return` expressions rather than tuple positions,
all seven lose their claim. Comparing position by position would recover the six. It was
not done: it is a second gate for seven of the 123 claims that moved.

The number that watches all of this is **alias recall**, and it fell from 65.9% to
64.3%. That is the price, and it is stated rather than buried.

## 6. The numbers

Measured on climlab (72 files), same drivers as Steps 1a and 1b. The baseline was
re-run first and reproduced Step 1b exactly — 279 confirmed, 51 contradicted, 84.5%,
65.9% — before anything was changed.

### The artifact

| | before | copies fixed | both fixed |
| --- | ---: | ---: | ---: |
| objects | 1,866 | 1,866 | **1,868** |
| access edges | 4,239 | 4,239 | **4,239** |
| lineage edges | 1,175 | 1,173 | 1,169 |
| objects with an `alias_of` | 422 | 420 | 420 |
| edges saying "made from" | 0 | 3 | **126** |

The two new objects are the point rather than an accident. In
`_prepare_general_arguments`, `tlay` was previously filed as *being* the parameter of
the function that built it, so `tlay.shape` was recorded against that parameter. `tlay`
is now its own thing and its `shape` belongs to it.

### Whether the claims are true

| | before | copies fixed | both fixed |
| --- | ---: | ---: | ---: |
| claims checked | 1,597 | 1,590 | 1,463 |
| confirmed | 279 | 277 | 250 |
| **contradicted** | **51** | **46** | **17** |
| unobserved | 690 | 690 | 682 |
| **alias precision** | **84.5%** | 85.8% | **93.6%** |
| alias recall | 65.9% | 65.9% | 64.3% |

By claim source — read the `local_assign` row, which is the one Step 1b singled out as
"the claim the extractor gets wrong":

| source | before | after |
| --- | ---: | ---: |
| `local_assign` — a value assigned to a name | 78 right, **31 wrong** | 54 right, **4 wrong** |
| `arg_to_param` — an argument becomes a parameter | 106 right, 6 wrong | 102 right, 4 wrong |
| `alias_of` — the derived column | 51 right, 10 wrong | 51 right, 8 wrong |
| `state_assign` | 8 right, 3 wrong | 8 right, 1 wrong |
| `tuple_unpack` | 36 right, 1 wrong | 35 right, **0 wrong** |

126 edges are now excluded from scoring by name, printed in the report as
`relation derived_from is not an identity claim` — so nothing was hidden, it was
reclassified in the open.

### Everything else held still

| | |
| --- | --- |
| Step 1a access recall | **69.0%**, unchanged |
| Step 1a confirmed claims | **75.2%**, unchanged |
| Step 1a falsified claims | **1**, unchanged |
| `check-data-access-determinism`, five shuffled seeds | byte-identical, **DETERMINISTIC** |
| test suite | 345 passing, 7 of them new |

The Step 1a numbers holding still is the specific check that this step did not pay for
its precision with somebody else's recall. The one pre-existing test failure
(`test_cluster_structural_graph.py::test_parse_args_uses_structural_config_and_cli_overrides`)
fails identically without these changes; the only edits to that file are comments.

## 7. The hand-check

All 17 surviving contradictions were read against the source before the number was
recorded, as both earlier instruments require. They are five underlying facts, each
showing up once as an `alias_of` row and once as the lineage row it came from.

| what it is | rows | verdict |
| --- | ---: | --- |
| the claim is true, on a branch the drivers never ran (`domain.py` 533/542/597, `akmaev_adjustment.py` 58) | 10 | the instrument's limit, not a defect — Step 1b §6.3 |
| `super(MutableAttr, self).__setattr__` resolved to the wrong method | 2 | a real defect, and Step 5's, not this one's |
| `Iceline(state=self.state, ...)` — a constructor builds a new dictionary | 2 | a real merge, from the shared-state path; Step 7 |
| an argument being passed, never seen at both ends at the same moment (`advection_diffusion.py` 178, and the tridiagonal one) | 3 | structurally right; the instrument could not confirm it |

So of the 17, **none is the defect this step set out to fix**, two belong to Step 5, two
to Step 7, and thirteen are the instrument reaching its limits rather than the extractor
being wrong.

`akmaev_adjustment.py` is worth spelling out, because it is the clearest example of why
"contradicted" is a lead and not a proof. The claim is that
`Akmaev_adjustment_multidim`'s local `theta` is `Akmaev_adjustment`'s parameter `theta`.
Reading the source, `Akmaev_adjustment` has one `return theta` and never rebinds
`theta`, so the claim is **correct**. The drivers only ever ran multi-column models,
which take the other branch, so the two names were never seen holding one object.

## 8. Corrections to `code_review.md` §7

- **It is two defects, not one.** They are different sizes, they carry different risks,
  and landing them together would have made the movement unattributable. Measured
  separately: copies 5 contradictions, return paths 29.
- **"Alias only when every return path returns the same source" is incomplete.** A
  function that falls off the end returns `None` on that path, and no amount of listing
  `return` statements will show it. §4 above.
- **The alias is made in three places, and one of them undoes the other two.** §7's Step
  4 note describes the fix as changes to where the claim is *written*. Without also
  changing the pass that *rebuilds* `alias_of` from the lineage graph afterwards, both
  fixes would have measured as doing nothing at all.

One correction to Step 1b's own §10, which called cause (a) "four lines and the cheapest
fix in Step 4". The change to the copy rule is indeed four lines. Making those four
lines have any effect on the artifact took the whole of §4's vocabulary change first.

## 9. What this does not fix

- **Branch blindness itself** (`code_review.md` §4.4). This step works around it for one
  question — "does every way out return the same thing?" — by counting `return`
  statements. Everything else about the missing branch model is untouched, and it is
  still the one finding in the review with no step of its own.
- **§6.2's mis-resolved `super()` call**, which is Step 5's cross-stage work.
- **The shared-state constructor merge** (`Iceline(state=self.state)`), which comes
  through the registration path and needs Step 7's derived shared-state gate.
- **Tuple returns, position by position.** §5.
- **Function bodies that only leave through a `return` inside `while True:`.** Treated as
  falling off the end, so their claims are withheld. Conservative, and not seen on
  climlab.

## 10. What changed in the code

### Edited

| file | what |
| --- | --- |
| `data_access/models.py` | `IDENTITY_RELATIONS` and the relation names, in one place; `ValueOrigin`; `ExprRef.derived` |
| `data_access/generate_data_access_ast.py` | the copy rule, the return check (`_returns_one_source`, `_always_returns`), `derived_from` at the five sites that record lineage, and the two-reading alias rule |
| `data_access/infer_shared_containers.py` | the same filter on its own copy of the root walk |
| `data_access/identity_comparison.py` | takes `IDENTITY_RELATIONS` from `models` instead of listing it again |
| `data_access/ontology.md` | section 4 rewritten: two kinds of relation, and 4.7 for `derived_from` |
| `configs/weight_profiles/default.json` | a `derived_from` weight, at today's fallback value so nothing moves by accident |
| `cluster_structural_graph.py`, `clustering/leiden_reweighted.py` | comments only — both must-link sets are allowlists and already ignore an unknown name |
| `tests/test_data_access_ast.py` | 7 tests |

### The seven tests

Each pins something that would otherwise fail quietly: a copy is `derived_from` and
keeps its object *and its access edges*; a two-branch function gives its caller no
alias; a one-`return` function still does; falling off the end counts as a second path;
a generator never qualifies; the alias rule ignores a `derived_from` edge; and — the
other direction — a `derived_from` edge still erases an alias that has become ambiguous.

### Verified unchanged

- `check-data-access-determinism` on climlab: byte-identical over five seeds.
- Step 1a re-scored: recall 69.0%, confirmed 75.2%, falsified 1 — exactly as recorded.
- The vocabulary change was landed on its own first, with nothing yet emitting
  `derived_from`, and the artifacts came out **byte-identical**. That is the evidence
  that rewriting the alias rule changed no behaviour by itself.

## 11. One note for whoever runs this next

The runtime trace must be started from the analysed project's own directory. The pytest
driver is given paths relative to the working directory, and run from anywhere else it
fails with a usage error, silently falls back to the notebooks alone, and produces a
baseline roughly half the size — 80,177 identity observations instead of 148,511. The
run does say `driver issue: pytest exited with code 4`, in one line among twenty. Worth
either fixing in the tracer or repeating here.

## 12. What is next

Step 4's remaining items are untouched: §1.6 (unanchored suffix matching), §1.7 (file
objects keyed on source text), §1.8 (class bodies have no scope), §1.9 (lambda
parameters), §1.10 (`self.index`), §1.11 (dynamic `getattr` fan-out), §1.12 (one bad
file kills the run), §1.14 (default arguments attributed to the callee, which Step 1a
found and which its oracle will confirm directly), and §5.5 (lambdas as callables).

Step 3 is now clear to go, and its effect on the identity numbers will be readable,
because 93.6% is a baseline that was hand-checked rather than assumed.
