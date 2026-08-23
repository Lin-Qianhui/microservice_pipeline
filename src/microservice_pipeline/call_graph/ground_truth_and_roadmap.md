# Ground truth, current state, and what's left to do

This document answers one question that turned out to matter more than expected:

> **Is the runtime call graph the right thing to measure the static call graph against?**

Short answer: it is the right oracle for *one* of our two goals, and the wrong
one for the other. Confusing the two leads to optimizing the wrong number. The
rest of this document explains where things stand, why that distinction matters,
and what to build next.

**Reference codebase throughout:** `/Users/qianhuilin/Desktop/Envision/climlab`
(the [climlab](https://github.com/climlab/climlab) process-oriented climate
modelling package, installed editable into this project's `.venv`).

---

## 1. Where things stand

The static extractor now resolves nearly everything climlab really executes.

| Measure | Before | After the four fixes | Now (2026-08-02) |
| --- | ---: | ---: | ---: |
| Runtime edges found (recall) | 70.8% | 98.7% | **99.7%** |
| Runtime edges missed | 110 | 5 | **1** |
| Unconfirmed static edges | — | 168 | **53** |
| Falsified static edges | not measurable | not measurable | **5** |
| Static edges produced | 805 | 1141 | 1144 |
| `registered_invoke` (was a framework hardcode; now derived) | 4 | 21 | 21 |

Against the boundary oracle, which is the metric that actually matters (§2):

| `evaluate` | Before | Now |
| --- | ---: | ---: |
| ARI | 0.281 | **0.337** |
| V-measure | 0.569 | **0.607** |
| Pairwise F1 | 0.339 | **0.395** |
| BCubed F1 | 0.429 | **0.460** |

All five remaining falsified edges are *labelled* over-approximation
(`virtual_override` ×4, `self_method` ×1), which §6 says is acceptable. The two
that were not — unlabelled `super_method` inventions at full weight — are gone;
see Step 2.

Four fixes got us to the middle column:

1. **Re-export aliases.** A base class imported through a package `__init__.py`
   was recorded under the re-export path, which no resolver recognized, so every
   inherited method call on that class vanished. Worth 55 of the 110 missing
   edges on its own.
2. **Downward virtual dispatch.** `self.hook()` written in a base class really
   dispatches to subclass overrides. Emitted as a distinct `virtual_override`
   relation. Worth 49 edges.
3. **A container dimension.** Class attributes and parameters now track what a
   collection *holds*, not just what it is — plus attribute lookups walk the
   inheritance chain, and nested containers are flattened.
4. **Type annotations.** Read for parameters and annotated assignments.

Current edge mix on climlab:

```
362 self_method       107 direct            73 constructor      21 registered_invoke
184 import             95 imported          69 super_method      9 inferred_type
131 property_getter    88 virtual_override                       4 dunder_*
                                                                 1 cooperative_super
```

### The measuring instrument

`trace-runtime` runs climlab under `sys.monitoring` (PEP 669, Python 3.12+) and
records the calls the interpreter actually dispatches. `compare-graphs` diffs
that against `edges.csv`.

```bash
cd /Users/qianhuilin/Desktop/Envision/climlab
microservice-pipeline call-graph      --config configs/microservice_pipeline/extraction.jsonc
microservice-pipeline trace-runtime   --config configs/microservice_pipeline/extraction.jsonc
microservice-pipeline compare-graphs  --config configs/microservice_pipeline/extraction.jsonc
```

Drivers are climlab's own `pytest -m fast` suite plus three courseware
notebooks, all run in-process (a subprocess would be invisible to
`sys.monitoring`).

---

## 2. Three different graphs, and which one we actually want

The confusion comes from treating "call graph" as one thing. There are three.

### Graph A — what the source code says (the static graph)

What our extractor produces. Covers the whole codebase, including code no test
ever runs.

### Graph B — what the interpreter does (the runtime graph)

What `sys.monitoring` records. **This is a good oracle for one specific question:
is our call-resolution machinery correct?** If Python dispatched `A → B` and we
did not infer it, we have a real bug. That is how we found the re-export and
virtual-dispatch failures, and it is why recall went from 70.8% to 98.7%.

But Graph B has three hard limits:

- **It is a lower bound.** It only sees what the drivers exercised — 258 of 486
  callables, roughly half the codebase. Code with no test and no notebook is
  simply invisible.
- **It cannot express some real dependencies.** Module bodies are executed by the
  import machinery in C, so no `import` edge ever appears. Dunder methods
  implemented in C (`dict.__setitem__`) are equally invisible.
- **It is full of framework plumbing.** More on this below — it is the important
  one.

### Graph C — what depends on what (the coupling graph)

This is what the pipeline is actually for. We are not trying to reproduce
Python's call sequence; we are trying to decide **which pieces of code could be
separated into different services.** That is a question about *dependency between
units of domain logic*, and it is not the same graph as B.

**Graph B and Graph C genuinely disagree on climlab.** Here is the case that
makes it concrete.

#### The registry-coupling example

`climlab/surface/albedo.py:344-358` wires a model together:

```python
class StepFunctionAlbedo(...):
    def __init__(self, Tf=-10., a0=0.3, a2=0.078, ai=0.62, **kwargs):
        ...
        self.add_subprocess('iceline', Iceline(Tf=Tf, state=self.state, timestep=self.timestep))
        warm = P2Albedo(a0=a0, a2=a2, domains=sfc, timestep=self.timestep)
        cold = ConstantAlbedo(albedo=ai, domains=sfc, timestep=self.timestep)
        self.add_subprocess('warm_albedo', warm)
        self.add_subprocess('cold_albedo', cold)
```

At runtime, the parent's computation reaches the children like this
(`climlab/process/time_dependent_process.py`):

```
EBM.step_forward()
  └─ TimeDependentProcess.compute()               # line 165 — base class
       ├─ _compute_type('explicit')               # line 256
       │    └─ for proc in self.process_types[t]:
       │         └─ proc.compute()                # line 272 — recurses on the CHILD
       │              └─ StepFunctionAlbedo._compute()
       └─ self._compute()                         # line 241 — the PARENT's own science
            └─ EBM._compute()
```

So `EBM._compute` and `StepFunctionAlbedo._compute` are **siblings**. Both are
called by the same base-class driver at different levels of its recursion.
Neither calls the other, and they are never in a caller/callee relation on the
stack. The trace confirms it: it contains `TimeDependentProcess.compute →
EBM._compute` and `TimeDependentProcess.compute → StepFunctionAlbedo._compute`,
and never an edge between the two.

Now compare what each graph offers the clustering step:

**Graph B (runtime) gives a hub — 20 edges:**

```
TimeDependentProcess.compute → EBM._compute
TimeDependentProcess.compute → StepFunctionAlbedo._compute
TimeDependentProcess.compute → Iceline._compute
TimeDependentProcess.compute → P2Albedo._compute            ... ×20
```

Every concrete process hangs off **one shared node**. This says all 20 classes
are equally close to a single dispatcher — almost no boundary information. Worse,
it is exactly the topology our hub policy exists to neutralize, and
`TimeDependentProcess.compute` **is** in fact flagged as one of the 27 callable
hubs in
`artifacts/structural_dependency_graph/callable_hub_nodes.csv`. Once hub policy
isolates it, all 20 edges' structural signal is discarded and the process tree
looks disconnected again.

**Graph C (coupling) wants pairs — what `registered_invoke` emits, 21 edges:**

```
EBM._compute                → StepFunctionAlbedo._compute
StepFunctionAlbedo._compute → Iceline._compute
StepFunctionAlbedo._compute → P2Albedo._compute
StepFunctionAlbedo._compute → ConstantAlbedo._compute
```

This encodes the **tree structure** — which parent owns which children. That is
precisely the signal clustering needs: the albedo subtree groups together, and
separately from radiation.

**Conclusion:** these 21 edges score **zero** on recall (they never happen at
runtime) while being *more* valuable than the hub edges they stand in for. Recall
is therefore not a valid acceptance test for this part of the work.

### The oracle for Graph C already exists

`/Users/qianhuilin/Desktop/Envision/climlab/configs/microservice_pipeline/manual_mapping_labeled.csv`
— 1061 hand-labelled nodes across 11 buckets:

```
S1  Process Orchestration & Time-Integration Engine
S2  Radiation Service
S3  Insolation & Orbital Service
S4  Convection Service
S5  Dynamics / Heat & Moisture Transport Service
S6  Surface Processes Service
S7  Domain & Grid Service
S8  Thermodynamics & Constants Kernel
S9  State Export / Interoperability Service
S10 Model Assembly / Configuration
CC  Cross-Cutting / Shared Infrastructure
```

`microservice-pipeline evaluate` already compares cluster assignments against
these labels. **That is the ground truth for Graph C, and it is the number that
should judge coupling-model changes** — not recall.

### Summary

| Question | Oracle | Metric |
| --- | --- | --- |
| Does our resolver find the calls Python makes? | runtime trace | recall (now 98.7%) |
| Do our clusters match a human decomposition? | `manual_mapping_labeled.csv` | `evaluate` output |

Runtime is **necessary but not sufficient**. It validates the plumbing. It cannot
validate the abstraction.

---

## 3. Current issues in detail

### 3.1 The 5 remaining "missing" edges — and why 2 are not really bugs

**(a) Computed default arguments — an attribution disagreement, not a gap.**

`climlab/domain/domain.py:403`:

```python
class SlabAtmosphere(Atmosphere):
    def __init__(self, axes=make_slabatm_axis(), **kwargs):
        super(SlabAtmosphere, self).__init__(axes=axes, **kwargs)
```

`make_slabatm_axis()` is a **default argument**, so Python evaluates it once when
the `class` statement runs — inside the class body, not inside `__init__`. The
runtime therefore reports the caller as `SlabAtmosphere` (the class-body code
object). We report:

```
climlab.domain.domain.SlabAtmosphere.__init__ → climlab.domain.domain.make_slabatm_axis   [direct]
```

**The edge is not missing — both graphs found the same dependency and disagree
only on which node to attribute it to.** For boundary purposes the two
attributions are equivalent (either way, `SlabAtmosphere` depends on
`make_slabatm_axis`). Recall penalizes us for a non-problem. `code_review.md:20`
notes the same attribution issue independently. Two of the five fall in this
class (`SlabAtmosphere`, `SlabOcean`).

**(b) Cooperative multiple inheritance — a genuine gap.**

`climlab/radiation/radiation.py:202,216,269`:

```python
class _Radiation_SW(_Radiation):
    def __init__(self, **kwargs):
        super(_Radiation_SW, self).__init__(**kwargs)   # line 216

class _Radiation_LW(_Radiation):
    def __init__(self, **kwargs):
        super(_Radiation_LW, self).__init__(**kwargs)   # line 273
```

`_Radiation_SW` and `_Radiation_LW` are siblings — neither inherits from the
other. But `climlab/radiation/rrtm/rrtmg.py:8` and `climlab/radiation/cam3.py:48`
do this:

```python
class RRTMG(_Radiation_SW, _Radiation_LW):
class CAM3(_Radiation_SW, _Radiation_LW):
```

When `self` is an `RRTMG`, the `super()` call inside `_Radiation_SW.__init__`
resolves by C3 MRO to `_Radiation_LW.__init__`, **not** to `_Radiation`. Our
`ProjectIndex.resolve_super_method_targets` walks the *lexical* bases of the
class the code is written in, so it cannot see this. This is the MRO limitation
`code_review.md:9` flags, and it is a real missing edge. The remaining three
misses are in this family or similar one-offs.

> **Correction (2026-08-02): this recorded only half the defect.** The same
> lexical-base walk that *misses* the edge above also **manufactures** edges, and
> that half went unwritten for as long as recall was the only measurement — a
> missing edge shows up in recall, an invented one shows up nowhere.
>
> Unioning over `C`'s bases returns a target for each of them, and Python takes
> one. Both `CAM3.__init__` (L56) and `RRTMG.__init__` (L125) emitted edges to
> `_Radiation_SW.__init__` *and* `_Radiation_LW.__init__`; C3 says only
> `_Radiation_SW` runs. Four edges, two of them describing calls that never
> happen — carried as `super_method`, which is a relation for certainties and
> takes no discount at all.
>
> Both halves are fixed in Step 2, and §3.5 is how the manufacturing half became
> visible rather than argued about.

### 3.2 The "unconfirmed" static edges — 168, then 53

Static edges whose caller ran but which never dispatched. Mostly untaken
branches — a weak signal by design.

**Two thirds of the 168 were never candidates for confirmation at all**, and
counting them was manufacturing the very gaps the section warns against:

- `property_getter` (97), `dunder_*` and `dunder_call` are dispatched through
  C-level descriptor and type slots, so **no `CALL` event ever fires**. Verified
  directly: `c.p`, `c[0]`, `c + c` and `c()` produce zero events while an
  ordinary `c.method()` on the same object produces one. Widening
  `UNOBSERVABLE_RELATIONS` past the single `import` entry moved scoped precision
  from **0.689 to 0.869** without changing the graph at all.
- `registered_invoke` (21) is *synthetic*: it will never be confirmable, for the
  reason in §2. Now reported under its own heading rather than folded in with the
  above — "the instrument cannot see it" and "the model deliberately departs from
  the interpreter" are different facts and conflating them loses one.

What remains is a real weak signal, and §3.5 splits the decidable part out of it.

`virtual_override` is still the largest contributor: 88 edges emitted, 46
runtime-confirmed. The rest is over-approximation — a base with ten subclasses
yields ten edges though one instance takes one. Its weight was **0.4** and is now
**0.8**; see Step 7 for why the discount moved and what pays for it.

### 3.5 Falsified static edges — the other direction, now measurable

Everything above concerns edges the trace did not confirm. Until 2026-08-02 there
was no way to say a static edge was *wrong*, only unexercised, so the instrument
could measure misses and not inventions — and over-approximation therefore looked
free.

A subset is decidable. If the trace watched the exact `(caller, line, column)`
site an edge claims, and that site dispatched elsewhere and never to this callee,
the branch was taken and went somewhere else. Reported as **falsified**, separate
from unconfirmed. See Step 6 for the three ways this number can lie and how each
is closed.

Current: **5 falsified**, at 54.7% site coverage — `virtual_override` ×4 and
`self_method` ×1, all labelled and weighted. The two that were *not* labelled,
`CAM3.__init__` and `RRTMG.__init__` both claiming `_Radiation_LW.__init__` at
`super_method` and full weight, are exactly the inventions §3.1b half-described,
and Step 2 removed them.

### 3.3 The framework hardcode is gone (done — see Step 4)

`_add_subprocess_compute_edge` has been replaced by `registration.py`, which
derives the same coupling from evidence. The relation is now `registered_invoke`
and all 21 edges are reproduced with identical caller, callee, file and line; the
rest of `edges.csv` is unchanged. No behavioural reference to `add_subprocess`
remains in `call_graph/` — the surviving mentions are prose explaining the
motivating shape.

Generic registry dispatch had already been working with no framework knowledge:
a fixture whose wiring method is called `register` resolves completely; see
`test_generic_registry_dispatch_is_fully_resolved_statically` in
`tests/test_dynamic_trace.py`. What was missing was the *attribution* — which
parent owns which child — and that is what the escape/invoke join now supplies.

### 3.4 Type annotations do nothing for climlab

climlab has **0 annotated parameters out of 935**, and zero annotated
assignments. The annotation support is real and tested, but it is an investment
in other codebases. Do not expect it to move any climlab number.

---

## 4. The aim

1. **Keep the resolver honest.** Recall on the runtime trace should stay at or
   above 98.7%. This is a regression gate, not a target to push toward 100% —
   the residue is attribution differences and rare MRO shapes.
2. **Make the coupling model framework-independent.** The pairwise registry
   edges should be derived from evidence rather than from the string
   `add_subprocess`, so the tool works on torch, sklearn, dask, and plugin
   registries. This is the actual generalization goal.
3. **Judge the coupling model by the boundary oracle.** Changes to how coupling
   is modelled must be evaluated with `microservice-pipeline evaluate` against
   `manual_mapping_labeled.csv`, because recall is blind to them.
4. **Widen coverage of the trace.** Half the codebase is never exercised. More
   drivers means recall covers more of the graph.

---

## 5. Plan

### Step 1 — Establish the boundary baseline (do this first, it is cheap)

Before changing anything else, record what `evaluate` says *today*:

```bash
cd /Users/qianhuilin/Desktop/Envision/climlab
microservice-pipeline structural-graph   --config configs/microservice_pipeline/structural_graph.jsonc
microservice-pipeline structural-cluster --config configs/microservice_pipeline/structural_clustering.jsonc
microservice-pipeline evaluate           --config configs/microservice_pipeline/evaluation.jsonc
```

Without this number, none of the coupling work below can be shown to help. It
also settles an open question: whether the 88 new `virtual_override` edges and
the 336 extra edges overall *improved* or *degraded* the clustering. Recall says
the graph is more faithful; only `evaluate` can say it is more useful.

### Step 2 — Fix cooperative-MRO `super()` resolution — **DONE (2026-08-02)**

§3.1b recorded only half of this gap. It described the *missing* edge and did
not notice that the same defect was also **manufacturing** edges: walking `C`'s
lexical bases and unioning whatever they defined returns two targets for any
class with two bases, and Python takes one. On climlab both `CAM3.__init__`
(L56) and `RRTMG.__init__` (L125) emitted edges to `_Radiation_SW.__init__` *and*
`_Radiation_LW.__init__`; C3 says only `_Radiation_SW` runs, so one of each pair
was invented — at `super_method`, which carries no discount at all.

Implemented as `ProjectIndex.linearize` (C3 merge, memoized) plus
`next_in_mro`. `resolve_super_method_targets` now returns at most one target.
The cooperative case is separate: `resolve_cooperative_super_targets` asks, for
every known subclass `T` of `C`, what comes after `C` in `mro(T)`. Which subclass
is instantiated is genuinely unknown, so that over-approximates and is emitted as
**`cooperative_super`** rather than folded into `super_method`.

`linearize` never raises. Alias canonicalization can collapse two classes onto
one ID and produce a hierarchy Python would reject, and an exception escaping
into the edge pass would lose a whole run's edges over one bad base list; C3
failure or a cycle falls back to `class_and_ancestors`. Bases outside the project
(`builtins.object`, `numpy.ndarray`) are kept as opaque leaves rather than
dropped — dropping one only matters if it defines the method, which is unknowable
either way, while keeping it preserves the order of everything around it.

`_is_super_call` now reads `super(Other, self)`'s first argument. That was
harmless while resolution unioned; under C3 the starting class decides the
answer.

*Acceptance, met in full:* both `super_method` pairs collapse to a single
`_Radiation_SW.__init__` target; `_Radiation_SW.__init__ → _Radiation_LW.__init__`
appears as `cooperative_super`; recall rose 99.5% → **99.7%** (376/377); and the
two invented edges disappeared from the falsified set (§3.5), which is the first
time this repo could demonstrate an invention was removed rather than argue it.

**`resolve_method_targets` was deliberately left alone.** Measured over all 77
climlab classes it already returns exactly one target for every `(class, method)`
pair, so switching it to a single MRO winner would be unmeasurable here while
risking four consumers (`resolve_constructor_targets`, `virtual_override`,
`registration.py`, `data_access/registration_lineage.py`). Revisit it on a
codebase with wide diamonds, where the `seen.copy()` blowup `code_review.md:9`
flags is also a real cost that C3 removes.

### Step 3 — Attribute default arguments to the enclosing scope — **DONE (2026-08-02)**

Default-argument, decorator and annotation expressions were visited with
`current_callable` set to the function being defined, because `_enter_callable`
set the scope and *then* called `generic_visit`. They are now visited by the
enclosing scope, via the shared `signature_expressions` helper.

Matching the interpreter needed one more thing: a class body is its own code
object, and `SlabAtmosphere`'s default argument runs *there*, not in `__init__`
and not at module scope. Classes whose body actually runs a call therefore become
nodes of kind `class_body`, keyed by the class ID so the tracer's `co_qualname`
still agrees. Gated on `class_body_evaluates_calls` so only 8 of climlab's 77
classes qualify — indexing all 77 would add isolated leaves and drag down every
degree-based threshold downstream for no information.

One trap worth recording: a class body must be a node but must **never** enter
the resolution universe. Adding class IDs to `callable_ids` made `SlabOcean(...)`
resolve to the body instead of `__init__` and silently deleted five constructor
edges. `models.resolvable_callable_ids` now filters them out — a class body can
be an edge's caller, never its callee.

*Acceptance, met:* both `Slab*` edges appear under the class-body caller; recall
rose 98.7% → 99.2%.

### Step 3b — Canonicalize callable re-export aliases — **DONE (2026-08-02)**

`add_reexport_class_aliases` solved this for classes and left callables out.
`from climlab.domain import column_state` records the target as
`climlab.domain.column_state`, while the function lives at
`climlab.domain.initial.column_state`, so the call resolved to nothing and the
edge was dropped. `build_callable_aliases` runs the same fixpoint over
`callable_ids`. Worth the third of the five remaining misses; recall 99.2% →
99.5%.

### Step 3 — Attribute default arguments to the enclosing scope

Per `code_review.md:20`, default-argument and decorator expressions are currently
attributed to the function being defined rather than the scope that evaluates
them. Fixing it removes two of the five misses and makes the static graph agree
with the interpreter about *when* code runs.

*Acceptance:* the two `Slab*` edges appear under the class-body caller; recall
does not drop.

### Step 4 — Derive the registry pairing (the real generalization) — **DONE**

*Outcome, 2026-08-01.* Implemented in `registration.py` plus escape/invoke
recording in `summary_collectors.py`. Acceptance met in full:

1. All 21 pairwise edges reproduced under the relation `registered_invoke`,
   matching the old `subprocess_compute` set exactly on caller, callee, file and
   line. `edges.csv` is otherwise byte-identical (1141 edges either way).
2. `evaluate` unchanged from the Step 1 baseline: ARI 0.281, V-measure 0.569,
   Pairwise F1 0.339, BCubed F1 0.429. Recall also unchanged at 98.7%.
3. Fixtures for a torch-like `add_module`, a `Pipeline(steps=[...])` list, a
   plain dict registry, a two-attribute relay, a delegating wrapper, and two
   negative shapes (`self.config = config`, `self.parent = parent`).

Three departures from the plan below, all forced by what climlab actually does:

- **The invoke summary is not optional.** Escape alone matches every
  `self.config = config` and emits an inverted edge for `self.parent = parent`.
  Requiring that the stored values are invoked is the gate, and it also supplies
  the hook name, so no framework hook name is in config.
- **A third fact was needed:** elements flow between attributes. climlab stores
  into `self.subprocess` and invokes out of `self.process_types`, copying between
  them in `_build_process_type_list`. An invoke summary keyed on the escaped
  attribute finds nothing. `RegistryFacts.element_flow` records the copies and
  the join closes over them, and over the class hierarchy — `Process` writes the
  attribute that `TimeDependentProcess` reads.
- **Third-party frameworks needed a new pass.** `nn.Module.add_module` lives in
  site-packages, which `iter_analysis_files` never reads, so escape facts for it
  were unreachable. `call_graph.summary_packages` names packages to parse for
  facts only; they contribute no nodes and no edges.

Original plan follows.

Replace the `add_subprocess` trigger with two derived facts:

- **Escape summary** — "parameter 1 of this method ends up inside `self.<attr>`."
  Requires tracking parameter *identity* (not type) through the method body. In
  climlab this is `self.subprocess.update({name: proc})` at
  `climlab/process/process.py:253`. This is what marks a call site as a
  registration.
- **Invoke summary** — "elements of `(class, attr)` receive method `m`, called
  from `X`." In climlab: elements of `self.subprocess` receive `.compute()`.

Join them **at the call site**, where parent and child types are known exactly —
that is where the current hardcode gets its precision, and it must be preserved.
Then apply **template-method re-projection**: the invoked method resolves to the
shared base (`TimeDependentProcess.compute`), which is useless as a node, so
follow the base's delegation one hop to the hook the concrete class overrides
(`compute` calls `self._compute()` at
`climlab/process/time_dependent_process.py:241`).

*Suggested sequencing:* build the escape summary first and keep the hook name in
config. That isolates the hard half and answers whether the invoke summary is
needed at all, or whether a declared hook is good enough for the frameworks you
care about.

*Acceptance — note this is NOT recall:*
1. All 21 pairwise edges reproduced with the hardcode removed and the string
   `add_subprocess` absent from `call_graph/`.
2. `evaluate` score no worse than the Step 1 baseline.
3. Fixtures for other shapes: a torch-like `add_module`, a `Pipeline(steps=[...])`
   list, a plain dict registry.

~~Keep `subprocess_coupling.py` — `data_access/subprocess_lineage.py` still uses it
for shared-state lineage, which is a separate concern.~~ **Superseded — see Step 4b.**

### Step 4b — Retire the last hardcode, in data access — **DONE**

*Outcome, 2026-08-02.* Step 4 left one consumer behind: `data_access` still
recognised a registration by the literal method name. Deriving it there too meant
`subprocess_coupling.py` had no job left, so it is deleted rather than renamed.

- `data_access/subprocess_lineage.py` → `registration_lineage.py`;
  `SubprocessStateLineageMixin` → `RegisteredStateLineageMixin`. No module, class,
  function or constant now carries "subprocess" in climlab's sense — the word
  collided with the stdlib module, which this repo also imports.
- The shared analysis prefix is factored into `analyze_analysis_files` →
  `CallGraphAnalysis`. Data access calls it and uses `.registration_rules` and
  `.project_index`. Computed in-process rather than read from an artifact: an
  artifact would make `call-graph` a hard prerequisite and would silently produce
  wrong lineage when stale.
- `ProjectIndex` is what makes this work at all. Data access's own `known_classes`
  is a flat set with no ancestors, so `self.add_subprocess(...)` on a
  `StepFunctionAlbedo` could never reach `Process.add_subprocess` three classes up.
- The registration *slot* (`'albedo'`) is derived too, via
  `FunctionEscapeSummary.key_params` → `RegistrationRule.slot_param`, so the
  lineage label survives the loss of `add_subprocess_name`.
- **The shared-state gate is unchanged.** Registration says two objects are
  coupled; only `state=` passed at the call site says they share an object.

*Result:* climlab's `data_access.json` is byte-for-byte identical — same 1154
lineage edges, same 3 registration edges with the same slots — and `evaluate` is
unchanged. Data-access wall clock 5.9s against 6.1s before; the added analysis is
inside the noise.

### Does it generalise? Measured on matplotlib, 2026-08-02

Every fixture up to this point was one we wrote. Running the derivation over
matplotlib (248 files, 9434 callables, none of it written for this tool) gives the
first honest read.

**A bug surfaced immediately.** `iter_python_files` tested the *absolute* path for
hidden directories, so every `summary_packages` entry returned **zero files** —
installed packages live under `.venv`. The feature had a passing test only because
the fixture package sat in a directory with no leading dot. Now judged below the
scan root, with a regression test.

**4 rules derived, 4 correct** — verified by reading the source:

| registrar | child | invoked with |
|---|---|---|
| `patheffects.PathEffectRenderer.__init__` | elements of `path_effects` | `.draw_path()` |
| `collections.Collection.set_paths` | elements of `paths` | `.iter_segments()` |
| `collections.PathCollection.__init__` | elements of `paths` | `.iter_segments()` |
| `_mathtext.List.__init__` | elements of `elements` | `.shrink()` |

`_mathtext.List` is the climlab shape exactly — a composite tree that recurses
into `self.children`. No false positives.

**The instructive miss is `Axes.add_artist`.** The escape is clean
(`self._children.append(a)`), but the invoke path is
`self._children` → returned by `get_children()` as a *new list* → local `artists`
→ sliced and rebuilt → passed as an argument to
`mimage._draw_list_compositing_images`, which iterates and calls `.draw()`.
`RegistryFacts.element_flow` follows attribute→attribute relays only. It does not
follow a value out through a **return value** or a **function parameter**, and
this path needs both. The gate declines rather than guessing, so the cost is a
missing rule, not a wrong one.

### Cross-call provenance — **DONE**, same day

The miss above was fixed by carrying attribute provenance across the two
boundaries it died at. `RegistryFacts` gained:

- `returned_attrs` — "this callable hands back the contents of `(class, attr)`",
  recorded in `TypeSummaryCollector.visit_Return`.
- `param_attrs` — "callers fill this parameter with the contents of
  `(class, attr)`", recorded at call sites.

plus an `attr_container` origin for a whole collection whose contents came from
an attribute (what `artists = self.get_children()` binds), which iterating turns
back into element origins. Starred elements inside a literal now contribute their
*contents* rather than themselves, so `return [*self._children, ...]` is seen
through.

*Result on matplotlib:* **4 rules → 6**, still no false positives.
`figure.FigureBase.add_artist` is now derived — `self.artists.append(artist)`,
then `Artist.pick()` does `for a in self.get_children(): a.pick(...)`, a chain
that crosses the return-value boundary and was previously invisible.
`artist.Artist.set_path_effects` is the other gain. climlab is untouched: 1141
edges, 21 `registered_invoke`, `data_access.json` byte-identical, `evaluate`
unchanged.

**The binding constraint has moved.** `_AxesBase.add_artist` is still missed, but
no longer for want of provenance — `_AxesBase._children` is now seen receiving
*three* methods (`get_visible`, `_get_in_autoscale`, `pick`) and
`build_registration_rules` requires exactly one, so it declines rather than
guessing which carries the coupling. Picking among several would need a
principled tie-break; preferring the one subclasses actually override is the
obvious candidate, since `_registration_hook` already uses that test for
re-projection. Deliberately not done here — it changes what gets emitted and
should be judged by `evaluate`, not chosen in passing.

**Scale.** 43.9s for 248 files (33.8s before this extension), and both fixed
points hit their iteration caps (`max_iterations` 3 and 5) with the
non-convergence warnings firing. Fine for a summary-only pass whose output is a
handful of rules, but the caps were tuned for climlab-sized projects and would
need revisiting before trusting this on a large codebase as the primary analysis.

### Step 5 — Widen trace coverage

Only 258 of 486 callables are currently exercised. Add the remaining courseware
notebooks (some need the `climlab-rrtmg` / `climlab-cam3-radiation` wheels or
network access for orbital tables) and the `compiled`-marked tests once those
wheels are installed. More coverage makes recall a statement about more of the
codebase.

---

### Step 6 — Make inventions measurable — **DONE (2026-08-02)**

The instrument was asymmetric by construction. Recall measured misses; nothing
measured inventions, because `graph_comparison` (correctly) refuses to call an
unconfirmed edge false — an edge absent from the trace usually means an untaken
branch. So over-approximation looked free.

It is not free, and one case is decidable. If the trace watched the *exact call
site* an edge claims, and that site dispatched to other callees and never to this
one, "untaken branch" is ruled out: the call ran and went elsewhere. The tracer
now records `(caller, line, column)` for every dispatch (`dynamic_sites.csv`) and
the comparison reports **falsified** separately from **unconfirmed**.

Three things this needed, each of which would have made the number a lie:

* **Column, not just line.** 718 of 2413 climlab call expressions (29.8%) share a
  line with another call. Keyed on line alone, ~30% of "falsified" verdicts would
  be a different call on the same line. `Edge` gained `col_offset`; the tracer
  reads `co_positions()[offset // 2]`. Measured agreement between the two: 2351
  of 2353 real call instructions match exactly.
* **Retired sites excluded.** The adaptive disable rule stops watching a site
  after `disable_after` uneventful hits, so its callee set is truncated by
  design. Falsifying against one turns a cost-saving heuristic into a false
  accusation — it was the difference between **29** falsified edges and **7**.
* **Unobservable relations excluded first.** See §3.5.

### Step 7 — Label uncertainty, and make the label count — **DONE (2026-08-02)**

`relation` says how an edge was *derived*; it was also being asked to say how
*certain* it is, and those are different questions. A receiver with one possible
type and a receiver with five both emitted `inferred_type`, and the structural
graph stamped every call edge `confidence="high"` regardless.

`Edge.confidence` now grades a call site by its fan-out, measured at the emission
point. The thresholds come from data, not taste. Confirmation rate by fan-out `k`
on climlab:

| relation | k=1 | k=2 | k>=3 |
| --- | --- | --- | --- |
| `constructor` | 62/63 | 3/3 | 1/1 |
| `direct` | 60/64 | 6/6 | — |
| `imported` | 58/73 | 9/9 | 4/4 |
| `self_method` | 266/290 | 23/24 | **6/14** |
| `super_method` | 59/59 | 2/4 | — |
| `virtual_override` | — | 14/16 | **36/60** |

Fan-out is **not** monotone with wrongness at the low end: two-target sites
confirm as often as or better than one-target sites. A binary `k==1 high,
k>1 medium` rule would have demoted the more reliable bucket. Degradation starts
at three, so the grading does: `k<=2 -> high`, `3..5 -> medium`, `>5 -> low`.

Validated against the falsification instrument from Step 6, which is independent
of it: **7.14%** of `medium` edges are falsified versus **0.10%** of `high` — a
70x separation.

Two follow-through steps that are easy to skip and would have left the field
decorative:

* The call branch of `generate_structural_dependency_graph` computed `weight`
  without ever multiplying by `confidence_weight`, unlike the data-access branch.
  It now does.
* `virtual_override`'s **0.4** was a *proxy* for ambiguity that confidence now
  measures directly, so leaving both discounted it twice. Swept against
  `evaluate`: 0.4 → ARI 0.288, 0.6 → 0.288, **0.8 → 0.325**, 1.0 → 0.296. Set to
  **0.8**, which wins on ARI, Pairwise F1 and BCubed F1 simultaneously. Re-derive
  on a second codebase before treating either the thresholds or this weight as
  settled.

### Step 8 — Widen the lattice to class ids ∪ callable ids — **DONE (2026-08-02)**

The one item on `code_review.md`'s list that had never been started, and the root
of a whole family: no `__call__`, no `functools.partial`, no dispatch tables, no
callback registration. Everything the analyzer knew about a value was a set of
class ids, so a value that *is* code was not lost through a bug — it was
inexpressible.

Built as a **callable dimension parallel to the container dimension**, which was
added the same way (§1 item 3). `TypeEnv` gains a sixth stack;
`FunctionReturnSummary`, `FunctionParamSummary` and `ClassAttrTypes` each gain
callable maps; `_infer_callable_ids_from_value` mirrors
`_infer_class_types_from_value` case for case.

**A separate stack, not a tagged union**, and not for tidiness. Five consumers
break concretely if a callable id reaches a class-id set — most sharply
`collector/resolution.py`'s `if var_types:` short-circuit, which would take the branch,
fabricate the callee `mod.f.method`, and shadow the class-based rungs below that
would have resolved correctly. `registration.py`'s `len(...) != 1` gates would
silently *delete* `registered_invoke` edges, and `data_access` would mint a
phantom data node from every callable id in `ClassAttrTypes`.

Two traps that fail **silently** and are worth remembering:

* `copy_param_summaries` / `copy_class_attr_types` must copy *every* field. One
  left out is shared between the pending and previous summaries, the convergence
  test is trivially satisfied on that field, and the fixpoint exits after one
  round with no warning.
* `return_links.FLAT_RETURN_FIELDS` / `SLOT_RETURN_FIELDS` are explicit
  frozensets. A field omitted there advances one hop per pass and is truncated by
  `max_iterations`, again silently.

Resolution gained two rungs, placed after every class-based rule so a known
receiver still resolves as a method call: **`inferred_callable`** (the expression
holds a callable) and **`dunder_call`** (the expression is an instance of a class
defining `__call__`). Lambdas are indexed as
`<enclosing>.<locals>.<lambda>` — CPython's own `co_qualname`, so tracer and
static IDs keep agreeing by construction; sibling lambdas collide exactly as they
do in CPython, which is the price of that agreement.

**climlab cannot validate any of this**: 0 lambdas, 0 `functools.partial`, 0
higher-order library calls, and only 5 calls to a value at all. Correctness is
carried by fixtures covering all three review pictures plus dispatch tables,
`self.handler = fn`, `partial`, functors and lambdas. climlab's role was
no-regression, which it passed — and it did surface one genuine bug the fixtures
would not have: `cls()` inside a classmethod was resolving to `Attr.__call__`,
because `cls` is seeded with its class id like `self` but denotes the *class
object*. `cls(...)` constructs; it is now a `constructor` edge, worth 3 more
correct edges and dropping unresolvable calls from 5 to 2.

**Not done, and deliberately:** the invoke-gated edge at the *passing* site
(`Step 8b` below), and matplotlib as a second oracle.

### Step 8b — Invoke-gated callable arguments — open

A callable passed as an argument currently produces an edge inside the callee
(`Crypto.apply -> cryptops.encrypt`) but none at the call site that passed it.
The blanket alternative is not acceptable: on climlab, emitting an edge for every
callable-valued argument would add up to **1759** speculative edges to a
1144-edge graph, unfalsifiable by construction. The gate should mirror
`registration.py` — an edge only where there is evidence the receiving parameter
is invoked — which is the same escape/invoke join that stops `self.config =
config` looking like a registration.

### Step 8c — matplotlib as a second oracle — open

Required before any of Step 7's thresholds or Step 8's rungs can be called
validated. 248 files, 9434 callables, and unlike climlab it actually contains
lambdas, `partial` and `__call__`. Note both fixpoints already hit their caps
there with non-convergence warnings firing; treat those as hard failures for this
work rather than notes.

## 6. Things worth remembering

- **Recall validates plumbing, not abstraction.** A coupling model that
  deliberately departs from the runtime call sequence cannot be scored by recall.
  Use `evaluate`.
- **A perfect recall score is not the goal.** Faithfully reproducing base-class
  hubs would raise recall and *harm* clustering, since hub policy then discards
  them.
- **Over-approximation is acceptable when it is labelled.** `virtual_override`
  earns its place by carrying a distinct relation and a lower weight, so the
  clustering step can discount it. Unlabelled over-approximation silently fuses
  services.
- **An instrument that can only find one kind of error will only find one kind
  of error.** Recall measured misses and nothing measured inventions, so
  over-approximation looked free for as long as nobody checked. Falsification at
  the *site* level is decidable and cost two extra columns.
- **A number that cannot fail is not a measurement.** `unresolved: 0` was printed
  as a perfect score and was a tautology: with `include_external` off, an
  unresolved edge is dropped before it can be counted. Tally at the point of
  loss, never from the artifact.
- **Check whether a heuristic can lie before trusting what it says.** The
  tracer's disable rule truncates a site's callee set; falsifying against a
  retired site turns a cost saving into a false accusation, and was the
  difference between 29 falsified edges and 7.
- **Derive thresholds, don't choose them.** Fan-out is not monotone with
  wrongness at the low end — two-target sites confirm better than one-target
  sites — so the obvious binary rule would have discounted the wrong bucket.
- **The trace is a lower bound, always.** An edge absent from it may simply never
  have run. That is why the report calls those edges *unconfirmed* rather than
  *false*.
