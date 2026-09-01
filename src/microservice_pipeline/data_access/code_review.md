# Code review — `data_access/`

`call_graph/` has been through a full review cycle: `code_review.md` catalogued the
defects, `ground_truth_and_roadmap.md` sequenced them, and a dynamic oracle
(`dynamic_trace.py` + `graph_comparison.py`) made every claim about progress
falsifiable. `data_access/` has had none of that. It is 4,921 lines across 8 files,
3,024 of them in one module and ~1,700 in one class, and it is the input to the
clustering this whole pipeline exists to produce — so a false coupling here does not
stay here, it lands in `evaluate`.

This document is the equivalent catalogue. Findings are ordered by how much damage
they do, not by where they live in the file.

**How to read a finding.** Each one is tagged:

- **CONFIRMED** — reproduced by running the code. The transcript is included.
- **REASONED** — read off the source, not executed. Stated as mechanism, not as a
  measurement.

Nothing here is a measurement of *how often* these fire on a real codebase, because
there is currently no instrument that could say. That gap is §6, and closing it is
Step 1a of the plan for a reason.

§5 is separate from the rest and answers a specific question: **which of these did
your call-graph revision cause, and which does it now hand you the fix for?** Three
of its seven findings are defects the call graph has already solved, in code this
package was handed and does not call.

---

## 1. Correctness

### 1.1 The container-family classifier throws away the common case — CONFIRMED

`pyright_family_from_type_text` (`pyright_type_probe.py:82-169`) decides which family
a container belongs to by collecting substring hits and then refusing to answer if it
found more than one. Two independent rules make it refuse constantly:

```
'dict[str, Any]'              -> unknown
'dict[str, list[int]]'        -> unknown
'defaultdict[str, list[int]]' -> unknown
'dict[str, DataFrame]'        -> unknown
'DataFrame'                   -> dataframe
'list[str]'                   -> list
'Dataset'                     -> object
```

- `has_unknown` tests for the substring `" any"`, which is present in *every*
  parameterised generic whose value type is `Any`. `dict[str, Any]` is not an unknown
  type; it is a dict.
- Any text mentioning two families at all collapses to `unknown`, so a dict of lists
  reports neither.

The family is not cosmetic. `_field_kind` reads it to choose between `df_col`,
`dict_key` and `container_field`, so a correctly annotated dictionary is demoted to
the bucket the artifact README describes as "the base container family remained
unknown". The better a project's type annotations are, the more of them this drops.

The shape of the type text is a nesting, and the answer wanted is the *outermost*
constructor. Parse it as such — take the head before the first `[` — instead of
unioning substring hits across the whole string, and reserve `unknown` for a head
that is genuinely unrecognised.

`'Dataset' -> object` is a second, smaller instance: the xarray branch matches only
fully-qualified spellings (`xarray.Dataset`, `xr.Dataset`), so a bare `Dataset` from
`from xarray import Dataset` is not xarray.

> **FIXED by Step 3 (2026-08-31).** Parsed as a union of nestings, classified on the head
> of each member, with uninformative members (`Unknown`, `Any`, `None`, `Never`) dropped
> before the rest are required to agree. Answers thrown away as unknown **2,220 → 1,637**;
> `dict` 34 → 263. The bare-`Dataset` instance is fixed by handing the classifier each
> module's import map, so the name is resolved to where it came from — worth two answers
> on climlab and no artifact rows, since those two locals are never modelled (§1.16).
> One thing this section does not say: the same substring scanning was reading type names
> out of *function signatures*, so numpy's `moveaxis` was a list and `os.path.join` a path.

### 1.2 One class gets two different object IDs depending on which file is analysed — CONFIRMED

`infer_split_class_owners` is called from `collect_data_access_from_tree` once per
file and walks `tree.body`, so it only ever knows about classes defined in *that*
file:

```
class Coordinator, defined in module a:
  infer_split_class_owners(tree_a, "a") -> {'a.Coordinator'}
  infer_split_class_owners(tree_b, "b") -> set()
```

`registration_lineage._class_state_object_id_for_owner` then consults that set for an
owner class which, at a registration call site, is very often defined somewhere else:

```python
split_class_state = owner in self.split_class_owners
object_id = f"class_attr_state:{owner}:state" if split_class_state else f"class_state:{owner}"
```

So the same class is `class_attr_state:X:state` while its own file is being walked and
`class_state:X` while any other file is. Two node identities for one thing: lineage
edges recorded from file B point at an ID that the object table — populated from file
A — never registered, and the class's state is split across two nodes that downstream
clustering has no reason to join.

The same per-file set also drives `_class_attr_ref`, so the split/unsplit choice for
ordinary `self.foo` accesses is likewise file-local.

Fix: compute split owners once, project-wide, before the collection passes, and hand
the result in — the same move `ProjectIndex` made for the call graph.

### 1.3 `_lineage_roots` caches an answer it computed inside a cycle — CONFIRMED

```python
if object_id in cache:
    return cache[object_id]
...
if object_id in local_seen:
    return set()          # cycle guard: correct for *this* traversal only
...
cache[object_id] = roots  # ... but the truncated answer is cached anyway
```

The cycle guard returns an empty set for a node already on the stack. That truncated
answer propagates up to the *ancestor*, and the ancestor's value is then written to
`cache` as though it were complete. Every later query gets the polluted value:

```
graph: R -> A, A -> B, B -> A (cycle), B -> C

roots(C) with a cold cache        = ['B', 'R']
roots(C) after roots(A) was asked = ['A']
identical? False
```

The result depends on which node happened to be queried first, and the query order is
`for object_id, obj in objects.items()` — i.e. insertion order, i.e. the order files
were processed. `_apply_lineage_aliases` uses it to decide `alias_of`, and a
single-root answer becomes an alias while a multi-root answer erases one, so this
silently flips alias assignments between runs on any project with a lineage cycle
(mutual recursion, a value threaded back into its own source parameter).

Either do not cache nodes whose computation touched an active `seen` entry, or
compute strongly-connected components once and resolve roots per component.

The function is duplicated almost verbatim as `_lineage_root_ids` in
`infer_shared_containers.py:100-127`, so the inferred shared-container config inherits
the same defect independently. Fixing one and not the other is the likely outcome of
leaving them both in place — see §4.3.

### 1.4 Cross-file object merges keep only `confidence` — REASONED

`collect_data_access_from_analysis_files:2788-2795`:

```python
if object_id not in objects:
    objects[object_id] = data_object
else:
    objects[object_id].confidence = _confidence_max(...)
```

Within one file, `_register_object` has careful merge logic: it upgrades a
placeholder `kind == "unknown"`, fills in a missing `lineno`, takes an `alias_of` or
`access_path` if it did not have one, and promotes `structural_role`. None of that
runs across files. A `file:` object, a class-state object, or any ID reachable from
two modules keeps whatever the first file said and discards every later refinement —
including a known `inferred_type` arriving to replace `unknown`.

First-file-wins is only deterministic if file order is, which makes this the same
class of defect as call-graph review item 16.

Fix: make `_register_object`'s merge a function on two `DataObject`s and use it in
both places.

### 1.5 The tuple-return fixpoint can converge early, silently — REASONED

The two snapshot functions that decide convergence do not compare the same fields:

```python
_return_summary_snapshot        -> (object_id, inferred_type, confidence, display_name, access_path)
_return_tuple_summary_snapshot  -> (object_id, inferred_type, confidence, display_name)
```

A pass that changes only a tuple slot's `access_path` produces an identical snapshot,
so the loop declares convergence and stops. `access_path` is not decorative — it is
what `_field_object_id` and `_root_relative_access_path` build field IDs from.

Neither fixpoint (`collect_data_access_from_source:2664`,
`collect_data_access_from_analysis_files:2741`) records whether it converged or merely
exhausted `MAX_RETURN_SUMMARY_PASSES`. The call-graph roadmap's own conclusion, after
both of its fixpoints hit their caps on matplotlib with warnings firing, is that a
cap-hit is a hard failure rather than a note. Here there is not even the warning.

Two fixes, both small: include every field in both snapshots, and return a
`converged: bool` that the caller reports.

### 1.6 Suffix matching is unanchored, so ordinary method calls mint container types — CONFIRMED

`_infer_type_from_value` classifies a call by testing the *end* of the dotted name:

```python
if call_name in {"set", "builtins.set"} or call_name.endswith(".set"):
    return FAMILY_SET, "", "high"
```

```
def f(ax, node):
    style = ax.set(1)      ->  local_exposed:sample.f:style   inferred=set
    kids  = node.list()    ->  local_exposed:sample.f:kids    inferred=list
```

`ax.set(...)` is matplotlib's bulk property setter and `node.list()` is anything at
all; neither builds a collection. The same pattern appears for `.dict`,
`endswith("DataFrame")`, and `endswith(tuple(FILE_READ_FUNCS))` — the last of which
matches a project's own `my_read_csv` and declares the result a DataFrame at
confidence `high`. `_is_open_call` has it worst: any call whose name ends in `.open`
is a file open, so `self.db.open(conn)` mints a file object.

`rules.py` already contains the correct test:

```python
def _call_name_matches(call_name: str, names: Set[str]) -> bool:
    return any(call_name == name or call_name.endswith(f".{name}") for name in names)
```

It is used for the xarray and pooch branches and nowhere else. The fix is to use it
everywhere, which also removes the `endswith(tuple(...))` idiom entirely.

### 1.7 File objects are keyed on source text, so unrelated callables share one node — CONFIRMED

`_file_ref` builds the ID from the unparsed argument expression when it is not a
literal:

```
def load_users(path):    open(path)  ->  file:path
def load_invoices(path): open(path)  ->  file:path
```

Two callables that share no file at runtime now share a data object, and the
structural graph reads that as coupling. Any project where the conventional parameter
name is `path`, `filename` or `fname` collapses its entire filesystem surface onto a
handful of nodes.

This one is worth ranking above its apparent size: the extractor's output exists to
be clustered, and this manufactures exactly the kind of edge that fuses services.

The literal case (`open("data/users.csv")`) is genuinely shared and should stay
global. The non-literal case is an *unresolved* file whose identity is unknown —
either scope the ID to the callable, or resolve the expression through the existing
lineage machinery to whatever the parameter was bound to at its call sites, which
`param_bindings` already computes.

> **Measurable since Step 1b (2026-08-29), but it does not fire on climlab.** The oracle
> records the path *string* at each site — object identity cannot settle this, since two
> equal paths are one file whether or not they are one `str`. climlab has 5 `file:` objects,
> only one is touched by more than one callable, and its callables never disagreed. The
> defect is still real (the test fixture reproduces it exactly); climlab just does not open
> files from parameters. Judging the fix needs a project that does, or `evaluate`.

### 1.8 Class bodies have no scope, so class attributes leak into the module — CONFIRMED

`visit_ClassDef` sets `current_class` and walks the body, but never pushes a `Scope`.
Class-level assignments therefore land in the enclosing scope's `locals`:

```
class A:
    registry = {}

registry['k'] = 1     # NameError at runtime
```
```
object dict_key:sample.<module>:registry:k   owner=local_exposed:sample.<module>:registry
edge   sample.<module>  write  assign  dict_key:sample.<module>:registry:k
```

Three consequences, in increasing order of severity:

- `A.registry` is never modelled as class state at all — the one place a class
  attribute is unambiguously class-level data, and it is filed under the module.
- The binding outlives the class body, so module-level code that could not possibly
  reach it resolves against it, as above.
- Two classes in one file that use the same attribute name overwrite each other's
  binding. A third probe with `class A: registry = {'a': 1}` and
  `class B: registry = {'b': 2}` produces **no objects at all**.

Fix: push a scope for the class body, and register its container assignments as
`class_attr_state` / `class_state` for the owning class.

### 1.9 Lambda parameters resolve to enclosing locals — CONFIRMED

`_visit_comprehension` models target shadowing explicitly and restores it afterwards.
`visit_Lambda` does not — it just walks the children:

```
def f(rows):
    data = {'a': 1}
    return sorted(rows, key=lambda data: data['a'])
```
```
read subscript_load  dict_key:sample.f:data:a   evidence="data['a']"
```

The lambda's own parameter is attributed to the outer dictionary, inventing a key
access on an object the lambda never touches. The machinery to fix it already exists;
`visit_Lambda` needs the same `scope.shadowed` treatment the comprehensions get.

### 1.10 `self.index` and `self.columns` are unobservable on every receiver — CONFIRMED

`visit_Attribute:2033` returns early for `PANDAS_INDEXER_ATTRS | {"index", "columns"}`
regardless of what the receiver is:

```
class Grid:
    def use(self):
        a = self.index      -> no edge
        b = self.columns    -> no edge
        c = self.spacing    -> read attribute_load class_state:sample.Grid
```

The suppression is right for a DataFrame, where `df.loc` is an indexer rather than
data. It is wrong for every other object, and `index`/`columns` are ordinary attribute
names. Gate it on the receiver's inferred family.

### 1.11 A dynamic `getattr` fans out to every callable in a module — REASONED

`_dynamic_getattr_callable_ids` falls back to a prefix scan when the attribute name is
not a literal:

```python
prefix = f"{imported_base}."
return sorted(callable_id for callable_id in known_ids if callable_id.startswith(prefix))
```

`_record_confirmed_param_lineage` then deliberately bypasses its own uniqueness gate
for this case (`len(candidates) != 1 and not is_dynamic_getattr`) and records
`arg_to_param` lineage against *all* of them. One `getattr(mod, name)(x)` against a
large module emits a lineage edge per callable per argument.

Those edges are not inert. They feed `_apply_lineage_aliases`, where more than one
root causes an existing alias to be **erased**, and they feed the shared-container
inference through `data_access.json`. An over-approximation that deletes information
elsewhere is worse than a missing edge; the call-graph review reached the same
conclusion about unlabelled over-approximation in general.

Either cap the fan-out and record nothing beyond it, or mark these edges with a
distinct relation so consumers can discount them — the `virtual_override` precedent.

### 1.12 One bad file kills the run — REASONED

`ast_utils.parse_python_file` is `ast.parse(py_file.read_text(encoding="utf-8"))`.
A `SyntaxError` in a vendored or generated file, or a latin-1 source with a PEP-263
coding cookie, raises out through every caller here. Data access calls it from four
places, none of which catch it.

These are call-graph review items 14 and 15, still open. `ast.parse(path.read_bytes())`
honours the coding cookie, and per-file failures belong in a report rather than in a
traceback.

### 1.13 Receiver reads are recorded twice — CONFIRMED

`visit_Call` records a receiver read and then visits the receiver expression, which
records it again:

```
def f(items):
    items.count(3)
```
```
read  method:count:receiver  param:sample.f:items
read  load                   param:sample.f:items
```

Edges are never deduplicated — call-graph review item 19, also still open. Both
survive into the structural graph, and because `_add_edge` keys on
`(src, dst, edge_type, relation, access, operation)` and the two differ in
`operation`, they do not collapse into one edge with `evidence_count = 2`; they
become **two separate edges**, each carrying its own weight. `write_report` counts
them separately too. Whether the fix is dedup or suppressing one of the two, the two
edges currently describe one syntactic fact.

### 1.14 Default-argument expressions are attributed to the callee — CONFIRMED, found by Step 1a

Added 2026-08-24. This was not in the original catalogue; the access oracle produced it as
its only surviving falsified claim on climlab, and the source confirms it.

`_enter_callable` sets `current_callable` to the function being defined and *then* visits
the defaults:

```python
# Defaults can read module globals or outer-scope data.
for default in list(node.args.defaults) + list(node.args.kw_defaults):
    if default is not None:
        self.visit(default)
```

Visiting them is right — they genuinely do read outer-scope data. Attributing them to the
new callable is not. Python evaluates a default **once, in the enclosing scope, when the
`def` statement executes**, so the reader is the module body (or the enclosing function),
never the callee. climlab:

```python
# climlab/radiation/rrtm/rrtmg_sw.py:95, inside the __init__ signature
def __init__(self, ..., bndsolvar = np.ones(nbndsw), **kwargs):
```

```
static : climlab...RRTMG_SW.__init__  reads module_global nbndsw
runtime: no LOAD_* of `nbndsw` exists anywhere in __init__'s bytecode;
         the read is in climlab.radiation.rrtm.rrtmg_sw.<module>
```

The edge is attached to the wrong node, so a module-level constant looks like a
per-constructor dependency. On a project that uses computed defaults heavily this
misattributes an edge per default per callable, all pointing into the class rather than the
module.

Fix: visit defaults *before* entering the callable, while `current_callable` is still the
definer — which is also where `visit_Lambda` will need them once §5.5 lands.

Belongs with the rest of the cheap, local, independently testable fixes in **Step 4**.

### 1.15 `object` is a catch-all that swallows scalars — CONFIRMED, found by Step 3

Added 2026-08-31. `pyright_family_from_type_text` answers `FAMILY_OBJECT` for any type
whose head it does not recognise, and `_field_kind` treats everything that is not
`dataframe`, `dict` or `xarray` as a container. So a `float` is modelled as something
that could hold fields.

On climlab this is **450 of 3,183 probe answers** — 288 `float`, 87 `int`, 44 `bool`,
31 `str`. A number is not a container, and `x.real` on one is not a data access in any
sense this pipeline cares about.

Deliberately not fixed in Step 3, and the reason is the reason it is worth a finding of
its own: the fix *removes* objects and access edges, while Step 3's fix creates them, so
landing them together would have moved the Step 1a recall figure for two opposite reasons
with no way to separate them. It is also the first change in this document that would
lower recall on purpose, which needs its own argument and its own measurement.

Fix: a `scalar` family that `_field_kind` refuses to treat as a container, judged by
Step 1a's recall and by `evaluate` rather than by recall alone.

### 1.16 A local's family can be known while the local is never modelled — CONFIRMED, found by Step 3

Added 2026-08-31. `_local_ref` asks `_family_for_object_id` for a local's family, but
whether a local becomes an object at all is decided by its caller from the shape of the
assignment — chiefly whether `_infer_type_from_value` recognised the right-hand side.
The two are independent, and the second gate is the blunter one.

Concretely, in `climlab/domain/xarray.py`:

```python
ds = Dataset()          # pyright: xarray.Dataset. No object is created.
da = DataArray(...)     # pyright: xarray.DataArray. No object is created.
```

Step 3 taught the probe to recognise both, and neither reaches the artifact, because
`_infer_type_from_value` has no rule for "a call to a class that is a known container
type" — it has rules for `dict(...)`, `list(...)`, `DataFrame(...)` and the xarray *open*
functions, matched by name. The type checker's answer is available and unused.

This is the constructive half of §1.6: that finding is about the name-matching being too
loose, and this one is about it being the only mechanism. Fix: let a known probe family
expose a local, so the two gates become one.

---

## 2. Pyright probing

This subsystem has an unusual failure mode: everything below degrades to
`FAMILY_UNKNOWN` rather than raising, and `FAMILY_UNKNOWN` is a legitimate value that
the artifacts document. A total probe failure and a correctly-typed unknown are
indistinguishable in the output.

### 2.1 The probe sandbox is built for one project layout — REASONED

`probe_pyright_targets:328-331` copies three hardcoded directories:

```python
for folder_name in ("src", "scripts", "tests"):
```

climlab — this repo's own reference codebase — keeps its package at `<repo>/climlab/`
and matches none of them. Each *probed* file is then written into the temp tree
individually, so what survives is the set of files that happened to have a probe
target. A module with no probe targets (an `__init__.py` of pure imports is the
common case) never reaches the sandbox, so every import routed through it fails, so
every project type behind it is `Unknown`. `reportMissingImports: "none"` in the
generated config suppresses the diagnostic that would have said so.

Copy the source roots the config already knows about, and assert that a non-zero
fraction of probes resolved before accepting the result.

> **FIXED by Step 3 (2026-08-31).** The sandbox now receives every file the stage
> analyses, probed or not. On climlab the eight files that never reached it were **all
> package `__init__.py`** — the files every intra-project import passes through — and
> adding them is worth **+113** resolved answers, which measured as the whole of that
> slice's effect. The assertion refuses a run only when *nothing* resolved, and prints
> the counts otherwise; a fraction would have been a chosen threshold (§4.5).

### 2.2 The generated config drops the project's environment — REASONED

`_write_probe_config` emits `include`, `pythonVersion`, `typeCheckingMode` and two
suppressions. It does not emit `venvPath`, `venv` or `extraPaths`, and the temp root
is not the project root, so pyright's own venv discovery has nothing to find. pandas,
xarray and every other third-party type — precisely the types this analysis exists to
read — resolve only if they happen to be installed in the ambient interpreter running
the pipeline.

> **FIXED by Step 3 (2026-08-31), and it changed nothing measurable.** `extraPaths` is
> derived per file from the module name, a project venv beside the source is passed as
> `venvPath`/`venv`, and otherwise the interpreter running the pipeline is named with
> `--pythonpath` rather than left to be found. Measured on its own: **zero** additional
> answers on climlab, because the temp root already serves as the import root and the
> ambient interpreter already had numpy. The mechanism is real and nothing should depend
> on that luck; it is simply not a source of lost families here.

### 2.3 A source root outside `project_root` crashes the run — REASONED

```python
rel_path = file_path.relative_to(project_root)
```

`config.py` accepts absolute source roots and entrypoints with no requirement that
they sit under `project_root`. When one does not, this raises `ValueError` rather than
degrading. Given that everything else in this subsystem degrades silently, the
inconsistency is worth noting on its own.

### 2.4 Probes are inserted where the type is unreachable — REASONED

`probe_insert_lineno` walks up to the nearest enclosing statement and takes its
`end_lineno`; `_apply_probes_to_source` inserts the probe after that line. For an
attribute inside a `return`, `raise`, `break` or `continue`, the probe lands after
control has left:

```python
return self.data['x']
__msp_probe_7 = self.data          # unreachable
reveal_type(__msp_probe_7)         # pyright: Unknown
```

Pyright types unreachable code as `Unknown`, so the family is lost and nothing
anywhere reports a problem. Terminal statements need the probe *before* them.

> **FIXED by Step 3 (2026-08-31).** 153 probes on climlab (134 after `return`, 19 after
> `raise`). One correction to the wording: pyright does not answer `Unknown` for these —
> it does not answer **at all**, which is why 155 probes had no reply. Moving them in
> front took answers 3,183 → 3,318 and typed answers 1,761 → 1,857. A probe that would
> land inside `if x: return y` is dropped rather than placed, which is the remaining 20.

### 2.5 The callable-ID convention is written three times — REASONED

`collect_pyright_probe_targets.Visitor._callable_id`,
`pyright_type_probe._callable_node_map.Visitor._callable_id` and
`DataAccessCollector._callable_id_for_node` each implement the same rule with a
different formulation of the base case (`current_callable is not None` in two of them,
`is not None and != self.module_callable` in the third).

They agree today; I checked the three branches. The point is what happens if they ever
stop agreeing: probe target IDs stop matching the keys `_family_for_object_id` looks
up, every family becomes unknown, and no test fails, because "unknown" is a valid
answer. One shared function, or one test asserting the three agree on a fixture.

### 2.6 A project name is hardcoded in the shared classifier — CONFIRMED by reading

`pyright_type_probe.py:143`:

```python
or "climlab" in lowered
```

inside the branch that decides whether a type called `Field` is a data container. This
is the same class of framework hardcode that Steps 4 and 4b deliberately removed from
`call_graph/` — and it sits in the module with the widest reuse in this package.

> **FIXED by Step 3 (2026-08-31), ahead of its own step.** It sat inside the function
> §1.1 rewrites, so leaving it would have meant measuring that rewrite against
> climlab-special-cased behaviour. Removed on its own first: **byte-identical artifacts**,
> because the three other clauses in the same condition already matched climlab's `Field`.
> It was dead code, which is worth knowing rather than assuming. §2.7 is untouched and
> remains Step 7's.

### 2.7 The shared-state gate is still a literal keyword — REASONED

`RegisteredStateLineageMixin` matches on the name `state` in two places:

```python
if isinstance(node, ast.Attribute) and node.attr == "state":
if keyword.arg == "state":
```

Step 4b derived the *registration* half of this analysis from evidence and recorded
that "the shared-state gate is unchanged". Restating it as a finding rather than a
note: registration says two objects are coupled, and the `state=` test is the only
thing that says they share an object — so on any framework that does not spell it
`state`, the mixin is inert and produces nothing. This is the remaining obstacle to
roadmap goal 2, and it is the last hardcode of its kind in the package.

The derivable version is the same shape as the escape/invoke join: a parameter whose
identity ends up reachable from both objects' attribute graphs.

---

## 3. Performance

### 3.1 Whole-corpus sets rebuilt per call site — REASONED

`_candidate_callable_ids_for_call:1585`, executed for **every `ast.Call` node**:

```python
known_ids = set(self.callable_map) | set(self.callable_params) | set(self.return_summaries) | set(self.return_tuple_summaries)
```

`_dynamic_getattr_callable_ids:1611` builds the identical set again, and
`_known_class_ids` unions two more sets on every class-reference resolution. On a
matplotlib-sized callable map (~9,400 callables) that is four full-corpus set
constructions per call site per pass. This is very likely the dominant cost of the
whole stage.

Build it once per collector — or better, once per run, alongside the frozen index
§1.2 also wants.

### 3.2 `known_classes` is rebuilt per collector — CONFIRMED

`DataAccessCollector.__init__:906` recomputes the project's class set from the entire
callable map, and a collector is constructed per file per pass:

```
3 source files -> 9 DataAccessCollector instances (3 passes x 3 files)
```

This is precisely the defect `project_index.py` fixed on the call-graph side, where
the same set was being rebuilt 504 times per climlab run.

### 3.3 Every file is parsed ~4× at minimum, ~10× at the cap — CONFIRMED

Counting `parse_python_file` calls inside the data-access stage alone, with Pyright
disabled, on a 3-file package whose fixpoint converged after two passes:

```
parses of each file: {'__init__.py': 4, 'a.py': 4, 'b.py': 4}
attach_parents (top-level) calls: 9
```

That is one parse for `collect_attrdict_classes_from_analysis_files`, one per fixpoint
pass, and one for the final pass, each followed by a fresh recursive `attach_parents`
over the whole tree. With `MAX_RETURN_SUMMARY_PASSES = 8` the worst case is ten. The
count excludes the Pyright target-collection pass (a further one per file) and the
`analyze_analysis_files` prefix that `run_from_extraction_config` runs first.

`call_graph.ast_utils.ParsedFileCache` exists, is documented for exactly this
situation, notes that `attach_parents` is idempotent so one call at parse time
suffices — and is not used anywhere in `data_access/`. **See §5.3: it is worse than
an unused utility. `run_from_extraction_config` is already holding a fully populated
instance and drops it on the floor.**

### 3.4 `None` overloaded as an error policy — REASONED

`_pyright_families_for_config` returns `{}` for "disabled", a dict for "computed", and
`None` for "the fail policy is `error`, so let the recomputation downstream raise".
The third meaning is invisible at the call site and costs a second full parse of every
file to reach the exception. Raise here instead.

---

## 4. Structure

### 4.1 Four entry points, thirteen-to-fifteen parameters each — REASONED

`collect_data_access_from_source`, `collect_data_access_from_tree`,
`collect_data_access_from_analysis_files` and `collect_data_access` all thread the same
eight mutable dictionaries by hand. `DataAccessCollector.__init__` takes 15
parameters, twelve of them `Optional` with an `x if x is not None else {}` line each.

The divergence matters more than the verbosity: `collect_data_access_from_source` is
the path every unit test uses, and it passes neither `registration_rules` nor
`project_index` nor `param_bindings` during the fixpoint. **The tests exercise a
different analysis from production.**

This is item (a) of the call-graph review, unaddressed here. The remedy is the same: a
frozen `DataAccessContext` carrying the callable map, project index, families,
attrdict classes, split owners and config, with file discovery a caller concern.

### 4.2 `DataAccessCollector` fuses five concerns — REASONED

One class, ~1,700 lines, doing: AST walking; the type and scope environment
(`Scope`, the six per-scope dictionaries, and their manual push/pop); name and
attribute resolution; object identity minting (`_field_object_id`,
`_root_relative_access_path`, `_safe_path_id_part`); and lineage recording.

The call-graph split is the template — `collector/`, 15 files, largest 379 lines,
verified at each step by a byte-identical artifact diff. The natural seams here:

```
data_access/
  objects.py     # ID minting + the object registry and its merge rule (1.4)
  typeenv.py     # Scope stacks, plus the branch merging they do not do (4.4)
  resolve.py     # _resolve_name / _resolve_attribute / _resolve_subscript_fields
  lineage.py     # lineage edges, param bindings, alias solving
  passes/        # returns.py  params.py  edges.py  fixpoint.py
  pyright/       # targets.py  probe.py  families.py
```

Do this **after** §6, not before. The call-graph history is explicit that a split
performed before the measurement exists preserves defects rather than removing them —
its own two rounds of modularisation left the `_resolve_callees` ↔ `_infer_class_types`
cycle exactly where it was.

### 4.3 Duplicated helpers, drifting independently — REASONED

- `collect_module_imports` (`generate_data_access_ast.py:823`) and
  `attrdict._module_imports` are near-identical; only the first handles star imports,
  so attrdict detection silently misses star-imported bases.
- `_lineage_roots` and `infer_shared_containers._lineage_root_ids` are the same
  function with the same bug (§1.3).
- `_confidence_max` and `_confidence_weight` are one-line pass-throughs to `models`.
- The three callable-ID visitors of §2.5.
- Class-reference resolution exists three times — `_resolve_class_reference_name`,
  `attrdict._resolve_class_reference`, and `ProjectIndex.resolve_class_reference_name`
  — and only the third is correct. §5.2.

### 4.4 No control-flow modelling — REASONED, and partly by design

One mutable scope, last-assignment-wins. There is no `if`/`else` merge (the call graph
at least merges `If`), no `try`/`except` (which is where optional-dependency imports
and fallback bindings live), no loop re-analysis, and `global`/`nonlocal` are ignored.
A binding created in an `if` body is live in the `else` branch and after the
statement.

Recorded here as a limitation rather than a bug, because fixing it properly means
building the join lattice that `typeenv.py` would own. But it should be *written
down*: `limitation.md` on the call-graph side exists for exactly this, and this
package has no equivalent.

> **Step 1b (2026-08-29) found one place where this stops being a limitation and becomes a
> bug.** Last-assignment-wins over return statements makes a callable with one
> `return <parameter>` branch claim it returns that parameter *always*, and the caller turns
> that into `alias_of` — so a value merely computed from an argument is recorded as being
> that argument. That is not coarseness, it is a false merge, and it is the single largest
> source of them in the artifact. It is scheduled at the top of Step 4, and it does **not**
> need the join lattice this section describes: enumerating a function's `Return` nodes is
> enough to tell "returns its argument on some path" from "returns its argument".
>
> **Fixed by Step 4a (2026-08-31).** The last sentence needed one correction: enumerating
> `Return` nodes is not quite enough, because a function that falls off the end returns
> `None` on that path and no `Return` node records it. With that added, the claim holds —
> no join lattice was needed, and contradicted identity claims went 51 → 17. **The rest of
> this section is untouched and remains the one finding in this document with no step of
> its own.**

### 4.5 Thresholds chosen rather than derived — REASONED

`COORDINATOR_ATTR_THRESHOLD = 4`, `COORDINATOR_METHOD_THRESHOLD = 3`,
`COORDINATOR_CONTAINER_THRESHOLD = 2` decide whether a class's state is one node or
many — i.e. they decide object identity — and nothing records where they came from.
`infer_shared_containers`'s `jaccard < 0.6 and overlap < 0.8` is the same. The
roadmap's own rule, learned from the fan-out confirmation table: derive thresholds,
do not choose them.

### 4.6 `unknown`-kind objects are an unmeasured diagnostic — REASONED

`_record_access` mints a placeholder object whenever an edge names an ID that no pass
materialised. The artifact README documents `unknown` as a legitimate object kind,
which normalises it. It is better read as a defect counter: every one of them is a
lineage or return summary pointing at an object that does not exist, which is exactly
the residue §1.2, §1.4 and §1.5 leave behind. Count them, print the count, and drive
it down.

---

## 5. What the call-graph revision changed here

`data_access/` imports fifteen symbols from `call_graph/` and consumes three of its
data structures, so the modularisation and the fixes that followed it did not stop at
the package boundary. They split cleanly into two groups, and the split is the useful
part:

**What propagated for free.** `dde4cc1` (pass exclude globs) landed in
`discovery.py`, which data access *calls* — so it inherited the fix without anyone
touching this package. The same is true of `ast_utils`, `import_resolution` and
`ProjectIndex` itself.

**What did not.** Every fix that landed in `collector/` stopped at the boundary,
because data access does not call that code — it reimplements it. The three
reimplementations are `_candidate_callable_ids_for_call`,
`_resolve_class_reference_name`, and the callable-ID convention of §2.5. All three
now lag their call-graph counterparts, and the gap is no longer theoretical.

### 5.1 The module-alias fix has an unfixed twin here — CONFIRMED by comparison

`ca9c1fc` is the most recent call-graph commit. Its own docstring states the case:

> `import parcels._sgrid as sgrid` followed by `sgrid.get_n_faces()` names
> `parcels._sgrid.get_n_faces`, but the function is defined in
> `parcels._sgrid.core` and merely re-exported by the package `__init__`. Only the
> defining path is a known callable, so without the same alias canonicalization the
> `ast.Name` branch already does, the target looks external and the edge is dropped.

The fix routes every dotted-alias branch through
`project_index.canonical_callable_id`. Data access has the *pre-fix* code, at
`generate_data_access_ast.py:1569-1572`:

```python
if "." in call_name:
    prefix, _, suffix = call_name.partition(".")
    if prefix in self.module_imports:
        candidates.append(f"{self.module_imports[prefix]}.{suffix}")
```

No canonicalization, and `known_ids` is the raw `set(self.callable_map) | ...`. The
alias path is not a key in any of those maps, so the same call that used to drop an
edge in the call graph currently drops, in data access:

- `_return_ref_from_call` → `None`, so the value flowing out of the call has no type
  and no object, so the local it binds is never tracked as a container. **Objects and
  their access edges vanish**, not just a relation.
- `_record_confirmed_param_lineage` → no candidate, so no `arg_to_param` lineage,
  so `_apply_confirmed_param_aliases` has nothing to work with.

Data access is *already handed* `project_index`; the fix is the same one-line
canonicalization. Note that this is the failure mode the call graph measured at 55 of
110 missing edges before Step 3b — the largest single gap it had.

### 5.2 `ProjectIndex.resolve_class_reference_name` supersedes the local copy — CONFIRMED by comparison

Side by side, `ProjectIndex.resolve_class_reference_name` (`project_index.py:130`)
against `DataAccessCollector._resolve_class_reference_name`
(`generate_data_access_ast.py:933`):

| | ProjectIndex | data_access |
| --- | --- | --- |
| class universe | `module_index.classes` | flat set derived from `callable_map` |
| re-export aliases | `known_class_id(...)` canonicalizes | none |
| star imports | `resolve_star_import_targets` | none |
| ordering | `sorted` | unordered `set` |

The star-import gap is the odd one: data access *does* collect `star_imports` and
*does* use them in `_candidate_callable_ids_for_call`, so star imports resolve for
callables and not for classes, in the same collector.

What this costs is concentrated in one place. `_class_types_from_expr` feeds
`registration_lineage`, whose gates are `len(parent_types) != 1` and
`len(child_types) != 1`. A class referenced through a re-export alias resolves to the
empty set, the gate declines, and **registration lineage silently produces nothing** —
no edge, no warning, no diagnostic. climlab reaches its base classes through package
`__init__` re-exports, which is exactly what Step 3b was written for.

That makes the low registration-lineage count on climlab (3 edges, against 21
`registered_invoke` edges in the call graph) a **hypothesis worth testing** rather
than something explained solely by the `state=` gate of §2.7. Test it by swapping in
`project_index.resolve_class_reference_name` and counting.

`attrdict._resolve_class_reference` is a third implementation, weaker than both — it
returns a single string and never checks whether the result is a known class.

### 5.3 The parsed-file cache is built, handed over, and dropped — CONFIRMED

`CallGraphAnalysis` is a frozen dataclass whose first field is
`cache: ParsedFileCache`, and whose docstring names this package as the reason it
exists:

> Split out from graph construction because the facts have a second consumer: the
> data-access stage derives its registration lineage from `registration_rules`, and
> needs `project_index` to resolve a call onto them.

By the time `analyze_analysis_files` returns, that cache holds every file, parsed
once, with `attach_parents` already applied. `run_from_extraction_config:2875-2888`
takes `project_nodes()`, `registration_rules` and `project_index` off the result —
and never touches `.cache`, then re-parses every file 4–10 times (§3.3).

This upgrades §3.3 from "adopt an available utility" to "stop discarding a populated
value you are already holding," and it makes the fix roughly a signature change:
thread `analysis.cache` through `collect_data_access_from_analysis_files` and replace
the four `parse_python_file` calls with `cache.get`.

### 5.4 Class-body nodes are now in `callable_map`, with no `resolvable_callable_ids` filter — LATENT

Step 3 added `class_body` nodes, keyed by the class ID itself (`definitions.py:184-196`),
and the call graph needed a guard for them immediately:

> Class bodies are nodes … but they are never call targets … Leaving them in the
> resolution universe makes every construction of such a class resolve to the body
> instead of the constructor, which silently deletes the real edge.

`project_nodes()` does not apply that filter — `resolvable_callable_ids` is a separate
function the call graph calls at its own resolution sites. Data access receives the
unfiltered map and has no equivalent, and
`_candidate_callable_ids_for_call` puts `mod.ClassName` **ahead of**
`mod.ClassName.__init__` in the candidate list, while `_return_ref_from_call` and
`_return_refs_from_call` both take the first candidate that hits.

I traced this and **it does not fire today**: the maps those two consult
(`return_summaries`, `return_tuple_summaries`, `callable_params`) are populated by data
access's own traversal, which never enters a class body, so a class-body ID can never
be a key in them. It is recorded as latent rather than live — but it is one
`_enter_callable` change away from silently deleting constructor-return inference, and
the call graph's own experience is that this exact trap cost it five constructor edges
before anyone noticed.

### 5.5 Lambdas are call-graph nodes with no data-access counterpart — REASONED

Step 8 made `definitions.visit_Lambda` index every lambda as
`<enclosing>.<locals>.<lambda>`, CPython's own `co_qualname`. Data access's
`visit_Lambda` still just walks the children in the enclosing scope and enters no
callable.

So for every lambda in the analyzed project, the structural graph now has a callable
node carrying call edges and **zero** access edges, while the enclosing function
carries access edges that belong to the lambda. The two stages disagree about which
callable touched the data. Compounded by §1.9, where the lambda's own parameters
resolve to enclosing locals, the accesses attributed to the enclosing function are not
merely misplaced but wrong.

### 5.6 `resolve_method_targets` is on notice, and this package is a named consumer — REASONED

Step 2's closing paragraph:

> `resolve_method_targets` was deliberately left alone … switching it to a single MRO
> winner would be unmeasurable here while risking four consumers
> (`resolve_constructor_targets`, `virtual_override`, `registration.py`,
> `data_access/registration_lineage.py`).

`_registration_rule_for_call` iterates `resolve_method_targets(...)` and returns the
**first** target that carries a rule. Narrowing that union to the single C3 winner can
therefore *remove* a registration rule — if the winner has no rule while some other
member of the union did — and delete lineage edges. The change is planned, the
direction of the effect is not obvious, and there is currently no data-access test
that would catch it.

Concretely: before that change lands, this package needs a fixture with a diamond
hierarchy where the C3 winner and the union disagree, asserting what registration
lineage does.

### 5.7 The tunable confidence weights are dead for data-access edges — CONFIRMED

Step 7 turned `confidence` into a graded, derived signal on the call-graph side and
gave the structural graph a configurable `confidence.weights` mapping. The two
branches of `generate_structural_dependency_graph` read it differently:

```python
# call edges (line 550)
confidence_weight = weight_config.confidence_weight(confidence)

# data-access edges (line 575)
confidence_weight = _confidence_weight(edge, weight_config)
```

and `_confidence_weight` short-circuits on the row's own column:

```python
if "confidence_weight" in row:
    return _as_float(row.get("confidence_weight"), weight_config.confidence_fallback_weight)
```

`outputs._edges_payload` **always** emits that column, computed from the hardcoded
`models.CONFIDENCE_WEIGHT = {"low": 0.25, "medium": 0.6, "high": 1.0}`. So the
short-circuit always wins: **retuning `confidence.weights` moves the call half of the
structural graph and leaves the data half exactly where it was.** Anyone sweeping that
knob against `evaluate` — as Step 7 swept `virtual_override`'s — will read the result
as "data-access confidence does not matter" when the truth is that the knob was never
connected.

Two vocabularies have also drifted apart: `call_graph.models` now has four confidence
values including `CONFIDENCE_UNKNOWN`, while `data_access.models.CONFIDENCE_RANK` has
three. Nothing mints `unknown` on the data side today, so this is a divergence to
close rather than a live bug.

---

## 6. There is no way to tell whether any of this matters

Every finding above is a statement about mechanism. None is a statement about impact,
because `data_access/` has no oracle: no ground truth, no precision or recall, no
regression gate. The call graph was in exactly this position, and the thing that
changed it was not any individual fix — it was `dynamic_trace.py`. Recall went 70.8%
→ 99.7% because the instrument said which gaps were expensive, and the two most
valuable lessons in `ground_truth_and_roadmap.md` are both about instruments rather
than about analysis:

> An instrument that can only find one kind of error will only find one kind of error.
>
> A number that cannot fail is not a measurement.

The equivalent instrument here is buildable, and most of it already exists.
`dynamic_trace.py` runs the analyzed project in-process under `sys.monitoring` and
already maps runtime frames onto the same callable IDs the static passes emit, via
`module_map_from_analysis_files`. What data access needs is a different event set on
the same harness: `INSTRUCTION` events filtered to `LOAD_ATTR` / `STORE_ATTR` /
`BINARY_SUBSCR` / `STORE_SUBSCR`, with `code.co_positions()` giving the
`(line, col_offset)` that Step 6 already taught `Edge` to carry. That yields observed
(callable, attribute-or-key, access-kind) triples to score `access_edges.csv` against,
broken down by object kind — which is the breakdown that says whether §1.1 or §1.7 is
the expensive one.

Three properties of the call-graph comparison transfer unchanged and should not be
re-litigated: the trace is a **lower bound**, an unconfirmed edge is **not** a false
one, and relations the interpreter cannot expose must be excluded before scoring
rather than counted as gaps. Two things are new here and need deciding before the
number means anything:

- **Key identity.** The static side names a key by its literal (`df['mass_g']`); the
  runtime side sees the value. Literal-keyed accesses compare directly; computed keys
  do not, and should be scored separately rather than folded in.
- **The object-identity question is not the access question.** Whether
  `local_exposed:f:df` and `param:g:frame` are *the same object* is what `alias_of`
  and the lineage graph claim, and a trace can settle it directly with `id()` at the
  observed access — which would make §1.3 and §1.7 measurable rather than arguable.
  This is the data-access analogue of the falsification instrument from Step 6, and
  worth the same effort.

### 6.1 The two instruments are not equally ready — added 2026-08-23, after Step 0

That second bullet draws a distinction this section then stops short of acting on. The
access instrument and the identity instrument have **different prerequisites**, and
building them together would hold the ready one hostage to the blocked one.

**The access instrument is ready now.** Its unit is a
`(callable, attribute-or-key, access-kind)` triple read off the *access site*. Step 0
is the evidence that this is the stable half of the artifact: a change that added 12
objects and 18 access edges removed **nothing**, and every field that did move was on an
object rather than an edge. More to the point, the triple stays meaningful even where
§1.2 shifts the object ID an edge points at, because the object ID is not part of it.

**The identity instrument is blocked on Step 2.** What it scores is `alias_of` and the
lineage graph — which is exactly what §1.3 (roots cached from inside a cycle) and §1.4
(first-file-wins cross-file merge) make a function of file order rather than of the
input. Step 0 measured that directly: a strictly additive change moved **22 derived
fields** on climlab, 18 of them `alias_of`. An instrument calibrated against that today
is calibrated against something that moves when the file order does, and its baseline
would not be reproducible.

So §6 splits into **Step 1a** and **Step 1b**, with Step 2 between them. See §7.

> **Both are built (2026-08-24 and 2026-08-29).** The split was the right call for the reason
> given, and for one this section did not anticipate: the two instruments turned out to need
> *opposite* sampling rules. The access tracer retires each site on first hit, because an
> instruction that fetches `spacing` always fetches `spacing`. Carrying that rule into the
> identity tracer inflated its contradiction count by a factor of three, because a loop binds
> a different object each pass. Building them as one instrument would have hidden that.

**What the harness already provides** (checked 2026-08-23), which is why 1a is a smaller
piece of work than this section makes it sound:

| needed | status in `call_graph/dynamic_trace.py` |
| --- | --- |
| in-process execution under `sys.monitoring` | `CallTracer.__enter__` / `__exit__` |
| runtime frame → static callable ID | `module_map_from_analysis_files`, `_code_id` |
| bytecode offset → `(line, col_offset)` | `_position`, already using `co_positions()` |
| drivers (pytest / notebook / script) | `run_pytest`, `run_notebook`, `run_script`, `trace_all` |
| headless plotting, cwd handling | `_headless_matplotlib`, `_pushd` |
| climlab drivers configured | `extraction.jsonc` → `trace.pytest_args` + `notebook_globs` |

What is missing is the event set: it registers `events.CALL` only. 1a adds `INSTRUCTION`
filtered to `LOAD_ATTR` / `STORE_ATTR` / `BINARY_SUBSCR` / `STORE_SUBSCR`, and a
comparison module beside `graph_comparison.py`.

---

## 7. Plan

Ordered so that each step is checkable by the time it lands. §5 changed this
ordering: it moved alias canonicalization to the front, because those findings are
already-solved problems whose fix is sitting on an object this package is handed, and
because two of them make data access *silently emit nothing* rather than emit
something wrong.

**Execution order** (revised 2026-08-31, after Step 3):

```
0  ->  1a  ->  2  ->  1b  ->  4a  ->  3  ->  4  ->  5  ->  7  ->  8
DONE   DONE   DONE   DONE   DONE   DONE  <- next
       |       |      |      |      |
       |       |      |      |      +- container families; also removed 2.6
       |       |      |      +- the false-merge fix, taken out of order: see below
       |       |      +- identity oracle: needs Step 2's determinism to have a
       |       |         reproducible baseline
       |       +- the one gate that needs no oracle, so it can go early
       +- access oracle: ready today, and everything after 3 needs it
```

Step numbers are unchanged from the original plan so existing references still resolve;
what changed is that **Step 1 split into 1a and 1b with Step 2 between them** (§6.1),
**Step 6 left the sequence** — it is now opportunistic, see below — and **the top item
of Step 4 was pulled out ahead of Step 3 as Step 4a**.

> **Why 4a jumped Step 3 (2026-08-31).** Not because it is cheap. Step 3 changes
> container families, which changes `_field_kind`, which changes the object IDs the Step
> 1b baseline is entirely composed of. Run in the planned order, the identity numbers
> would have moved for reasons unrelated to false merges and the fix could not have been
> credited with anything. Run first, it was judged against a two-day-old baseline that
> reproduced to the last digit. The two steps are scored by *different* instruments
> — Step 3 by 1a's recall, this by 1b's precision — so nothing else in the ordering
> depended on it.

Two things Step 0 settled that this ordering now assumes:

- **§2.7 is no longer the prime suspect** for climlab's low registration-lineage count.
  Class-reference resolution was the larger cause (3 → 13 edges). Step 7 is still worth
  doing; it is no longer the explanation for anything.
- **Byte-identical artifact diff does not currently work as a verification method.**
  That is Step 8's stated technique, so Step 8 is blocked on Step 2, not merely on the
  oracle.

**Step 0 — Adopt what the call graph already fixed: §5.1, §5.2, §5.3.** Cheapest work
in this document and the only work that is already known-correct elsewhere. Route
callee candidates through `project_index.canonical_callable_id`, replace
`_resolve_class_reference_name` with `project_index.resolve_class_reference_name`, and
thread `analysis.cache` instead of re-parsing. Acceptance: registration lineage edge
count on climlab is recorded before and after — §5.2 predicts it rises, and if it does
not, that hypothesis is dead and §2.7 is the whole explanation. Do this first because
it is also the cleanest test of whether the two packages can share resolution at all.

> **DONE (2026-08-23) — see [`step0_adopt_call_graph_fixes.md`](revision_progess/step0_adopt_call_graph_fixes.md).**
> Registration lineage on climlab went **3 → 13**, so the §5.2 hypothesis survived and
> §2.7's `state=` gate is *not* the whole explanation. Objects +12, access edges +18,
> lineage edges +21, **nothing lost**; parses per file 4 → 1, or 0 with the shared cache.
> Two departures from the wording above, both explained there: §5.2 became a *union* with
> the local resolver rather than a replacement (`ProjectIndex` does not know attrdict
> classes), and §3.1's `known_ids` hoist was deliberately left for Step 6 because those
> sets are mutated mid-traversal. Note also that the new lineage moved 22 derived
> `alias_of` / `access_path` fields via §1.3 and §1.4 — further evidence for doing Step 2
> before anything judged by artifact diff.

**Step 1a — The access oracle (§6, first half).** Observed
`(callable, attribute-or-key, access-kind)` triples from a traced run, scored against
`access_edges.csv`. Nothing after this can be shown to help until it exists, and the
call-graph history is that the instrument found defects nobody had listed. Ready to
build today — see §6.1 for what the harness already provides. Acceptance: a
per-object-kind recall report against a traced run of climlab, plus a baseline recorded
in this document. Keep the three transferred properties: the trace is a lower bound, an
unconfirmed edge is not a false one, and inexpressible relations are excluded before
scoring rather than counted as gaps. Score computed keys separately from literal ones.

> **DONE (2026-08-24) — see [`step1a_access_oracle.md`](revision_progess/step1a_access_oracle.md).**
> Baseline on climlab: **recall 69.0%** (1,598 of 2,317 observed accesses found),
> **75.2% of static claims confirmed**, **1 falsified**. 98.0% of the artifact is scored
> (4,156 of 4,239 rows); every exclusion is counted and printed, in both directions.
> Literal keys 88.6%, computed keys **1.7%** — scored apart, as required. Trace runs in 9 s.
>
> Three departures from the wording above, all explained there. (i) `BINARY_SUBSCR` **does
> not exist** on this repo's Python 3.14 — it is `BINARY_OP` with `NB_SUBSCR`. (ii) The four
> opcodes named in §6 cover only ~36% of the artifact; `load` alone is 1,892 edges, 1,409 of
> them on `param`, the largest object kind. A **name tier** (`LOAD_FAST`/`LOAD_NAME`/
> `LOAD_GLOBAL`) was added, taking scoreable coverage to 98%. (iii) The comparison gains a
> verdict the call-graph one cannot have: because a code object's full instruction set is
> fixed at compile time, **falsified** is provable without coverage.
>
> The instrument found a defect listed in no section of this document: **default-argument
> expressions are attributed to the callee rather than to the definer**
> (`RRTMG_SW.__init__` claims a read of `nbndsw`, which Python evaluates in the module body
> at `def` time). It also puts a number on §5.5 — 84 observed accesses in lambdas and
> genexps that `access_edges.csv` has no row for.
>
> Note for anyone reading a future run: the first climlab pass reported **89** falsified
> claims and all 89 were the *instrument's* fault — `LOAD_NAME`/`STORE_NAME` (module and
> class bodies, 2,284 instructions) and `STORE_FAST_STORE_FAST` (tuple unpacking) were not
> decoded at all. Hand-checking the falsified list against the source is what caught it, and
> it is the reason this instrument reports named refutable claims rather than only a score.
> No line of `generate_data_access_ast.py` was changed in this step.

**Step 2 — Determinism and ID consistency: §1.2, §1.3, §1.4, §1.5.** These make the
output a function of the input rather than of file order. Do them before anything that
would be evaluated by comparing artifacts, because until they land a diff cannot
distinguish a real change from a reordering. Acceptance: two runs with shuffled file
order produce byte-identical artifacts.

> This is the **only** acceptance gate in this plan that needs no oracle, which is why
> it can sit ahead of Step 1b rather than behind the whole of Step 1. Step 0 supplied
> the evidence that it is needed early: a strictly additive change moved 22 derived
> `alias_of` / `access_path` fields on climlab.

> **DONE (2026-08-29) — see [`step2_determinism.md`](revision_progess/step2_determinism.md).**
> The gate passes: shuffled climlab runs are byte-identical over five seeds, where before
> the shuffle moved 2 objects and 6 access edges in or out of existence. Objects
> 1870 → 1866 — five `class_state:X` nodes collapsed into the `class_attr_state:X:state`
> nodes they had been duplicating, which is §1.2's stated cost being removed, not
> information lost. Access edges and lineage edges **unchanged** (4239, 1175, every
> relation kind); §4.6's `kind == unknown` counter **4 → 0**. The Step 1a oracle is
> unmoved — recall 69.0%, confirmed 75.2%, falsified 1 — exactly as §6.1 predicted, since
> the access triple never mentions an object ID.
>
> §3.1 and §3.2 were folded in as §7's Step 6 suggested, and **Step 6 is now fully
> discharged**: stage time 0.904 s → 0.431 s. §3.1's fix is *not* the frozen index proposed
> above — that would reintroduce the mid-traversal staleness Step 0 deferred it for. The set
> is only used for membership tests, so testing the four dicts directly is equivalent and
> cannot go stale.
>
> Three departures and two additions, all explained there. (i) The call graph was verified
> order-stable **first**, so every difference found afterwards was known to be this package's;
> `--check-inputs` keeps that test. (ii) §1.2's claim that `_class_attr_ref` is file-local is
> not quite right — its owner is always a class in the file being walked. The real holes were
> `registration_lineage._class_state_object_id_for_owner` and **nested classes**, which
> `infer_split_class_owners` never saw because it walked `tree.body` only. (iii) §1.3 was fixed
> by not caching a cycle-truncated answer rather than by computing SCCs; the same fix was applied
> to the `infer_shared_containers` copy, which §4.3 predicts would otherwise drift.
>
> **The instrument found two defects in no section of this document.** The larger: the return
> fixpoint was run *without* `project_index`, while the final pass has it — so the final pass
> resolved calls the fixpoint could not, learned new return summaries **while running**, and
> handed them only to the files it reached afterwards. Convergence was therefore meaningless: the
> loop had settled on a smaller question. That is what made the last two objects order-dependent.
> The smaller: two output sort keys omitted columns, and Python's stable sort kept tied rows in
> file order — an artifact-layer difference with no analysis change behind it.
>
> Note also that `call_graph`'s own type-summary fixpoint hits `max_iterations=5` on climlab and
> warns. Its outputs are still order-stable, so nothing here depends on it, but it belongs on that
> package's list.
>
> **`check-data-access-determinism` is permanent, not scaffolding for this revision** — the
> argument is in §9 of that document. Short version: order-independence is an invariant with no
> finish line, it breaks silently, and Steps 3, 4, 7 and 8 are the most likely things to break it —
> Step 8 in particular *depends* on it, so retiring the check when Step 8 ends would delete the
> thing that made Step 8 checkable. The fixture test is a CI guard, not a replacement: a fixture
> only holds shapes someone thought to write down, and both unlisted defects above were found on
> climlab.

> **UNBLOCKED (2026-08-29) by Step 2.** The baseline this needs is now reproducible.

**Step 1b — The object-identity oracle (§6, second half).** `id()` at the observed
access, settling whether two static object IDs are one runtime object — which makes
§1.3 and §1.7 measurable rather than arguable, and is the only thing that can judge
whether §1.7's `file:path` collapse manufactures real coupling. Deliberately **after**
Step 2: what it scores is `alias_of` and the lineage graph, the two things §1.3 and §1.4
make order-dependent, so a baseline taken before Step 2 would not be reproducible.
Acceptance: an alias precision/recall report, and a recorded count of static alias
claims the trace contradicts.

> **DONE (2026-08-29) — see [`step1b_identity_oracle.md`](revision_progess/step1b_identity_oracle.md).**
> Baseline on climlab: **alias precision 84.5%** (279 confirmed, **51 contradicted**, 690
> unobserved of 1,597 claims), **alias recall 65.9%** — 7,915 of 23,225 runtime aliasings are
> split across unconnected parts of the static graph. 76.4% of objects name a site the
> instrument can read. Step 1a's numbers are unmoved (recall 69.0%, confirmed 75.2%,
> falsified 1) and the Step 2 determinism gate still passes byte-identical.
>
> Six departures from the wording above, all explained there. The load-bearing ones: (i)
> "`id()` at the observed access" is not possible — the operand stack is unreachable from
> Python — but `sys._getframe` inside the callback is, so values are read out of the running
> frame and out of `self.__dict__`, never via `getattr`, which would execute properties.
> (ii) A raw `id()` is not an identity, because CPython recycles addresses; the weakref fix
> does not work since `list` and `dict` are not weak-referenceable, so one strong reference
> per recorded object is kept, under a reported cap. (iii) **§1.7 cannot be answered by
> `id()` at all** — file identity is a path *value* — so a second value channel was added.
>
> **"Contradicted" here is not coverage-independent, unlike Step 1a's "falsified".** There is
> no compile-time fact to fall back on, so the 51 is a lower bound carrying that caveat, and
> all 51 were hand-checked against the source before the number was recorded.
>
> **The instrument was wrong the first time, in a new way.** Carrying over Step 1a's
> retire-on-first-hit rule reported **152** contradictions at 52.4% precision; an identity
> site is not like an access site, because a loop binds a different object each pass.
> Sweeping the sample budget gives 8 → 152, 64 → 56, 512 → 51, 4096 → 51, so **two thirds of
> the first count was the instrument**, the default is now calibrated at 512, and the run
> reports how many sites still exhausted their budget.
>
> **The instrument found a defect in no section of this document, and it is a large one:**
> the lineage graph records "made from" as "is". A value produced by a call is claimed to
> *be* one of that call's arguments — `dic = self.state.copy()`, `self.timeave =
> self.state.copy()`, and ~18 lines of `cam3.py` of the form `Tatm =
> self._climlab_to_cam3(self.Tatm)`. The cause is `_merge_return_summary` keeping a single
> "best" return summary, so one `if np.isscalar(field): return field` branch makes a whole
> function "returns its own argument" at every call site — §4.4 producing wrong identity
> rather than merely coarse identity. Scored by claim source, `arg_to_param` is 106/6 while
> `local_assign` is 78/31: **passing an argument is the claim the extractor gets right, and
> assigning a call result is the one it gets wrong.** It also gives §5.6 a concrete example
> (`super(MutableAttr, self).__setattr__` resolved to `MutableAttr._setattr`).
>
> **§1.7 does not fire on climlab.** Only one `file:` node is touched by more than one
> callable and its callables never disagreed on the path. That is a property of climlab, not
> a refutation of §1.7 — the test fixture reproduces the defect exactly — so §1.7's fix must
> be judged elsewhere, or by `evaluate` as this section already prescribes.
>
> No line of `generate_data_access_ast.py` was changed in this step.

**Step 3 — Container families: §1.1, §2.1, §2.2, §2.4.** The single largest precision
lever, and Step 1a makes it measurable. Acceptance: the share of objects with
`inferred_type == unknown` falls, and per-kind recall rises with it. Add the
probe-resolution assertion from §2.1 so a silent total failure can never look like a
clean run again.

> **DONE (2026-08-31) — see [`step3_container_families.md`](revision_progess/step3_container_families.md).**
> Objects with an unknown family **759 (40.6%) → 556 (28.7%)**; named `dict_key` objects
> 69 → 187 and anonymous `container_field` objects 141 → 25. At the probe level, answers
> Pyright could type went 1,648 → 1,857 and answers thrown away as unknown 2,220 → 1,637.
> **Access edges unchanged at 4,239 at every slice**, and the Step 1a oracle is unmoved
> (recall 69.0%, confirmed 75.2%, falsified 1). Determinism byte-identical over five
> seeds. §2.6's `"climlab"` hardcode was removed here, on its own, and produced
> byte-identical artifacts — it was dead code.
>
> **The acceptance criterion above is half wrong, and the correction matters.** "Per-kind
> recall rises with it" cannot happen: the access oracle's unit is a
> `(callable, attribute-or-key, access-kind)` triple that never mentions an object, which
> is the same property that kept it still through Steps 2 and 4a. What families change is
> *which kind of object the confirmed accesses are filed under* — named dictionary keys
> 36 → 226, anonymous container fields 167 → 29, with the total confirmed identical at
> 3,124. A redistribution, not a gain, and it has to be read as one.
>
> Four departures from the wording above. (i) §2.1's sandbox fix is the whole of slice 1's
> effect (+113 answers) and §2.2's environment settings contributed **nothing** on climlab
> — the temp root is already the import root and the ambient interpreter already had
> numpy; they are kept as correctness, not as a measured gain. (ii) The probe-resolution
> assertion refuses a run only when *nothing at all* resolved, and prints the counts
> otherwise; any threshold short of that would be §4.5's mistake. (iii) §1.1's second
> instance (a bare `Dataset` from `from xarray import Dataset`) is fixed by handing the
> classifier each module's import map, and it changes **two probe answers and zero
> artifact rows** on climlab — real, unit-tested, and not exercised here, as with §1.7.
> (iv) Hand-checking all 626 reclassified answers found one regression the totals hid:
> pyright writes a method's own class as `Self@AttrDict`, which the new head-parsing read
> as an unknown name.
>
> **Alias precision 93.6% → 93.1%, contradicted claims 17 → 20.** All three new ones are
> `AplusBT(state=self.state)`, `StepFunctionAlbedo(state=self.state)` and
> `MeridionalHeatDiffusion(state=self.state)` in `ebm.py` — the same shared-state
> constructor defect Step 4a already found as `Iceline(state=self.state)` and assigned to
> **Step 7**. They are not new: `class_attr_state:...EBM:state` did not exist before this
> step, so the claim had no node to land on and could not be scored. Verified against both
> artifacts directly.
>
> **Two findings in no section of this document**, both recorded above as §1.15 and §1.16:
> `object` swallows scalars, and a local's family can be known while the local is never
> modelled as an object at all.

**Step 4 — Cheap correctness: §1.6, §1.7, §1.8, §1.9, §1.10, §1.11, §1.12, §1.13, §1.14,
plus §5.5, plus the "made from" / "is" defect Step 1b found.** Each is local and
independently testable.

> **Added by Step 1b (2026-08-29): the lineage graph records "made from" as "is".** A value
> produced by a call is claimed to *be* one of that call's arguments. Measured: 31 of the 78
> scored `local_assign` claims on climlab are contradicted by the trace, against 6 of 106 for
> `arg_to_param` — **passing an argument is the claim the extractor gets right, and assigning
> a call result is the one it gets wrong.** This belongs at the *top* of this step, not the
> bottom: it is the largest source of false merges in the artifact, and false merges are why
> §1.7 was ranked above its apparent size. Judge it with the Step 1b oracle's precision
> figure, currently 84.5%.
>
> It is **two defects**, and the first is the cheapest fix in this whole step:
>
> **(a) `.copy()` is hard-coded as an alias.** `_infer_type_from_value`, in the `ast.Call`
> branch, returns the base object as `alias_of` for any `x.copy()`. A copy is the one call in
> the language whose entire purpose is to *not* be the same object. Four lines, no judgement
> call, and it accounts for 5 of the 51 contradictions (`dic = self.state.copy()`,
> `self.timeave = self.state.copy()`, and the two `alias_of` rows derived from them).
>
> **(b) A single return path speaks for the whole callable.** The same branch hands
> `_return_ref_from_call(...).object_id` back as `alias_of`, and `_merge_return_summary`
> keeps one "best" summary per callable, so `if np.isscalar(field): return field` makes
> `_climlab_to_cam3` "returns its own argument" at all eighteen of its call sites. The
> remaining ~46. The fix is to alias only when *every* return path returns the same source.
>
> **Neither fix needs §4.4.** §4.4 is the underlying cause and is recorded there as a
> limitation rather than a bug, but "do all return statements return this parameter?" is
> answerable by enumerating `Return` nodes — no join lattice, so this is not blocked on the
> one item in this document that has no step.

> **DONE as Step 4a (2026-08-31) — see [`step4a_false_merges.md`](revision_progess/step4a_false_merges.md).**
> Taken ahead of Step 3 on purpose; the argument is with the execution order above.
> Contradicted identity claims on climlab **51 → 17**, alias precision **84.5% → 93.6%**,
> and the `local_assign` row that this finding is *about* went 78/31 → 54/4. Objects
> 1866 → 1868, **access edges unchanged at 4239**, and the Step 1a oracle is unmoved
> (recall 69.0%, confirmed 75.2%, falsified 1) — this fix did not pay for its precision
> with somebody else's recall. Determinism gate still byte-identical over five seeds.
> Alias recall 65.9% → 64.3%: the price, and it is stated there rather than buried.
>
> Three departures from the wording above, all explained there. (i) **The alias is minted
> in three places and one of them undoes the other two** — `_apply_lineage_aliases` runs
> afterwards and rebuilds `alias_of` by composing the surviving lineage edges, so without
> a relation-aware rule there both fixes measure as doing nothing. That rule reads the
> graph twice, with everything and with identity relations only, and requires the two to
> agree; filtering alone would have *added* aliases by hiding ambiguity. (ii) **Enumerating
> `Return` nodes is not sufficient**: a function that falls off the end returns `None` on
> that path, so the gate also asks whether the body always returns, and excludes
> generators. (iii) The fix is a new non-identity relation, `derived_from`, rather than a
> deleted edge — `expose = bool(alias_of)` means deleting the claim deletes the object and
> its access edges with it.
>
> **One thing this document does not record, found while planning it:** `local_assign` and
> `state_assign` are **must-link relations** in `cluster_structural_graph` and
> `leiden_reweighted`. A false "is" claim was therefore a hard constraint fusing two
> clusters, not merely a heavier edge — which is the strongest form of the argument §1.7
> makes for ranking false coupling above its apparent size.
>
> All 17 survivors were hand-checked: **none is this defect.** Two are §6.2's mis-resolved
> `super().__setattr__` (Step 5), two are the `Iceline(state=self.state)` shared-state merge
> (Step 7), and thirteen are the instrument reaching its limits on branches the drivers
> never took — §6.3's caveat, now the majority of what is left.

§1.14 is the one Step 1a's oracle
found; it is also the cheapest, and the oracle will confirm the fix directly. §1.7 and §1.11 are the two that
generate false coupling, so judge them with `evaluate` rather than with recall — the
registry-coupling lesson from the call-graph roadmap applies unchanged. §5.5 (lambda
callables) belongs here because the fix — entering a callable for a lambda, keyed the
way `definitions.visit_Lambda` keys it — is also what fixes §1.9.

**Step 5 — Cross-stage consistency: §5.7, then §5.4 and §5.6 as guards.** Connect the
`confidence.weights` knob to data-access edges before anyone sweeps it, or the sweep
will conclude the wrong thing. Add the `resolvable_callable_ids` filter (§5.4) and the
diamond-hierarchy registration fixture (§5.6) as regression guards rather than fixes —
neither is broken today, and both are one upstream change away from breaking quietly.

**Step 6 — Performance: §3.1, §3.2, §3.3.** ***DONE — folded into Step 2, as suggested below.***
§3.3 was discharged by Step 0; §3.1 and §3.2 landed with Step 2, taking the stage from 0.904 s to
0.431 s on climlab. The paragraphs below are kept because their reasoning is what decided *where*
the work went, and because the caveat in the last one is the reason §3.1's fix is a membership
test rather than the hoist this section imagines. *Original text follows.*

*No longer sequenced — do it opportunistically.*
§3.3 is discharged: Step 0 took parses per file from 4 to 1, or 0 when `analysis.cache`
is threaded, so this step's original acceptance criterion is already met. What remains is
§3.1 and §3.2, and both are the same move: hoist the per-call-site sets into one frozen
per-run index.

Profiled on climlab 2026-08-23, `_candidate_callable_ids_for_call` is **0.555s of the
stage's 2.126s of profiled self-time — 26%, the single largest self-cost**, essentially
all of it the `known_ids` rebuild. So §3.1 is real. But the stage runs in 0.9s unprofiled
on climlab, so fixing it saves ~0.2s that nobody is waiting for, and it produces no
correctness signal, so it gates nothing.

Two reasons to do it anyway, when convenient: the cost is O(corpus) per call site, so it
grows badly at matplotlib scale (~9,400 callables); and the frozen index it needs is the
same one **§1.2 needs in Step 2**. Folding it into Step 2 is the cheapest moment.

Note that hoisting is *not* the one-liner it appears to be: `return_summaries` and
`return_tuple_summaries` are mutated by the collector during its own traversal, so a
per-collector snapshot would go stale mid-walk. This is why Step 0 left it alone.

**Step 7 — Framework independence: §2.6, §2.7.** Derive the shared-state gate the way
registration was derived. Judge with `evaluate` on a second codebase; climlab cannot
validate a generalization it is the special case of.

**Step 8 — Structure: §4.1, §4.2, §4.3.** Last, deliberately. Verify each step by
byte-identical artifacts, as the `collector/` split was. §5 is the argument for going
further than a split here: three of the four regressions in that section exist because
this package reimplements resolution that `call_graph` already owns. The split should
end with data access *calling* `ProjectIndex` rather than paraphrasing it. Step 0
removed two of the three paraphrases, so the remaining one is the callable-ID convention
of §2.5.

> **~~Blocked on Step 2, not merely on the oracle.~~ UNBLOCKED (2026-08-29).** The verification
> method named above did not work before Step 2. Step 0 added zero objects and removed zero, and
> 22 derived fields still moved on climlab, because §1.3 and §1.4 made `alias_of` and
> `access_path` depend on file processing order. A refactor that reorders anything therefore
> produced a diff that could not be read. Attempting this step before Step 2 would have meant a
> ~1,700-line split with no working regression gate — precisely the failure the call-graph
> history records, where two rounds of modularisation preserved the `_resolve_callees` ↔
> `_infer_class_types` cycle intact.
>
> Step 2 closed that. Run `microservice-pipeline check-data-access-determinism --config <config>`
> after each slice of the split; a byte-identical result now means the slice changed nothing,
> which is what this step's verification method assumed all along.

Written 2026-08-23. Findings above are against the tree at that date; line numbers
will drift.
