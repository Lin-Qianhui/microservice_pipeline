Precision gaps

Annotations are completely ignored. You infer param types only from call sites, but def solve(mesh: Mesh) -> Solution: is free, reliable, and increasingly present. Also self.grid: Grid in AnnAssign (you only read node.value). Biggest cheap win by a wide margin.
No __call__. model(x), loss(y, ŷ), functors, anything torch-shaped — currently falls through _resolve_callee to (name, "direct", False). Silent, large hole.
Higher-order flow. solve_ivp(rhs, ...), minimize(objective, x0), Pool.map(f, xs), Parallel(delayed(f)(...)), Thread(target=f), functools.partial, @singledispatch. Your FunctionParamSummary fixpoint is exactly the right substrate — widen the lattice from class ids to class ids ∪ callable ids, so a param bound to a known function resolves f(x) inside the callee. That one generalization subsumes your add_subprocess special case and turns it into a config rule.
Dict-valued registries. You track list/set/tuple elements but not dict values. SOLVERS = {"cg": ConjugateGradient} → SOLVERS[cfg.name]() is the dominant config-driven dispatch idiom in scientific code.
Decorators. You only recognise property/staticmethod/classmethod. @njit, @jax.jit, @dask.delayed, @functools.wraps, @register("...") all need transparency rules. Also @property.setter — you emit getter edges but assignments to properties produce nothing.
Synthesized __init__. dataclasses / attrs / pydantic / NamedTuple have no __init__ in the AST, so _resolve_constructor_targets returns nothing and constructor edges vanish.
MRO. _resolve_method_targets unions all matching bases rather than picking the C3 winner — over-approximates on mixins, which scientific frameworks love. The seen.copy() per base is also exponential on wide diamonds.
Control flow. Only If merges type state. No try/except/finally (optional-dependency imports!), while, match, or comprehension scopes.

Robustness / correctness

parse_python_file raises on the first SyntaxError and kills the run — vendored, generated, or py2 files will do this. Collect failures per file into a report.
read_text(encoding="utf-8") crashes on legacy latin-1 files. Use ast.parse(path.read_bytes()), which honours PEP 263 coding cookies.
Non-determinism: class_attr_types is mutated during the final edge pass (set_attribute_types), so edge output depends on file iteration order. Freeze the summaries into an immutable snapshot before collecting edges.
Fixpoint isn't monotone. collected is rebuilt from scratch each round rather than joined into the previous state, so it can oscillate and silently truncate at max_iterations. Union into the prior state, and record whether it actually converged.
~10× reparsing. 1 (defs) + 3 (returns) + 5 (types) + 1 (edges) full parses of the tree. Cache ASTs in memory, or persist keyed by content hash. Free 5–10× speedup.
Edges aren't deduplicated; add col_offset and/or a count so repeated call sites collapse.
Decorator expressions and default arguments are visited with current_callable set to the decorated function, so they're attributed to the callee rather than the enclosing scope.
Native boundary (.pyx, f2py, pybind11) should be its own node kind, not lumped into "unresolved".
Nothing reads .ipynb — SourceCallResolver exists but has no front door.

The thing I'd do first, before any of the above: build a dynamic ground truth. Run your test suite under sys.monitoring (PY_START/CALL events, 3.12+) or sys.setprofile, collect actual caller→callee pairs, and compute precision/recall against the static graph broken down by relation kind. That tells you which of the gaps above actually matter for your codebases, instead of guessing. The PyCG micro-benchmark suite (135 small programs with expected graphs) is a good complementary fixture set.

3. Modularisation

The file has three structural problems, and fixing them deletes more code than a naive split would.

(a) Duplicated build_X / build_X_from_analysis_files pairs. You have four of these, each with 10+ parameters. Keep only the _from_analysis_files form and make file discovery a caller concern. Replace the parameter soup with a single frozen AnalysisContext(callable_map, module_map, known_classes, return_summaries, param_summaries, class_attr_types, config). That's a few hundred lines gone.

(b) CallCollector fuses three concerns: AST walking, the type environment (*_types_stack, merge/copy helpers), and name/method resolution (_resolve_*). Split into Visitor + TypeEnv + Resolver collaborating objects. ReturnSummaryCollector and TypeSummaryCollector currently subclass CallCollector purely to suppress edge emission — that's inheritance-as-configuration. Replace with sinks: one visitor, three consumers (EdgeSink, ReturnSink, ParamSink). Three near-identical constructors collapse to one.

(c) Extraction and resolution are entangled. The highest-leverage change: make pass 1 emit a CallSite IR — (caller, receiver_expr, name, syntactic_kind, loc) — and pass 2 resolve it. This gives you unit-testable resolution without ASTs, and it's what makes the resolver swappable for scip/pyright later.

Suggested layout:

call_graph/
  model.py          # CallableDef, Edge, CallSite, summaries
  discovery.py      # AnalysisFile, globs, project root
  parsing.py        # cached parse, PEP-263 bytes, parent links, error report
  definitions.py    # DefinitionCollector
  typeenv.py        # scope stacks, join/copy, branch merging
  resolve/
    names.py        # imports, star imports, class refs
    methods.py      # MRO (C3), method targets, super, constructors
    dunders.py      # operator tables — pure data
  passes/
    returns.py  params.py  edges.py  fixpoint.py   # generic driver
  rules/            # framework plugins: subprocess, joblib, dask, torch, registries
  backends/         # ast_backend.py, scip_backend.py  (same CallSite → Edge interface)
  pipeline.py  outputs.py  cli.py

Two smaller wins: the four dunder dicts become one table op → (forward, reverse, inplace); and the rules/ package is where add_subprocess goes, declared rather than hard-coded ({match: "*.add_subprocess", child_arg: 0, link: "_compute → _compute"}).

### Status (2026-07-31)

A behaviour-preserving split has landed. collectors.py went from 2,108 lines to
1,565, verified by a byte-identical edges.csv on climlab after every step:

- `dunders.py` — the operator tables, moved verbatim. The one-table `op →
  (forward, reverse, inplace)` collapse suggested above is still open.
- `project_index.py` — `ProjectIndex`, holding the class hierarchy plus the
  resolution that depends only on definition facts (`resolve_method_targets`,
  `class_and_ancestors`, subclass overrides, super, star imports, class
  references). This also fixed a real inefficiency: those indexes were rebuilt
  inside `CallCollector.__init__`, i.e. **504 times per climlab run**, once per
  collector. Now built once and threaded through the passes alongside
  `ParsedFileCache`.
- `type_env.py` — `TypeEnv`, the scope stacks and their copy/merge machinery.
- `summary_collectors.py` — the two edge-free subclasses.

So (b) is half done: the type environment and the *static* half of resolution
are now separate collaborating objects.

**What is left of (b) and (c), and why it stopped here.** The remaining
resolution cannot be lifted out by moving code. `_resolve_callees` calls
`_infer_class_types_from_value` → `_infer_class_types_from_call` →
`_resolve_callees`: callee resolution and type inference are mutually recursive,
because resolving `build().submit()` requires the inferred type of `build()`
and inferring that type requires resolving `build`. A `resolve/names.py` +
`resolve/methods.py` split at that layer produces an import cycle, not a
separation of concerns.

The CallSite IR from (c) is the thing that actually breaks the cycle — pass 1
emits `(caller, receiver_expr, name, syntactic_kind, loc)`, pass 2 resolves it —
and the sink refactor from (b) should ride along with it rather than land first,
since one visitor feeding three sinks is only worth doing once the visitor has
stopped resolving. Neither is attempted here. (a) is untouched.

### Status (2026-08-06)

`collectors.py` is gone. Its 2,493 lines are now `collector/`, a sub-package of
15 files (largest 379 lines) assembled from one mixin per concern, verified at
every step by a byte-identical four-artifact diff on climlab. See
**`modularisation_plan.md`** for the layout, the invariants, and an honest
account of what mixins do and do not buy.

That does *not* change the analysis above. The mixins are slices of one object,
so the `_resolve_callees` ↔ `_infer_class_types_from_*` cycle is untouched — it
now spans `resolution.py` and `inference.py`, which are the two files a CallSite
IR would sit between. (b)'s sink refactor and (c)'s IR remain open and remain
coupled; the split deliberately *preserved* the private subclass contract that
`summary_collectors` depends on rather than narrowing it, since narrowing it is
a behavioural change. (a) is still untouched.

a simplified method-resolution algorithm. It handles direct definitions and recursively indexed bases, but it is not a complete implementation of Python’s C3 method resolution order. In ambiguous multiple-inheritance cases, it can return several targets rather than selecting exactly the one Python would at runtime