# microservice-pipeline

Reusable static-analysis pipeline for identifying candidate microservice
boundaries in Python projects. It combines a Python call graph with generalized data-access evidence, so the analyzed project does not need a database.
In-memory dictionaries, lists, object state, dataframes, arrays, and
file-backed resources can all contribute structural evidence.

The extractors parse source code without executing the analyzed application.
The pipeline produces reviewable CSV, JSON, and Markdown artifacts at each
stage.

## Install

For development from this checkout, install the base package:

```bash
python3 -m pip install -e .
```

The default workflow uses dependency-light label propagation. If you need Leiden, Infomap, or hierarchical clustering, run the editable install with the
`clustering` extra instead:

```bash
python3 -m pip install -e '.[clustering]'
```

That command installs the package plus the optional clustering dependencies.
You do not need to run both install commands. If you already installed the base package, rerunning the command with `[clustering]` is enough to add the extra dependencies.

Pyright is optional but recommended. It improves container-family inference
during data-access extraction:

```bash
npm install pyright
```

When Pyright is unavailable, either keep the default
`"fail_policy": "fallback_unknown"` behavior or set
`"data_access": {"pyright": {"enabled": false}}` in `extraction.jsonc`.

## Create Project Config

Run the bootstrap command from the Python project that you want to analyze:

```bash
microservice-pipeline init-config \
  --project-root . \
  --outdir configs/microservice_pipeline
```

The command writes seven editable starter files:

| File | Purpose |
| --- | --- |
| `extraction.jsonc` | Source discovery, call-graph output, data-access output, and Pyright behavior |
| `structural_graph.jsonc` | Heterogeneous graph inputs, weight profile, and hub detection |
| `structural_clustering.jsonc` | Clustering algorithm, sweeps, hub policy, and clustering weights |
| `evaluation.jsonc` | Standalone evaluation inputs, matching rules, scope, and output path |
| `notebook_task_analysis.jsonc` | Optional notebook-derived task overlays and refinements |
| `shared_containers.jsonc` | Optional reviewed manual aliases for shared dataframe and dictionary containers |
| `manual_mapping.csv` | Empty worksheet for manually adjudicated microservice labels |

The command refuses to overwrite existing files unless `--force` is supplied.
The `.jsonc` files support `//` comments, `/* block comments */`, and trailing commas.

Start by editing `extraction.jsonc`. `init-config` writes its `project_root`
relative to the generated config directory. Paths inside that file resolve
from the configured project root. For the structural graph, structural
clustering, and notebook-task commands, relative paths resolve from
`--project-root`, or from the current working directory when that flag is
omitted.

Most settings in this README are JSONC keys inside one of the generated config
files:

| Setting or section | Edit this file |
| --- | --- |
| `source.roots`, `source.package_prefixes`, `source.entrypoints`, `source.include_globs`, `source.exclude_globs` | `configs/microservice_pipeline/extraction.jsonc` |
| `call_graph.include_external`, `call_graph.outdir` | `configs/microservice_pipeline/extraction.jsonc` |
| `data_access.pyright`, `data_access.shared_containers_config`, data-access output paths | `configs/microservice_pipeline/extraction.jsonc` |
| Reviewed `df_col` and `dict_key` shared-container aliases | `configs/microservice_pipeline/shared_containers.jsonc` |
| Structural graph input/output paths, `weighting.weight_config`, `hub_detection` thresholds | `configs/microservice_pipeline/structural_graph.jsonc` |
| Clustering algorithm, `sweep`, `sweep_best`, `hub_policy`, clustering `weighting` overrides | `configs/microservice_pipeline/structural_clustering.jsonc` |
| Standalone evaluation paths, label column, node matching, NA labels, and evaluation scope | `configs/microservice_pipeline/evaluation.jsonc` |
| Notebook paths, notebook source resolver, task/refinement/pruning options | `configs/microservice_pipeline/notebook_task_analysis.jsonc` |
| Manual labels used for sweep or standalone evaluation | `configs/microservice_pipeline/manual_mapping.csv` |

### Configure Source Discovery

Edit `configs/microservice_pipeline/extraction.jsonc`. The snippet below lives
under the top-level `source` key.

Use `source.roots` to list the Python trees to scan. If a root points inside a
package, set that root's `module_prefix`:

```jsonc
{
  "source": {
    "roots": [
      {
        "path": "src/shop",
        "module_prefix": "shop"
      }
    ],
    "package_prefixes": ["shop"],
    "entrypoints": ["scripts/run_shop.py"]
  }
}
```

`module_prefix` controls how file paths become importable module names. For
example, with `"path": "src"`, `src/shop/orders.py` naturally becomes
`shop.orders`, so `module_prefix` should stay `null`. With
`"path": "src/shop"`, the same file would otherwise become `orders`; set
`"module_prefix": "shop"` to keep the module name as `shop.orders`. There is no
`model_prefix` option; use `module_prefix` for this purpose.

`package_prefixes` is a top-level `source` filter for internal namespaces. Set
it to the analyzed package names, such as `["shop"]`, when you want the call
graph to keep only internal resolved calls. Leave it empty to avoid
package-prefix filtering.

`entrypoints` adds scripts outside the scanned roots. `include_globs` and
`exclude_globs` control source selection without requiring code changes.

## Core Workflow

### 1. Extract Call Relationships

```bash
microservice-pipeline call-graph \
  --config configs/microservice_pipeline/extraction.jsonc
```

Default output: `artifacts/call_graph/`.

Important artifacts:

| Artifact | Contents |
| --- | --- |
| `call_graph.json` | Complete machine-readable call graph |
| `nodes.csv` | Discovered callable definitions |
| `edges.csv` | Caller-to-callee relationships and resolution metadata |

Set `call_graph.include_external` to `true` when unresolved or external calls
should remain visible as edges.

#### Optional: measure the call graph against runtime ground truth

The extractors never execute the analyzed project, so a call the interpreter
resolves at runtime through a registry, a base class, or a container can be
invisible to them. When the analyzed project *can* be run, `trace-runtime`
records the calls it really dispatches, using `sys.monitoring` (PEP 669,
requires Python 3.12+), and `compare-graphs` reports what the static pass
missed:

```bash
microservice-pipeline trace-runtime \
  --config configs/microservice_pipeline/extraction.jsonc

microservice-pipeline compare-graphs \
  --config configs/microservice_pipeline/extraction.jsonc
```

Drivers are declared under `call_graph.trace` (`pytest_args`,
`notebook_globs`, `scripts`) and all run **inside** the pipeline process,
because `sys.monitoring` cannot observe a subprocess. Failing tests and failing
notebook cells are tolerated: each still contributes the edges it reached.

| Artifact | Contents |
| --- | --- |
| `dynamic_edges.csv` | Caller-to-callee pairs actually dispatched at runtime |
| `dynamic_trace.json` | Callables entered, driver problems, tracing statistics |
| `graph_comparison.md` | Recall, missing edges grouped by owner, per-relation confirmation |

Read the two directions differently. **Recall** — runtime edges the static pass
never inferred — is the real measurement, and every miss is a genuine gap.
**Unconfirmed static edges** are a weak signal: a static edge that never
dispatched usually means the branch did not run, not that it is wrong. Edges the
runtime cannot express are excluded from both directions, namely `import` edges
(module bodies are executed by the import machinery in C) and unresolved edges
(their callee is a name, not a project callable).

#### Optional: cross-check against PyCG

When the analyzed project *cannot* be run, there is no ground truth to measure
against — but a second static extractor still catches mistakes the first one
makes alone. [`scripts/`](scripts/README.md) runs
[PyCG](https://github.com/vitsalis/PyCG), which uses a different technique
(inter-procedural points-to analysis rather than an AST pass with targeted type
inference), and reports the edges as `both` / `ours_only` / `pycg_only`.

Read it as a cross-check, **not** as ground truth. Neither graph is authoritative,
so disagreements are triaged by hand in both directions, and a clean result
bounds nothing about what both tools missed together. Runtime tracing above is
the stronger signal whenever the project can be executed.

PyCG is unmaintained and needs its own Python 3.11 environment; see
[`scripts/README.md`](scripts/README.md) for the setup and the pins it depends on.

### 2. Extract Raw Data Access

Run the first data-access extraction without a shared-container mapping:

```bash
microservice-pipeline data-access \
  --config configs/microservice_pipeline/extraction.jsonc
```

Default output: `artifacts/data_access/`.

Important artifacts:

| Artifact | Contents |
| --- | --- |
| `data_access.json` | Complete callable, object, access-edge, and lineage payload |
| `data_objects.csv` | Discovered service-relevant data objects |
| `access_edges.csv` | Callable-to-data-object accesses |
| `callable_data_access.csv` | Denormalized review table |
| `data_access_report.md` | Human-readable extraction guide and summary |

The raw run intentionally keeps uncertain field containers local to their
callables. This avoids collapsing unrelated dictionaries or dataframes merely
because they use the same field name.

### 3. Infer And Review Shared Containers

Generate a draft mapping:

```bash
microservice-pipeline infer-shared-containers \
  --config configs/microservice_pipeline/extraction.jsonc
```

Review `artifacts/inferred_shared_containers.md`. The draft JSON mapping is
written to `artifacts/inferred_shared_containers.json`.

The mapping has two sections. `df_col` is for dataframe-like containers whose
columns should be treated as one shared data object across callables.
`dict_key` is for dictionary-like containers whose keys should be treated the
same way.

```jsonc
{
  "df_col": {
    "Results": "simulation_results",
    "results_df": "simulation_results",
    "output_table": "simulation_results",
    "emissions": "emissions_table",
    "emissions_df": "emissions_table"
  },
  "dict_key": {
    "configuration": "run_config",
    "config": "run_config",
    "settings": "run_config",
    "rates": "rate_parameters",
    "rate_params": "rate_parameters"
  }
}
```

Each key is an observed container name and each value is the canonical shared
name that should appear in the re-extracted artifacts. Multiple observed names
can point to the same canonical name. The observed-name match is
case-insensitive, and the canonical name does not have to be an existing
variable name. Copy only accepted mappings into
`configs/microservice_pipeline/shared_containers.jsonc` when you want a stable,
reviewed project config. Existing plain JSON mappings remain supported.

### 4. Re-Extract With Accepted Mappings

Rerun data-access extraction into a separate directory:

```bash
microservice-pipeline data-access \
  --config configs/microservice_pipeline/extraction.jsonc \
  --shared-containers-config artifacts/inferred_shared_containers.json \
  --outdir artifacts/data_access_inferred
```

For a reviewed persistent mapping, pass:

```bash
microservice-pipeline data-access \
  --config configs/microservice_pipeline/extraction.jsonc \
  --shared-containers-config configs/microservice_pipeline/shared_containers.jsonc \
  --outdir artifacts/data_access_inferred
```

The generated `structural_graph.jsonc` expects
`artifacts/data_access_inferred/data_access.json`.

Instead of passing `--shared-containers-config` every time, you can set
`data_access.shared_containers_config` in
`configs/microservice_pipeline/extraction.jsonc` to the reviewed
`shared_containers.jsonc` path.

Shared-container mapping precedence:

1. `--shared-containers-config` on the `data-access` command wins for that run.
2. If the CLI flag is omitted, `data_access.shared_containers_config` from
   `configs/microservice_pipeline/extraction.jsonc` is used.
3. If both are omitted or `null`, no shared-container mapping is applied.

Paths in `extraction.jsonc` resolve from the configured `project_root`. The
`--shared-containers-config` CLI path resolves from the current working
directory.

### 5. Build The Structural Graph

```bash
microservice-pipeline structural-graph \
  --project-root . \
  --config configs/microservice_pipeline/structural_graph.jsonc
```

Default output: `artifacts/structural_dependency_graph/`.

The structural graph combines callable nodes and data nodes with call,
data-access, and data-lineage edges. It also reports broad fan-out callables and widely shared data objects as hub candidates.

Important artifacts:

| Artifact | Contents |
| --- | --- |
| `structural_graph.json` | Complete heterogeneous graph and resolved options |
| `nodes.csv` | Callable and data nodes |
| `edges.csv` | Weighted structural edges |
| `callable_hub_nodes.csv` | Callable hub candidates and reasons |
| `data_hub_nodes.csv` | Data hub candidates and reasons |
| `README.md` | Generated artifact guide |

Hub detection is descriptive at this stage. Whether a hub should be excluded
is a clustering policy decision.

### 6. Cluster Candidate Boundaries

```bash
microservice-pipeline structural-cluster \
  --project-root . \
  --config configs/microservice_pipeline/structural_clustering.jsonc
```

Default output: `artifacts/structural_microservice_candidates/`.

Important artifacts:

| Artifact | Contents |
| --- | --- |
| `cluster_assignments.csv` | Node-to-candidate assignments |
| `cluster_summary.csv` | Candidate sizes, cohesion, coupling, and warnings |
| `cluster_edges.csv` | Candidate-to-candidate dependencies |
| `excluded_nodes.csv` | Nodes removed by policy |
| `hub_nodes.csv` | Hub decisions applied during clustering |
| `hub_cluster_links.csv` | Edges between kept/dropped hub candidates and candidate clusters |
| `must_link_groups.csv` | Nodes contracted before clustering |
| `cycle_findings.csv` | Machine-readable node-level and candidate-level cycle review |
| `cycle_findings.md` | Node-level and candidate-level cycle review |
| `clusters.json` | Complete machine-readable clustering result |

The generated template uses `label_propagation`, which does not require the
optional clustering dependencies. Available structural algorithms are:

| Algorithm | Intended use |
| --- | --- |
| `label_propagation` | Dependency-light starting point |
| `leiden` | Weighted graph community detection |
| `leiden_reweighted` | Ownership-biased Leiden variant |
| `leiden_multiplex` | Separate call, data-access, and lineage layers |
| `infomap` | Flow-based clustering |
| `hac_callable_projection` | Hierarchical clustering over callable projection |

## Weight Profiles

Edit `weighting.weight_config` in
`configs/microservice_pipeline/structural_graph.jsonc` for structural graph
generation and in
`configs/microservice_pipeline/structural_clustering.jsonc` for clustering.
Both stages accept packaged profile aliases:

```jsonc
{
  "weighting": {
    "weight_config": "builtin:default"
  }
}
```

Supported aliases:

| Alias | Purpose |
| --- | --- |
| `builtin:default` | General-purpose generation weights and clustering scales |
| `builtin:ownership_biased` | Stronger create/write evidence and ownership-oriented settings |

Filesystem paths remain supported for project-specific JSON overrides. Custom
profiles are deep-merged over `builtin:default`.

The structural graph stage uses the profile's `generation` weights to convert
call, data-access, data-lineage, and confidence evidence into edge weights. The
clustering stage uses the profile's `clustering.edge_type_scales`, plus
algorithm-specific profile settings for modes such as `leiden_reweighted` and
`leiden_multiplex`. Keep both config files on the same profile unless you are
intentionally comparing weighting behavior.

In `structural_clustering.jsonc`, `weighting.call_weight_scale`,
`weighting.data_access_weight_scale`, and
`weighting.data_lineage_weight_scale` are optional clustering-only overrides.
Leave them `null` to use the selected profile.

## Hub Policy

Hub detection thresholds live in
`configs/microservice_pipeline/structural_graph.jsonc` under
`hub_detection`. Review `callable_hub_nodes.csv` and `data_hub_nodes.csv` after
structural graph generation. Hub drop-or-keep decisions live in
`configs/microservice_pipeline/structural_clustering.jsonc` under
`hub_policy`.

Callable hub policy options:

| Value | Behavior |
| --- | --- |
| `null` or `keep` | Report callable hub candidates without dropping them |
| `drop-all` | Exclude all detected callable hubs |
| `drop-configured` | Exclude only explicitly listed callable hubs |

Use `drop_callable_hub`, `keep_callable_hub`, or
`callable_hub_decisions` for reviewed exceptions. Explicit keeps win. Set
`drop_data_hubs` when broadly shared data objects should be excluded from one
base run. Parameter sweeps compare both data-hub policies automatically.

## Parameter Sweeps

Enable `sweep.run_sweep` in `structural_clustering.jsonc` to compare
algorithm-specific parameters:

```jsonc
{
  "sweep": {
    "run_sweep": true,
    "range": "0.1:1.5:0.1"
  }
}
```

`range` means Leiden resolution, Infomap Markov time, or HAC target cluster
count according to the selected algorithm. Multiplex Leiden can sweep
layer-specific `call_resolutions`, `data_access_resolutions`, and
`data_lineage_resolutions`.

When a manually adjudicated mapping CSV is available, set
`paths.manual_mapping` and `sweep.evaluation_enabled`. Set
`sweep_best.enabled` to `true` when one selected sweep row should also be
materialized as complete clustering artifacts.

Generated config directory defaults:

| Config key | Default path | Written when |
| --- | --- | --- |
| `paths.outdir` | `artifacts/structural_microservice_candidates/` | Always, for the base clustering run |
| `paths.sweep_outdir` | `artifacts/structural_microservice_candidates_sweep/` | Only when `sweep.run_sweep` is `true` |
| `sweep_best.outdir` | `artifacts/structural_microservice_candidates_sweep/best/` | Only when `sweep.run_sweep` and `sweep_best.enabled` are both `true` |

### Clustering And Evaluation Output Layout

The base `structural-cluster` run always writes complete clustering artifacts
to `paths.outdir`, which defaults to
`artifacts/structural_microservice_candidates/`. This happens whether the
sweep is enabled or disabled.

When `sweep.run_sweep` is `false`, `paths.outdir` is the only clustering output:

```text
artifacts/structural_microservice_candidates/
  cluster_assignments.csv
  cluster_summary.csv
  cluster_edges.csv
  excluded_nodes.csv
  hub_nodes.csv
  hub_cluster_links.csv
  must_link_groups.csv
  cycle_findings.csv
  cycle_findings.md
  clusters.json
```

When `sweep.run_sweep` is `true`, the command still writes the base output
above, then adds sweep-level artifacts under `paths.sweep_outdir`:

```text
artifacts/structural_microservice_candidates/
  ...base run artifacts...

artifacts/structural_microservice_candidates_sweep/
  parameter_sweep.csv
  parameter_sweep.md
  parameter_sweep.json
  sweep_best_selection.json
  sweep_best_selection.md
  best/
    ...complete clustering artifacts for the selected sweep row, only if enabled...
```

`parameter_sweep.csv` has one row per swept parameter value and data-hub
policy. If `sweep.evaluation_enabled` is `true` and `paths.manual_mapping`
points to an existing manual mapping CSV, the sweep rows include
`evaluation_*` metric columns. Sweep evaluation does not create a separate
evaluation directory for each row.

The generated config sets `sweep_best.enabled` to `false`, so enabling the
sweep alone does not write `best/`. If `sweep_best.enabled` is set to `true`,
the best output stays inside the sweep directory by default. It only goes
elsewhere if `sweep_best.outdir` is changed.

## Build A Manual Mapping

`init-config` generates an empty
`configs/microservice_pipeline/manual_mapping.csv` worksheet:

```csv
microservice_id,node,node_type,label,kind,module
```

Build the worksheet after running `structural-cluster`. The source of truth for
its `node` values is:

```text
artifacts/structural_microservice_candidates/cluster_assignments.csv
```

Copy the `node`, `node_type`, `label`, `kind`, and `module` columns from the
rows you want to adjudicate. Assign your own domain label in
`microservice_id`. Do not copy `cluster_id` into `microservice_id`:
`cluster_id` is the pipeline prediction being evaluated, while
`microservice_id` is the independent manual decision.

Example:

```csv
microservice_id,node,node_type,label,kind,module
inventory,callable:shop.inventory.Inventory.reserve,callable,reserve,method,shop.inventory
inventory,data:class_attr_state:shop.inventory.Inventory:stock,data,stock,class_attr_state,
checkout,callable:shop.checkout.place_order,callable,place_order,function,shop.checkout
NA,callable:shop.checkout.experimental_helper,callable,experimental_helper,function,shop.checkout
```

The CSV contract is:

| Column | Requirement |
| --- | --- |
| `node` | Required. Copy the exact value from `cluster_assignments.csv`. Each mapped node must appear only once. |
| `microservice_id` | Required by the generated worksheet. Assign a stable manual service label. The evaluator also accepts `manual_microservice_id`, `manual_label`, or `service_id`. |
| `node_type` | Recommended. Copy from `cluster_assignments.csv`; values include `callable` and `data`. |
| `label` | Recommended for human review. Copy from `cluster_assignments.csv`. |
| `kind` | Recommended for evaluation filtering. Copy from `cluster_assignments.csv`. |
| `module` | Recommended for human review. Copy from `cluster_assignments.csv`. |

Use `NA`, an empty value, or another configured `--na-label` for rows that
have not been adjudicated. These rows are excluded from primary metrics but
remain visible in coverage reports.

Preserving the exact `node` value supports callable and data-node evaluation.
The default `--node-mode auto` chooses exact matching when possible.
`--node-mode callable` is only for callable-only mappings that omit the
`callable:` prefix; it excludes data nodes.

To use the generated worksheet during parameter sweeps, fill it and update
`configs/microservice_pipeline/structural_clustering.jsonc`:

```jsonc
{
  "paths": {
    "manual_mapping": "configs/microservice_pipeline/manual_mapping.csv"
  },
  "evaluation": {
    "enabled": false
  },
  "sweep": {
    "evaluation_enabled": true
  }
}
```

The generated sweep config evaluates both callable and data rows. Adjust
`sweep.evaluation_node_types`, `sweep.evaluation_kind_tokens`, or
`sweep.all_evaluation_nodes` when a narrower or broader adjudication scope is
needed.

Sweep evaluation settings live in `structural_clustering.jsonc` because they
are used while `structural-cluster` is producing `parameter_sweep.csv`.
Standalone detailed evaluation uses `evaluation.jsonc`.

## Optional Notebook Task Analysis

Use notebook-task analysis when tutorial or workflow notebooks should inform
candidate review. Configure source paths and notebook include rules in
`notebook_task_analysis.jsonc`, then run:

```bash
microservice-pipeline notebook-tasks \
  --project-root . \
  --config configs/microservice_pipeline/notebook_task_analysis.jsonc
```

The stage extracts heading-based tasks, resolves notebook calls to structural
nodes, computes task-to-cluster overlays, recommends refinements, and can prune zero-in-degree callables not observed from notebooks.

Keep `refinement.mode` set to `none` for an audit-only first run. Review
`refinement_recommendations.csv`, then use `selected` with accepted refined
group IDs or use `all` when the recommendations should be applied together.

Default output: `artifacts/notebook_task_analysis/`.

## Optional Evaluation

Evaluate candidate assignments against a manually adjudicated mapping:

```bash
microservice-pipeline evaluate \
  --project-root . \
  --config configs/microservice_pipeline/evaluation.jsonc
```

The evaluator treats cluster IDs as arbitrary partition labels. Use
`--manual-label-column`, `--node-mode`, repeated `--na-label`, and the
evaluation-scope flags when the mapping schema needs customization. Standalone
evaluation defaults to callable rows plus data rows whose `kind` contains
`class_attr_state`.

Values in `evaluation.jsonc` override the built-in defaults. Explicit CLI flags
override the config for one run. This lets one config evaluate multiple
assignment sets, for example:

```bash
microservice-pipeline evaluate \
  --project-root . \
  --config configs/microservice_pipeline/evaluation.jsonc \
  --clusters artifacts/notebook_task_analysis/refined_cluster_assignments.csv \
  --outdir artifacts/notebook_task_analysis/evaluation
```

Useful evaluation flags:

| Flag | Meaning |
| --- | --- |
| `--manual-label-column` | Column in the manual CSV that contains the human microservice label. Omit it to auto-detect `microservice_id`, `manual_microservice_id`, `manual_label`, or `service_id`. |
| `--node-mode auto` | Default. Chooses the join mode with the best overlap, preferring exact `node` values when possible. |
| `--node-mode exact` | Joins manual and cluster rows by the exact `node` value, including prefixes such as `callable:` and `data:`. Use this for the generated worksheet. |
| `--node-mode callable` | Strips a leading `callable:` prefix and ignores data nodes. Use this only for older callable-only mappings. |
| `--na-label LABEL` | Treat `LABEL` as unknown or unadjudicated. Repeat the flag for multiple labels. If you pass this flag, include every label you want treated as NA; otherwise the defaults are empty value, `NA`, `N/A`, `nan`, and `None`. |

The sweep equivalents live in `structural_clustering.jsonc` as
`sweep.manual_label_column`, `sweep.node_mode`, and `sweep.na_labels`.

Standalone evaluation output is separate from sweep evaluation. The
`microservice-pipeline evaluate` command writes to its own `--outdir`:

```text
artifacts/microservice_clustering_evaluation/
  metrics_summary.csv
  metrics_summary.md
  contingency_table.csv
  contingency_table_with_na.csv
  per_microservice_best_match.csv
  na_cluster_review.csv
  joined_assignments.csv
  evaluation.json
```

Use standalone evaluation when you want detailed files for one specific
assignment set, such as the base
`artifacts/structural_microservice_candidates/cluster_assignments.csv` or the
selected best sweep row under
`artifacts/structural_microservice_candidates_sweep/best/cluster_assignments.csv`.

## Callable-Only Diagnostic

Structural clustering is the primary workflow. Callable-only clustering
remains available as a diagnostic:

```bash
microservice-pipeline call-cluster \
  --nodes artifacts/call_graph/nodes.csv \
  --edges artifacts/call_graph/edges.csv \
  --outdir artifacts/call_graph_clusters
```

cd /Users/qianhuilin/Desktop/Envision
source .venvs/msa-analysis-climlab/bin/activate
