# Data-access ontology

This document explains the concepts used by the static data-access extractor.
Here, **ontology** means the small vocabulary and set of relationships that the
extractor uses to describe source code. It is not an RDF or OWL ontology.

The extractor does not run the analyzed application. It reads Python syntax and
builds a graph that answers two questions:

1. Which callable appears to access which data?
2. How can the identity of that data flow from one source-code location to
   another?

The central graph is:

```text
Callable ── AccessEdge ──> DataObject

Data object or virtual flow point ── LineageEdge ──>
    Data object or virtual flow point
```

For example:

```text
shop.orders.calculate_total
    └── read ──> orders['price']

the object returned by load_orders()
    └── arg_to_param ──> calculate_total(orders)
```

The canonical schema classes are in [models.py](models.py). The AST visitor that
creates these records is `DataAccessCollector` in
[generate_data_access_ast.py](generate_data_access_ast.py).

## 1. The four main concepts

### 1.1 Callable

A callable is a function, method, or synthetic module body. Callable IDs are
shared with the call-graph extractor.

Examples:

```text
shop.orders.load_orders
shop.orders.OrderService.save
shop.orders.<module>
```

`<module>` represents statements that execute at module level.

### 1.2 DataObject

A `DataObject` is a symbolic piece of data discovered in source code. It is not
the actual Python object and it does not contain the runtime value.

Examples of symbolic data objects are:

- A function parameter named `config`
- All ordinary state belonging to `OrderService`
- The `price` column of a particular DataFrame
- The `timeout` key of a particular dictionary
- A file represented by the source expression `output_path`

Two source expressions have the same data identity when the extractor gives
them the same `DataObject.id`. Possible relationships between different IDs are
kept in `alias_of` and `LineageEdge` records.

`DataObject` has these fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable machine-readable identity used by access edges. |
| `kind` | Source/data category, such as `param`, `dict_key`, or `file`. |
| `display_name` | Human-readable name, usually close to the source spelling. |
| `scope` | Where the data lives: `callable`, `module`, `class`, `object`, `field`, `external`, or `unknown`. |
| `owner` | Callable, class, object, or shared container that owns the data. |
| `container` | Parent container ID for a field or nested object. It can be empty. |
| `field` | Parameter, local, attribute, dictionary-key, or column name when applicable. |
| `file` | File where the object was first registered. |
| `lineno` | Line where the object was first registered. |
| `inferred_type` | Coarse family such as `dict`, `dataframe`, `xarray`, `object`, or `unknown`. |
| `confidence` | How directly the identity was inferred: `high`, `medium`, or `low`. |
| `alias_of` | The one underlying object this object unambiguously aliases. Empty when unknown or ambiguous. |
| `access_path` | Source-like path, for example `model.results['mass_g']`. |
| `structural_role` | `primary`, `precise`, or `coarse`; explained below. |

The `id`, `display_name`, and `access_path` serve different purposes:

```text
id
    Stable graph identity, for example
    object_state:param:shop.run:model:results

display_name
    Readable label, for example model.results

access_path
    The source path represented by the object, also model.results in this case
```

#### Structural roles

- `primary` is the normal role for a data object.
- `precise` means the object describes a specific nested path, such as
  `model.results.summary`.
- `coarse` is a broader placeholder, such as all state under `model`. A coarse
  object can coexist with more useful precise children.

### 1.3 AccessEdge

An `AccessEdge` connects a callable to a `DataObject` that it appears to use.

```text
Callable ── read/write/create/read_write ──> DataObject
```

Its fields are:

| Field | Meaning |
| --- | --- |
| `callable` | Callable performing the access. |
| `object_id` | ID of the accessed `DataObject`. |
| `access` | Broad effect: `read`, `write`, `create`, or `read_write`. |
| `operation` | More specific source-level reason for the edge. |
| `file` | File containing the evidence. |
| `lineno` | Line containing the evidence. |
| `confidence` | Confidence in this particular edge. |
| `evidence` | Short source-like expression generated from the AST. |

Access edges are evidence records, not a deduplicated set of semantic facts.
One source expression can produce more than one edge. For example,
`config['timeout']` can produce a read of the precise `timeout` key and a read
of the base `config` parameter.

### 1.4 LineageEdge

A `LineageEdge` describes possible identity or value flow between symbolic data
locations.

```text
source object ── relation ──> destination object
```

Lineage is separate from access. Passing an object to a function connects the
argument to the callee parameter, but that connection is not itself a read or
write by the callee.

Its fields are:

| Field | Meaning |
| --- | --- |
| `src_object_id` | Source data ID or virtual flow-point ID. |
| `dst_object_id` | Destination data ID or virtual flow-point ID. |
| `relation` | Kind of flow, such as `arg_to_param`. |
| `file` | File containing the evidence. |
| `lineno` | Line containing the evidence. |
| `caller` | Caller at the source location, when relevant. |
| `callee` | Resolved callee, mainly for `arg_to_param`. |
| `slot` | Parameter name, tuple position, or registration slot when relevant. |

Not every lineage endpoint is a row in the `objects` collection. A symbolic
parameter or assignment target can be mentioned by lineage before it is
materialized as a `DataObject`, and return points are always virtual IDs:

```text
return:shop.orders.load_orders
return:shop.orders.load_pair:0
return:shop.orders.load_pair:1
```

They let the graph connect a returned object to later assignments without
pretending that a Python return slot is a normal `DataObject`.

## 2. DataObject kinds and source concepts

The following table maps common source concepts to object kinds and IDs. The
examples use a fictional `sample` module.

| Source concept | `kind` | Example source | Example ID |
| --- | --- | --- | --- |
| Parameter | `param` | `def run(config): ...` | `param:sample.run:config` |
| Module global | `module_global` | `SETTINGS = {...}` | `module_global:sample.SETTINGS` |
| Normal class state | `class_state` | `self.name`, `self.volume` | `class_state:sample.Model` |
| Split class attribute | `class_attr_state` | `self.results` in a coordinator class | `class_attr_state:sample.Workflow:results` |
| Parameter/local object attribute | `object_state` | `model.results` | `object_state:param:sample.run:model:results` |
| DataFrame column | `df_col` | `df['mass_g']` | `df_col:sample.run:df:mass_g` |
| Dictionary key | `dict_key` | `config['solver']` | `dict_key:sample.run:config:solver` |
| Unknown container field | `container_field` | `data['solver']` when `data` has unknown type | `container_field:sample.run:data:solver` |
| External file | `file` | `open(path)` | `file:path` |
| Escaped local container | `local_exposed` | `items = []; return items` | `local_exposed:sample.build:items` |
| Fallback object | `unknown` | Edge found before metadata can be classified | An ID supplied by the edge-producing rule |

### 2.1 Parameter

A parameter object is created when a parameter is used as data.

```python
def calculate(rate):
    return rate * 2
```

```text
DataObject
    id: param:sample.calculate:rate
    kind: param
    scope: callable
    owner: sample.calculate
    field: rate

AccessEdge
    sample.calculate --read/load--> param:sample.calculate:rate
```

`self` and `cls` are not modeled as ordinary parameters. Their attributes are
handled as class state.

### 2.2 Module global

A name assigned directly at module level is eligible to become module-global
data when a callable uses it.

```python
SETTINGS = {"timeout": 30}

def timeout():
    return SETTINGS["timeout"]
```

The base object is:

```text
module_global:sample.SETTINGS
```

If Pyright or syntax identifies it as a dictionary, the key is a `dict_key`
whose container is that module-global object.

### 2.3 Normal class state

For an ordinary class, top-level `self` and `cls` attributes are rolled into one
class-state object:

```python
class Model:
    def describe(self):
        return self.name, self.volume
```

Both reads point to:

```text
class_state:sample.Model
```

This means “reads the state of `Model`.” It does not mean that `name` and
`volume` are the same Python attribute.

### 2.4 Split class attribute

Large coordinator classes often own several independent containers. The
extractor splits a class by top-level attribute when it sees at least:

- Four different attributes
- Attributes used across three methods
- Two container-like attributes

For example:

```python
class Workflow:
    def __init__(self):
        self.config = {}
        self.data = {}
        self.results = {}
        self.lookup = {}
```

The objects can become:

```text
class_attr_state:sample.Workflow:config
class_attr_state:sample.Workflow:data
class_attr_state:sample.Workflow:results
class_attr_state:sample.Workflow:lookup
```

This prevents every field owned by a large coordinator from being collapsed
into one giant state node.

### 2.5 Parameter or local object attribute

State reached through a parameter or tracked local is kept as a precise path.

```python
def summarize(model):
    return model.results.summary
```

Possible objects include:

```text
object_state:param:sample.summarize:model
    structural_role: coarse

object_state:param:sample.summarize:model:results
    access_path: model.results
    structural_role: precise

object_state:param:sample.summarize:model:results.summary
    access_path: model.results.summary
    structural_role: precise
```

The precise object is the useful evidence. The coarse object is a structural
placeholder for the broader root.

### 2.6 DataFrame column

If a base expression is inferred as `dataframe`, a literal field access becomes
a `df_col`.

```python
def total(df):
    return df["price"].sum()
```

```text
df_col:sample.total:df:price
```

Pandas indexers such as these are also recognized:

```python
df.loc[:, "price"]
df.at[0, "price"]
```

The extractor looks for literal string fields. A dynamic expression such as
`df[prefix + suffix]` normally cannot become one precise column object.

### 2.7 Dictionary key

If a base expression is inferred as `dict`, a literal field becomes a
`dict_key`:

```python
def solver(config):
    return config["solver"]
```

```text
dict_key:sample.solver:config:solver
```

Classes that expose mapping keys through `__getattr__` are detected by
[attrdict.py](attrdict.py). For such a class, `state.temperature` can be treated
like `state['temperature']` and become a `dict_key`.

### 2.8 Unknown container field

When the syntax clearly selects a field but the base family is unknown, the
extractor keeps the evidence instead of guessing `dict` or `dataframe`:

```python
def inspect(container):
    return container["solver"]
```

```text
container_field:sample.inspect:container:solver
```

This means “literal keyed access on an unknown kind of container.”

### 2.9 External file

Files are identified from the source expression used as a path:

```python
open("settings.json")  -> file:settings.json
open(path)             -> file:path
```

A literal path is high-confidence. A dynamic expression such as `path` is
usually medium-confidence because the runtime filename is not known.

File identity is textual. `file:path` means “the file expression spelled
`path`,” not proof that every variable named `path` points to the same physical
file.

### 2.10 Escaped local container

The extractor intentionally hides most scratch locals. A local container is
promoted to `local_exposed` when the container itself becomes relevant outside
its immediate calculation, for example when it is:

- Returned
- Passed to a resolved callable
- Assigned into object or class state
- An alias of another meaningful data object

```python
def build():
    items = []
    items.append("ready")
    return items
```

```text
local_exposed:sample.build:items
```

By contrast, a temporary dictionary that is mutated and then discarded may
remain hidden. This filtering keeps the graph focused on data that can connect
different callables or components.

## 3. Access modes and operation labels

The words **access** and **operation** are related but not interchangeable.

- `access` is the broad data effect.
- `operation` explains which source pattern caused the edge.

### 3.1 Access modes

| `access` | Meaning | Example |
| --- | --- | --- |
| `read` | The callable uses the existing data. | `value = config['timeout']` |
| `write` | The callable overwrites data or a field. | `config['timeout'] = 60` |
| `create` | The extractor sees creation-like evidence for the object. For assignment-managed state and exposed locals, this is normally the first such assignment encountered. | `self.results = {}`, exposing `items = []`, or `open(path, 'w')` |
| `read_write` | The operation needs the old value and changes it. | `items.append(x)` or `count += 1` |

`create` is a source-analysis label. It does not prove that the runtime allocated
a new Python object. For assignment-managed state and exposed locals, later
creation-like assignments to the same object ID are changed to `write`.
Write-mode `open` calls are labeled `create` directly, so repeated write-mode
opens can each provide `create` evidence.

### 3.2 Common operation labels

`operation` is an extensible string, not a closed enum. These are the main forms
emitted by the current extractor:

| Operation label | Why it appears | Example |
| --- | --- | --- |
| `load` | A name is loaded. | `config` |
| `attribute_load` | An attribute is loaded. | `model.results` |
| `subscript_load` | A key, column, or indexer field is loaded. | `config['solver']` |
| `assign` | Assignment target is handled. | `self.results = value` |
| `delete` | A tracked attribute or field is deleted. | `del config['old']` |
| `return` | An internal local is exposed and read by returning it. | `return items` |
| `passed_arg` | An internal local is exposed and read as a positional argument. | `consume(items)` |
| `passed_kwarg` | An internal local is exposed and read as a keyword argument. | `consume(items=items)` |
| `escape_assign` | An internal local is exposed by assignment into reachable state. | `self.items = items` |
| `open` | Built-in or open-like file access. | `open(path, 'w')` |
| Call name | Known reader function. | `pd.read_csv`, `xr.open_dataset`, `pooch.retrieve` |
| `json.load` / `json.dump` | JSON file-handle access. | `json.load(handle)` |
| `method:<name>` | Known mutating method. | `method:append` |
| `method:<name>:inplace` | Known Pandas method with literal `inplace=True`. | `method:fillna:inplace` |
| `method:<name>:receiver` | Ordinary method reads its receiver. | `method:copy:receiver` |
| `method:<name>:labeled_access` | Xarray labeled selection. | `method:sel:labeled_access` |

Because edges are evidence rows, an expression can produce a specific edge and
a broader base-object edge. This is expected:

```python
config["timeout"]
```

can produce:

```text
read/subscript_load -> dict_key:...:config:timeout
read/load           -> param:...:config
```

## 4. Lineage relations

The current lineage vocabulary contains six relations.

### 4.1 `arg_to_param`

Direction:

```text
caller argument object -> callee parameter object
```

Example:

```python
def consume(items):
    return len(items)

def run(results):
    return consume(results)
```

Possible edge:

```text
src_object_id: param:sample.run:results
dst_object_id: param:sample.consume:items
relation: arg_to_param
caller: sample.run
callee: sample.consume
slot: items
```

The edge is only recorded when the call can be matched safely to known
parameters. Ordinary ambiguous calls are skipped. Dynamic
`getattr(imported_module, name)(...)` dispatch is handled conservatively by
recording possible parameter flows to each matching callable in that module.

### 4.2 `return_value`

Direction:

```text
returned object -> virtual callable return point
```

Example:

```python
def build():
    items = []
    return items
```

Possible edge:

```text
local_exposed:sample.build:items
    --return_value-->
return:sample.build
```

The destination is a virtual flow point, not a `DataObject` row.

### 4.3 `return_slot`

`return_slot` is the tuple-aware form of `return_value`.

Direction:

```text
returned tuple element -> virtual indexed return point
```

Example:

```python
def build_pair():
    items = []
    errors = []
    return items, errors
```

Possible edges:

```text
local_exposed:sample.build_pair:items
    --return_slot, slot=0-->
return:sample.build_pair:0

local_exposed:sample.build_pair:errors
    --return_slot, slot=1-->
return:sample.build_pair:1
```

### 4.4 `local_assign`

Direction:

```text
source object or virtual return point -> assigned local object
```

Example:

```python
def run():
    produced = build()
    return produced
```

Possible lineage includes:

```text
return:sample.build
    --local_assign-->
local_exposed:sample.run:produced
```

When a direct returned-object summary is known, the extractor can also connect
that concrete object to `produced`. Keeping both edges preserves the explicit
return boundary and the best known object identity.

### 4.5 `state_assign`

The normal direction is:

```text
assigned source object or return point -> object/class state destination
```

Example:

```python
class Holder:
    def attach(self, items):
        self.items = items
```

Possible edge:

```text
param:sample.Holder.attach:items
    --state_assign-->
class_state:sample.Holder
```

There is one special registration form. When a parent registers a child that
was constructed with the parent's state, the current implementation stores:

```text
child class state --state_assign--> parent class state
```

For example:

```python
child = Child(state=self.state)
self.register("child", child)
```

can produce:

```text
class_state:sample.Child
    --state_assign-->
class_state:sample.Parent
```

This stored direction should be read as shared-state identity evidence, not as
the chronological direction in which the constructor argument was passed.

### 4.6 `tuple_unpack`

Direction:

```text
tuple element or virtual indexed return point -> unpack target
```

Example:

```python
items, errors = build_pair()
```

Possible edges:

```text
return:sample.build_pair:0
    --tuple_unpack, slot=0-->
local_exposed:sample.run:items

return:sample.build_pair:1
    --tuple_unpack, slot=1-->
local_exposed:sample.run:errors
```

### 4.7 `alias_of` compared with lineage

`DataObject.alias_of` is a convenient single-object answer. A lineage graph can
contain several possible sources.

The extractor sets `alias_of` only when it can find one unambiguous root. If two
different roots can reach the object, `alias_of` remains empty while the lineage
edges preserve both possibilities.

Therefore:

```text
alias_of    = one safe summary
lineage     = complete recorded flow evidence, including ambiguity
```

## 5. Internal resolver concepts

`Scope`, `LocalBinding`, and `ExprRef` are internal analysis records. They are
not rows in `data_access.json`.

### 5.1 Scope

A `Scope` is the collector's working memory for one callable.

It contains:

| Field | Meaning |
| --- | --- |
| `callable_id` | Callable currently being visited. |
| `params` | Parameter names other than `self` and `cls`. |
| `locals` | Local name to `LocalBinding`. |
| `attr_bindings` | Attribute path to `LocalBinding`, mainly for `self`/`cls` assignments. |
| `local_class_types` | Project class types a local may contain. |
| `local_element_class_types` | Project class types that elements of a local collection may contain. |
| `local_shared_state_owner_types` | Classes whose shared state a local carries. |
| `local_element_shared_state_owner_types` | Shared-state owners carried by collection elements. |
| `attr_class_types` | Project class types remembered for attribute paths. |
| `attr_shared_state_owner_types` | Shared-state owners remembered for attribute paths. |
| `shadowed` | Names temporarily hidden by comprehension targets. |

The collector keeps a stack of scopes so nested functions can be visited. A
comprehension target is added to `shadowed`, preventing this code:

```python
def summarize(data, row):
    return [row["mass"] for row in data]
```

from incorrectly reading the outer parameter `row` inside the comprehension.

### 5.2 LocalBinding

A `LocalBinding` answers:

> What does this local name or remembered attribute currently refer to?

Its main fields are:

| Field | Meaning |
| --- | --- |
| `object_id` | Symbolic ID reserved for the local or bound state. |
| `inferred_type` | Coarse container family. |
| `confidence` | Confidence in the binding. |
| `alias_of` | Underlying object ID, if known. |
| `display_name` | Readable name inherited from the source object when useful. |
| `access_path` | Source-like path carried through the alias. |
| `exposed` | Whether the local should be emitted as a `DataObject`. |
| `node` | AST node where the binding originated. |
| `class_types` | Project classes the value may instantiate or reference. |

Example:

```python
results = load_results()
```

After return-summary resolution, `results` might have a binding like:

```text
object_id: local_exposed:sample.run:results
alias_of: local_exposed:sample.load_results:df
inferred_type: dataframe
access_path: df
exposed: true
```

Bindings can exist with `exposed: false`. This lets the analyzer follow a local
container without adding a noisy graph node unless it later escapes.

### 5.3 ExprRef

An `ExprRef` is the short-lived answer returned when the collector resolves one
AST expression.

It contains:

| Field | Meaning |
| --- | --- |
| `object_id` | Data object represented by the expression. |
| `inferred_type` | Coarse family of the expression. |
| `confidence` | Confidence in the resolution. |
| `display_name` | Readable name. |
| `access_path` | Precise source-like path. |
| `coarse_object_id` | Optional broader state object created alongside a precise object. |

For:

```python
model.results
```

the expression reference can point to:

```text
object_id: object_state:param:sample.run:model:results
access_path: model.results
coarse_object_id: object_state:param:sample.run:model
```

`ExprRef` is passed from expression resolution to field creation, assignment,
return summaries, and lineage recording. It is not serialized.

## 6. Method-call processing

For every `ast.Call`, the collector runs the following handlers in order. More
than one handler can contribute evidence for the same call.

### 6.1 File operations

The extractor recognizes:

- `open(...)` and open-like calls
- Pandas readers: `read_csv`, `read_json`, `read_excel`, `read_table`, and
  `read_parquet`
- `json.load` and `json.dump`
- Xarray `open_dataset` and `open_dataarray`
- `pooch.retrieve`
- Pandas writers: `to_csv`, `to_json`, `to_excel`, `to_parquet`, and
  `to_pickle`

Examples:

```python
pd.read_csv(path)       # read file:path
open(path, "w")         # create file:path
df.to_csv(output_path)  # read df; write file:output_path
```

For `open`, any mode containing `w`, `a`, `x`, or `+` is treated as write-like.

### 6.2 Xarray labeled operations

When the receiver is known to be Xarray, labeled operations expose dimension or
indexer names:

```python
ds.sel(latitude=45)
ds.isel(level=0)
ds.loc[{"time": "January"}]
```

These can create key-like objects for `latitude`, `level`, and `time`. If the
receiver is not known to be Xarray, the extractor records an ordinary receiver
read instead of guessing labeled fields.

### 6.3 Mutating methods

Known general mutators include:

```text
append, extend, insert, update, setdefault, pop, remove,
clear, add, discard, sort
```

They create `read_write` edges on the receiver.

Known Pandas methods such as `drop`, `fillna`, `replace`, `rename`,
`reset_index`, `set_index`, and `sort_values` create `read_write` only when the
call contains the literal keyword `inplace=True`.

These rules are based on method names. They are useful heuristics, not proof of
the receiver's runtime type.

### 6.4 Ordinary method receiver reads

A method not recognized as mutating normally reads its receiver:

```python
model.results.copy()
```

Possible access:

```text
read/method:copy:receiver -> model.results
```

The receiver read is suppressed when a more specific handler has already
represented the relevant effect, such as a known mutator, file writer, or
successful Xarray labeled access.

### 6.5 Argument-to-parameter lineage

If the callee resolves to known project code, actual arguments are matched to
formal parameters:

```python
consume(results)
```

can produce:

```text
results object --arg_to_param--> consume's parameter object
```

Both ordinary positional arguments and explicitly named keyword arguments are
handled. This is a lightweight binder; it is not a complete model of every
`*args` and `**kwargs` case.

### 6.6 Registered shared-state lineage

The call-graph stage derives registration rules for methods or constructors that
retain a child and later invoke it. The data-access stage then asks whether the
registered child was also given the parent's state.

Both conditions are required:

1. The call matches a derived registration rule.
2. The child received the parent's state, for example through `state=self.state`.

Only then is shared-state `state_assign` lineage emitted. Registration by itself
means coupling, not shared data.

### 6.7 Ordinary argument and function-expression traversal

Finally, the visitor walks:

- Every positional argument
- Every keyword value
- The receiver of an attribute call, or the function expression of a direct call

This catches ordinary reads inside the call. If a local container is passed to a
known callable, it is first exposed as `local_exposed` so the caller and callee
can be connected through lineage.

Because the specialized handlers and ordinary traversal both record evidence,
one call can legitimately create several edges.

## 7. End-to-end example

Assume Pyright identifies `config` as a dictionary and `df` as a DataFrame:

```python
import pandas as pd


def update(
    config: dict[str, float],
    df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    df["mass_g"] = config["default_mass"]
    df.fillna(0, inplace=True)
    df.to_csv(output_path)
    return df
```

Important `DataObject` rows are approximately:

```text
param:sample.update:config
param:sample.update:df
param:sample.update:output_path
dict_key:sample.update:config:default_mass
df_col:sample.update:df:mass_g
file:output_path
```

Important access edges are approximately:

```text
sample.update --read/subscript_load-->
    dict_key:sample.update:config:default_mass

sample.update --read/load-->
    param:sample.update:config

sample.update --write/assign-->
    df_col:sample.update:df:mass_g

sample.update --read_write/method:fillna:inplace-->
    param:sample.update:df

sample.update --read/method:to_csv:receiver-->
    param:sample.update:df

sample.update --write/method:to_csv-->
    file:output_path

sample.update --read/load-->
    param:sample.update:output_path
```

The return produces lineage:

```text
param:sample.update:df
    --return_value-->
return:sample.update
```

There may be additional base-object read evidence because the visitor also
walks the receiver and arguments after specialized call handling.

## 8. Type families

Pyright and syntax-based inference reduce values to these broad families:

```text
dataframe
dict
list
set
file
path
field
xarray
object
unknown
```

The family affects field classification:

```text
dataframe base -> df_col
dict base      -> dict_key
xarray base    -> dict_key for labeled fields
unknown base   -> container_field
```

Pyright helps classify expressions; it does not find the read or write. The AST
visitor finds the source pattern, and the family determines how that pattern is
named. See [pyright_type_probe.py](pyright_type_probe.py) for the family
classifier and temporary `reveal_type` probes.

## 9. Shared container identity

Raw extraction keeps same-name containers separate by default:

```text
df_col:sample.load:results:mass_g
df_col:sample.report:results:mass_g
```

This avoids merging unrelated data just because variables share a common name.

A reviewed shared-container configuration can give them a canonical identity:

```json
{
  "df_col": {
    "results": "simulation_results"
  }
}
```

After re-extraction, the shared field can be:

```text
df_col:simulation_results:mass_g
```

`infer_shared_containers.py` proposes conservative same-name mappings using
field overlap, lineage/call support, and Pyright family agreement. Its output is
a draft for review, not automatic proof that two runtime objects are identical.

## 10. Confidence

Confidence describes how directly an identity or edge was inferred:

| Confidence | Weight | Meaning |
| --- | ---: | --- |
| `high` | `1.0` | Direct syntax, such as a literal key on a resolved base. |
| `medium` | `0.6` | Propagated alias or clear but indirect inference. |
| `low` | `0.25` | Dynamic or weakly resolved identity. |

Confidence is not a measured probability. It also does not mean the data family
is known. A literal keyed access can be high-confidence while its kind remains
`container_field` because the base family is unknown.

## 11. Output representation

[outputs.py](outputs.py) serializes the ontology into:

| Artifact | Contents |
| --- | --- |
| `data_objects.csv` | One row per `DataObject`. |
| `access_edges.csv` | One row per `AccessEdge`. |
| `callable_data_access.csv` | Access edges joined with object metadata. |
| `data_access.json` | Callables, objects, access edges, and lineage edges. |
| `data_access_report.md` | Human-readable summary by callable and object. |

For investigation, start with `callable_data_access.csv` and inspect:

```text
callable
access
operation
object_id
object_kind
access_path
evidence
file
lineno
confidence
```

Then use `data_access.json` to follow `alias_of` and `lineage_edges`.

## 12. Implementation map

The ontology is implemented across these files:

| File | Responsibility |
| --- | --- |
| [models.py](models.py) | `DataObject`, `AccessEdge`, `LineageEdge`, `Scope`, `LocalBinding`, `ExprRef`, and confidence helpers. |
| [generate_data_access_ast.py](generate_data_access_ast.py) | Source discovery integration, AST traversal, expression resolution, object identity, access detection, return summaries, and most lineage creation. |
| [rules.py](rules.py) | Container families, recognized method/function tables, and small AST field/indexer helpers. |
| [pyright_type_probe.py](pyright_type_probe.py) | Temporary `reveal_type` probes and conversion from Pyright type text to coarse families. |
| [attrdict.py](attrdict.py) | Detection of mapping classes that expose keys as attributes. |
| [registration_lineage.py](registration_lineage.py) | Shared-state lineage for derived child-registration patterns. |
| [infer_shared_containers.py](infer_shared_containers.py) | Conservative draft inference for canonical shared DataFrame and dictionary containers. |
| [outputs.py](outputs.py) | CSV, JSON, Markdown report, and generated artifact-guide writers. |

## 13. Interpretation rules and limits

Keep these rules in mind when reading the graph:

1. A `DataObject` is a source-derived identity, not a runtime value.
2. `AccessEdge` means “this source evidence appears to access this object.”
3. `LineageEdge` means possible identity or value flow, not an access by itself.
4. `alias_of` is set only for one unambiguous root; empty can mean unknown or
   ambiguous.
5. The extractor is static. It does not know which branch actually runs.
6. Literal keys and columns are more precise than dynamic selectors.
7. Method mutation is recognized from a finite method-name table.
8. Ordinary class attributes are intentionally rolled into one class-state
   object unless the coordinator heuristic splits the class.
9. File IDs are based on source text, not resolved filesystem paths.
10. Scratch locals are intentionally filtered until they escape.
11. Access edges are evidence rows and can contain repeated or overlapping
    evidence.
12. Pyright improves the family label, but the AST visitor remains responsible
    for detecting the access.

These choices make the graph suitable for architecture analysis: it favors
reviewable, service-relevant data relationships over a complete trace of every
Python variable and value.
