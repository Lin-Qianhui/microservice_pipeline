import ast
from pathlib import Path

from microservice_pipeline.data_access.generate_data_access_ast import (
    collect_module_imports,
    collect_data_access,
    collect_data_access_from_source,
    collect_data_access_from_tree,
)
from microservice_pipeline.call_graph.generate_call_graph_ast import build_indices


def _collect(source, shared_config=None, pyright_families=None):
    objects, edges = collect_data_access_from_source(
        source,
        module="sample",
        shared_containers_config=shared_config,
        pyright_families=pyright_families,
    )
    return objects, edges


def _edge_exists(edges, object_part, access=None, operation=None):
    for edge in edges:
        if object_part not in edge.object_id:
            continue
        if access is not None and edge.access != access:
            continue
        if operation is not None and edge.operation != operation:
            continue
        return True
    return False


def _matching_edges(edges, object_id, access=None):
    return [
        edge
        for edge in edges
        if edge.object_id == object_id and (access is None or edge.access == access)
    ]


def test_data_access_import_collection_resolves_relative_imports():
    tree = ast.parse(
        """
from .helper import load
from ..common import shared
from . import sibling
"""
    )

    imports, star_imports = collect_module_imports(tree, "pkg.feature.consumer")

    assert star_imports == []
    assert imports["load"] == "pkg.feature.helper.load"
    assert imports["shared"] == "pkg.common.shared"
    assert imports["sibling"] == "pkg.feature.sibling"


def test_self_attributes_params_and_dict_keys():
    objects, edges = _collect(
        """
class Model:
    def configure(self, config):
        self.config = config
        selected = self.config["MPdensity_kg_m3"]
        return selected
""",
        pyright_families={
            "param:sample.Model.configure:config": "dict",
            "class_attr:sample.Model:config": "dict",
        },
    )

    assert "class_state:sample.Model" in objects
    assert any(
        obj.kind == "dict_key"
        and obj.field == "MPdensity_kg_m3"
        and obj.container == "class_state:sample.Model"
        for obj in objects.values()
    )
    assert "param:sample.Model.configure:config" in objects
    assert not any(obj_id.endswith(":selected") for obj_id in objects)
    assert _edge_exists(edges, "class_state:sample.Model", access="create")
    assert _edge_exists(edges, "param:sample.Model.configure:config", access="read")
    assert _edge_exists(edges, "MPdensity_kg_m3", access="read")


def test_parameters_are_read_but_scalar_temporaries_are_filtered():
    objects, edges = _collect(
        """
def calculate(a, b):
    c = a + b
    return c
"""
    )

    assert "param:sample.calculate:a" in objects
    assert "param:sample.calculate:b" in objects
    assert not any(obj_id.endswith(":c") for obj_id in objects)
    assert _edge_exists(edges, "param:sample.calculate:a", access="read")
    assert _edge_exists(edges, "param:sample.calculate:b", access="read")


def test_dataframe_columns_are_read_and_written_from_subscripts_and_indexers():
    objects, edges = _collect(
        """
import pandas as pd

def load(path):
    df = pd.read_csv(path)
    df["mass_g"] = 1
    total = df.loc[:, "mass_g"]
    again = df.at[0, "mass_g"]
    return df
"""
    )

    df_columns = [obj for obj in objects.values() if obj.kind == "df_col"]
    assert any(obj.field == "mass_g" for obj in df_columns)
    assert _edge_exists(edges, "file:path", access="read")
    assert _edge_exists(edges, "mass_g", access="write")
    assert _edge_exists(edges, "mass_g", access="read")


def test_dict_key_reads_and_writes_are_recorded():
    objects, edges = _collect(
        """
def update_config(config):
    config["solver"] = "SteadyState"
    return config["solver"]
""",
        pyright_families={"param:sample.update_config:config": "dict"},
    )

    assert "dict_key:sample.update_config:config:solver" in objects
    assert _edge_exists(edges, "solver", access="write")
    assert _edge_exists(edges, "solver", access="read")


def test_unknown_family_subscript_becomes_container_field():
    objects, edges = _collect(
        """
def inspect(container):
    return container["solver"]
"""
    )

    assert "container_field:sample.inspect:container:solver" in objects
    assert _edge_exists(edges, "container_field:sample.inspect:container:solver", access="read")


def test_file_handles_json_load_and_dump_are_recorded():
    objects, edges = _collect(
        """
import json

def save_and_load(path, data):
    with open(path, "w") as f:
        json.dump(data, f)
    with open(path) as f:
        return json.load(f)
"""
    )

    assert "file:path" in objects
    assert _edge_exists(edges, "file:path", access="create", operation="open")
    assert _edge_exists(edges, "file:path", access="write", operation="json.dump")
    assert _edge_exists(edges, "file:path", access="read", operation="open")
    assert _edge_exists(edges, "file:path", access="read", operation="json.load")


def test_dataframe_aliases_propagate_to_source_object():
    objects, edges = _collect(
        """
class Processor:
    def process(self):
        Results = self.R[["mass_g"]]
        Results_extended = Results.copy()
        Results_extended.loc[:, "mass_fraction"] = Results["mass_g"]
        self.Results_extended = Results_extended
""",
        pyright_families={"class_attr:sample.Processor:R": "dataframe"},
    )

    assert "class_state:sample.Processor" in objects
    assert any(obj.kind == "df_col" and obj.field == "mass_g" for obj in objects.values())
    assert any(obj.kind == "df_col" and obj.field == "mass_fraction" for obj in objects.values())
    assert not any(obj_id.startswith("class_attr_state:sample.Processor:") for obj_id in objects)
    assert _edge_exists(edges, "mass_g", access="read")
    assert _edge_exists(edges, "mass_fraction", access="write")


def test_same_class_attributes_roll_up_to_single_state_object():
    objects, edges = _collect(
        """
class Compartment:
    def describe(self):
        return self.Cname, self.CsurfaceArea_m2

    def resize(self, value):
        self.Cvolume_m3 = value
"""
    )

    assert "class_state:sample.Compartment" in objects
    assert not any("Compartment.Cname" in obj_id for obj_id in objects)
    assert not any("Compartment.CsurfaceArea_m2" in obj_id for obj_id in objects)
    assert _edge_exists(edges, "class_state:sample.Compartment", access="read")
    assert _edge_exists(edges, "class_state:sample.Compartment", access="create")


def test_large_coordinator_class_splits_by_top_level_attribute():
    objects, edges = _collect(
        """
class Workflow:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.results = {}
        self.lookup = {}

    def load(self):
        self.results["mass_g"] = self.data["mass_g"]

    def summarize(self):
        return self.config["solver"], self.results["mass_g"]

    def index(self):
        return self.lookup["Air"]
""",
        pyright_families={
            "param:sample.Workflow.__init__:config": "dict",
            "param:sample.Workflow.__init__:data": "dict",
            "class_attr:sample.Workflow:config": "dict",
            "class_attr:sample.Workflow:data": "dict",
            "class_attr:sample.Workflow:results": "dict",
            "class_attr:sample.Workflow:lookup": "dict",
        },
    )

    assert "class_state:sample.Workflow" not in objects
    assert "class_attr_state:sample.Workflow:config" in objects
    assert "class_attr_state:sample.Workflow:data" in objects
    assert "class_attr_state:sample.Workflow:results" in objects
    assert "class_attr_state:sample.Workflow:lookup" in objects
    assert _edge_exists(edges, "class_attr_state:sample.Workflow:config", access="read")
    assert _edge_exists(edges, "class_attr_state:sample.Workflow:results", access="write")


def test_split_class_attribute_reassignment_is_write_after_first_creator():
    objects, edges = _collect(
        """
class Workflow:
    def __init__(self, value):
        self.Cvolume_m3 = value
        self.config = {}
        self.results = {}
        self.lookup = {}

    def calc_volume(self):
        if self.Cvolume_m3 is None:
            self.Cvolume_m3 = 1

    def calc_vol_fromBox(self):
        self.Cvolume_m3 = 2

    def summarize(self):
        return self.config, self.results, self.lookup
"""
    )

    object_id = "class_attr_state:sample.Workflow:Cvolume_m3"
    assert object_id in objects
    assert [edge.callable for edge in _matching_edges(edges, object_id, access="create")] == [
        "sample.Workflow.__init__"
    ]
    assert {edge.callable for edge in _matching_edges(edges, object_id, access="write")} == {
        "sample.Workflow.calc_volume",
        "sample.Workflow.calc_vol_fromBox",
    }
    assert _edge_exists(edges, object_id, access="read")


def test_split_class_none_initializer_is_first_creator_not_later_assignment():
    objects, edges = _collect(
        """
class ResultsProcessor:
    def __init__(self):
        self.Results_extended = None
        self.Results = {}
        self.tables = {}
        self.lookup = {}

    def process_results(self):
        self.Results_extended = {}

    def load(self):
        self.Results["mass_g"] = 1

    def summarize(self):
        return self.lookup, self.tables, self.Results_extended
"""
    )

    object_id = "class_attr_state:sample.ResultsProcessor:Results_extended"
    assert object_id in objects
    assert [edge.callable for edge in _matching_edges(edges, object_id, access="create")] == [
        "sample.ResultsProcessor.__init__"
    ]
    assert [edge.callable for edge in _matching_edges(edges, object_id, access="write")] == [
        "sample.ResultsProcessor.process_results"
    ]


def test_literal_container_attrs_make_coordinator_splittable_without_pyright():
    objects, edges = _collect(
        """
class Compartment:
    def __init__(self, name):
        self.name = name
        self.particles = {
            "free": [],
            "bound": [],
        }
        self.processes = [
            "settling",
            "rising",
        ]
        self.links = []

    def add_particle(self, particle):
        self.particles[particle.kind].append(particle)

    def summarize(self):
        return self.name, self.processes, self.links

    def count(self):
        return len(self.particles)
"""
    )

    assert "class_state:sample.Compartment" not in objects
    assert "class_attr_state:sample.Compartment:particles" in objects
    assert "class_attr_state:sample.Compartment:processes" in objects
    assert "class_attr_state:sample.Compartment:links" in objects
    assert _edge_exists(edges, "class_attr_state:sample.Compartment:particles", access="read_write")


def test_mutating_container_methods_do_not_expose_scratch_locals_by_themselves():
    objects, edges = _collect(
        """
def build():
    items = []
    items.append(1)
    mapping = {}
    mapping.update({"a": 1})
    return items
"""
    )

    assert "local_exposed:sample.build:items" in objects
    assert "local_exposed:sample.build:mapping" not in objects
    assert _edge_exists(edges, "local_exposed:sample.build:items", operation="return")
    assert not _edge_exists(edges, "local_exposed:sample.build:mapping")


def test_filters_dynamic_selectors_and_index_slices_do_not_become_objects():
    objects, edges = _collect(
        """
def analyze(processor, R_comp, tables_outputFlows_mass, e_comp, proc):
    concentration = processor.Results_extended["concentration_g_m3"][
        processor.Results_extended["Compartment"] == R_comp
    ]
    flow = tables_outputFlows_mass[e_comp].loc[:, "k_" + proc]
    prefix = tables_outputFlows_mass[e_comp].index[1:2]
    return concentration, flow, prefix
""",
        pyright_families={"attr_expr:sample.analyze:processor.Results_extended": "dataframe"},
    )

    display_names = [obj.display_name for obj in objects.values()]
    assert "processor.Results_extended['concentration_g_m3']" in display_names
    assert "processor.Results_extended['Compartment']" in display_names
    assert not any(" == R_comp" in name for name in display_names)
    assert not any("'k_' + proc" in name for name in display_names)
    assert not any("index['1:2']" in name for name in display_names)
    assert _edge_exists(edges, "concentration_g_m3", access="read")
    assert _edge_exists(edges, "Compartment", access="read")


def test_row_indexer_then_column_normalizes_to_column_object():
    objects, edges = _collect(
        """
def summarize(results_by_comp, i):
    compartment = results_by_comp.iloc[i]["Compartments"]
    inflows = results_by_comp.iloc[i].inflows_g_s.values()
    return compartment, inflows
""",
        pyright_families={"param:sample.summarize:results_by_comp": "dataframe"},
    )

    display_names = [obj.display_name for obj in objects.values()]
    assert "results_by_comp['Compartments']" in display_names
    assert not any("results_by_comp.iloc['Compartments']" in name for name in display_names)
    assert not any(name == "results_by_comp.iloc state" for name in display_names)
    assert _edge_exists(edges, "Compartments", access="read")


def test_dict_entry_and_entry_attribute_share_same_object():
    objects, edges = _collect(
        """
def mixing(model):
    area = model.dict_comp["Bulk_Freshwater"].CsurfaceArea_m2
    comp = model.dict_comp["Bulk_Freshwater"]
    return area, comp
""",
        pyright_families={"attr_expr:sample.mixing:model.dict_comp": "dict"},
    )

    object_id = "dict_key:sample.mixing:model.dict_comp:Bulk_Freshwater"
    assert object_id in objects
    assert not any(obj_id == f"object_state:{object_id}" for obj_id in objects)
    assert _edge_exists(edges, object_id, access="read", operation="subscript_load")
    assert _edge_exists(edges, object_id, access="read", operation="attribute_load")


def test_nested_model_dict_comp_paths_are_precise_for_deposition_helpers():
    objects, edges = _collect(
        """
def mixing(model):
    ocean = model.dict_comp["Ocean_Mixed_Water"].CsurfaceArea_m2
    coast = model.dict_comp["Coast_Column_Water"].CsurfaceArea_m2
    fresh = model.dict_comp["Bulk_Freshwater"].CsurfaceArea_m2
    return ocean, coast, fresh

def dry_deposition(model):
    return model.dict_comp["Air"].CsurfaceArea_m2

def wet_deposition(model):
    return model.dict_comp["Air"].CsurfaceArea_m2
"""
    )

    expected_paths = {
        "model.dict_comp['Ocean_Mixed_Water'].CsurfaceArea_m2",
        "model.dict_comp['Coast_Column_Water'].CsurfaceArea_m2",
        "model.dict_comp['Bulk_Freshwater'].CsurfaceArea_m2",
        "model.dict_comp['Air'].CsurfaceArea_m2",
    }
    observed_paths = {obj.access_path for obj in objects.values()}

    assert expected_paths <= observed_paths
    assert all(
        obj.structural_role == "precise"
        for obj in objects.values()
        if obj.kind == "object_state" and obj.access_path in expected_paths
    )
    assert _edge_exists(edges, "model.dict_comp:Air", access="read", operation="subscript_load")
    assert _edge_exists(edges, "CsurfaceArea_m2", access="read", operation="attribute_load")


def test_same_named_containers_are_not_global_by_default():
    objects, edges = _collect(
        """
class Processor:
    def read_self(self):
        return self.Results_extended["Compartment"]

def summarize(model, processor, Results_extended):
    a = model.Results_extended["Compartment"]
    b = processor.Results_extended["Compartment"]
    c = Results_extended["Compartment"]
    return a, b, c
""",
        pyright_families={
            "class_attr:sample.Processor:Results_extended": "dataframe",
            "attr_expr:sample.summarize:model.Results_extended": "dataframe",
            "attr_expr:sample.summarize:processor.Results_extended": "dataframe",
            "param:sample.summarize:Results_extended": "dataframe",
        },
    )

    compartment_objects = [
        obj_id
        for obj_id, obj in objects.items()
        if obj.kind == "df_col" and obj.field == "Compartment"
    ]
    assert len(compartment_objects) == 4
    assert all(objects[obj_id].display_name.endswith("['Compartment']") for obj_id in compartment_objects)
    assert len([edge for edge in edges if edge.object_id in compartment_objects and edge.access == "read"]) == 4


def test_shared_named_containers_canonicalize_receiver_and_callable_prefixes_when_configured():
    objects, edges = _collect(
        """
class Processor:
    def read_state(self):
        return self.R["mass_g"], self.data["MPdensity_kg_m3"]

def analyze(model, sp2, RC_df, data):
    a = model.R["mass_g"]
    b = sp2.RateConstants["k_fragmentation"]
    c = model.dict_comp["Air"].CsurfaceArea_m2
    d = RC_df["advection"]
    e = data["MPdensity_kg_m3"]
    return a, b, c, d, e
""",
        shared_config={
            "df_col": {"R": "R", "RC_df": "RC_df"},
            "dict_key": {
                "data": "data",
                "RateConstants": "RateConstants",
                "dict_comp": "dict_comp",
            },
        },
        pyright_families={
            "class_attr:sample.Processor:R": "dataframe",
            "class_attr:sample.Processor:data": "dict",
            "attr_expr:sample.analyze:model.R": "dataframe",
            "attr_expr:sample.analyze:sp2.RateConstants": "dict",
            "attr_expr:sample.analyze:model.dict_comp": "dict",
            "param:sample.analyze:RC_df": "dataframe",
            "param:sample.analyze:data": "dict",
        },
    )

    expected_ids = {
        "df_col:R:mass_g",
        "df_col:RC_df:advection",
        "dict_key:RateConstants:k_fragmentation",
        "dict_key:data:MPdensity_kg_m3",
        "dict_key:dict_comp:Air",
    }
    assert expected_ids <= set(objects)
    assert not any("self.R:mass_g" in obj_id for obj_id in objects)
    assert not any("model.R:mass_g" in obj_id for obj_id in objects)
    assert not any("sp2.RateConstants:k_fragmentation" in obj_id for obj_id in objects)

    assert _edge_exists(edges, "df_col:R:mass_g", access="read")
    assert _edge_exists(edges, "dict_key:dict_comp:Air", access="read")


def test_field_objects_keep_object_id_without_rollup():
    objects, edges = _collect(
        """
def calculate():
    alpha_heter = {"freeMP": 0.01, "biofMP": 0.02}
    return alpha_heter["biofMP"] + alpha_heter["freeMP"]

def summarize(Results_extended, data):
    return Results_extended["Compartment"], data["MPdensity_kg_m3"]
""",
        pyright_families={
            "param:sample.summarize:Results_extended": "dataframe",
            "param:sample.summarize:data": "dict",
        },
    )

    expected_ids = {
        "dict_key:sample.calculate:alpha_heter:biofMP",
        "dict_key:sample.calculate:alpha_heter:freeMP",
        "df_col:sample.summarize:Results_extended:Compartment",
        "dict_key:sample.summarize:data:MPdensity_kg_m3",
    }
    assert expected_ids <= set(objects)
    assert "local_exposed:sample.calculate:alpha_heter" not in objects
    assert _edge_exists(edges, "dict_key:sample.calculate:alpha_heter:biofMP", access="read")
    assert _edge_exists(edges, "df_col:sample.summarize:Results_extended:Compartment", access="read")


def test_configured_shared_container_changes_field_object_identity_without_cluster_id():
    objects, edges = _collect(
        """
def summarize(Results_extended, data):
    return Results_extended["Compartment"], data["MPdensity_kg_m3"]
""",
        shared_config={
            "df_col": {"Results_extended": "Results_extended"},
            "dict_key": {"data": "data"},
        },
        pyright_families={
            "param:sample.summarize:Results_extended": "dataframe",
            "param:sample.summarize:data": "dict",
        },
    )

    assert "df_col:Results_extended:Compartment" in objects
    assert "dict_key:data:MPdensity_kg_m3" in objects
    assert objects["df_col:Results_extended:Compartment"].container == "shared_container:Results_extended"
    assert objects["dict_key:data:MPdensity_kg_m3"].container == "shared_container:data"


def test_local_container_is_exposed_only_when_it_escapes():
    objects, edges = _collect(
        """
def direct_return():
    items = []
    return items

def mutation():
    mapping = {}
    mapping.update({"x": 1})
    return 1
"""
    )

    assert "local_exposed:sample.direct_return:items" in objects
    assert "local_exposed:sample.mutation:mapping" not in objects
    assert _edge_exists(edges, "local_exposed:sample.direct_return:items", operation="return")
    assert not _edge_exists(edges, "local_exposed:sample.mutation:mapping")


def test_local_container_is_exposed_when_assigned_into_class_state():
    objects, edges = _collect(
        """
class Holder:
    def attach(self):
        items = []
        self.items = items
        return self.items
"""
    )

    assert "local_exposed:sample.Holder.attach:items" in objects
    assert _edge_exists(
        edges,
        "local_exposed:sample.Holder.attach:items",
        operation="escape_assign",
    )
    assert _edge_exists(edges, "class_state:sample.Holder", access="create")


def test_simple_return_value_alias_propagates_to_call_site():
    objects, edges = _collect(
        """
def build_results():
    df = {}
    return df

def use_results():
    results = build_results()
    return results["mass_g"]
"""
    )

    object_id = "dict_key:sample.build_results:df:mass_g"
    alias_id = "local_exposed:sample.use_results:results"
    assert object_id in objects
    assert alias_id in objects
    assert _edge_exists(edges, object_id, access="read")
    assert objects[alias_id].alias_of == "local_exposed:sample.build_results:df"


def test_transitive_return_value_alias_reaches_outer_call_site():
    objects, edges = _collect(
        """
def build_results():
    df = {}
    return df

def passthrough():
    return build_results()

def use_results():
    results = passthrough()
    return results["mass_g"]
"""
    )

    object_id = "dict_key:sample.build_results:df:mass_g"
    alias_id = "local_exposed:sample.use_results:results"
    assert object_id in objects
    assert alias_id in objects
    assert _edge_exists(edges, object_id, access="read")
    assert objects[alias_id].alias_of == "local_exposed:sample.build_results:df"


def test_ambiguous_return_value_sources_do_not_merge_downstream_param():
    objects, edges = _collect(
        """
def source():
    df = {}
    return df

def process(df, flag):
    other = {}
    if flag:
        return other
    return df

def analyze(df):
    return df["mass_g"]

def main(flag):
    raw = source()
    processed = process(raw, flag)
    return analyze(processed)
""",
        pyright_families={
            "param:sample.process:df": "dict",
            "param:sample.analyze:df": "dict",
        },
    )

    producer_object = "local_exposed:sample.source:df"
    alternate_object = "local_exposed:sample.process:other"
    processed_object = "local_exposed:sample.main:processed"
    analyze_param = "param:sample.analyze:df"
    analyzed_field = "dict_key:sample.analyze:df:mass_g"

    assert producer_object in objects
    assert alternate_object in objects
    assert processed_object in objects
    assert analyze_param in objects
    assert analyzed_field in objects
    assert objects[processed_object].alias_of == ""
    assert objects[analyze_param].alias_of == ""
    assert objects[analyzed_field].container == analyze_param


def test_ambiguous_return_value_sources_propagate_through_local_copy():
    objects, edges = _collect(
        """
def source():
    df = {}
    return df

def process(df, flag):
    other = {}
    if flag:
        return other
    return df

def analyze(df):
    return df["mass_g"]

def main(flag):
    raw = source()
    processed = process(raw, flag)
    forwarded = processed
    return analyze(forwarded)
""",
        pyright_families={
            "param:sample.process:df": "dict",
            "param:sample.analyze:df": "dict",
        },
    )

    forwarded_object = "local_exposed:sample.main:forwarded"
    analyze_param = "param:sample.analyze:df"
    analyzed_field = "dict_key:sample.analyze:df:mass_g"

    assert forwarded_object in objects
    assert analyze_param in objects
    assert analyzed_field in objects
    assert objects[forwarded_object].alias_of == ""
    assert objects[analyze_param].alias_of == ""
    assert objects[analyzed_field].container == analyze_param


def test_returning_copy_does_not_expose_underlying_local():
    objects, edges = _collect(
        """
def build():
    data = {}
    return data.copy()
"""
    )

    object_id = "local_exposed:sample.build:data"
    assert object_id not in objects
    assert not _edge_exists(edges, object_id)


def test_comprehension_iteration_variables_do_not_shadow_as_outer_data():
    objects, edges = _collect(
        """
def summarize(data, row):
    return [row["mass_g"] for row in data if row["keep"]]
"""
    )

    assert "param:sample.summarize:data" in objects
    assert _edge_exists(edges, "param:sample.summarize:data", access="read")
    assert "param:sample.summarize:row" not in objects
    assert not any("row:mass_g" in object_id for object_id in objects)


def test_non_mutating_method_call_records_receiver_read():
    objects, edges = _collect(
        """
def copy_results(model):
    return model.results.copy()
"""
    )

    assert _edge_exists(
        edges,
        "object_state:param:sample.copy_results:model:results",
        access="read",
        operation="method:copy:receiver",
    )


def test_mass_balance_style_model_paths_keep_loop_element_identity():
    objects, edges = _collect(
        """
def massBalance(model):
    for p in model.system_particle_object_list:
        if p.Pcode[0] == "a":
            loss = p.RateConstants["k_fragmentation"]
    m_ss = model.R["mass_g"]
    in_flow = list(model.input_flows_g_s.values())
    return loss, m_ss, in_flow
"""
    )

    rate_object = "container_field:sample.massBalance:model.system_particle_object_list[].RateConstants:k_fragmentation"
    result_object = "container_field:sample.massBalance:model.R:mass_g"
    input_state = "object_state:param:sample.massBalance:model:input_flows_g_s"
    coarse_state = "object_state:param:sample.massBalance:model"

    assert rate_object in objects
    assert result_object in objects
    assert input_state in objects
    assert objects[rate_object].access_path == "model.system_particle_object_list[].RateConstants['k_fragmentation']"
    assert objects[result_object].access_path == "model.R['mass_g']"
    assert objects[input_state].access_path == "model.input_flows_g_s"
    assert objects[rate_object].structural_role == "precise"
    assert objects[coarse_state].structural_role == "coarse"
    assert _edge_exists(edges, rate_object, access="read", operation="subscript_load")
    assert _edge_exists(edges, input_state, access="read", operation="method:values:receiver")


def test_callee_param_descendants_keep_raw_ids_with_param_alias_path():
    objects, edges = _collect(
        """
def process_particle(particle):
    code = particle.Pcode
    rate = particle.RateConstants["k_fragmentation"]
    return code, rate

def caller(model):
    for particle in model.system_particle_object_list:
        direct = particle.Pcode
        return direct, process_particle(particle)
"""
    )

    caller_code = "object_state:param:sample.caller:model:system_particle_object_list[].Pcode"
    callee_code = "object_state:param:sample.process_particle:particle:Pcode"
    callee_rate = "container_field:sample.process_particle:particle.RateConstants:k_fragmentation"

    assert caller_code in objects
    assert callee_code in objects
    assert callee_rate in objects
    assert objects[f"param:sample.process_particle:particle"].access_path == "model.system_particle_object_list[]"
    assert (
        objects[f"param:sample.process_particle:particle"].alias_of
        == "object_state:param:sample.caller:model:system_particle_object_list"
    )
    assert objects[callee_code].access_path == "particle.Pcode"
    assert objects[callee_rate].access_path == "particle.RateConstants['k_fragmentation']"
    assert _edge_exists(edges, callee_code, access="read", operation="attribute_load")
    assert _edge_exists(edges, callee_rate, access="read", operation="subscript_load")


def test_split_class_nested_self_paths_keep_descendant_identity_and_alias_root():
    objects, edges = _collect(
        """
class Processor:
    def __init__(self, model):
        self.model = model
        self.config = {}
        self.results = {}
        self.lookup = {}

    def process(self):
        for particle in self.model.system_particle_object_list:
            return particle.Pcode

    def load(self):
        self.results["mass_g"] = self.config["mass_g"]

    def summarize(self):
        return self.lookup, self.results

def run(model):
    processor = Processor(model)
    return processor.process()
""",
        pyright_families={
            "class_attr:sample.Processor:config": "dict",
            "class_attr:sample.Processor:results": "dict",
            "class_attr:sample.Processor:lookup": "dict",
        },
    )

    self_code = "object_state:class_attr_state:sample.Processor:model:self.model.system_particle_object_list[].Pcode"
    run_code = "object_state:param:sample.run:model:system_particle_object_list[].Pcode"

    assert self_code in objects
    assert objects[self_code].access_path == "self.model.system_particle_object_list[].Pcode"
    assert objects["class_attr_state:sample.Processor:model"].alias_of == "param:sample.run:model"
    assert run_code not in objects
    assert _edge_exists(edges, self_code, access="read", operation="attribute_load")


def test_solver_style_parent_particle_paths_are_recorded_only_when_touched():
    objects, edges = _collect(
        """
def solve_ODES_SS(system_particle_object_list):
    for p in system_particle_object_list:
        mass = p.parentMP.parentMP.Pvolume_m3
        density = p.parentMP.Pdensity_kg_m3
    return mass, density
"""
    )

    volume_state = "object_state:param:sample.solve_ODES_SS:system_particle_object_list:[].parentMP.parentMP.Pvolume_m3"
    density_state = "object_state:param:sample.solve_ODES_SS:system_particle_object_list:[].parentMP.Pdensity_kg_m3"

    assert volume_state in objects
    assert density_state in objects
    assert objects[volume_state].access_path == "system_particle_object_list[].parentMP.parentMP.Pvolume_m3"
    assert objects[density_state].access_path == "system_particle_object_list[].parentMP.Pdensity_kg_m3"
    assert not any("parentMP.parentMP.parentMP" in obj.access_path for obj in objects.values())
    assert _edge_exists(edges, volume_state, access="read", operation="attribute_load")
    assert _edge_exists(edges, density_state, access="read", operation="attribute_load")


def test_source_order_placeholder_local_is_upgraded_from_unknown():
    objects, edges = _collect(
        """
def caller():
    return build()

def build():
    items = []
    return items
"""
    )

    object_id = "local_exposed:sample.build:items"
    assert object_id in objects
    assert objects[object_id].kind == "local_exposed"
    assert objects[object_id].display_name == "items"
    assert _edge_exists(edges, object_id, access="read")


def test_source_order_placeholder_param_is_upgraded_from_unknown():
    objects, edges = _collect(
        """
def caller(results):
    return consume(results)

def consume(results):
    return results["mass_g"]
""",
        pyright_families={"param:sample.consume:results": "dict"},
    )

    object_id = "param:sample.consume:results"
    assert object_id in objects
    assert objects[object_id].kind == "param"
    assert objects[object_id].display_name == "results"
    assert _edge_exists(edges, object_id, access="read")


def test_hidden_local_receiver_does_not_materialize_unknown_local_exposed_object():
    objects, edges = _collect(
        """
import pandas as pd

def load_csv_column(path, column_name):
    df = pd.read_csv(path, usecols=[column_name])
    return df[column_name].tolist()
"""
    )

    object_id = "local_exposed:sample.load_csv_column:df"
    assert object_id not in objects
    assert not _edge_exists(edges, object_id)


def test_attrdict_attribute_load_becomes_dict_key_read():
    objects, edges = _collect(
        """
class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


class Model:
    def __init__(self):
        self.state = AttrDict()

    def run(self):
        return self.state.Ts
"""
    )

    object_id = "dict_key:class_state:sample.Model:self.state:Ts"
    assert object_id in objects
    assert objects[object_id].kind == "dict_key"
    assert objects[object_id].confidence == "medium"
    assert _edge_exists(edges, object_id, access="read", operation="attribute_load")


def test_pyright_dict_class_attr_attribute_load_becomes_dict_key_read():
    objects, edges = _collect(
        """
class Model:
    def run(self):
        return self.state.Ts
""",
        pyright_families={"class_attr:sample.Model:state": "dict"},
    )

    object_id = "dict_key:class_state:sample.Model:self.state:Ts"
    assert object_id in objects
    assert _edge_exists(edges, object_id, access="read", operation="attribute_load")


def test_pyright_dict_param_attribute_load_becomes_dict_key_read():
    objects, edges = _collect(
        """
def read_state(state):
    return state.Ts
""",
        pyright_families={"param:sample.read_state:state": "dict"},
    )

    object_id = "dict_key:sample.read_state:state:Ts"
    assert object_id in objects
    assert objects[object_id].confidence == "medium"
    assert _matching_edges(edges, object_id, access="read")


def test_returned_attrdict_preserves_access_path_for_attribute_key_read():
    objects, edges = _collect(
        """
class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

def build():
    state = AttrDict()
    return state

def read():
    return build().Ts
"""
    )

    matches = [obj for obj in objects.values() if obj.kind == "dict_key" and obj.field == "Ts"]
    assert matches
    assert any(obj.access_path == "state['Ts']" for obj in matches)
    assert any(_matching_edges(edges, obj.id, access="read") for obj in matches)


def test_xarray_open_and_labeled_access_are_recorded():
    objects, edges = _collect(
        """
import xarray as xr

def load(path):
    ds = xr.open_dataset(path)
    by_lat = ds.sel(lat=45.0)
    by_lev = ds.isel(lev=0)
    by_time = ds.loc[{"time": "jan"}]
    return by_lat, by_lev, by_time
"""
    )

    assert _edge_exists(edges, "file:path", access="read", operation="xr.open_dataset")
    expected = {
        "dict_key:sample.load:ds:lat": "method:sel:labeled_access",
        "dict_key:sample.load:ds:lev": "method:isel:labeled_access",
        "dict_key:sample.load:ds:time": "subscript_load",
    }
    for object_id, operation in expected.items():
        assert object_id in objects
        assert any(edge.operation == operation for edge in _matching_edges(edges, object_id, access="read"))


def test_xarray_labeled_call_with_unknown_receiver_records_receiver_read():
    objects, edges = _collect(
        """
def load(ds):
    return ds.sel(lat=45.0)
"""
    )

    receiver_id = "param:sample.load:ds"
    assert receiver_id in objects
    assert any(
        edge.object_id == receiver_id
        and edge.access == "read"
        and edge.operation == "method:sel:receiver"
        for edge in edges
    )
    assert not any(obj.kind == "dict_key" and obj.field == "lat" for obj in objects.values())


def test_xarray_open_dataarray_and_pooch_retrieve_are_recorded():
    objects, edges = _collect(
        """
import pooch
import xarray as xr

def load(path):
    fetched = pooch.retrieve(url="https://example.com/data.nc", known_hash=None)
    da = xr.open_dataarray(path)
    return da.isel(lev=0), fetched
"""
    )

    assert _edge_exists(edges, "https://example.com/data.nc", access="read", operation="pooch.retrieve")
    assert _edge_exists(edges, "file:path", access="read", operation="xr.open_dataarray")
    assert any(obj.kind == "dict_key" and obj.field == "lev" for obj in objects.values())


def test_add_subprocess_records_state_assign_lineage(tmp_path):
    (tmp_path / "sample.py").write_text(
        """
class Child:
    def _compute(self):
        pass


class Parent:
    def __init__(self):
        child = Child()
        self.add_subprocess("child", child)

    def _compute(self):
        pass

    def add_subprocess(self, name, proc):
        pass
""",
        encoding="utf-8",
    )

    callable_map, _module_map, _known_classes = build_indices(tmp_path)
    objects, _edges, lineage = collect_data_access(
        tmp_path,
        callable_map=callable_map,
        pyright_families={},
    )

    assert "class_state:sample.Child" in objects
    assert "class_state:sample.Parent" in objects
    assert any(
        edge.src_object_id == "class_state:sample.Child"
        and edge.dst_object_id == "class_state:sample.Parent"
        and edge.relation == "state_assign"
        and edge.slot == "child"
        for edge in lineage
    )


def test_add_subprocess_resolves_loop_and_with_bound_child_types(tmp_path):
    (tmp_path / "sample.py").write_text(
        """
class Child:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def _compute(self):
        pass


class LoopParent:
    def __init__(self):
        children = [Child()]
        for child in children:
            self.add_subprocess("loop_child", child)

    def _compute(self):
        pass

    def add_subprocess(self, name, proc):
        pass


class WithParent:
    def __init__(self):
        with Child() as child:
            self.add_subprocess("with_child", child)

    def _compute(self):
        pass

    def add_subprocess(self, name, proc):
        pass
""",
        encoding="utf-8",
    )

    callable_map, _module_map, _known_classes = build_indices(tmp_path)
    _objects, _edges, lineage = collect_data_access(
        tmp_path,
        callable_map=callable_map,
        pyright_families={},
    )

    lineage_tuples = {
        (edge.src_object_id, edge.dst_object_id, edge.relation, edge.slot)
        for edge in lineage
    }
    assert (
        "class_state:sample.Child",
        "class_state:sample.LoopParent",
        "state_assign",
        "loop_child",
    ) in lineage_tuples
    assert (
        "class_state:sample.Child",
        "class_state:sample.WithParent",
        "state_assign",
        "with_child",
    ) in lineage_tuples


def test_add_subprocess_state_lineage_uses_split_class_state_namespace(tmp_path):
    (tmp_path / "sample.py").write_text(
        """
class Child:
    def _compute(self):
        pass


class Parent:
    def __init__(self):
        self.state = {}
        self.config = {}
        self.results = {}
        self.lookup = {}
        child = Child()
        self.add_subprocess("child", child)

    def load(self):
        self.results["mass_g"] = self.state["mass_g"]

    def summarize(self):
        return self.config["solver"], self.results["mass_g"]

    def index(self):
        return self.lookup["Air"]

    def _compute(self):
        pass

    def add_subprocess(self, name, proc):
        pass
""",
        encoding="utf-8",
    )

    callable_map, _module_map, _known_classes = build_indices(tmp_path)
    objects, _edges, lineage = collect_data_access(
        tmp_path,
        callable_map=callable_map,
        pyright_families={},
    )

    assert "class_attr_state:sample.Parent:state" in objects
    assert any(
        edge.src_object_id == "class_state:sample.Child"
        and edge.dst_object_id == "class_attr_state:sample.Parent:state"
        and edge.relation == "state_assign"
        and edge.slot == "child"
        for edge in lineage
    )


def test_unique_passed_object_lineage_rolls_param_state_up_to_producer_for_clustering():
    objects, edges = _collect(
        """
def build():
    items = []
    return items

def consume(items):
    for item in items:
        return item.value

def orchestrate():
    produced = build()
    return consume(produced)
"""
    )

    producer_object = "local_exposed:sample.build:items"
    consumer_param = "param:sample.consume:items"
    consumer_state = "object_state:param:sample.consume:items"

    assert producer_object in objects
    assert consumer_param in objects
    assert consumer_state in objects
    assert objects[consumer_param].alias_of == producer_object
    assert objects[consumer_state].container == consumer_param


def test_ambiguous_passed_object_lineage_does_not_merge_param_clustering():
    objects, edges = _collect(
        """
def build_a():
    items = []
    return items

def build_b():
    items = []
    return items

def consume(items):
    for item in items:
        return item.value

def orchestrate():
    first = build_a()
    second = build_b()
    consume(first)
    return consume(second)
"""
    )

    consumer_param = "param:sample.consume:items"
    consumer_state = "object_state:param:sample.consume:items"

    assert consumer_param in objects
    assert consumer_state in objects
    assert objects[consumer_param].alias_of == ""
    assert objects[consumer_state].container == consumer_param


def test_tuple_return_unpack_and_self_assignment_preserve_passed_object_lineage():
    objects, edges = _collect(
        """
class Runner:
    def run(self):
        self.items, self.other = build()
        return consume(self.items)

def build():
    items = []
    other = []
    return items, other

def consume(items):
    for item in items:
        return item.value
"""
    )

    producer_object = "local_exposed:sample.build:items"
    consumer_param = "param:sample.consume:items"
    consumer_state = "object_state:param:sample.consume:items"

    assert producer_object in objects
    assert consumer_param in objects
    assert consumer_state in objects
    assert objects[consumer_param].alias_of == producer_object
    assert objects[consumer_state].container == consumer_param


def test_dynamic_getattr_module_dispatch_records_param_lineage_for_possible_callees():
    source = """
import sample_ops as ops

def dispatch(particle, model, proc):
    return getattr(ops, proc)(particle, model)
"""
    lineage_edges = []
    param_bindings = {}
    collect_data_access_from_tree(
        ast.parse(source, filename="sample.py"),
        module="sample",
        file=Path("sample.py"),
        callable_map={
            "sample_ops.alpha": object(),
            "sample_ops.beta": object(),
        },
        callable_params={
            "sample_ops.alpha": ("particle", "model"),
            "sample_ops.beta": ("particle", "model"),
        },
        param_bindings=param_bindings,
        lineage_edges=lineage_edges,
    )

    lineage = {
        (edge.src_object_id, edge.dst_object_id, edge.callee, edge.slot)
        for edge in lineage_edges
        if edge.relation == "arg_to_param"
    }

    assert (
        "param:sample.dispatch:particle",
        "param:sample_ops.alpha:particle",
        "sample_ops.alpha",
        "particle",
    ) in lineage
    assert (
        "param:sample.dispatch:model",
        "param:sample_ops.alpha:model",
        "sample_ops.alpha",
        "model",
    ) in lineage
    assert (
        "param:sample.dispatch:particle",
        "param:sample_ops.beta:particle",
        "sample_ops.beta",
        "particle",
    ) in lineage
    assert (
        "param:sample.dispatch:model",
        "param:sample_ops.beta:model",
        "sample_ops.beta",
        "model",
    ) in lineage


def test_data_access_includes_explicit_entrypoint_without_scanning_all_docs(tmp_path):
    src = tmp_path / "src" / "utopia"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "runner.py").write_text(
        """
def run(config):
    return config["solver"]
""",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    entrypoint = docs / "run_example.py"
    entrypoint.write_text(
        """
from utopia.runner import run

config = {"solver": "SteadyState"}
run(config)
""",
        encoding="utf-8",
    )
    (docs / "ignored.py").write_text(
        """
def ignored():
    return 1
""",
        encoding="utf-8",
    )

    callable_map, _module_map, _known_classes = build_indices(
        src,
        module_prefix="utopia",
        entrypoints=(entrypoint,),
    )
    objects, edges, _lineage = collect_data_access(
        src,
        callable_map=callable_map,
        module_prefix="utopia",
        pyright_families={},
        entrypoints=(entrypoint,),
    )

    assert "docs.run_example.<module>" in callable_map
    assert "docs.ignored.<module>" not in callable_map
    assert any(edge.callable == "docs.run_example.<module>" for edge in edges)
    assert any(
        obj.owner == "docs.run_example.<module>" and obj.display_name == "config"
        for obj in objects.values()
    )
