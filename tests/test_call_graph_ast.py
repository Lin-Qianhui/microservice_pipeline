from microservice_pipeline.call_graph.generate_call_graph_ast import (
    build_indices,
    build_return_summaries,
    build_type_summaries,
    collect_edges,
    iter_python_files,
)


def _edge_exists(edges, caller, callee, relation=None):
    for edge in edges:
        if edge.caller != caller or edge.callee != callee:
            continue
        if relation is not None and edge.relation != relation:
            continue
        return True
    return False


def _build_call_graph(tmp_path):
    nodes, module_map, known_classes = build_indices(tmp_path)
    return_summaries = build_return_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
    )
    param_summaries, class_attr_types = build_type_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
    )
    edges = collect_edges(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=False,
        package_prefix=None,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
    )
    return nodes, edges


def test_source_excluded_files_are_config_driven(tmp_path):
    src = tmp_path / "src" / "utopia"
    src.mkdir(parents=True)
    included = src / "used.py"
    included.write_text("def used():\n    pass\n", encoding="utf-8")
    excluded = src / "preprocessing" / "rc_sea_spray.py"
    excluded.parent.mkdir()
    excluded.write_text("def unused():\n    pass\n", encoding="utf-8")

    assert {path.relative_to(src) for path in iter_python_files(src)} == {
        included.relative_to(src),
        excluded.relative_to(src),
    }

    scanned = {
        path.relative_to(src)
        for path in iter_python_files(
            src,
            project_root=tmp_path,
            exclude_globs=["src/utopia/preprocessing/rc_sea_spray.py"],
        )
    }

    assert scanned == {included.relative_to(src)}


def test_explicit_entrypoint_is_included_without_scanning_all_docs(tmp_path):
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
    (docs / "ignored.py").write_text("def ignored():\n    pass\n", encoding="utf-8")

    nodes, module_map, known_classes = build_indices(
        src,
        module_prefix="utopia",
        entrypoints=(entrypoint,),
    )
    return_summaries = build_return_summaries(
        src,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        module_prefix="utopia",
        entrypoints=(entrypoint,),
    )
    param_summaries, class_attr_types = build_type_summaries(
        src,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
        module_prefix="utopia",
        entrypoints=(entrypoint,),
    )
    edges = collect_edges(
        src,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=False,
        package_prefix=None,
        module_prefix="utopia",
        entrypoints=(entrypoint,),
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
    )

    assert "docs.run_example.<module>" in nodes
    assert "docs.ignored.<module>" not in nodes
    assert _edge_exists(
        edges,
        "docs.run_example.<module>",
        "utopia.runner.run",
        "imported",
    )


def test_relative_import_from_sibling_module_resolves_imported_callable(tmp_path):
    package = tmp_path / "pkg" / "feature"
    package.mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text(
        """
def direct():
    pass
""",
        encoding="utf-8",
    )
    (package / "consumer.py").write_text(
        """
from .helper import direct


def run():
    direct()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "pkg.feature.consumer.run",
        "pkg.feature.helper.direct",
        "imported",
    )


def test_relative_import_from_parent_package_resolves_imported_callable(tmp_path):
    package = tmp_path / "pkg" / "feature"
    package.mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "common.py").write_text(
        """
def shared():
    pass
""",
        encoding="utf-8",
    )
    (package / "consumer.py").write_text(
        """
from ..common import shared


def run():
    shared()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "pkg.feature.consumer.run",
        "pkg.common.shared",
        "imported",
    )


def test_relative_import_from_current_package_resolves_module_alias_call(tmp_path):
    package = tmp_path / "pkg" / "feature"
    package.mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sibling.py").write_text(
        """
def module_call():
    pass
""",
        encoding="utf-8",
    )
    (package / "consumer.py").write_text(
        """
from . import sibling


def run():
    sibling.module_call()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "pkg.feature.consumer.run",
        "pkg.feature.sibling.module_call",
        "imported",
    )


def test_inherited_methods_from_returned_list_items_are_resolved(tmp_path):
    (tmp_path / "producer.py").write_text(
        """
class Base:
    def __init__(self):
        pass

    def work(self):
        pass


class Child(Base):
    def __init__(self):
        super().__init__()


def make_items():
    items = []
    items.append(Child())
    return items
""",
        encoding="utf-8",
    )
    (tmp_path / "consumer.py").write_text(
        """
from producer import make_items


def run():
    items = make_items()
    for item in items:
        item.work()
""",
        encoding="utf-8",
    )

    nodes, module_map, known_classes = build_indices(tmp_path)
    return_summaries = build_return_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
    )
    param_summaries, class_attr_types = build_type_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
    )
    edges = collect_edges(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=False,
        package_prefix=None,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
    )

    assert _edge_exists(
        edges, "producer.Child.__init__", "producer.Base.__init__", "super_method"
    )
    assert _edge_exists(
        edges, "producer.make_items", "producer.Child.__init__", "constructor"
    )
    assert _edge_exists(
        edges, "consumer.run", "producer.Base.work", "inferred_type"
    )


def test_nested_attribute_methods_from_return_slot_and_param_alias_are_resolved(tmp_path):
    (tmp_path / "objects.py").write_text(
        """
class Particulates:
    def calc_numConc(self):
        pass
""",
        encoding="utf-8",
    )
    (tmp_path / "factory.py").write_text(
        """
from objects import Particulates


def generate_objects():
    spm = Particulates()
    return [], spm
""",
        encoding="utf-8",
    )
    (tmp_path / "rates.py").write_text(
        """
def heteroaggregation(model):
    model.spm.calc_numConc()
""",
        encoding="utf-8",
    )
    (tmp_path / "model.py").write_text(
        """
from factory import generate_objects
from rates import heteroaggregation


class Model:
    def run(self):
        _particles, self.spm = generate_objects()
        heteroaggregation(self)
""",
        encoding="utf-8",
    )

    nodes, module_map, known_classes = build_indices(tmp_path)
    return_summaries = build_return_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
    )
    param_summaries, class_attr_types = build_type_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
    )
    edges = collect_edges(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=False,
        package_prefix=None,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
    )

    assert _edge_exists(
        edges, "rates.heteroaggregation", "objects.Particulates.calc_numConc", "inferred_type"
    )


def test_bound_method_argument_types_skip_implicit_self(tmp_path):
    (tmp_path / "sample.py").write_text(
        """
class Particle:
    def assign_compartment(self, comp):
        pass


class Compartment:
    def add_particles(self, particle):
        particle.assign_compartment(self)


def run():
    comp = Compartment()
    particle = Particle()
    comp.add_particles(particle)
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "sample.Compartment.add_particles",
        "sample.Particle.assign_compartment",
        "inferred_type",
    )


def test_staticmethod_arguments_do_not_get_bound_method_offset(tmp_path):
    (tmp_path / "sample.py").write_text(
        """
class Target:
    def use(self):
        pass


class Helper:
    @staticmethod
    def forward(target):
        target.use()

    def run(self, target):
        self.forward(target)


def run():
    helper = Helper()
    target = Target()
    helper.run(target)
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "sample.Helper.forward",
        "sample.Target.use",
        "inferred_type",
    )


def test_branch_assignments_merge_inferred_receiver_types(tmp_path):
    (tmp_path / "sample.py").write_text(
        """
class Apple:
    def __init__(self):
        pass

    def do_thing(self):
        pass


class Banana:
    def __init__(self):
        pass

    def do_thing(self):
        pass


def run(cond):
    if cond:
        x = Apple()
    else:
        x = Banana()
    x.do_thing()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "sample.run", "sample.Apple.do_thing", "inferred_type")
    assert _edge_exists(edges, "sample.run", "sample.Banana.do_thing", "inferred_type")


def test_property_and_dunder_accesses_are_recorded(tmp_path):
    (tmp_path / "sample.py").write_text(
        """
class Child:
    def use(self):
        pass


class Box:
    def __init__(self):
        pass

    @property
    def value(self):
        return Child()

    def __getitem__(self, key):
        return Child()

    def __contains__(self, key):
        return True

    def __iter__(self):
        return iter(())

    def __add__(self, other):
        return self


def run():
    box = Box()
    child = box.value
    child.use()
    box["key"]
    "key" in box
    for _item in box:
        pass
    box + box
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "sample.run", "sample.Box.value", "property_getter")
    assert _edge_exists(edges, "sample.run", "sample.Child.use", "inferred_type")
    assert _edge_exists(edges, "sample.run", "sample.Box.__getitem__", "dunder_getitem")
    assert _edge_exists(edges, "sample.run", "sample.Box.__contains__", "dunder_contains")
    assert _edge_exists(edges, "sample.run", "sample.Box.__iter__", "dunder_iter")
    assert _edge_exists(edges, "sample.run", "sample.Box.__add__", "dunder_operator")


def test_module_level_code_and_local_imports_are_recorded(tmp_path):
    (tmp_path / "dep.py").write_text(
        """
def init():
    pass


init()
""",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        """
import dep


class Registrar:
    def __init__(self):
        pass

    def register(self):
        pass


registry = Registrar()
registry.register()


def build():
    pass


build()
""",
        encoding="utf-8",
    )

    nodes, edges = _build_call_graph(tmp_path)

    assert "main.<module>" in nodes
    assert nodes["main.<module>"].kind == "module"
    assert _edge_exists(edges, "main.<module>", "dep.<module>", "import")
    assert _edge_exists(edges, "dep.<module>", "dep.init", "direct")
    assert _edge_exists(edges, "main.<module>", "main.Registrar.__init__", "constructor")
    assert _edge_exists(edges, "main.<module>", "main.Registrar.register", "inferred_type")
    assert _edge_exists(edges, "main.<module>", "main.build", "direct")
