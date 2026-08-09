import warnings

from microservice_pipeline.call_graph.generate_call_graph_ast import (
    CallGraphHealth,
    ProjectIndex,
    build_call_graph_from_analysis_files,
    build_indices,
    build_registration_rules,
    build_return_summaries,
    build_type_summaries,
    collect_edges,
    iter_analysis_files,
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
    summaries = build_type_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
    )
    project_index = ProjectIndex(module_map, known_classes, set(nodes.keys()))
    edges = collect_edges(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=False,
        package_prefix=None,
        return_summaries=return_summaries,
        param_summaries=summaries.params,
        class_attr_types=summaries.class_attrs,
        registry_facts=summaries.registry,
        registration_rules=build_registration_rules(
            summaries.escapes, summaries.registry, project_index
        ),
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
    summaries = build_type_summaries(
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
        param_summaries=summaries.params,
        class_attr_types=summaries.class_attrs,
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


def test_registering_a_child_couples_the_invoked_hook(tmp_path):
    """The wiring method's name must not matter -- only what it does with the value.

    ``wire_up`` is deliberately meaningless. What makes this a registration is
    that its ``proc`` parameter is retained in ``self.slots`` and that elements
    of ``self.slots`` are later invoked.
    """
    (tmp_path / "sample.py").write_text(
        """
class Child:
    def run(self):
        pass


class Parent:
    def __init__(self):
        self.slots = {}
        child = Child()
        self.wire_up("child", child)

    def wire_up(self, name, proc):
        self.slots.update({name: proc})

    def run(self):
        for name, proc in self.slots.items():
            proc.run()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "sample.Parent.run",
        "sample.Child.run",
        "registered_invoke",
    )


def test_registration_survives_a_relay_between_two_attributes(tmp_path):
    """The attribute registered into need not be the one invoked.

    climlab stores children in ``self.subprocess`` but iterates
    ``self.process_types`` when it runs them, having copied between the two in a
    third method. Keying the invoke fact on the attribute the child escaped into
    would find nothing here.
    """
    (tmp_path / "sample.py").write_text(
        """
class Child:
    kind = "fast"

    def run(self):
        pass


class Parent:
    def __init__(self):
        self.slots = {}
        self.by_kind = {"fast": [], "slow": []}
        self.attach("child", Child())

    def attach(self, name, proc):
        self.slots.update({name: proc})

    def regroup(self):
        for name, proc in self.slots.items():
            self.by_kind[proc.kind].append(proc)

    def run(self):
        for proc in self.by_kind["fast"]:
            proc.run()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "sample.Parent.run",
        "sample.Child.run",
        "registered_invoke",
    )


def test_registration_survives_leaving_through_a_return_value_and_an_argument(tmp_path):
    """The registry's contents are handed out of the class, then invoked elsewhere.

    This is matplotlib's ``add_artist`` shape. Nothing is stored in a second
    attribute, so attribute-to-attribute relaying finds nothing: the children
    leave via ``get_children()``'s return value, get bound to a local, are passed
    as an argument to a free function, and are only invoked there.
    """
    (tmp_path / "sample.py").write_text(
        """
class Child:
    def draw(self):
        pass


def draw_all(items):
    for item in items:
        item.draw()


class Parent:
    def __init__(self):
        self.children = []
        self.add_child(Child())

    def add_child(self, artist):
        self.children.append(artist)

    def get_children(self):
        return [*self.children]

    def draw(self):
        drawable = self.get_children()
        draw_all(drawable)
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges, "sample.Parent.draw", "sample.Child.draw", "registered_invoke"
    )


def test_registration_reprojects_through_a_template_method(tmp_path):
    """A hook shared by both ends is useless, so follow the base's delegation.

    ``run`` is defined only on ``Node`` and overridden by neither concrete class,
    so linking it would produce an edge from a node to itself. The base hands off
    to ``self._step()``, which subclasses do override, and that is what carries
    which process this is.
    """
    (tmp_path / "sample.py").write_text(
        """
class Node:
    def __init__(self):
        self.slots = {}

    def attach(self, name, proc):
        self.slots.update({name: proc})

    def run(self):
        for name, proc in self.slots.items():
            proc.run()
        self._step()

    def _step(self):
        pass


class Child(Node):
    def _step(self):
        pass


class Parent(Node):
    def __init__(self):
        Node.__init__(self)
        self.attach("child", Child())

    def _step(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges, "sample.Parent._step", "sample.Child._step", "registered_invoke"
    )
    assert not _edge_exists(
        edges, "sample.Node.run", "sample.Node.run", "registered_invoke"
    )


def test_torch_style_module_registration_is_derived(tmp_path):
    """A different framework shape, with no shared code and no shared names."""
    (tmp_path / "sample.py").write_text(
        """
class Module:
    def __init__(self):
        self._modules = {}

    def add_module(self, name, module):
        self._modules[name] = module

    def forward(self):
        for name, module in self._modules.items():
            module.forward()


class Linear(Module):
    def forward(self):
        pass


class Net(Module):
    def __init__(self):
        Module.__init__(self)
        self.add_module("fc", Linear())

    def forward(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges, "sample.Net.forward", "sample.Linear.forward", "registered_invoke"
    )


def test_constructor_time_list_registration_is_derived(tmp_path):
    """``Pipeline(steps=[...])`` registers at construction rather than by a setter."""
    (tmp_path / "sample.py").write_text(
        """
class Scaler:
    def fit(self):
        pass


class Pipeline:
    def __init__(self, steps):
        self.steps = []
        self.steps.extend(steps)

    def fit(self):
        for step in self.steps:
            step.fit()


def build():
    return Pipeline([Scaler()])
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges, "sample.Pipeline.fit", "sample.Scaler.fit", "registered_invoke"
    )


def test_retained_but_never_invoked_state_is_not_registration(tmp_path):
    """The invoke summary is the gate, and this is what it is gating out.

    ``self.config = config`` retains its parameter exactly as a registration
    does. Nothing ever calls a method on it, so it is state, not coupling.
    """
    (tmp_path / "sample.py").write_text(
        """
class Config:
    def load(self):
        pass


class Service:
    def __init__(self, config):
        self.config = config

    def load(self):
        pass


def build():
    return Service(Config())
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert not any(edge.relation == "registered_invoke" for edge in edges)


def test_back_reference_to_a_parent_is_not_registration(tmp_path):
    """A scalar back-reference points the opposite way from the coupling wanted.

    ``self.parent = parent`` retains the parent and later calls a method on it.
    Treating it as a registration would emit ``Child.notify -> Parent.notify``,
    naming the child as the owner of its own owner.
    """
    (tmp_path / "sample.py").write_text(
        """
class Parent:
    def notify(self):
        pass


class Child:
    def __init__(self, parent):
        self.parent = parent

    def notify(self):
        self.parent.notify()


def build():
    return Child(Parent())
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert not any(edge.relation == "registered_invoke" for edge in edges)


def test_hidden_directory_check_applies_below_the_scan_root(tmp_path):
    """A tree living under a dot-directory must still be scanned.

    Judging "hidden" on the absolute path makes discovery depend on where the
    checkout happens to sit -- and silently returns nothing for every
    ``summary_packages`` entry, since installed packages live under ``.venv``.
    """
    root = tmp_path / ".venv" / "lib" / "pkg"
    root.mkdir(parents=True)
    (root / "mod.py").write_text("def run():\n    pass\n", encoding="utf-8")
    hidden = root / ".hidden"
    hidden.mkdir()
    (hidden / "skipped.py").write_text("def skipped():\n    pass\n", encoding="utf-8")

    found = {path.name for path in iter_python_files(root)}

    assert found == {"mod.py"}


def test_summary_only_package_supplies_rules_without_becoming_graph(tmp_path, monkeypatch):
    """A framework's registration method is readable without it entering the graph.

    ``add_module`` stores into ``self._modules`` in the framework's own source,
    which is the only place that fact exists. Reading it is what lets the call in
    the project become coupling -- but nothing from the framework may appear as a
    node or an edge, or the graph would describe someone else's code.
    """
    framework = tmp_path / "site" / "frameworklib"
    framework.mkdir(parents=True)
    (framework / "__init__.py").write_text(
        """
class Module:
    def __init__(self):
        self._modules = {}

    def add_module(self, name, module):
        self._modules[name] = module

    def forward(self):
        for name, module in self._modules.items():
            module.forward()
""",
        encoding="utf-8",
    )

    project = tmp_path / "proj"
    project.mkdir()
    (project / "net.py").write_text(
        """
from frameworklib import Module


class Linear(Module):
    def forward(self):
        pass


class Net(Module):
    def __init__(self):
        Module.__init__(self)
        self.add_module("fc", Linear())

    def forward(self):
        pass
""",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    nodes, edges = build_call_graph_from_analysis_files(
        list(iter_analysis_files(project)),
        summary_packages=("frameworklib",),
    )

    assert _edge_exists(
        edges, "net.Net.forward", "net.Linear.forward", "registered_invoke"
    )
    assert not any(node.startswith("frameworklib") for node in nodes)
    assert not any(
        edge.caller.startswith("frameworklib") or edge.callee.startswith("frameworklib")
        for edge in edges
    )


def test_registration_through_a_delegating_wrapper(tmp_path):
    """The store can be a call away from the method the caller uses."""
    (tmp_path / "sample.py").write_text(
        """
class Child:
    def run(self):
        pass


class Parent:
    def __init__(self):
        self.slots = {}
        self.attach("child", Child())

    def attach(self, name, proc):
        self._store(name, proc)

    def _store(self, name, proc):
        self.slots.update({name: proc})

    def run(self):
        for name, proc in self.slots.items():
            proc.run()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges, "sample.Parent.run", "sample.Child.run", "registered_invoke"
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
    summaries = build_type_summaries(
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
        param_summaries=summaries.params,
        class_attr_types=summaries.class_attrs,
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
    summaries = build_type_summaries(
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
        param_summaries=summaries.params,
        class_attr_types=summaries.class_attrs,
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


def test_deep_return_chain_resolves_beyond_the_pass_limit(tmp_path):
    """A return chain longer than max_iterations still resolves.

    Each summary pass advances a chain by one hop, so before the return-link
    back-fill this five-deep chain outran the default max_iterations=3 and the
    ``order.submit()`` edge was silently lost.
    """
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def level5():
    return Order()


def level4():
    return level5()


def level3():
    return level4()


def level2():
    return level3()


def level1():
    return level2()


def checkout():
    order = level1()
    order.submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.checkout", "shop.Order.submit", "inferred_type")


def test_deep_container_return_chain_carries_element_types(tmp_path):
    """Element types survive a chain too, not just directly returned objects."""
    (tmp_path / "shop.py").write_text(
        """
class Item:
    def price(self):
        return 0


def level4():
    return [Item()]


def level3():
    return level4()


def level2():
    return level3()


def level1():
    return level2()


def total():
    for item in level1():
        item.price()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.total", "shop.Item.price", "inferred_type")


def test_returned_call_carries_tuple_slot_types(tmp_path):
    """``return make_pair()`` forwards the callee's per-slot types.

    Only a literal tuple used to produce slot facts, so a function that returned
    another function's tuple lost the shape and destructuring it inferred nothing.
    """
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Cart:
    def clear(self):
        return None


def make_pair():
    return Order(), Cart()


def forward():
    return make_pair()


def use():
    order, cart = forward()
    order.submit()
    cart.clear()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.use", "shop.Order.submit", "inferred_type")
    assert _edge_exists(edges, "shop.use", "shop.Cart.clear", "inferred_type")


def test_mutually_recursive_returns_terminate_without_warning(tmp_path):
    """Cyclic return dependencies must settle instead of looping forever."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def ping(flag):
    if flag:
        return Order()
    return pong(flag)


def pong(flag):
    return ping(flag)


def run():
    order = pong(True)
    order.submit()
""",
        encoding="utf-8",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_awaited_call_return_types_are_inferred(tmp_path):
    """``return await build()`` carries the callee's return types.

    ``ast.Await`` used to fall through every shape test, so an async codebase
    lost the return type of any function that awaited another -- and lost the
    deferred return link with it, since inference never reached the call.
    """
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


async def make_order():
    return Order()


async def get_order():
    return await make_order()


async def run():
    order = await get_order()
    order.submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_awaited_call_carries_tuple_slot_types(tmp_path):
    """Awaiting must not lose the per-slot shape either."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Cart:
    def clear(self):
        return None


async def make_pair():
    return Order(), Cart()


async def forward():
    return await make_pair()


async def use():
    order, cart = await forward()
    order.submit()
    cart.clear()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.use", "shop.Order.submit", "inferred_type")
    assert _edge_exists(edges, "shop.use", "shop.Cart.clear", "inferred_type")


def test_conditional_return_infers_both_branches(tmp_path):
    """A ternary return is knowable on both arms, so both must be recorded."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Cart:
    def clear(self):
        return None


def make_cart():
    return Cart()


def pick(flag):
    return Order() if flag else make_cart()


def run(flag):
    value = pick(flag)
    value.submit()
    value.clear()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")
    assert _edge_exists(edges, "shop.run", "shop.Cart.clear", "inferred_type")


def test_boolop_return_infers_every_operand(tmp_path):
    """``return cached or build()`` can evaluate to either operand."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def build():
    return Order()


def get(cached):
    return cached or build()


def run(cached):
    order = get(cached)
    order.submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_indexing_a_collection_yields_its_element_type(tmp_path):
    """``orders[0]`` is one element; ``orders[1:]`` is still the collection."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def list_orders():
    return [Order()]


def first():
    orders = list_orders()
    return orders[0]


def rest():
    orders = list_orders()
    return orders[1:]


def use_first():
    order = first()
    order.submit()


def use_rest():
    for order in rest():
        order.submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.use_first", "shop.Order.submit", "inferred_type")
    assert _edge_exists(edges, "shop.use_rest", "shop.Order.submit", "inferred_type")


def test_walrus_return_is_transparent(tmp_path):
    """``return (order := build())`` returns what ``build()`` returned."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def build():
    return Order()


def get():
    return (order := build())


def run():
    order = get()
    order.submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_method_called_on_a_call_result_resolves(tmp_path):
    """``build().submit()`` resolves without binding the result to a name first.

    Receiver inference only understood names and attributes, so calling a
    method straight off a call result produced no edge at all.
    """
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def build():
    return Order()


def run():
    build().submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_chained_calls_resolve_through_each_receiver(tmp_path):
    """Each link of ``a().b().c()`` is the receiver of the next."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Repo:
    def find(self):
        return Order()


def get_repo():
    return Repo()


def run():
    get_repo().find().submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Repo.find", "inferred_type")
    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_receiver_type_is_not_mistaken_for_the_return_type(tmp_path):
    """``return build().submit()`` returns submit's value, not build's.

    Resolving the receiver runs the same inference that records return links, so
    it must not be attributed to the enclosing return.
    """
    (tmp_path / "shop.py").write_text(
        """
class Receipt:
    def show(self):
        return None


class Order:
    def submit(self):
        return Receipt()


def build():
    return Order()


def run():
    return build().submit()
""",
        encoding="utf-8",
    )

    nodes, module_map, known_classes = build_indices(tmp_path)
    summaries = build_return_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
    )

    assert summaries["shop.run"].class_types == {"shop.Receipt"}


def test_chain_through_local_variables_resolves_past_the_pass_cap(tmp_path):
    """``x = f(); return x`` links, so long chains no longer need one pass each.

    Assigning to a local before returning it used to record no return link, so
    the chain advanced a single hop per pass and died at ``max_iterations``.
    Warnings are errors here: the run must reach a fixed point, not just happen
    to get far enough.
    """
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def make():
    return Order()


def a():
    x = make()
    return x


def b():
    x = a()
    return x


def c():
    x = b()
    return x


def d():
    x = c()
    return x


def run():
    order = d()
    order.submit()
""",
        encoding="utf-8",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_variable_assigned_in_both_branches_links_to_both(tmp_path):
    """A name bound in each arm of an ``if`` depends on both expressions.

    Pinned at ``max_iterations=1`` deliberately. With the default three passes
    the pass loop re-derives these types directly, which would hide the second
    arm's provenance overwriting the first's; one pass forces the answer to
    arrive over the links alone.
    """
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Cart:
    def clear(self):
        return None


def base_order():
    return Order()


def o1():
    x = base_order()
    return x


def base_cart():
    return Cart()


def c1():
    x = base_cart()
    return x


def pick(flag):
    if flag:
        value = o1()
    else:
        value = c1()
    return value
""",
        encoding="utf-8",
    )

    nodes, module_map, known_classes = build_indices(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summaries = build_return_summaries(
            tmp_path,
            callable_map=nodes,
            module_map=module_map,
            known_classes=known_classes,
            max_iterations=1,
        )

    assert summaries["shop.pick"].class_types == {"shop.Order", "shop.Cart"}


def test_destructured_local_keeps_its_slot(tmp_path):
    """``order, cart = pair(); return order`` links to slot 0, not the whole value.

    Pinned at ``max_iterations=1`` for the same reason as the branch test above:
    only then must the tuple position survive on the link itself rather than
    being re-derived by a later pass.
    """
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Cart:
    def clear(self):
        return None


def base():
    return Order(), Cart()


def p1():
    return base()


def first():
    order, cart = p1()
    return order


def second():
    order, cart = p1()
    return cart
""",
        encoding="utf-8",
    )

    nodes, module_map, known_classes = build_indices(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summaries = build_return_summaries(
            tmp_path,
            callable_map=nodes,
            module_map=module_map,
            known_classes=known_classes,
            max_iterations=1,
        )

    assert summaries["shop.first"].class_types == {"shop.Order"}
    assert summaries["shop.second"].class_types == {"shop.Cart"}


def test_collection_returned_via_a_local_keeps_element_types(tmp_path):
    """Provenance covers element types, not just object types."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


def base():
    return [Order()]


def l1():
    x = base()
    return x


def l2():
    x = l1()
    return x


def l3():
    x = l2()
    return x


def l4():
    x = l3()
    return x


def run():
    for order in l4():
        order.submit()
""",
        encoding="utf-8",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.run", "shop.Order.submit", "inferred_type")


def test_rebinding_a_local_drops_the_old_provenance(tmp_path):
    """A rebound name no longer depends on what it used to hold."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Cart:
    def clear(self):
        return None


def make_order():
    return Order()


def make_cart():
    return Cart()


def run():
    value = make_order()
    value.submit()
    value = make_cart()
    return value
""",
        encoding="utf-8",
    )

    nodes, module_map, known_classes = build_indices(tmp_path)
    summaries = build_return_summaries(
        tmp_path,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
    )

    assert summaries["shop.run"].class_types == {"shop.Cart"}


def test_tuple_shape_survives_a_local_variable(tmp_path):
    """``pair = make_pair(); return pair`` keeps the per-slot types."""
    (tmp_path / "shop.py").write_text(
        """
class Order:
    def submit(self):
        return None


class Cart:
    def clear(self):
        return None


def make_pair():
    return Order(), Cart()


def forward():
    pair = make_pair()
    return pair


def use():
    order, cart = forward()
    order.submit()
    cart.clear()
""",
        encoding="utf-8",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "shop.use", "shop.Order.submit", "inferred_type")
    assert _edge_exists(edges, "shop.use", "shop.Cart.clear", "inferred_type")


def test_base_class_reached_through_package_reexport_resolves_inherited_methods(tmp_path):
    """A base imported from a package must not lose its inherited methods.

    ``from pkg import Base`` records the base as ``pkg.Base``, but the class is
    defined at ``pkg.base.Base``. Without alias canonicalization the MRO walk
    stops at the unrecognized name and every inherited call vanishes -- this was
    55 of 110 missing edges on climlab.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .base import Base\n", encoding="utf-8")
    (pkg / "base.py").write_text(
        """
class Base:
    def register(self, item):
        pass
""",
        encoding="utf-8",
    )
    (tmp_path / "child.py").write_text(
        """
from pkg import Base


class Child(Base):
    def __init__(self):
        self.register("thing")
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "child.Child.__init__", "pkg.base.Base.register", "self_method")


def test_reexport_chain_through_nested_packages_resolves(tmp_path):
    """Re-exports chain, so alias resolution has to reach a fixpoint."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (outer / "__init__.py").write_text("from .inner import Base\n", encoding="utf-8")
    (inner / "__init__.py").write_text("from .impl import Base\n", encoding="utf-8")
    (inner / "impl.py").write_text(
        """
class Base:
    def hook(self):
        pass
""",
        encoding="utf-8",
    )
    (tmp_path / "leaf.py").write_text(
        """
from outer import Base


class Leaf(Base):
    def go(self):
        self.hook()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "leaf.Leaf.go", "outer.inner.impl.Base.hook", "self_method")


def test_real_class_definition_wins_over_a_same_named_alias(tmp_path):
    """An alias may never shadow a class actually defined in that module."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    # pkg/__init__.py both defines Base and imports a different Base. The local
    # definition is what ``pkg.Base`` means at runtime.
    (pkg / "__init__.py").write_text(
        """
from .other import Base as _Other


class Base:
    def local_hook(self):
        pass
""",
        encoding="utf-8",
    )
    (pkg / "other.py").write_text(
        """
class Base:
    def other_hook(self):
        pass
""",
        encoding="utf-8",
    )
    (tmp_path / "leaf.py").write_text(
        """
from pkg import Base


class Leaf(Base):
    def go(self):
        self.local_hook()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "leaf.Leaf.go", "pkg.Base.local_hook", "self_method")
    assert not _edge_exists(edges, "leaf.Leaf.go", "pkg.other.Base.other_hook")


def test_self_call_resolves_to_subclass_overrides_when_base_has_no_definition(tmp_path):
    """The Template Method shape: base declares the hook only by calling it."""
    (tmp_path / "flux.py").write_text(
        """
class Base:
    def run(self):
        self.compute_flux()


class Sensible(Base):
    def compute_flux(self):
        pass


class Latent(Base):
    def compute_flux(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "flux.Base.run", "flux.Sensible.compute_flux", "virtual_override")
    assert _edge_exists(edges, "flux.Base.run", "flux.Latent.compute_flux", "virtual_override")


def test_self_call_reaches_overrides_even_when_the_base_defines_the_hook(tmp_path):
    """``self`` may be a subclass instance, so the override is a real target.

    A fallback-only rule (overrides *only* when the base defines nothing) misses
    this entirely, and on climlab that is the common case -- the base almost
    always supplies a default implementation that subclasses replace.
    """
    (tmp_path / "insol.py").write_text(
        """
class Base:
    def refresh(self):
        self.compute_fixed()

    def compute_fixed(self):
        pass


class Steady(Base):
    def compute_fixed(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "insol.Base.refresh", "insol.Base.compute_fixed", "self_method")
    assert _edge_exists(edges, "insol.Base.refresh", "insol.Steady.compute_fixed", "virtual_override")


def test_virtual_override_reaches_indirect_subclasses(tmp_path):
    """Overrides can appear at any depth below the class holding the call."""
    (tmp_path / "deep.py").write_text(
        """
class A:
    def go(self):
        self.hook()


class B(A):
    pass


class C(B):
    def hook(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "deep.A.go", "deep.C.hook", "virtual_override")


def test_virtual_override_does_not_duplicate_an_inherited_self_method_target(tmp_path):
    """A subclass that does not override must not produce a second edge."""
    (tmp_path / "plain.py").write_text(
        """
class Base:
    def run(self):
        self.helper()

    def helper(self):
        pass


class Child(Base):
    pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    matching = [
        edge for edge in edges
        if edge.caller == "plain.Base.run" and edge.callee == "plain.Base.helper"
    ]
    assert len(matching) == 1
    assert matching[0].relation == "self_method"
    assert not any(edge.relation == "virtual_override" for edge in edges)


def test_function_imported_through_a_package_reexport_resolves(tmp_path):
    """A callable's re-export path is not where it is defined.

    ``from pkg import helper`` records the target as ``pkg.helper``, but the
    function is defined at ``pkg.impl.helper``. Only the second is a known
    callable, so without an alias map the call resolves to nothing and the edge
    is dropped entirely. ``add_reexport_class_aliases`` has always done this for
    classes; this is the same fixpoint for callables.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from .impl import helper\n", encoding="utf-8"
    )
    (package / "impl.py").write_text(
        """
def helper():
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """
from pkg import helper


def run():
    helper()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "app.run", "pkg.impl.helper", "imported")


def test_reexport_aliases_chain_through_nested_packages(tmp_path):
    """Two levels of ``__init__.py`` re-export must still reach the definition."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (outer / "__init__.py").write_text(
        "from .inner import helper\n", encoding="utf-8"
    )
    (inner / "__init__.py").write_text(
        "from .impl import helper\n", encoding="utf-8"
    )
    (inner / "impl.py").write_text(
        """
def helper():
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """
from outer import helper


def run():
    helper()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "app.run", "outer.inner.impl.helper", "imported")


def test_computed_default_argument_is_attributed_to_the_class_body(tmp_path):
    """A default argument runs when the ``class`` statement runs, not on call.

    Python evaluates ``axes=make_axis()`` once, inside the class body's code
    object, so the interpreter reports the caller as ``Slab`` -- not
    ``Slab.__init__``, which may never be invoked at all. Attributing it to the
    method credits the callee with work its definition site did.
    """
    (tmp_path / "dom.py").write_text(
        """
def make_axis():
    pass


class Slab:
    def __init__(self, axes=make_axis()):
        self.axes = axes
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "dom.Slab", "dom.make_axis", "direct")
    assert not _edge_exists(edges, "dom.Slab.__init__", "dom.make_axis", "direct")


def test_a_class_body_node_is_never_a_call_target(tmp_path):
    """Constructing the class must still reach ``__init__``, not the body.

    The class body is a node so that edges can be attributed to it, but Python
    offers no way to *call* one. If it leaks into the resolution universe,
    ``Slab(...)`` resolves to the body and the constructor edge disappears.
    """
    (tmp_path / "ctor.py").write_text(
        """
def make_axis():
    pass


class Slab:
    def __init__(self, axes=make_axis()):
        self.axes = axes


def build():
    return Slab()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "ctor.build", "ctor.Slab.__init__", "constructor")
    assert not _edge_exists(edges, "ctor.build", "ctor.Slab")


def test_a_class_without_body_calls_gets_no_node(tmp_path):
    """Only class bodies that actually run something become nodes.

    Otherwise every class in the project becomes an isolated leaf node, which
    shifts every degree-based threshold downstream for no information gained.
    """
    (tmp_path / "plainc.py").write_text(
        """
class Quiet:
    x = 1

    def go(self):
        pass
""",
        encoding="utf-8",
    )

    nodes, _edges = _build_call_graph(tmp_path)

    assert "plainc.Quiet.go" in nodes
    assert "plainc.Quiet" not in nodes


def test_decorator_expression_is_attributed_to_the_enclosing_scope(tmp_path):
    """A decorator expression runs at definition time, where the ``def`` sits.

    Only the *expression* is asserted here. Applying a bare ``@register`` is a
    call the interpreter makes but the AST has no ``Call`` node for, so it
    produces no edge either before or after this change.
    """
    (tmp_path / "deco.py").write_text(
        """
def make_registrar(tag):
    def register(fn):
        return fn
    return register


@make_registrar("worker")
def worker():
    pass


class Holder:
    @make_registrar("method")
    def method(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "deco.<module>", "deco.make_registrar", "direct")
    assert not _edge_exists(edges, "deco.worker", "deco.make_registrar", "direct")
    # A method's decorator belongs to the class body, which is where it runs.
    assert _edge_exists(edges, "deco.Holder", "deco.make_registrar", "direct")
    assert not _edge_exists(
        edges, "deco.Holder.method", "deco.make_registrar", "direct"
    )


def test_virtual_override_argument_types_land_in_the_right_parameter(tmp_path):
    """An argument passed through a subclass override must skip the ``self`` slot.

    ``virtual_override`` arises from ``self.handle(...)`` exactly as
    ``self_method`` does, so its targets are bound methods whose argument 0 is
    the first *real* parameter. If the implicit receiver is not accounted for,
    ``Order`` is filed against ``self`` instead of ``item``, and the
    ``item.submit()`` below silently resolves to nothing -- a corrupted fact
    rather than a missing one, which is why this is asserted directly.
    """
    (tmp_path / "disp.py").write_text(
        """
class Order:
    def submit(self):
        pass


class Base:
    def run(self):
        order = Order()
        self.handle(order)


class Child(Base):
    def handle(self, item):
        item.submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "disp.Base.run", "disp.Child.handle", "virtual_override")
    assert _edge_exists(edges, "disp.Child.handle", "disp.Order.submit", "inferred_type")


def test_registry_element_types_cross_method_boundaries(tmp_path):
    """Types stored into a container attribute must survive to another method.

    ``register`` puts a child into ``self.children``; ``go`` iterates it. Without
    an element dimension on class attributes the fact dies at the method
    boundary and ``child.run()`` resolves to nothing.
    """
    (tmp_path / "reg.py").write_text(
        """
class Child:
    def run(self):
        pass


class Parent:
    def __init__(self):
        self.children = {}

    def register(self, name, child):
        self.children[name] = child

    def go(self):
        for child in self.children.values():
            child.run()


def build():
    parent = Parent()
    parent.register("a", Child())
    return parent
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "reg.Parent.go", "reg.Child.run", "inferred_type")


def test_registry_populated_by_dict_update_and_walked_with_items(tmp_path):
    """The climlab shape: ``dict.update`` to store, ``.items()`` to traverse."""
    (tmp_path / "tree.py").write_text(
        """
class Leaf:
    def compute(self):
        pass


class Node:
    def __init__(self):
        self.parts = {}

    def attach(self, name, part):
        self.parts.update({name: part})

    def run(self):
        for name, part in self.parts.items():
            part.compute()


def assemble():
    node = Node()
    node.attach("leaf", Leaf())
    return node
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "tree.Node.run", "tree.Leaf.compute", "inferred_type")


def test_container_element_types_flow_through_a_parameter(tmp_path):
    """A list of objects handed across a call keeps its element types."""
    (tmp_path / "batch.py").write_text(
        """
class Order:
    def submit(self):
        pass


def process(items):
    for item in items:
        item.submit()


def run():
    process([Order()])
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "batch.process", "batch.Order.submit", "inferred_type")


def test_attribute_types_are_found_on_base_classes(tmp_path):
    """A registry filled by a base method must be readable from a subclass.

    ``Base.attach`` records into ``self.parts``, so the fact is keyed on
    ``Base``. ``Derived.run`` reads ``self.parts`` and would find nothing if the
    lookup did not walk up to the base that owns the attribute -- which is
    exactly how climlab splits ``Process.add_subprocess`` from the
    ``TimeDependentProcess`` code that traverses the result.
    """
    (tmp_path / "split.py").write_text(
        """
class Part:
    def compute(self):
        pass


class Base:
    def __init__(self):
        self.parts = {}

    def attach(self, name, part):
        self.parts.update({name: part})


class Derived(Base):
    def run(self):
        for part in self.parts.values():
            part.compute()


def build():
    obj = Derived()
    obj.attach("p", Part())
    return obj
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "split.Derived.run", "split.Part.compute", "inferred_type")


def test_nested_container_attributes_flatten_for_dispatch(tmp_path):
    """``self.buckets[key].append(x)`` then iterating ``self.buckets[key]``.

    This two-hop shape is how climlab bins subprocesses into
    ``self.process_types[time_type]`` before walking them.
    """
    (tmp_path / "bins.py").write_text(
        """
class Job:
    def run(self):
        pass


class Scheduler:
    def __init__(self):
        self.queues = {"fast": [], "slow": []}
        self.jobs = {}

    def submit(self, name, job):
        self.jobs.update({name: job})

    def rebuild(self):
        for name, job in self.jobs.items():
            self.queues["fast"].append(job)

    def drain(self):
        for job in self.queues["fast"]:
            job.run()


def build():
    scheduler = Scheduler()
    scheduler.submit("a", Job())
    return scheduler
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "bins.Scheduler.drain", "bins.Job.run", "inferred_type")


def test_parameter_annotations_resolve_method_calls(tmp_path):
    """A bare annotation is enough; no call site has to be observed."""
    (tmp_path / "solver.py").write_text(
        """
class Mesh:
    def refine(self):
        pass


def solve(mesh: Mesh):
    mesh.refine()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "solver.solve", "solver.Mesh.refine", "inferred_type")


def test_optional_and_union_annotations_unwrap_to_the_inner_class(tmp_path):
    (tmp_path / "opt.py").write_text(
        """
from typing import Optional


class Grid:
    def build(self):
        pass


def a(grid: Optional[Grid]):
    grid.build()


def b(grid: Grid | None):
    grid.build()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "opt.a", "opt.Grid.build", "inferred_type")
    assert _edge_exists(edges, "opt.b", "opt.Grid.build", "inferred_type")


def test_container_annotations_give_element_types_not_object_types(tmp_path):
    """``List[Order]`` means it holds orders -- iterating resolves, calling does not."""
    (tmp_path / "coll.py").write_text(
        """
from typing import Dict, List


class Order:
    def submit(self):
        pass


def each(orders: List[Order]):
    for order in orders:
        order.submit()


def mapped(orders: Dict[str, Order]):
    for order in orders.values():
        order.submit()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "coll.each", "coll.Order.submit", "inferred_type")
    assert _edge_exists(edges, "coll.mapped", "coll.Order.submit", "inferred_type")
    # The parameter itself is a list, so it must not be treated as an Order.
    assert not _edge_exists(edges, "coll.each", "coll.Order.submit", "self_method")


def test_annotated_attribute_declaration_without_a_value_is_used(tmp_path):
    (tmp_path / "decl.py").write_text(
        """
class Engine:
    def start(self):
        pass


class Car:
    engine: Engine

    def go(self):
        self.engine.start()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "decl.Car.go", "decl.Engine.start", "inferred_type")


def _build_health(tmp_path, *, include_external=False):
    """Run the real pipeline and return what it could not resolve."""
    health = CallGraphHealth()
    analysis_files = list(iter_analysis_files(tmp_path))
    build_call_graph_from_analysis_files(
        analysis_files,
        include_external=include_external,
        package_prefix=None,
        health=health,
    )
    return health


def test_unresolved_calls_are_counted_even_though_they_are_dropped(tmp_path):
    """The drop must be tallied before the filter that discards it.

    With ``include_external`` off, an unresolved edge never reaches the edge
    list, so any count taken from the artifact reports zero unresolved calls no
    matter how many there were. That is a tautology printed as a perfect score.
    """
    (tmp_path / "ext.py").write_text(
        """
import numpy


def run():
    numpy.array([1, 2, 3])
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)
    health = _build_health(tmp_path)

    # The artifact cannot see the loss...
    assert all(edge.resolved for edge in edges)
    # ...but the tally taken at the point of loss can.
    assert health.dropped_unresolved["imported"] >= 1


def test_calling_a_parameter_is_recorded_as_an_inexpressible_callee(tmp_path):
    """Calling a value is a different failure from failing to look up a name.

    ``func`` resolves perfectly well as a name; the abstract domain is a set of
    *class* ids and simply has no way to say that a variable holds a function.
    Counting it beside unresolved ``numpy`` calls would bury the one gap that is
    a limitation of the lattice rather than of the configuration.
    """
    (tmp_path / "crypto.py").write_text(
        """
import cryptops


class Crypto:
    def __init__(self, key):
        self.key = key

    def apply(self, msg, func):
        return func(self.key, msg)


crp = Crypto("secretkey")
encrypted = crp.apply("hello world", cryptops.encrypt)
""",
        encoding="utf-8",
    )

    health = _build_health(tmp_path)

    assert health.unresolvable_calls["name"] == 1
    assert sum(health.unresolvable_calls.values()) == 1


def test_fanout_is_counted_per_site_not_per_line(tmp_path):
    """Two independent calls on one line are two sites, not one site of two.

    Nearly a third of real call expressions share a line with another call, so a
    fan-out histogram derived from ``edges.csv`` measures co-location rather than
    resolution ambiguity. Counting at the emission point is what keeps the number
    meaning what it says.
    """
    (tmp_path / "fan.py").write_text(
        """
def a():
    pass


def b():
    pass


def run():
    return (a(), b())
""",
        encoding="utf-8",
    )

    health = _build_health(tmp_path)

    assert health.site_fanout[1] == 2
    assert health.site_fanout[2] == 0


def test_a_virtual_dispatch_site_reports_its_real_fanout(tmp_path):
    """One site with several possible targets must be counted once, at k."""
    (tmp_path / "poly.py").write_text(
        """
class Base:
    def run(self):
        self.hook()

    def hook(self):
        pass


class First(Base):
    def hook(self):
        pass


class Second(Base):
    def hook(self):
        pass
""",
        encoding="utf-8",
    )

    health = _build_health(tmp_path)

    # Base.hook plus two overrides, all from the single self.hook() site.
    assert health.site_fanout[3] == 1


def test_super_resolves_by_c3_not_by_unioning_bases(tmp_path):
    """Python picks one target; unioning the bases invents the others.

    ``Mixed(Left, Right)`` linearizes to ``Mixed, Left, Right, Base``, so
    ``super().run()`` inside ``Mixed`` reaches ``Left.run`` and nothing else.
    Walking the lexical bases reports both ``Left.run`` and ``Right.run``, and
    one of those edges describes a call that never happens.
    """
    (tmp_path / "mro.py").write_text(
        """
class Base:
    def run(self):
        pass


class Left(Base):
    def run(self):
        super().run()


class Right(Base):
    def run(self):
        super().run()


class Mixed(Left, Right):
    def run(self):
        super().run()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    supers = [
        edge
        for edge in edges
        if edge.caller == "mro.Mixed.run" and edge.relation == "super_method"
    ]
    assert len(supers) == 1
    assert supers[0].callee == "mro.Left.run"


def test_cooperative_super_reaches_a_sibling_through_a_subclass_mro(tmp_path):
    """``self`` inside ``Left`` need not be a ``Left``.

    When ``Mixed(Left, Right)`` exists, C3 threads ``super()`` inside ``Left``
    *sideways* into ``Right`` rather than up into ``Base``. No walk of ``Left``'s
    own bases can see that, and it is a real dispatch the interpreter makes.
    """
    (tmp_path / "coop.py").write_text(
        """
class Base:
    def run(self):
        pass


class Left(Base):
    def run(self):
        super().run()


class Right(Base):
    def run(self):
        pass


class Mixed(Left, Right):
    pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "coop.Left.run", "coop.Right.run", "cooperative_super")
    # Labelled separately from super_method: which subclass is instantiated is
    # genuinely unknown, so this is over-approximation and must be weightable.
    assert _edge_exists(edges, "coop.Left.run", "coop.Base.run", "super_method")


def test_explicit_super_argument_selects_the_starting_class(tmp_path):
    """``super(Other, self)`` starts the search after ``Other``, not the enclosing class."""
    (tmp_path / "expl.py").write_text(
        """
class A:
    def run(self):
        pass


class B(A):
    def run(self):
        pass


class C(B):
    def run(self):
        super(B, self).run()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "expl.C.run", "expl.A.run", "super_method")
    assert not _edge_exists(edges, "expl.C.run", "expl.B.run", "super_method")


def test_a_cyclic_hierarchy_does_not_abort_the_run(tmp_path):
    """Linearization must never raise into a pass.

    Alias canonicalization can collapse two classes onto one ID and produce a
    hierarchy Python would reject. An exception escaping here would lose a whole
    run's edges over one malformed base list, so it falls back instead.
    """
    (tmp_path / "cyc.py").write_text(
        """
class A(B):
    def run(self):
        super().run()


class B(A):
    def run(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert any(edge.caller == "cyc.A.run" for edge in edges)


def test_a_certain_call_and_a_guess_are_not_labelled_the_same(tmp_path):
    """Two edges can share a relation while differing completely in certainty.

    ``a.func()`` has one possible target and is right. ``c.func()`` on a class
    with several candidate definitions is one guess among them. Before
    confidence, both arrived downstream as ``inferred_type`` at weight 1.0, and a
    guess pulled two unrelated classes together exactly as hard as a fact.
    """
    (tmp_path / "conf.py").write_text(
        """
class Handler:
    def run(self):
        pass


class Base:
    def go(self):
        self.hook()

    def hook(self):
        pass


class One(Base):
    def hook(self):
        pass


class Two(Base):
    def hook(self):
        pass


class Three(Base):
    def hook(self):
        pass


def certain():
    handler = Handler()
    handler.run()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    certain = [e for e in edges if e.caller == "conf.certain" and e.callee == "conf.Handler.run"]
    assert len(certain) == 1
    assert certain[0].confidence == "high"

    # One site, four possible targets: Base.hook plus three overrides.
    guesses = [e for e in edges if e.caller == "conf.Base.go"]
    assert len(guesses) == 4
    assert {e.confidence for e in guesses} == {"medium"}


def test_confidence_grades_by_fanout_not_by_relation(tmp_path):
    """A two-target site stays ``high``: fan-out is not monotone with wrongness.

    Measured on climlab, sites with two targets confirmed as often as or better
    than sites with one. Demoting everything above a single target would punish
    a bucket the evidence says is reliable.
    """
    (tmp_path / "two.py").write_text(
        """
class Base:
    def go(self):
        self.hook()

    def hook(self):
        pass


class Only(Base):
    def hook(self):
        pass
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    site = [e for e in edges if e.caller == "two.Base.go"]
    assert len(site) == 2
    assert {e.confidence for e in site} == {"high"}


def test_a_function_passed_as_an_argument_resolves_inside_the_callee(tmp_path):
    """The review's picture 1: a value that *is* code.

    ``Crypto.apply`` calls whatever it was handed. The abstract domain was a set
    of class ids, so there was no way to write down "``func`` holds
    ``cryptops.encrypt``" -- the fact was not lost through a bug, it was
    inexpressible, and the edge simply never existed.
    """
    (tmp_path / "cryptops.py").write_text(
        """
def encrypt(key, msg):
    pass


def decrypt(key, msg):
    pass
""",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        """
import cryptops


class Crypto:
    def __init__(self, key):
        self.key = key

    def apply(self, msg, func):
        return func(self.key, msg)


crp = Crypto("secretkey")
encrypted = crp.apply("hello world", cryptops.encrypt)
decrypted = crp.apply(encrypted, cryptops.decrypt)
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges, "app.Crypto.apply", "cryptops.encrypt", "inferred_callable"
    )
    assert _edge_exists(
        edges, "app.Crypto.apply", "cryptops.decrypt", "inferred_callable"
    )


def test_a_callable_stored_on_self_resolves_when_invoked(tmp_path):
    """``self.handler = fn`` then ``self.handler()`` -- callback registration."""
    (tmp_path / "cb.py").write_text(
        """
def on_event():
    pass


class Emitter:
    def __init__(self):
        self.handler = on_event

    def fire(self):
        self.handler()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "cb.Emitter.fire", "cb.on_event", "inferred_callable")


def test_a_dispatch_table_resolves_its_entries(tmp_path):
    """``SOLVERS[name]()`` -- the config-driven dispatch idiom."""
    (tmp_path / "disp.py").write_text(
        """
def conjugate_gradient():
    pass


def gauss_seidel():
    pass


SOLVERS = {"cg": conjugate_gradient, "gs": gauss_seidel}


def solve(name):
    solver = SOLVERS[name]
    return solver()
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges, "disp.solve", "disp.conjugate_gradient", "inferred_callable"
    )
    assert _edge_exists(edges, "disp.solve", "disp.gauss_seidel", "inferred_callable")


def test_a_functor_resolves_through_dunder_call(tmp_path):
    """``model(x)`` where ``model`` is an instance, not a function."""
    (tmp_path / "functor.py").write_text(
        """
class Model:
    def __call__(self, x):
        pass


def run():
    model = Model()
    return model(1)
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "functor.run", "functor.Model.__call__", "dunder_call")


def test_functools_partial_resolves_to_the_wrapped_function(tmp_path):
    """``partial(f, x)`` evaluates to ``f`` with arguments pre-bound."""
    (tmp_path / "part.py").write_text(
        """
from functools import partial


def worker(a, b):
    pass


def run():
    bound = partial(worker, 1)
    return bound(2)
""",
        encoding="utf-8",
    )

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(edges, "part.run", "part.worker", "inferred_callable")


def test_a_lambda_body_is_attributed_to_the_lambda(tmp_path):
    """A call inside a lambda belongs to the lambda, as the interpreter reports it.

    The ID is CPython's own ``co_qualname`` spelling so the runtime tracer and
    the static pass keep agreeing by construction.
    """
    (tmp_path / "lam.py").write_text(
        """
def helper():
    pass


def run():
    fn = lambda: helper()
    return fn()
""",
        encoding="utf-8",
    )

    nodes, edges = _build_call_graph(tmp_path)

    assert "lam.run.<locals>.<lambda>" in nodes
    assert _edge_exists(edges, "lam.run.<locals>.<lambda>", "lam.helper", "direct")
    assert _edge_exists(
        edges, "lam.run", "lam.run.<locals>.<lambda>", "inferred_callable"
    )


# ---------------------------------------------------------------------------
# Structural guards on the CallCollector mixin split.
#
# CallCollector's body lives in one module per concern (see
# call_graph/modularisation_plan.md). The mixins are slices of one object, so
# nothing about that is enforced by the type system -- these tests are what
# keeps the arrangement honest. The failure they exist to catch is silent: a
# visitor lost or shadowed during a move shows up only as a quiet drop in edge
# count, and no behavioural test would name it.
# ---------------------------------------------------------------------------

import ast as _ast

from microservice_pipeline.call_graph.collector.collector import CallCollector
from microservice_pipeline.call_graph.collector.state import CollectorState

_MIXINS = [base for base in CallCollector.__bases__ if base is not CollectorState]

# The visitors CallCollector had before the split, taken from the pre-split
# collectors.py. Adding a visitor is a real change and should update this list;
# losing one silently is the bug.
_EXPECTED_VISITORS = {
    "visit_AnnAssign", "visit_Assign", "visit_AsyncFor", "visit_AsyncFunctionDef",
    "visit_Attribute", "visit_AugAssign", "visit_BinOp", "visit_Call",
    "visit_ClassDef", "visit_Compare", "visit_For", "visit_FunctionDef",
    "visit_If", "visit_Import", "visit_ImportFrom", "visit_Lambda",
    "visit_Module", "visit_Subscript", "visit_UnaryOp",
}


def _own_names(cls):
    return {name for name in vars(cls) if not name.startswith("__")}


def test_collector_mixins_inherit_only_collector_state():
    """Rule 1: a mixin never inherits another mixin, so the MRO stays a single diamond."""
    assert _MIXINS, "CallCollector should be composed from mixins"
    for mixin in _MIXINS:
        assert mixin.__bases__ == (CollectorState,), (
            f"{mixin.__name__} must inherit CollectorState and nothing else, "
            f"got {mixin.__bases__}"
        )


def test_collector_state_is_last_in_the_mro():
    """Rule 2: CollectorState last, so C3 can linearise at all."""
    assert CallCollector.__mro__[-3:] == (CollectorState, _ast.NodeVisitor, object)


def test_no_collector_method_is_defined_twice():
    """Rule 3: nothing is overridden, which is what makes the mixin order inert."""
    owners = {}
    collisions = {}
    for cls in (*_MIXINS, CollectorState):
        for name in _own_names(cls):
            if name in owners:
                collisions.setdefault(name, [owners[name]]).append(cls.__name__)
            else:
                owners[name] = cls.__name__
    assert not collisions, (
        "these names are defined in more than one mixin, so the order in "
        f"collector.py silently decides which wins: {collisions}"
    )


def test_every_visitor_survived_the_split():
    """A visitor lost during a move is invisible except as missing edges."""
    found = {name for name in dir(CallCollector) if name.startswith("visit_")}
    assert found == _EXPECTED_VISITORS


def test_visit_call_children_is_still_reachable():
    """Subclass API with no in-package caller -- deleting it fails at runtime, not import."""
    assert hasattr(CallCollector, "_visit_call_children")


def test_records_registry_facts_is_a_class_attribute():
    """An instance attribute here would silently shadow TypeSummaryCollector's override."""
    assert "records_registry_facts" in vars(CollectorState)
    assert CollectorState.records_registry_facts is False
