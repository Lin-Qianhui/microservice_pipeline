import warnings

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


def test_add_subprocess_synthesizes_compute_edge(tmp_path):
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

    _nodes, edges = _build_call_graph(tmp_path)

    assert _edge_exists(
        edges,
        "sample.Parent._compute",
        "sample.Child._compute",
        "subprocess_compute",
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
