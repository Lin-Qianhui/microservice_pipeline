"""Tests for the object-identity oracle -- Step 1b.

The properties asserted here are the ones that fail *silently*. An identity
instrument that is wrong does not raise: it reports a plausible precision figure
computed from aliases it invented, or contradictions it manufactured. So these
tests are about the three ways this instrument could lie to itself:

* an ``id()`` recycled onto a new object, which merges two unrelated things;
* an interned value type, which merges every callable that mentions ``0``;
* a site the tracer could not read, reported as a contradiction rather than as
  nothing at all.

Aggregate scores are deliberately not asserted -- those live in the recorded
baseline, which is allowed to move when the extractor improves.
"""

import sys
import textwrap

import pytest

from microservice_pipeline.call_graph.discovery import iter_analysis_files
from microservice_pipeline.data_access.dynamic_access_trace import (
    TIER_ATTR,
    TIER_NAME,
    ObjectTokens,
    decode_access_instructions,
    iter_code_objects,
    module_map_from_analysis_files,
    trace_driver,
)
from microservice_pipeline.data_access.identity_comparison import (
    CONFIRMED,
    CONTRADICTED,
    UNOBSERVED,
    build_site_index,
    compare,
)


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="sys.monitoring requires Python 3.12+"
)


# -- ObjectTokens: the part that keeps id() honest ------------------------


def test_a_recycled_address_does_not_become_the_same_object():
    """The failure this whole class exists to prevent.

    CPython reuses addresses aggressively. If a freed list's address is handed to
    a new list and both get the same token, the comparison reports an alias
    between two callables that never shared anything -- the instrument inventing
    exactly the defect it was built to find.
    """
    unpinned = [1, 2, 3]
    unpinned_address = id(unpinned)
    del unpinned
    # Control: with nothing holding it, the address really does come back. If
    # this ever stops being true the test below proves nothing, so it is asserted
    # rather than assumed.
    recycled = any(id([1, 2, 3]) == unpinned_address for _ in range(5000))

    tokens = ObjectTokens()
    pinned = [1, 2, 3]
    pinned_address = id(pinned)
    assert tokens.token(pinned) == 0
    del pinned

    # Same allocation pattern, but this one was tokenised, so ObjectTokens is
    # holding it and the address must stay out of circulation.
    for _ in range(5000):
        assert id([1, 2, 3]) != pinned_address, (
            "a tokenised object's address was handed to a new object -- "
            "retention is not working and identities can merge"
        )

    assert recycled, "addresses were not being recycled at all; the control case did not hold"


def test_value_types_never_get_a_token():
    """Interning makes their identity an artifact of the interpreter.

    ``a is b`` for two equal small integers or two identical string literals says
    nothing about data flowing between callables, and counting it would alias
    together every function in the project that mentions ``0``.
    """
    tokens = ObjectTokens()

    for value in (0, 1.5, True, "path", b"raw", (1, 2), None, frozenset({1})):
        assert tokens.token(value) is None

    assert tokens.excluded_value_type == 8
    assert tokens.retained == 0


def test_the_retention_cap_is_reported_rather_than_applied_quietly():
    """A truncated measurement must announce itself.

    A cap that silently drops observations turns recall into a number that cannot
    fail, which is the failure ``code_review.md`` section 6 names by that phrase.
    """
    tokens = ObjectTokens(cap=2)

    assert tokens.token([1]) == 0
    assert tokens.token([2]) == 1
    assert tokens.token([3]) is None
    assert tokens.cap_hit is True
    assert tokens.excluded_cap == 1


# -- the decoder's receiver, which is how attributes become reachable -----


def _decode(source, qualname):
    for code in iter_code_objects(compile(textwrap.dedent(source), "sample.py", "exec")):
        if code.co_qualname == qualname:
            return decode_access_instructions(code)
    raise AssertionError(f"no code object named {qualname!r}")


def test_a_named_receiver_is_recorded_and_a_stacked_one_is_not():
    """``self.data`` is reachable from the frame; ``self.data.rows`` is not.

    The identity channel finds an attribute by looking its receiver up in the
    frame, so a receiver that only ever exists on the operand stack has to come
    back empty. Recording it anyway would attribute a value to the wrong object.
    """
    decoded = _decode("def f(self):\n    return self.data.rows\n", "f")
    by_name = {item.name: item for item in decoded if item.tier == TIER_ATTR}

    assert by_name["data"].receiver == "self"
    assert by_name["rows"].receiver == ""


def test_a_store_target_has_no_identity_site():
    """``INSTRUCTION`` fires before the instruction runs, so a store has no value yet.

    Scoring a write site would read whatever the name held *before* the
    assignment -- the previous loop iteration's object, or nothing -- and call it
    the assigned value. It is picked up at the next load instead.
    """
    decoded = _decode("def f(given):\n    made = given\n    return made\n", "f")
    sites = {item.name: item.identity_site("m.f") for item in decoded}

    assert sites["given"] == ("m.f", TIER_NAME, "given")
    # ``made`` appears as both a write and a read; the read is what survives.
    reads = [item for item in decoded if item.name == "made" and item.access == "read"]
    writes = [item for item in decoded if item.name == "made" and item.access == "write"]
    assert all(item.identity_site("m.f") is None for item in writes)
    assert all(item.identity_site("m.f") is not None for item in reads)


# -- mapping static objects onto sites ------------------------------------


def _object(object_id, kind, **extra):
    row = {
        "id": object_id,
        "kind": kind,
        "owner": "",
        "container": "",
        "field": "",
        "access_path": "",
        "alias_of": "",
    }
    row.update(extra)
    return row


CALLABLES = [
    {"id": "m.f", "module": "m", "class_name": None},
    {"id": "m.g", "module": "m", "class_name": None},
    {"id": "m.C.method", "module": "m", "class_name": "C"},
]


def test_deep_object_state_paths_are_excluded_not_approximated():
    """``a.b.c`` cannot be read from a frame, and its prefix is a different object.

    Falling back to ``a.b`` would score the container against a claim about the
    field inside it, so the honest answer is to exclude it and say so.
    """
    index = build_site_index(
        [
            _object("object_state:param:m.f:model:model_R", "object_state",
                    owner="param:m.f:model", access_path="model.R"),
            _object("object_state:param:m.f:model:model_R_units", "object_state",
                    owner="param:m.f:model", access_path="model.R.units"),
        ],
        CALLABLES,
    )

    assert index.for_object("object_state:param:m.f:model:model_R") == {
        ("m.f", TIER_ATTR, "model.R")
    }
    assert index.for_object("object_state:param:m.f:model:model_R_units") == set()
    assert index.excluded["object_state path not one attribute deep"] == 1


def test_class_attribute_state_maps_to_every_method_of_its_class():
    """One attribute is one object seen from every method that touches it.

    Mapping it to a single method would make a genuine alias between two methods
    look like a contradiction.
    """
    index = build_site_index([_object("class_attr_state:m.C:rows", "class_attr_state")], CALLABLES)

    assert index.for_object("class_attr_state:m.C:rows") == {
        ("m.C.method", TIER_ATTR, "self.rows"),
        ("m.C.method", TIER_ATTR, "cls.rows"),
    }


def test_unmappable_kinds_are_counted_rather_than_dropped():
    index = build_site_index(
        [
            _object("df_col:m.f:df:col", "df_col"),
            _object("class_state:m.C", "class_state"),
            _object("file:path", "file"),
        ],
        CALLABLES,
    )

    assert index.sites == {}
    assert sum(index.excluded.values()) == 3


# -- the three verdicts ---------------------------------------------------


def _compare(objects, identities, *, lineage=(), access_edges=(), values=(), sites_seen=()):
    return compare(
        objects,
        CALLABLES,
        list(lineage),
        list(access_edges),
        set(identities),
        set(values),
        set(sites_seen),
    )


def test_a_real_alias_is_confirmed():
    report = _compare(
        [
            _object("param:m.f:given", "param"),
            _object("param:m.g:taken", "param", alias_of="param:m.f:given"),
        ],
        {("m.f", TIER_NAME, "given", 7), ("m.g", TIER_NAME, "taken", 7)},
    )

    assert report.verdicts[CONFIRMED] == 1
    assert report.verdicts[CONTRADICTED] == 0


def test_two_sites_that_never_coincide_are_contradicted():
    report = _compare(
        [
            _object("param:m.f:given", "param"),
            _object("param:m.g:taken", "param", alias_of="param:m.f:given"),
        ],
        {("m.f", TIER_NAME, "given", 7), ("m.g", TIER_NAME, "taken", 8)},
    )

    assert report.verdicts[CONTRADICTED] == 1
    assert report.contradicted[0].claim.src == "param:m.g:taken"


def test_an_unobserved_site_is_not_a_contradiction():
    """The difference between "wrong" and "never ran" is the whole caveat.

    Unlike the access oracle, this instrument cannot refute a claim from the
    bytecode. A site the drivers never reached must land in ``unobserved``, or
    every untested code path in the project becomes a reported defect.
    """
    report = _compare(
        [
            _object("param:m.f:given", "param"),
            _object("param:m.g:taken", "param", alias_of="param:m.f:given"),
        ],
        {("m.f", TIER_NAME, "given", 7)},
    )

    assert report.verdicts[UNOBSERVED] == 1
    assert report.verdicts[CONTRADICTED] == 0


def test_an_alias_reached_through_two_lineage_hops_still_counts_as_found():
    """Recall asks whether the graph connects two objects, not whether one edge does.

    An alias the extractor recorded as ``a -> b -> c`` is an alias it found. A
    per-edge test would report the two-hop case as a miss and understate recall.
    """
    report = _compare(
        [
            _object("param:m.f:given", "param"),
            _object("local_exposed:m.g:middle", "local_exposed"),
            _object("param:m.g:taken", "param"),
        ],
        {
            ("m.f", TIER_NAME, "given", 7),
            ("m.g", TIER_NAME, "middle", 7),
            ("m.g", TIER_NAME, "taken", 7),
        },
        lineage=[
            {"src_object_id": "param:m.f:given", "dst_object_id": "local_exposed:m.g:middle",
             "relation": "arg_to_param", "file": "m.py", "lineno": 1},
            {"src_object_id": "local_exposed:m.g:middle", "dst_object_id": "param:m.g:taken",
             "relation": "local_assign", "file": "m.py", "lineno": 2},
        ],
    )

    assert report.runtime_classes == 1
    assert report.runtime_classes_connected == 1
    assert report.runtime_classes_split == 0


def test_an_aliasing_the_static_graph_missed_is_counted_as_a_split():
    report = _compare(
        [_object("param:m.f:given", "param"), _object("param:m.g:taken", "param")],
        {("m.f", TIER_NAME, "given", 7), ("m.g", TIER_NAME, "taken", 7)},
    )

    assert report.runtime_classes_split == 1
    assert report.recall == 0.0


def test_return_slot_endpoints_are_excluded_rather_than_scored():
    """``return:m.f`` is a lineage pseudo-node, not a data object.

    It is never registered in the object table, so it has no site and no value.
    Scoring it would produce a contradiction for every function that returns.
    """
    report = _compare(
        [_object("param:m.f:given", "param")],
        {("m.f", TIER_NAME, "given", 7)},
        lineage=[
            {"src_object_id": "param:m.f:given", "dst_object_id": "return:m.f",
             "relation": "return_value", "file": "m.py", "lineno": 1}
        ],
    )

    assert report.claims_excluded["endpoint is not a registered object"] == 1
    assert report.verdicts[CONTRADICTED] == 0


# -- section 1.7 ----------------------------------------------------------


def test_a_shared_file_node_whose_callables_saw_different_paths_is_reported():
    """Section 1.7, measured.

    ``load_users(path)`` and ``load_invoices(path)`` both produce ``file:path``,
    and the structural graph reads that as coupling. The runtime answer is a
    path *value*, not an object identity -- two equal paths are one file whether
    or not they are one string.
    """
    report = _compare(
        [_object("file:path", "file", access_path="path", display_name="path")],
        set(),
        access_edges=[
            {"callable": "m.f", "object_id": "file:path"},
            {"callable": "m.g", "object_id": "file:path"},
        ],
        values={("m.f", "path", "users.csv"), ("m.g", "path", "invoices.csv")},
    )

    assert report.file_nodes_multi_callable == 1
    assert report.file_nodes_disagreeing == 1
    assert report.file_disagreements[0][0] == "file:path"


def test_a_genuinely_shared_literal_file_is_not_reported():
    report = _compare(
        [_object("file:data.csv", "file", access_path="path", display_name="path")],
        set(),
        access_edges=[
            {"callable": "m.f", "object_id": "file:data.csv"},
            {"callable": "m.g", "object_id": "file:data.csv"},
        ],
        values={("m.f", "path", "data.csv"), ("m.g", "path", "data.csv")},
    )

    assert report.file_nodes_multi_callable == 1
    assert report.file_nodes_disagreeing == 0


# -- end to end, against a real traced run --------------------------------


PACKAGE_SOURCE = '''
class Holder:
    def __init__(self, rows):
        self.rows = rows

    @property
    def doubled(self):
        raise AssertionError("the tracer must never run a property")


def make():
    data = [1, 2, 3]
    return Holder(data), data


def consume(frame):
    return frame


def drive():
    holder, data = make()
    consume(data)
    return holder.rows
'''


def _trace_package(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sample.py").write_text(textwrap.dedent(PACKAGE_SOURCE), encoding="utf-8")

    analysis_files = list(iter_analysis_files(package, module_prefix="pkg"))
    module_by_file = module_map_from_analysis_files(analysis_files)

    def invoke():
        import importlib

        sys.path.insert(0, str(tmp_path))
        try:
            module = importlib.import_module("pkg.sample")
            module.drive()
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("pkg.sample", None)
            sys.modules.pop("pkg", None)
        return None

    return trace_driver("test", invoke, module_by_file, identity=True)


def test_one_object_threaded_through_four_names_becomes_one_identity_class(tmp_path):
    """The end-to-end property: the same list really is one token everywhere.

    ``data`` in ``make``, ``data`` in ``drive``, ``frame`` in ``consume``,
    ``rows`` in ``__init__`` and ``self.rows`` in the instance are five sites
    holding one list. If the tracer, the token table and the site keys agree,
    they share a token; if any of the three is wrong, they do not.
    """
    result = _trace_package(tmp_path)

    by_site = {}
    for callable_id, tier, path, token in result.identities:
        by_site.setdefault((callable_id, tier, path), set()).add(token)

    shared = by_site[("pkg.sample.make", TIER_NAME, "data")]
    assert shared, "the traced list produced no token"
    assert by_site[("pkg.sample.drive", TIER_NAME, "data")] & shared
    assert by_site[("pkg.sample.consume", TIER_NAME, "frame")] & shared
    assert by_site[("pkg.sample.Holder.__init__", TIER_NAME, "rows")] & shared
    assert by_site[("pkg.sample.drive", TIER_ATTR, "holder.rows")] & shared


LOOP_SOURCE = '''
def rebuild(n):
    made = []
    for _ in range(n):
        item = [0]
        made.append(item)
    return made


def drive():
    return rebuild(6)
'''


def _trace_source(tmp_path, source, name, **options):
    package = tmp_path / "pkg"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")

    analysis_files = list(iter_analysis_files(package, module_prefix="pkg"))
    module_by_file = module_map_from_analysis_files(analysis_files)

    def invoke():
        import importlib

        sys.path.insert(0, str(tmp_path))
        try:
            importlib.import_module(f"pkg.{name}").drive()
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop(f"pkg.{name}", None)
            sys.modules.pop("pkg", None)
        return None

    return trace_driver("test", invoke, module_by_file, identity=True, **options)


def test_one_sample_per_site_sees_only_the_first_object_a_loop_binds(tmp_path):
    """The failure that made the first climlab run accuse 96 innocent claims.

    An access site has one fixed attribute name, so seeing it once is enough. An
    identity site does not: a loop binds a different object every pass, and a
    budget of one turns "the drivers moved on" into "these are different
    objects" -- a contradiction the extractor did not earn. Sweeping the budget
    on climlab moved contradictions 152 -> 51 for exactly this reason, which is
    why the default is calibrated rather than picked.
    """
    stingy = _trace_source(tmp_path, LOOP_SOURCE, "loop", identity_samples=1)
    generous = _trace_source(tmp_path, LOOP_SOURCE, "loop", identity_samples=64)

    def tokens(result):
        return {key[3] for key in result.identities if key[2] == "item"}

    assert len(tokens(stingy)) == 1
    assert len(tokens(generous)) > 1, "raising the budget must reveal the later bindings"
    assert stingy.identity_offsets_at_cap > 0, "the truncation must be reported, not silent"


def test_a_property_is_never_executed_by_the_instrument(tmp_path):
    """Reading through ``getattr`` would run user code and change the program.

    The fixture's ``doubled`` property raises. A tracer that used ``getattr``
    instead of the instance ``__dict__`` would trip it, so this test passing at
    all is the assertion.
    """
    result = _trace_package(tmp_path)

    assert not result.driver_errors
    assert result.identities
