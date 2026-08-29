"""Tests for the data-access oracle: the runtime tracer and its comparison.

This instrument's failure mode is silence. If the callable IDs it builds drift
from the ones the static artifacts use, or if a decoding rule is wrong, nothing
raises -- the recall number simply comes out low, and a low number looks exactly
like a low number whether the extractor is at fault or the instrument is. So the
properties asserted here are the ones that fail silently, not aggregate scores.
"""

import sys
import textwrap

import pytest

from microservice_pipeline.call_graph.discovery import iter_analysis_files
from microservice_pipeline.data_access.access_comparison import (
    FALSIFIED,
    MATCHED,
    NOT_EXERCISED,
    UNDERIVABLE,
    UNEXECUTED,
    build_bytecode_index,
    compare,
    derive_static_claim,
    load_static_rows,
)
from microservice_pipeline.data_access.dynamic_access_trace import (
    ACCESS_READ,
    ACCESS_WRITE,
    COMPUTED_KEY,
    ROLE_ATTRIBUTE,
    ROLE_COMPUTED,
    ROLE_LITERAL,
    ROLE_LOCAL,
    ROLE_METHOD,
    ROLE_PARAM,
    ROLE_SUPER,
    TIER_ATTR,
    TIER_KEY,
    TIER_NAME,
    decode_access_instructions,
    iter_code_objects,
    module_map_from_analysis_files,
    trace_driver,
)
from microservice_pipeline.data_access.generate_data_access_ast import main as data_access_main


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="sys.monitoring requires Python 3.12+"
)


# A package small enough to reason about completely, containing one instance of
# every situation the comparison has to get right: a literal key, a computed key,
# a folded-tuple key, an augmented attribute assignment, a method call whose
# receiver is the real access, a branch that never runs, and a lambda.
PACKAGE_SOURCE = '''
CONFIG = {'mode': 'fast'}


class Model:
    def __init__(self, state):
        self.state = state
        self.count = 0

    def step(self, rows):
        temp = self.state['Ts']
        self.count += 1
        self.state.update({'seen': temp})
        totals = {}
        for row in rows:
            totals[row] = temp
        return totals

    def never_runs(self):
        return self.state['unused']


def drive():
    model = Model({'Ts': 1, 'unused': 2})
    return model.step(['a']), CONFIG['mode']
'''


def _decode(source, qualname):
    """Decode one named code object out of a source string."""
    for code in iter_code_objects(compile(textwrap.dedent(source), "sample.py", "exec")):
        if code.co_qualname == qualname:
            return decode_access_instructions(code)
    raise AssertionError(f"no code object named {qualname!r}")


def _triples(decoded):
    return {(item.tier, item.name, item.access, item.role) for item in decoded}


# -- the decoder ---------------------------------------------------------


def test_literal_and_computed_keys_are_distinguished():
    """The one distinction section 6 insists on before any number means anything.

    The static side names a key by its literal and cannot name a computed one, so
    folding them together would let a computed-key miss masquerade as a
    literal-key miss.
    """
    decoded = _triples(_decode("def f(d, k):\n    return d['lit'], d[k]\n", "f"))

    assert (TIER_KEY, "lit", ACCESS_READ, ROLE_LITERAL) in decoded
    assert (TIER_KEY, COMPUTED_KEY, ACCESS_READ, ROLE_COMPUTED) in decoded


def test_folded_constant_tuple_slice_yields_the_string_key():
    """``df.loc[:, 'col']`` compiles the whole slice to one constant tuple.

    Easy to miss, and pandas column access is exactly what this extractor exists
    to find, so a decoder that only understood bare string constants would drop
    the case that matters most.
    """
    decoded = _triples(_decode("def f(df):\n    return df.loc[:, 'col']\n", "f"))

    assert (TIER_KEY, "col", ACCESS_READ, ROLE_LITERAL) in decoded


def test_augmented_attribute_assignment_is_both_a_read_and_a_write():
    decoded = _triples(_decode("def f(self):\n    self.n += 1\n", "f"))

    assert (TIER_ATTR, "n", ACCESS_READ, ROLE_ATTRIBUTE) in decoded
    assert (TIER_ATTR, "n", ACCESS_WRITE, ROLE_ATTRIBUTE) in decoded


def test_method_loads_are_tagged_apart_from_data_attributes():
    """``x.f()`` and ``x.f`` are the same opcode and must not be scored alike.

    The static side records a read of the *receiver* for a method call, never of
    the method name. Without this distinction every method call in the project
    would be reported as an attribute access the extractor missed.
    """
    decoded = _triples(_decode("def f(x):\n    x.method()\n    return x.attribute\n", "f"))

    assert (TIER_ATTR, "method", ACCESS_READ, ROLE_METHOD) in decoded
    assert (TIER_ATTR, "attribute", ACCESS_READ, ROLE_ATTRIBUTE) in decoded


def test_parameters_and_ordinary_locals_are_tagged_apart():
    """Both are ``LOAD_FAST``; only the parameter is comparable.

    Whether a local counts as an access is ``LocalBinding.exposed``, a static
    judgment the interpreter cannot see. Tagging is what lets the comparison
    exclude locals from the recall denominator without excluding them from the
    falsification test.
    """
    decoded = _triples(_decode("def f(given):\n    made = given\n    return made\n", "f"))

    assert (TIER_NAME, "given", ACCESS_READ, ROLE_PARAM) in decoded
    assert (TIER_NAME, "made", ACCESS_WRITE, ROLE_LOCAL) in decoded


def test_a_lambda_is_its_own_callable():
    """Review 5.5: the call graph indexes lambdas, data access does not.

    A lambda is a separate code object at runtime with the same
    ``<enclosing>.<locals>.<lambda>`` qualname the call graph uses, so its
    accesses land on the lambda and show up as missing there -- which is the
    instrument measuring 5.5 and 1.9 rather than tripping over them.
    """
    source = "def f(rows):\n    data = {'a': 1}\n    return sorted(rows, key=lambda item: item['a'])\n"

    lambda_decoded = _triples(_decode(source, "f.<locals>.<lambda>"))
    enclosing_decoded = _triples(_decode(source, "f"))

    assert (TIER_KEY, "a", ACCESS_READ, ROLE_LITERAL) in lambda_decoded
    assert (TIER_NAME, "item", ACCESS_READ, ROLE_PARAM) in lambda_decoded
    # The enclosing function never subscripts anything itself.
    assert not any(tier == TIER_KEY for tier, _name, _access, _role in enclosing_decoded)


def test_module_level_globals_use_load_name_not_load_global():
    """A module body addresses its namespace by name, never by global slot.

    Found by hand-checking the first climlab run's falsified list: 89 claims were
    reported wrong, and the first 7 were module-level globals in
    ``stommelbox`` and ``rrtmg_lw`` that the decoder simply could not see.
    ``LOAD_NAME``/``STORE_NAME`` is 2,284 instructions on climlab.
    """
    decoded = _triples(_decode("TABLE = {}\nVALUE = TABLE['k']\n", "<module>"))

    assert (TIER_NAME, "TABLE", ACCESS_WRITE, "global") in decoded
    assert (TIER_NAME, "TABLE", ACCESS_READ, "global") in decoded


def test_tuple_unpacking_records_every_target():
    """``STORE_FAST_STORE_FAST`` stores two locals in one instruction.

    The rest of the falsified list from the first climlab run: 34 unpack targets
    in ``RRTMG_LW._compute_heating_rates`` accused of not existing, because
    destructuring compiles almost entirely to the paired opcode.
    """
    decoded = _triples(_decode("def f(pair):\n    first, second, third = pair\n    return first\n", "f"))

    for name in ("first", "second", "third"):
        assert (TIER_NAME, name, ACCESS_WRITE, ROLE_LOCAL) in decoded


def test_super_attribute_access_gets_its_own_role():
    """``super().x`` is ``LOAD_SUPER_ATTR`` and is never a data access.

    The receiver is a dispatch proxy, and ``_resolve_attribute`` returns nothing
    for a call receiver — zero static rows on climlab mention ``super(``. Tagging
    it apart matters more than it looks: the method bit is set only for the
    *zero-argument* form, so ``super(C, self).__init__(**kwargs)`` decodes as a
    plain attribute read, which added 62 fake misses before this rule existed.
    """
    source = (
        "class C:\n"
        "    def m(self, **kw):\n"
        "        super().run()\n"
        "        super(C, self).__init__(**kw)\n"
        "        return super().value\n"
    )
    decoded = _triples(_decode(source, "C.m"))

    assert (TIER_ATTR, "value", ACCESS_READ, ROLE_SUPER) in decoded
    assert (TIER_ATTR, "run", ACCESS_READ, ROLE_SUPER) in decoded
    assert (TIER_ATTR, "__init__", ACCESS_READ, ROLE_SUPER) in decoded
    assert not any(role == ROLE_ATTRIBUTE for _t, _n, _a, role in decoded)


def test_comprehensions_stay_with_the_enclosing_callable():
    """PEP 709 inlines them from 3.12, so their accesses genuinely belong here.

    Asserted so that a future CPython that stops inlining is caught by a test
    rather than by an unexplained shift in the baseline.
    """
    decoded = _triples(_decode("def f(rows):\n    return [row['k'] for row in rows]\n", "f"))

    assert (TIER_KEY, "k", ACCESS_READ, ROLE_LITERAL) in decoded


# -- deriving the static claim -------------------------------------------


def test_evidence_beats_access_path_when_they_disagree():
    """``access_path`` belongs to the object; ``evidence`` belongs to the edge.

    A ``class_state`` object rolls every attribute up to the class, so its
    ``access_path`` is whichever path first registered it. Trusting it would
    compare the wrong attribute and report the extractor as wrong.
    """
    claim, reason = derive_static_claim(
        {
            "callable": "m.C.f",
            "object_id": "class_state:m.C",
            "object_kind": "class_state",
            "operation": "attribute_load",
            "access": "read",
            "field": "",
            "access_path": "self.time_type",
            "evidence": "self.state",
            "lineno": "12",
        }
    )

    assert reason == ""
    assert (claim.tier, claim.name) == (TIER_ATTR, "state")


def test_a_method_operation_is_a_claim_about_the_receiver():
    claim, _reason = derive_static_claim(
        {
            "callable": "m.C.f",
            "object_id": "class_state:m.C",
            "object_kind": "class_state",
            "operation": "method:items:receiver",
            "access": "read",
            "field": "",
            "evidence": "self.state.items()",
            "lineno": "3",
        }
    )

    assert (claim.tier, claim.name) == (TIER_ATTR, "state")


def test_exposure_rows_are_excluded_rather_than_scored():
    """``return x`` is not a memory operation and no instruction matches it.

    Section 6: relations the instrument cannot express are excluded *before*
    scoring rather than counted as gaps.
    """
    _claim, reason = derive_static_claim(
        {
            "callable": "m.f",
            "object_id": "local_exposed:m.f:rows",
            "object_kind": "local_exposed",
            "operation": "return",
            "access": "read",
            "field": "rows",
            "evidence": "return rows",
            "lineno": "4",
        }
    )

    assert reason == "exposure"


def test_a_row_yielding_no_name_is_counted_not_dropped():
    """A number that cannot fail is not a measurement.

    A row this module cannot parse is a defect in *this* module, so it has to
    surface as its own count rather than shrinking the denominator invisibly.
    """
    _claim, reason = derive_static_claim(
        {
            "callable": "m.f",
            "object_id": "unknown:x",
            "object_kind": "unknown",
            "operation": "load",
            "access": "read",
            "field": "",
            "evidence": "",
            "lineno": "1",
        }
    )

    assert reason == UNDERIVABLE


# -- the comparison, end to end ------------------------------------------


@pytest.fixture
def traced_project(tmp_path):
    """Extract, trace and compare one small package; return the report."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text(PACKAGE_SOURCE, encoding="utf-8")
    outdir = tmp_path / "out"

    data_access_main(["--root", str(tmp_path), "--outdir", str(outdir), "--no-pyright"])

    analysis_files = list(iter_analysis_files(tmp_path))
    module_by_file = module_map_from_analysis_files(analysis_files)

    sys.path.insert(0, str(tmp_path))
    try:
        def drive():
            import importlib

            module = importlib.import_module("pkg.mod")
            module.drive()
            return None

        result = trace_driver("fixture", drive, module_by_file)
    finally:
        sys.path.remove(str(tmp_path))
        for name in [key for key in sys.modules if key == "pkg" or key.startswith("pkg.")]:
            del sys.modules[name]

    index = build_bytecode_index(analysis_files)
    report = compare(
        load_static_rows(outdir / "callable_data_access.csv"),
        result.observed,
        result.executed_callables,
        index,
    )
    return report, result, index


def test_callable_ids_agree_with_the_static_artifacts(traced_project):
    """If the two ID conventions drift, every number below is zero and nothing says so.

    This is the assertion the whole comparison rests on, which is why it is not
    left to be inferred from a healthy-looking recall.
    """
    report, result, _index = traced_project

    assert "pkg.mod.Model.step" in result.executed_callables
    # A callable named by the artifacts but absent from the compiled bytecode
    # means the conventions disagree, not that the extractor is wrong.
    assert report.verdicts["no_bytecode"] == 0


def test_an_unrun_method_is_not_exercised_rather_than_falsified(traced_project):
    """The instrument must not cry wolf.

    ``never_runs`` accesses ``self.state['unused']`` and is never called. Its
    claims are real; the drivers simply did not reach them. Calling that
    falsified would make every uncovered branch look like a defect.
    """
    report, _result, _index = traced_project

    assert report.verdicts[FALSIFIED] == 0
    assert report.verdicts[NOT_EXERCISED] > 0


def test_an_invented_claim_is_falsified_without_needing_coverage(traced_project):
    """The strong signal, and the thing the call-graph comparison cannot do.

    ``spacing`` appears nowhere in ``step``'s bytecode, so the claim is wrong
    however little the drivers covered.
    """
    report, result, index = traced_project

    invented = {
        "callable": "pkg.mod.Model.step",
        "object_id": "class_state:pkg.mod.Model",
        "object_kind": "class_state",
        "operation": "attribute_load",
        "access": "read",
        "field": "",
        "evidence": "self.spacing",
        "lineno": "9",
    }
    rescored = compare([invented], result.observed, result.executed_callables, index)

    assert rescored.verdicts[FALSIFIED] == 1


def test_literal_and_computed_keys_are_reported_separately(traced_project):
    """``totals[row]`` is computed and ``self.state['Ts']`` is literal.

    They must not share a denominator: the extractor can name one and not the
    other, so merging them measures coverage of a distinction rather than of the
    extractor.
    """
    report, _result, _index = traced_project

    assert report.literal_key_recall[1] > 0
    assert report.computed_key_recall[1] > 0
    assert report.literal_key_recall != report.computed_key_recall


def test_self_is_not_scored_as_a_missed_parameter(traced_project):
    """Every method emits ``LOAD_FAST self``; the extractor models none of them.

    ``generate_data_access_ast`` strips ``self``/``cls`` from ``callable_params``
    on purpose. Scoring them would add one fake miss per method body and make
    recall a measure of how many methods the project has.
    """
    report, _result, _index = traced_project

    assert not any(name in {"self", "cls"} for _c, _t, name, _a in report.observed_missing)


def test_exclusions_are_counted_in_the_report(traced_project):
    """An exclusion nobody can see is a denominator chosen quietly."""
    report, _result, _index = traced_project

    assert report.scored_row_count < report.static_row_count
    assert sum(report.excluded.values()) == report.static_row_count - report.scored_row_count


def test_the_report_confirms_the_accesses_that_really_happened(traced_project):
    """A sanity floor: the fixture's real accesses are found, not merely counted."""
    report, _result, _index = traced_project

    assert report.verdicts[MATCHED] > 0
    assert report.recall > 0.5
    # ``self.state['Ts']`` is read by ``step`` and the extractor claims it.
    assert ("pkg.mod.Model.step", TIER_KEY, "Ts", ACCESS_READ) in report.observed_matched
