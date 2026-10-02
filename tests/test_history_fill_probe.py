"""纯合成探针接口测试；fake不作为真实API结果汇报。"""

import json

import pytest

from growrag.experiments.history_fill_probe import cases, compile_probe, probe_request
from growrag.experiments.history_runtime_v2 import _fill_request
from growrag.history_library import FrozenHistoryLibrary, HistoryCard, HistoryRecord
from growrag.macro_operators import GapField, OperatorSpec, QueryStep


def library(case):
    fields = tuple(
        GapField(name, "integer" if type(value) is int else "text")
        for name, value in case["expected_gap"].items()
    )
    template = " ".join("{" + f.name + "}" for f in fields)
    spec = OperatorSpec("SyntheticSpec", "1", ("lookup",), fields, (QueryStep("q", template),))
    card = HistoryCard(
        case["card_id"], "1", "Synthetic", "Synthetic only", "Keep parameters", operator_spec=spec
    )
    record = HistoryRecord(card, "reference", "synthetic/manual", "0" * 64, status="published")
    return FrozenHistoryLibrary("synthetic-probe", (), (record,))


def response(case):
    return {
        "intent": "lookup",
        "constraints": [],
        "gap_entries": [{"name": k, "value": v} for k, v in case["expected_gap"].items()],
        "bindings": [],
    }


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["case_id"])
def test_probe_original_request_and_v2_scope(case):
    lib = library(case)
    question, evidence, context, messages = probe_request(case, lib)
    wire, names, allowed_ids = _fill_request(messages)
    assert question.dataset == "synthetic_fill_probe"
    assert names == [] and allowed_ids == ["probe_current_passage"]
    assert "allowed_binding_names" in wire[1]["content"]
    plan = compile_probe(case, lib, context, evidence, json.dumps(response(case)))
    assert len(plan["requests"]) == 1
    assert context.offered_ids == (case["card_id"],)


def test_no_string_to_integer_cleanup():
    case = cases()[0]
    lib = library(case)
    _, evidence, context, _ = probe_request(case, lib)
    value = response(case)
    value["gap_entries"][1]["value"] = "2012"
    with pytest.raises(TypeError, match="integer"):
        compile_probe(case, lib, context, evidence, json.dumps(value))


def test_extra_gap_not_deleted():
    case = cases()[1]
    lib = library(case)
    _, evidence, context, _ = probe_request(case, lib)
    value = response(case)
    value["gap_entries"].append({"name": "unallowed", "value": "wrong"})
    with pytest.raises(ValueError):
        compile_probe(case, lib, context, evidence, json.dumps(value))


def test_wrong_entity_not_treated_as_success():
    case = cases()[1]
    lib = library(case)
    _, evidence, context, _ = probe_request(case, lib)
    value = response(case)
    value["gap_entries"][0]["value"] = "Wrong Entity"
    with pytest.raises(ValueError, match="preregistered"):
        compile_probe(case, lib, context, evidence, json.dumps(value))
