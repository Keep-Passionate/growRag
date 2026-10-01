"""合成选择合同/旧模板执行/来源迁移测试；不是API或HotpotQA效果测量。"""

import hashlib
import json
from dataclasses import replace

import pytest

from growrag.experiments.history_context import (
    decode_rule_query,
    plan_selected_template,
    prepare_history_context,
    prepare_rule_rewrite,
    resolve_history_selection,
)
from growrag.experiments.prepare_history_foundation import prepare_foundation
from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion
from growrag.history_library import (
    REPRESENTATIONS,
    FrozenHistoryLibrary,
    HistoryCard,
    HistoryCondition,
    HistoryExample,
    HistoryRecord,
)
from growrag.macro_operators import (
    GapField,
    GoalContract,
    GroundedBinding,
    OperatorSpec,
    QueryStep,
    RuntimeState,
)
from growrag.operator_bank import FrozenOperatorBank, OperatorRecord


@pytest.fixture
def question():
    return RuntimeQuestion("synthetic-query", "Where was the founder of Cedar Lab born?")


@pytest.fixture
def template():
    return HistoryCard(
        "HOP",
        "1",
        "Bridge lookup",
        "Use a current-evidence founder, not a historical person.",
        "Search the cited founder's birthplace.",
        (HistoryExample("Where was the founder of Pine Lab born?", "<founder> birthplace"),),
        (HistoryCondition("founder_known", "Founder found in current evidence"),),
        OperatorSpec(
            "HOP_SPEC",
            "1",
            ("lookup",),
            (),
            (QueryStep("next", "{founder} birthplace", requires_bindings=("founder",)),),
        ),
    )


@pytest.fixture
def library(template):
    rule = HistoryCard(
        "SEMANTIC",
        "1",
        "Semantic clarification",
        "Clarify current wording.",
        "Use precise synonymous wording without importing facts.",
        (HistoryExample("Pine Lab boss", "Pine Lab director"),),
    )
    return FrozenHistoryLibrary(
        "synthetic-only",
        (),
        tuple(
            HistoryRecord(card, "reference", "synthetic/reference", "0" * 64, status="published")
            for card in (template, rule)
        ),
    )


def context(library, question, **kwargs):
    return prepare_history_context(library, question, offered_ids=("HOP", "SEMANTIC"), **kwargs)


def choice(identity="HOP", reason="uncertain_match"):
    return json.dumps({"selected_card_id": identity, "reason": reason})


def test_same_candidates_same_template_body_in_all_views(library, question):
    evidence = (Evidence("e1", "Cedar Lab", 0, "Avery Finch founded Cedar Lab."),)
    states = [
        context(library, question, representation=v, evidence=evidence) for v in REPRESENTATIONS
    ]
    for ctx in states:
        assert ctx.offered_ids == ("HOP", "SEMANTIC")
        assert resolve_history_selection(choice(), ctx, library) == library.published_cards[0]
        plan = plan_selected_template(
            library,
            ctx,
            "HOP",
            goal=GoalContract(question.text, "lookup"),
            gap={},
            state=RuntimeState(evidence, (GroundedBinding("founder", "Avery Finch", ("e1",)),), 1),
        )
        assert plan.requests[0].query == "Avery Finch birthplace"
        assert plan.retrieval_cost == 1
    assert len({ctx.fingerprint for ctx in states}) == 3


def test_rule_execution_is_identical_after_different_selector_views(library, question):
    requests = [
        prepare_rule_rewrite(library, context(library, question, representation=v), "SEMANTIC")
        for v in REPRESENTATIONS
    ]
    assert requests[0] == requests[1] == requests[2]
    assert "examples" in json.loads(requests[0][1]["content"])["selected_rule"]


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_no_provenance_answer_score_or_raw_source_trajectory_is_served(
    library, question, representation
):
    ctx = context(library, question, representation=representation)
    encoded = json.dumps(ctx.payload)
    for forbidden in (
        "source_ref",
        "source_qids",
        "source_sha256",
        "gold_answer",
        "final_answer",
        "source_trace",
        "validation_ref",
        "synthetic/reference",
    ):
        assert forbidden not in encoded


def test_unknown_conditions_are_visible_not_automatically_satisfied(library, question):
    ctx = context(library, question)
    card = ctx.payload["candidate_cards"][0]
    assert card["applicability_status"] == "unknown"
    assert card["condition_observations"] == [
        {"condition_id": "founder_known", "status": "unknown"}
    ]
    assert resolve_history_selection(choice(), ctx, library) is not None


@pytest.mark.parametrize(
    "flag,status", [(True, "supported"), (False, "contradicted"), (None, "unknown")]
)
def test_external_condition_observations_are_not_card_reliability(library, question, flag, status):
    original = library.fingerprint
    ctx = context(library, question, observed_conditions={"HOP": {"founder_known": flag}})
    assert ctx.payload["candidate_cards"][0]["condition_observations"][0]["status"] == status
    assert library.fingerprint == original


@pytest.mark.parametrize(
    "output",
    [
        choice("NOT_OFFERED"),
        choice("HOP", "evidence_sufficient"),
        choice(None),
        '{"selected_card_id":"HOP","reason":"condition_match","operator_spec":{}}',
        '{"selected_card_id":"HOP","selected_card_id":null,"reason":"condition_match"}',
        "Use HOP please",
    ],
)
def test_bad_or_modified_model_choices_fail_without_repair(library, question, output):
    with pytest.raises((ValueError, TypeError)):
        resolve_history_selection(output, context(library, question), library)


def test_empty_pool_remains_empty_and_stop_is_valid(library, question):
    ctx = prepare_history_context(library, question, offered_ids=())
    assert ctx.payload["candidate_cards"] == []
    assert resolve_history_selection(choice(None, "no_suitable_card"), ctx, library) is None
    with pytest.raises(ValueError):
        resolve_history_selection(choice(), ctx, library)


def test_zero_budget_prevents_select_and_rule_rewrite(library, question):
    ctx = context(library, question, remaining_retrievals=0)
    with pytest.raises(ValueError, match="zero retrieval"):
        resolve_history_selection(choice(), ctx, library)
    with pytest.raises(ValueError, match="budget"):
        prepare_rule_rewrite(library, ctx, "SEMANTIC")


def test_revised_metadata_invalidates_prepared_context(library, question):
    ctx = context(library, question)
    first = library.records[0]
    changed = replace(
        library,
        records=(replace(first, card=replace(first.card, description="new")), library.records[1]),
    )
    with pytest.raises(ValueError, match="snapshot changed"):
        resolve_history_selection(choice(), ctx, changed)


def test_gold_record_is_not_a_runtime_question(library):
    with pytest.raises(TypeError, match="gold-free"):
        context(library, GoldRecord("q", ("not a question",)))


def test_context_payload_returns_a_defensive_copy(library, question):
    ctx = context(library, question)
    value = ctx.payload
    value["candidate_cards"][0]["name"] = "mutated"
    assert ctx.payload["candidate_cards"][0]["name"] != "mutated"


def test_template_execution_cannot_use_rule_body(library, question):
    with pytest.raises(ValueError, match="not a compiled template"):
        plan_selected_template(
            library,
            context(library, question),
            "SEMANTIC",
            goal=GoalContract(question.text, "lookup"),
            gap={},
            state=RuntimeState(),
        )


def test_template_execution_preserves_goal_and_budget(library, question):
    ctx = context(library, question)
    with pytest.raises(ValueError, match="original question"):
        plan_selected_template(
            library,
            ctx,
            "HOP",
            goal=GoalContract("another question", "lookup"),
            gap={},
            state=RuntimeState(),
        )
    with pytest.raises(ValueError, match="increase"):
        plan_selected_template(
            library,
            ctx,
            "HOP",
            goal=GoalContract(question.text, "lookup"),
            gap={},
            state=RuntimeState(remaining_retrievals=2),
        )


def test_binding_cannot_refer_to_new_or_changed_evidence(library, question):
    ctx = context(
        library, question, evidence=(Evidence("e1", "Cedar Lab", 0, "Avery is founder."),)
    )
    for identity, text in (("new", "Taylor is founder."), ("e1", "Taylor is founder.")):
        state = RuntimeState(
            (Evidence(identity, "Cedar Lab", 0, text),),
            (GroundedBinding("founder", "Taylor", (identity,)),),
            1,
        )
        with pytest.raises(ValueError, match="visible context|changed"):
            plan_selected_template(
                library, ctx, "HOP", goal=GoalContract(question.text, "lookup"), gap={}, state=state
            )


def test_binding_cannot_use_an_entity_only_in_hidden_evidence_tail(library, question):
    evidence = (Evidence("e1", "Cedar Lab", 0, "padding " * 10000 + "Avery Finch founded it."),)
    ctx = context(library, question, evidence=evidence)
    assert "Avery Finch" not in ctx.payload["evidence"][0]["text"]
    state = RuntimeState(evidence, (GroundedBinding("founder", "Avery Finch", ("e1",)),), 1)
    with pytest.raises(ValueError, match="visible window"):
        plan_selected_template(
            library, ctx, "HOP", goal=GoalContract(question.text, "lookup"), gap={}, state=state
        )


@pytest.mark.parametrize(
    "output",
    [
        '{"query":""}',
        '{"query":3}',
        '{"query":"valid","answer":"x"}',
        '{"query":"a","query":"b"}',
        '{"query":NaN}',
    ],
)
def test_bad_rewrite_json_is_not_silently_repaired(question, output):
    with pytest.raises((ValueError, TypeError)):
        decode_rule_query(output, question)


def test_rule_decode_has_explicit_original_plus_rewrite_policy(question):
    assert decode_rule_query('{"query":"Cedar Lab founder birthplace"}', question) == (
        question.text + " Cedar Lab founder birthplace"
    )
    assert decode_rule_query(json.dumps({"query": question.text}), question) == question.text
    assert decode_rule_query(
        '{"query":"Cedar Lab founder birthplace"}', question, hybrid=False
    ) == ("Cedar Lab founder birthplace")


def test_prompt_size_is_checked_without_dropping_or_truncating_cards(library, question):
    template = library.records[0]
    cards = tuple(
        replace(
            template,
            card=replace(template.card, card_id=f"CARD_{i}", transformation_rule="x" * 8000),
        )
        for i in range(5)
    )
    big = replace(library, records=cards)
    with pytest.raises(ValueError, match="byte budget"):
        prepare_history_context(big, question, offered_ids=tuple(r.card.card_id for r in cards))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    from growrag.experiments import prepare_history_foundation as module

    snapshot = tmp_path / "author"
    path = snapshot / "reformer/patterns/extracted_patterns.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps(
        [
            {
                "pattern_name": "Synthetic",
                "description": "Test data only.",
                "transformation_rule": "Preserve current entities.",
                "examples": [["x", "x clarified"]],
            }
        ]
    ).encode()
    path.write_bytes(raw)
    monkeypatch.setitem(
        module.PINNED_HASHES,
        "reformer/patterns/extracted_patterns.json",
        hashlib.sha256(raw).hexdigest(),
    )
    spec = OperatorSpec(
        "TEST",
        "1",
        ("lookup",),
        (GapField("term"),),
        (QueryStep("search", "{original_question} {term}"),),
    )
    bank = FrozenOperatorBank(
        "synthetic-source",
        ("source-only",),
        (
            OperatorRecord(
                spec,
                ("source-only",),
                "synthetic-source",
                "audit/x",
                "validated",
                "audit/validation",
            ),
        ),
    )
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(bank.to_json(), encoding="utf-8")
    manifest = {
        "official_splits": {"source": "train"},
        "roles": {
            "source": ["source-only"],
            "calibration": ["cal-only"],
            "evaluation": ["eval-only"],
        },
        "nested_source_ids": {"1": ["source-only"]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = {
        "schema_version": "growrag-operator-bank-bundle-v1",
        "phase": "source",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol": bank.protocol_id,
        "banks": {
            "1": {
                "bank_fingerprint": bank.fingerprint,
                "sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
            }
        },
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    return {
        "reformer_snapshot": snapshot,
        "bank_path": bank_path,
        "manifest_path": manifest_path,
        "bundle_path": bundle_path,
        "output": tmp_path / "new-output",
        "expected_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    }


def test_foundation_build_is_offline_and_inputs_are_preserved(inputs):
    before = {key: path.read_bytes() for key, path in inputs.items() if key.endswith("_path")}
    report = prepare_foundation(**inputs)
    assert report["api_calls"] == 0
    assert report["libraries"]["combined"]["published"] == 2
    assert report["libraries"]["combined"]["explicit_condition_cards"] == 0
    assert not report["runtime_connected_to_live_api"]
    assert before == {
        key: path.read_bytes() for key, path in inputs.items() if key.endswith("_path")
    }
    assert len(list(inputs["output"].iterdir())) == 4


def test_foundation_refuses_existing_output_without_overwrite(inputs):
    inputs["output"].mkdir()
    sentinel = inputs["output"] / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_foundation(**inputs)
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("fault", ["not_train", "scope", "manifest_hash", "bank_fingerprint"])
def test_foundation_rejects_wrong_source_role_or_audit_bindings(inputs, fault):
    manifest = json.loads(inputs["manifest_path"].read_text(encoding="utf-8"))
    bundle = json.loads(inputs["bundle_path"].read_text(encoding="utf-8"))
    if fault == "not_train":
        manifest["official_splits"]["source"] = "dev"
    if fault == "scope":
        manifest["roles"]["evaluation"].append("source-only")
    if fault in {"not_train", "scope"}:
        inputs["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
        bundle["manifest_sha256"] = hashlib.sha256(inputs["manifest_path"].read_bytes()).hexdigest()
    if fault == "manifest_hash":
        bundle["manifest_sha256"] = "0" * 64
    if fault == "bank_fingerprint":
        bundle["banks"]["1"]["bank_fingerprint"] = "0" * 64
    inputs["bundle_path"].write_text(json.dumps(bundle), encoding="utf-8")
    # 明确允许的外部指纹也不能绕过source/train/protected语义检查。
    if fault in {"not_train", "scope"}:
        inputs["expected_manifest_sha256"] = hashlib.sha256(
            inputs["manifest_path"].read_bytes()
        ).hexdigest()
        inputs["expected_bundle_sha256"] = hashlib.sha256(
            inputs["bundle_path"].read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError):
        prepare_foundation(**inputs)
    assert not inputs["output"].exists()


def test_self_consistent_modified_manifest_and_bundle_still_fail_external_anchor(inputs):
    manifest = json.loads(inputs["manifest_path"].read_text(encoding="utf-8"))
    manifest["roles"]["source"].append("an-extra-source")
    inputs["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    bundle = json.loads(inputs["bundle_path"].read_text(encoding="utf-8"))
    bundle["manifest_sha256"] = hashlib.sha256(inputs["manifest_path"].read_bytes()).hexdigest()
    inputs["bundle_path"].write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="externally pinned"):
        prepare_foundation(**inputs)
    assert not inputs["output"].exists()


def test_bank_byte_sha_is_checked_not_just_canonical_fingerprint(inputs):
    # 规范内容完全相同，字节变了；必须显式新版本声明，不能冒用旧审计。
    raw = inputs["bank_path"].read_text(encoding="utf-8")
    inputs["bank_path"].write_text(raw + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint/protocol"):
        prepare_foundation(**inputs)
    assert not inputs["output"].exists()
