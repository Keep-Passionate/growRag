"""Prompt provenance and optional ablation contracts: no live API or quality claim."""

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import ChatResponse
from growrag.experiments.fresh_baselines import (
    ALL_BASELINE_SPECS,
    BASELINE_SPECS,
    EXTRA_BASELINE_SPECS,
    RRR_PROMPT,
    baseline_generator,
)
from growrag.experiments.llm_adapters import APIReader
from growrag.experiments.prompt_registry import (
    PROMPT_REGISTRY,
    export_registry,
    main,
    prompt_identity,
    registry_document,
)
from growrag.experiments.protocol import (
    BackendCallError,
    Evidence,
    GoldRecord,
    RuntimeQuestion,
)

Q = RuntimeQuestion("fixture", "Which city is older, Eastlake or Westlake?")
E = Evidence("fixture-evidence", "Eastlake", 0, "Eastlake was founded earlier.")


class Client:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://test.invalid/v1", model="TEST")

    def __init__(self, content='{"query":"Eastlake Westlake city founding date"}'):
        self.content, self.requests = content, []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return ChatResponse(
            self.content,
            "TEST",
            "TEST-returned",
            "fixture-response",
            "fixture-request",
            10,
            20,
            0,
            Path("test-only-not-written.json"),
            "mock",
        )


def test_historical_baselines_remain_exactly_three_and_optional_variants_are_explicit():
    assert tuple(BASELINE_SPECS) == ("SIMPLE_PARAPHRASE", "RRR_KEYWORDS", "QUERY2DOC")
    assert tuple(EXTRA_BASELINE_SPECS) == ("RRR_MINIMAL", "RRR_QUERY_ANCHOR")
    assert set(ALL_BASELINE_SPECS) == set(BASELINE_SPECS) | set(EXTRA_BASELINE_SPECS)


@pytest.mark.parametrize("registry", [PROMPT_REGISTRY, ALL_BASELINE_SPECS, EXTRA_BASELINE_SPECS])
def test_all_registries_are_read_only(registry):
    with pytest.raises(TypeError):
        registry["REPLACEMENT"] = "invalid"


def test_prompt_records_and_export_copies_cannot_mutate_registry():
    record = PROMPT_REGISTRY["RRR_KEYWORDS"]
    with pytest.raises(FrozenInstanceError):
        record.system_prompt = "replacement"
    document = registry_document()
    document["prompts"][1]["system_prompt"] = "replacement"
    assert PROMPT_REGISTRY["RRR_KEYWORDS"].system_prompt == RRR_PROMPT


@pytest.mark.parametrize("variant", ALL_BASELINE_SPECS)
def test_export_system_prompt_and_payload_match_actual_adapter_messages(variant):
    content = '{"passage":"Hypothetical city history."}' if variant == "QUERY2DOC" else None
    client = Client(content) if content else Client()
    spec = ALL_BASELINE_SPECS[variant]
    baseline_generator(client, variant).generate(Q, spec.decision())
    messages, options = client.requests[0]
    record = PROMPT_REGISTRY[variant]
    assert messages[0] == {"role": "system", "content": record.system_prompt}
    assert options["prompt_version"] == record.prompt_version
    expected_payload = json.loads(record.user_payload_template)
    expected_payload["original_question"] = Q.text
    assert json.loads(messages[1]["content"]) == expected_payload
    assert (
        record.system_prompt_sha256
        == hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()
    )
    assert len(client.requests) == 1


def test_reader_export_is_actual_system_prompt_and_document_schema():
    client = Client('{"answer":"Eastlake", "cited_evidence_ids":["fixture-evidence"]}')
    APIReader(client).answer(Q, (E,))
    messages, options = client.requests[0]
    record = PROMPT_REGISTRY["READER"]
    assert messages[0]["content"] == record.system_prompt
    assert options["prompt_version"] == record.prompt_version
    assert json.loads(messages[1]["content"]) == {
        "original_question": Q.text,
        "evidence": [asdict(E)],
    }
    template = json.loads(record.user_payload_template)
    assert set(template["evidence"][0]) == set(asdict(E))


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("SIMPLE_PARAPHRASE", "998a04d42f3a2da32f11bbba9f561d268c47dbc9fe18bb1a016f4eb0b67699ce"),
        ("RRR_KEYWORDS", "b397e6d94672f39e465f441b6740c3c1e1d84235ca84523d7da76cf50e6f0b3f"),
        ("QUERY2DOC", "51bd4a178a2320fa8cf334a6bfdebcd8cf01d897678cdffb69f9d41dfc39ee2c"),
        ("READER", "379a03e647ac4c8b1e72ee196bc3fe62e370555ecff3b3b03b6ed8ea1f2f31eb"),
    ],
)
def test_preexisting_prompt_bytes_are_frozen(variant, expected):
    assert prompt_identity(variant)["system_prompt_sha256"] == expected


def test_prompt_ablation_and_composition_ablation_have_distinct_signatures():
    old, minimal, anchor = (
        prompt_identity(name) for name in ("RRR_KEYWORDS", "RRR_MINIMAL", "RRR_QUERY_ANCHOR")
    )
    assert minimal["system_prompt_sha256"] != old["system_prompt_sha256"]
    assert anchor["system_prompt_sha256"] == old["system_prompt_sha256"]
    assert len({x["execution_signature_sha256"] for x in (old, minimal, anchor)}) == 3
    assert ALL_BASELINE_SPECS["RRR_MINIMAL"].intent == BASELINE_SPECS["RRR_KEYWORDS"].intent


def test_minimal_prompt_removes_only_the_declared_reminder_block():
    removed = (
        "For comparisons keep both entities and the comparison property.\n"
        "For a question with an indirect entity description, keep the described link: do\n"
        "not guess the missing entity. Preserve dates, locations, relation direction and\n"
        "negation. "
    )
    assert removed in RRR_PROMPT
    assert PROMPT_REGISTRY["RRR_MINIMAL"].system_prompt == RRR_PROMPT.replace(removed, "", 1)


def test_query_anchor_changes_only_postprocessing_not_actual_model_messages():
    old, anchor = Client(), Client()
    raw = baseline_generator(old, "RRR_KEYWORDS").generate(
        Q, ALL_BASELINE_SPECS["RRR_KEYWORDS"].decision()
    )
    anchored = baseline_generator(anchor, "RRR_QUERY_ANCHOR").generate(
        Q, ALL_BASELINE_SPECS["RRR_QUERY_ANCHOR"].decision()
    )
    assert old.requests[0][0] == anchor.requests[0][0]
    assert anchored.value == Q.text + "\n" + raw.value
    assert old.requests[0][1]["prompt_version"] != anchor.requests[0][1]["prompt_version"]


@pytest.mark.parametrize("variant", EXTRA_BASELINE_SPECS)
def test_optional_query_only_variants_reject_gold_and_post_retrieval_inputs(variant):
    client = Client()
    generator = baseline_generator(client, variant)
    decision = ALL_BASELINE_SPECS[variant].decision()
    with pytest.raises(TypeError, match="no gold"):
        generator.generate(GoldRecord("fixture", ("secret reference answer",)), decision)
    with pytest.raises(ValueError, match="PRE"):
        generator.generate(Q, decision, evidence=(E,))
    with pytest.raises(ValueError, match="PRE"):
        generator.generate(Q, decision, previous_queries=("old query",))
    assert client.requests == []


def test_anchor_overlength_fails_without_truncation_or_retry():
    client = Client(json.dumps({"query": "x" * 1000}))
    question = RuntimeQuestion("long", "q" * 1500)
    with pytest.raises(BackendCallError, match="no retry"):
        baseline_generator(client, "RRR_QUERY_ANCHOR").generate(
            question, ALL_BASELINE_SPECS["RRR_QUERY_ANCHOR"].decision()
        )
    assert len(client.requests) == 1


def test_registry_exports_exact_json_and_human_readable_appendix_without_overwriting(tmp_path):
    json_path, markdown_path = export_registry(tmp_path / "appendix")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for entry in data["prompts"]:
        assert entry["system_prompt"] == PROMPT_REGISTRY[entry["variant_id"]].system_prompt
        assert (
            entry["system_prompt_sha256"]
            == hashlib.sha256(entry["system_prompt"].encode("utf-8")).hexdigest()
        )
    text = markdown_path.read_text(encoding="utf-8")
    assert "RRR_MINIMAL" in text and "READER" in text
    before = json_path.read_bytes(), markdown_path.read_bytes()
    with pytest.raises(FileExistsError):
        export_registry(tmp_path / "appendix")
    assert before == (json_path.read_bytes(), markdown_path.read_bytes())


def test_export_cli_never_instantiates_live_client_or_accepts_credentials(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("appendix export must not construct an API client")

    monkeypatch.setattr("growrag.experiments.api_client.LiveChatClient.__init__", forbidden)
    assert main(["--output", str(tmp_path / "appendix")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["api_calls"] == 0
    assert len(output["paths"]) == 2


def test_base_has_no_query_prompt_but_reader_has_one():
    assert prompt_identity("BASE")["system_prompt_sha256"] is None
    assert prompt_identity("READER")["system_prompt_sha256"]
    with pytest.raises(ValueError, match="unknown"):
        prompt_identity("invented-paper")
