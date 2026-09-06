"""Synthetic model clients only: these tests make NO network/API requests."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.hotpot import parse_hotpot_example
from growrag.experiments.pilot_engine import (
    GAP_PROMPT_VERSION,
    MEMORY_PROMPT_VERSION,
    run_pilot,
)
from growrag.experiments.protocol import GoldRecord


def example(identifier="source", work="Alpha", author="Writer", town="Lumen"):
    return parse_hotpot_example(
        {
            "_id": identifier,
            "question": f"{work} author birthplace?",
            "context": [
                [work, [f"{work} author {author}."]],
                [author, [f"{author} birthplace {town}."]],
                ["Distractor", ["Unrelated mountain geology."]],
            ],
            "answer": town,
            "supporting_facts": [[work, 0], [author, 0]],
            "type": "bridge",
            "level": "hard",
        },
        dataset="SYNTHETIC-NOT-HOTPOT",
    )


class SyntheticClient:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://test.invalid/v1", model="SYNTHETIC-MODEL")
    attempts = 0  # Actual API requests stay zero, unlike logical mock calls.

    def __init__(self, *, duplicate=False, fail_stage=None, fail_target_only=False):
        self.requests = []
        self.duplicate = duplicate
        self.fail_stage = fail_stage
        self.fail_target_only = fail_target_only

    def complete(self, messages, *, trace_id, prompt_version):
        payload = json.loads(messages[1]["content"])
        self.requests.append((messages, prompt_version, payload))
        question = payload.get("original_question", "")
        fail = self.fail_stage == prompt_version or (
            self.fail_stage == "reuse" and "optional_historical_procedure" in payload
        )
        if fail and (not self.fail_target_only or "Beta" in question):
            raise APIRequestError("synthetic failure", transport_source="mock")
        if prompt_version == GAP_PROMPT_VERSION:
            value = {"status": "insufficient", "missing_information": ["author birthplace"]}
        elif prompt_version == MEMORY_PROMPT_VERSION:
            value = {
                "operation": "Search the identified author's birthplace.",
                "applicability_conditions": ["Author identity is already evidenced."],
            }
        elif "query" in prompt_version:
            author = "Writer" if "Alpha" in question else "Skylark"
            value = {"query": question if self.duplicate else f"{author} birthplace"}
        else:
            town = "Lumen" if "Alpha" in question else "Nova"
            support = next((e for e in payload["evidence"] if town in e["text"]), None)
            value = {
                "answer": town if support else "",
                "cited_evidence_ids": [support["evidence_id"]] if support else [],
            }
        return ChatResponse(
            json.dumps(value),
            self.config.model,
            self.config.model,
            "SYNTHETIC-ID",
            None,
            None,
            None,
            0.0,
            Path("SYNTHETIC-NO-NETWORK-AUDIT"),
            "mock",
        )


def target():
    return example("target", "Beta", "Skylark", "Nova")


def pilot(client, path, sources=None, targets=None):
    return run_pilot(
        (example(),) if sources is None else sources,
        (target(),) if targets is None else targets,
        client,
        path,
        initial_top_k=1,
        repair_top_k=1,
    )


def test_source_library_freezes_before_targets_and_full_cost_scope_is_labelled(tmp_path):
    client = SyntheticClient()
    report = pilot(client, tmp_path)
    assert report["synthetic"] is True
    assert report["execution_kind"] == "mock"
    assert report["candidate_memory_count"] == 1
    assert len(client.requests) == 11  # Source five, target six logical model calls.
    assert report["usage"]["api_requests_known_sum"] == 0
    assert report["usage"]["client_attempts_delta"] == 0
    assert report["usage"]["input_tokens"] is None
    assert "shared-prefix" in report["usage"]["scope"]
    assert report["targets"][0]["memory_updates_allowed"] is False
    assert report["sources"][0]["memory_admission"]["status"] == "candidate_created"
    assert report["summary"]["branches"]["REUSE"]["available_count"] == 1
    assert report["summary"]["comparisons"]["REUSE_vs_FRESH"]["pair_denominator"] == 1
    assert report["summary"]["comparisons"]["REUSE_vs_FRESH"]["metrics"]["answer_em"] == {
        "operational_mean_delta_failure_as_zero": 0.0,
        "improved": 0,
        "degraded": 0,
        "tied": 1,
        "both_observed_pairs": 1,
        "missing_or_failed_pairs": 0,
    }
    library = json.loads((tmp_path / "memory_library.json").read_text(encoding="utf-8"))
    assert library["frozen_before_targets"] is True
    assert library["candidates"][0]["view"]["source_query_id"] == "source"
    assert "#branches/FRESH" in library["candidates"][0]["view"]["source_step_id"]
    assert report["targets"][0]["frozen_library_sha256"] == library["library_sha256"]
    assert len(list((tmp_path / "episodes").glob("*.json"))) == 2


def test_no_gold_type_or_level_reaches_gap_rewrite_reader_or_memory_prompt(tmp_path):
    client = SyntheticClient()
    changed = replace(
        target(),
        gold=GoldRecord(
            "target", ("SECRET-GOLD-ANSWER-NEVER-IN-PROMPTS",), (("SECRET-GOLD-TITLE", 99),)
        ),
    )
    report = pilot(client, tmp_path, targets=(changed,))
    for messages, version, payload in client.requests:
        serialized = json.dumps(messages)
        assert "SECRET-GOLD" not in serialized
        assert "supporting_facts" not in serialized
        assert "difficulty" not in payload
        assert "question_type" not in payload
        if version == GAP_PROMPT_VERSION:
            assert set(payload) == {"original_question", "evidence"}
        if version == MEMORY_PROMPT_VERSION:
            assert set(payload) == {"original_question", "gap", "executed_query", "new_evidence"}
            assert "answer" not in payload
    assert report["targets"][0]["branches"]["REUSE"]["raw_feedback"]["answer_em"] == 0.0
    assert report["candidate_memory_count"] == 1


def test_changed_target_gold_does_not_change_selected_memory_or_runtime_requests(tmp_path):
    normal_client, changed_client = SyntheticClient(), SyntheticClient()
    normal = pilot(normal_client, tmp_path / "normal")
    changed_target = replace(target(), gold=GoldRecord("target", ("Another answer",), ()))
    changed = pilot(changed_client, tmp_path / "changed", targets=(changed_target,))
    assert normal_client.requests == changed_client.requests
    assert normal["targets"][0]["selection"] == changed["targets"][0]["selection"]


def test_overlapping_source_and_target_rejected_before_any_request(tmp_path):
    client = SyntheticClient()
    with pytest.raises(ValueError, match="disjoint"):
        pilot(client, tmp_path, targets=(example(),))
    assert not client.requests
    assert not (tmp_path / "episodes").exists()


def test_empty_library_reports_reuse_unavailable_without_fake_card_or_target_promotion(tmp_path):
    client = SyntheticClient()
    report = pilot(client, tmp_path, sources=())
    assert len(client.requests) == 4
    assert report["candidate_memory_count"] == 0
    assert report["targets"][0]["selection"]["status"] == "reuse_unavailable"
    assert report["targets"][0]["branches"]["REUSE"]["effective_status"] == "unavailable"
    assert report["summary"]["branches"]["REUSE"]["unavailable_count"] == 1
    assert report["summary"]["comparisons"]["REUSE_vs_FRESH"]["pair_denominator"] == 0


def test_duplicate_query_keeps_raw_stop_and_explicit_base_fallback(tmp_path):
    client = SyntheticClient(duplicate=True)
    report = pilot(client, tmp_path)
    fresh = report["targets"][0]["branches"]["FRESH"]
    assert fresh["raw"]["status"] == "stopped"
    assert fresh["raw"]["stop_reason"] == "duplicate_query"
    assert fresh["raw"]["answer"] is None
    assert fresh["effective_fallback_to_BASE"] is True
    assert fresh["effective_feedback"] is not None
    assert report["summary"]["branches"]["FRESH"]["raw_stopped_count"] == 1
    assert report["summary"]["branches"]["FRESH"]["answer_em"]["denominator"] == 1
    assert report["candidate_memory_count"] == 0
    assert len(client.requests) == 6  # No retry, extraction or extra reader after stop.


def test_reuse_transport_error_remains_failure_not_successful_base_fallback(tmp_path):
    report = pilot(SyntheticClient(fail_stage="reuse"), tmp_path)
    reuse = report["targets"][0]["branches"]["REUSE"]
    assert reuse["raw"]["status"] == "error"
    assert reuse["effective_fallback_to_BASE"] is False
    assert reuse["effective_feedback"] is None
    metric = report["summary"]["branches"]["REUSE"]["answer_em"]
    assert metric["denominator"] == 1
    assert metric["operational_mean_failure_as_zero"] == 0.0
    assert metric["valid_output_mean"] is None
    assert (
        report["summary"]["comparisons"]["REUSE_vs_FRESH"]["metrics"]["answer_em"][
            "missing_or_failed_pairs"
        ]
        == 1
    )


def test_gap_error_keeps_failed_question_in_all_available_branch_denominators(tmp_path):
    client = SyntheticClient(fail_stage=GAP_PROMPT_VERSION, fail_target_only=True)
    report = pilot(client, tmp_path)
    assert report["targets"][0]["status"] == "error"
    assert report["summary"]["target_question_errors"] == 1
    assert len(report["targets"]) == 1
    for action in ("BASE", "FRESH", "REUSE"):
        assert report["targets"][0]["branches"][action]["raw"]["status"] == "not_run_error"
        assert report["summary"]["branches"][action]["answer_em"]["denominator"] == 1
    assert len(client.requests) == 6


def test_extraction_failure_is_recorded_and_does_not_create_card(tmp_path):
    report = pilot(SyntheticClient(fail_stage=MEMORY_PROMPT_VERSION), tmp_path)
    assert report["candidate_memory_count"] == 0
    assert report["sources"][0]["memory_admission"]["status"] == "extraction_error"
    assert report["sources"][0]["memory_extraction_calls"][0]["status"] == "error"
    assert report["targets"][0]["branches"]["REUSE"]["effective_status"] == "unavailable"


def test_gap_sufficient_still_runs_paired_diagnostic_not_a_deployed_router(tmp_path):
    class SufficientClient(SyntheticClient):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            if kwargs["prompt_version"] == GAP_PROMPT_VERSION:
                return replace(
                    response,
                    content=json.dumps(
                        {
                            "status": "sufficient",
                            "missing_information": [],
                        }
                    ),
                )
            return response

    report = pilot(SufficientClient(), tmp_path)
    assert report["gap_routes_branches"] is False
    assert report["targets"][0]["gap"]["status"] == "sufficient"
    assert report["targets"][0]["branches"]["FRESH"]["raw"]["status"] == "completed"


def test_bad_gap_json_is_error_and_never_retried_or_silently_inferred_from_gold(tmp_path):
    class BadGapClient(SyntheticClient):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            if kwargs["prompt_version"] == GAP_PROMPT_VERSION:
                return replace(response, content='{"status":"sufficient","extra":"bad"}')
            return response

    client = BadGapClient()
    report = pilot(client, tmp_path, sources=())
    assert report["targets"][0]["status"] == "error"
    assert len(client.requests) == 1


def test_output_refuses_overwrite_but_allows_caller_manifest(tmp_path):
    # A caller-owned manifest is a normal input artifact, not a pilot result.
    (tmp_path / "manifest.json").touch()
    client = SyntheticClient()
    pilot(client, tmp_path)
    requests = len(client.requests)
    with pytest.raises(FileExistsError):
        pilot(client, tmp_path)
    assert len(client.requests) == requests


def test_unlabelled_examples_rejected_without_calling_client(tmp_path):
    client = SyntheticClient()
    with pytest.raises(ValueError, match="matching gold"):
        pilot(client, tmp_path, sources=(replace(example(), gold=None),))
    assert not client.requests


def test_missing_transport_provenance_rejected(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        pilot(SimpleNamespace(), tmp_path)
