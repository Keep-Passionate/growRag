from __future__ import annotations

import json

import pytest

from growrag.models import EnvironmentFingerprint, QueryTransformation
from growrag.runtime_io import (
    PLAN_SCHEMA_VERSION,
    PlanBundle,
    RuntimeIOError,
    load_plan_bundle,
    load_requests,
    write_decision_jsonl,
)


def write_json(path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def signature_protocol() -> dict[str, str]:
    return {
        "signature_extractor_id": "query-only-rules",
        "signature_extractor_version": "1",
        "signature_input_scope": "query_only",
    }


def valid_plan_bundle() -> dict[str, object]:
    return {
        "schema_version": 1,
        "applier_id": "querygym-frozen",
        "application_version": "prompt-v3",
        "plans": [
            {
                "target_query_id": "target-1",
                "experience_id": "exp-1",
                "transformed_query": "Hamlet author playwright",
            },
            {
                "target_query_id": "target-1",
                "experience_id": "exp-2",
                "transformed_query": "Hamlet written by",
            },
        ],
    }


def experience() -> QueryTransformation:
    return QueryTransformation(
        experience_id="exp-1",
        source_query_id="source-1",
        source_query="Who wrote the novel?",
        transformed_query="novel author",
        atomic_units=("authorship relation",),
        environment=EnvironmentFingerprint(
            corpus_id="toy",
            corpus_version="1",
            retriever_id="bm25",
            retriever_version="1",
            rewriter_id="prompt",
            rewriter_version="1",
        ),
        provenance="fixture",
    )


def test_load_requests_from_strict_jsonl(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "target_query_id": "q-1",
                        "target_query": "Who wrote Hamlet?",
                        "signatures": ["authorship", "play"],
                        **signature_protocol(),
                    }
                ),
                json.dumps(
                    {
                        "target_query_id": "q-2",
                        "target_query": "Where was Austen born?",
                        "signatures": [],
                        **signature_protocol(),
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    requests = load_requests(path)

    assert [request.target_query_id for request in requests] == ["q-1", "q-2"]
    assert requests[0].signatures == ("authorship", "play")
    assert requests[1].signatures == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unknown": True}), "fields differ"),
        (lambda value: value.pop("target_query"), "fields differ"),
        (lambda value: value.update({"target_query": "  "}), "target_query must not be empty"),
        (lambda value: value.update({"signatures": "authorship"}), "signatures must be an array"),
        (lambda value: value.update({"signatures": [1]}), r"signatures\[0\] must be a string"),
        (lambda value: value.update({"signatures": [" "]}), r"signatures\[0\] must not be empty"),
        (
            lambda value: value.update({"signature_input_scope": "retrieval_results"}),
            "must be 'query_only'",
        ),
    ],
)
def test_request_schema_fails_closed(tmp_path, mutation, message: str) -> None:
    request = {
        "target_query_id": "q-1",
        "target_query": "Who wrote Hamlet?",
        "signatures": ["authorship"],
        **signature_protocol(),
    }
    mutation(request)
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeIOError, match=message):
        load_requests(path)


def test_duplicate_request_id_and_blank_line_are_rejected(tmp_path) -> None:
    request = {
        "target_query_id": "same",
        "target_query": "question",
        "signatures": [],
        **signature_protocol(),
    }
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        f"{json.dumps(request)}\n{json.dumps(request)}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeIOError, match="duplicate target_query_id"):
        load_requests(duplicate_path)

    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_text(f"{json.dumps(request)}\n\n{json.dumps(request)}", encoding="utf-8")
    with pytest.raises(RuntimeIOError, match="line 2 is blank"):
        load_requests(blank_path)


def test_duplicate_json_object_key_is_rejected_in_request(tmp_path) -> None:
    path = tmp_path / "duplicate-key.jsonl"
    path.write_text(
        '{"target_query_id":"a","target_query_id":"b","target_query":"question","signatures":[]}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeIOError, match="duplicate JSON key"):
        load_requests(path)


def test_request_batch_requires_one_frozen_signature_protocol(tmp_path) -> None:
    base = {
        "target_query": "question",
        "signatures": [],
        **signature_protocol(),
    }
    path = tmp_path / "mixed-protocol.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({**base, "target_query_id": "q1"}),
                json.dumps(
                    {
                        **base,
                        "target_query_id": "q2",
                        "signature_extractor_version": "2",
                    }
                ),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeIOError, match="one frozen signature protocol"):
        load_requests(path)


def test_load_plan_bundle_and_freeze_tuple_keys(tmp_path) -> None:
    path = tmp_path / "plans.json"
    write_json(path, valid_plan_bundle())

    bundle = load_plan_bundle(path)

    assert bundle.schema_version == PLAN_SCHEMA_VERSION
    assert bundle.applier_id == "querygym-frozen"
    assert bundle.application_version == "prompt-v3"
    assert bundle.plans[("target-1", "exp-1")] == "Hamlet author playwright"
    with pytest.raises(TypeError):
        bundle.plans[("new", "exp")] = "cannot mutate"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"future": 1}), "fields differ"),
        (lambda value: value.pop("applier_id"), "fields differ"),
        (lambda value: value.update({"schema_version": 2}), "unsupported schema_version"),
        (lambda value: value.update({"schema_version": True}), "must be an integer"),
        (lambda value: value.update({"application_version": " "}), "must not be empty"),
        (
            lambda value: value["plans"][0].update({"future": 1}),
            "fields differ",
        ),
        (
            lambda value: value["plans"][0].update({"transformed_query": ""}),
            "transformed_query must not be empty",
        ),
    ],
)
def test_plan_bundle_schema_fails_closed(tmp_path, mutation, message: str) -> None:
    raw = valid_plan_bundle()
    mutation(raw)
    path = tmp_path / "bad-plans.json"
    write_json(path, raw)

    with pytest.raises(RuntimeIOError, match=message):
        load_plan_bundle(path)


def test_duplicate_target_experience_plan_key_is_rejected(tmp_path) -> None:
    raw = valid_plan_bundle()
    raw["plans"].append(dict(raw["plans"][0]))
    path = tmp_path / "duplicate-plans.json"
    write_json(path, raw)

    with pytest.raises(RuntimeIOError, match="duplicate plan key"):
        load_plan_bundle(path)


def test_to_applier_requires_caller_budgets_and_preserves_versions(tmp_path) -> None:
    path = tmp_path / "plans.json"
    write_json(path, valid_plan_bundle())
    bundle = load_plan_bundle(path)

    applier = bundle.to_applier(
        retrieval_query_count=2,
        requested_top_k=13,
        context_token_budget=2048,
        estimated_cost=0.25,
    )
    prepared = applier.prepare(
        target_query_id="target-1",
        target_query="Who wrote Hamlet?",
        experience=experience(),
    )

    assert prepared.transformed_query == "Hamlet author playwright"
    assert prepared.generated_by == "querygym-frozen"
    assert prepared.application_version == "prompt-v3"
    assert prepared.retrieval_query_count == 2
    assert prepared.requested_top_k == 13
    assert prepared.context_token_budget == 2048
    assert prepared.estimated_cost == 0.25


def test_write_decision_jsonl_is_deterministic_and_atomic(tmp_path) -> None:
    path = tmp_path / "nested" / "decisions.jsonl"
    decisions = [
        {"target": "q-1", "action": "direct"},
        {"target": "q-2", "score": 0.75, "action": "reuse"},
    ]

    returned = write_decision_jsonl(path, decisions)

    assert returned == path
    assert path.read_text(encoding="utf-8") == (
        '{"action":"direct","target":"q-1"}\n{"action":"reuse","score":0.75,"target":"q-2"}\n'
    )
    assert not list(path.parent.glob("*.tmp"))


def test_write_decision_jsonl_rejects_non_dict_and_nan(tmp_path) -> None:
    path = tmp_path / "decisions.jsonl"
    with pytest.raises(RuntimeIOError, match="must be a dict"):
        write_decision_jsonl(path, [{"ok": True}, "not-a-dict"])
    with pytest.raises(RuntimeIOError, match="not JSON-safe"):
        write_decision_jsonl(path, [{"score": float("nan")}])
    assert not path.exists()


def test_direct_plan_bundle_rejects_malformed_tuple_key() -> None:
    with pytest.raises(RuntimeIOError, match="plan key"):
        PlanBundle(
            schema_version=1,
            applier_id="precomputed",
            application_version="1",
            plans={"not-a-tuple": "query"},
        )
