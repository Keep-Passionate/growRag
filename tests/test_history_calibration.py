"""开发运行器的保护与失败落盘；数据和模型响应均为合成。"""

import json
from types import SimpleNamespace

import pytest

from growrag.experiments import history_calibration as study
from growrag.experiments.pre_pilot import write_json
from growrag.experiments.protocol import RuntimeQuestion


@pytest.mark.parametrize("start,count", [(0, 5), (40, 20), (45, 5), (True, 5), (65, 20)])
def test_unregistered_ranges_fail_before_reading_files(tmp_path, start, count):
    with pytest.raises(ValueError, match="predeclares"):
        study.prepare(tmp_path, start=start, count=count)


def test_new_claim_is_not_replayed_under_another_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "check_unstarted", lambda *args: None)
    write_json(
        tmp_path / f"{study.PREFIX}old.claim.json",
        {
            "protocol": study.PROTOCOL,
            "question_ids": ["synthetic-q"],
            "arms": ["history"],
        },
    )
    with pytest.raises(ValueError, match="previously claimed"):
        study.check_claims(tmp_path, ["synthetic-q"])


class FakeClient:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def complete(self, messages, *, trace_id, prompt_version):
        self.calls.append({"trace_id": trace_id, "status": "completed", "input_tokens": 10})
        if self.fail:
            raise ValueError("synthetic local decode failure after HTTP")
        return SimpleNamespace(
            content=json.dumps({"answer": "", "supported": False, "evidence_ids": []})
        )


def test_base_searches_once_and_never_requests_planning(tmp_path):
    client, queries = FakeClient(), []

    def index(query, top_k):
        queries.append((query, top_k))
        return (SimpleNamespace(doc_id="current-doc", title="Cedar", text="Current evidence."),)

    target = tmp_path / "base.json"
    result = study.execute_arm(
        RuntimeQuestion("synthetic-q", "Where is Cedar?"),
        "base",
        index,
        client,
        library=None,
        reference=None,
        trace="synthetic/base",
        log=lambda _: None,
        target=target,
    )
    assert queries == [("Where is Cedar?", 6)]
    assert len(client.calls) == 1 and client.calls[0]["trace_id"].endswith("/reader")
    assert result["status"] == "completed" and not result["memory_updated"]
    assert not result["gold_loaded"]


def test_completed_http_cost_and_failed_arm_are_preserved_without_retry(tmp_path):
    client = FakeClient(fail=True)
    target = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="synthetic local"):
        study.execute_arm(
            RuntimeQuestion("synthetic-q", "Where is Cedar?"),
            "base",
            lambda *args: (),
            client,
            library=None,
            reference=None,
            trace="synthetic/base",
            log=lambda _: None,
            target=target,
        )
    saved = json.loads(target.read_bytes())
    assert saved["status"] == "failed" and saved["error_type"] == "ValueError"
    assert saved["calls"] == client.calls and len(client.calls) == 1
    assert saved["calls"][0]["status"] == "completed"
    assert saved["reader"] is None and saved["episode"] is not None


def test_expansion_rejects_claimed_success_with_no_predictions(tmp_path):
    root = tmp_path / f"{study.PREFIX}v1_0040_0045"
    root.mkdir()
    write_json(
        root / "SUMMARY.json",
        {
            "status": "completed",
            "completed_questions": 5,
            "source_sha256": "synthetic-sha",
            "prediction_sha256": {},
        },
    )
    write_json(
        root / "launch_plan.json",
        {
            **study.CONFIGURATION,
            "protocol": study.PROTOCOL,
            "manifest_sha256": study.MANIFEST_SHA256,
            "library_sha256": study.LIBRARY_SHA256,
            "source_sha256": "synthetic-sha",
            "question_ids": [f"q{i}" for i in range(5)],
            "arms": list(study.ARMS),
        },
    )
    with pytest.raises(ValueError, match="prediction set"):
        study.verify_first_batch(tmp_path, "synthetic-sha")
