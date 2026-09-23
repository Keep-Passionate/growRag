"""Gate orchestration tests: no credentials, network, or measured model quality."""

from types import SimpleNamespace

import pytest

from growrag.experiments import run_stability as runner
from growrag.experiments.protocol import CallResult, Usage


class FakeAssessor:
    def __init__(self, client):
        self.client = client
        self.records = []
        self.latest = None

    def __call__(self, state, reply):
        self.client.calls.append(
            {"api_requests": 0, "input_tokens": 0, "output_tokens": 0, "estimated_actual_cny": 0}
        )
        self.records.append({"status": "mock", "state": state, "reply": reply})
        if self.client.fail:
            raise ValueError("synthetic failure; do not expose arbitrary transport text")
        self.latest = {"sufficient": True}
        return CallResult(True, usage=Usage(0, 0, 0), transport_source="mock")


def setup_gate(monkeypatch, *, semantic_pass=True, fail=False):
    cases = tuple(SimpleNamespace(case_id=str(i), state="STATE", reply="REPLY") for i in range(2))
    monkeypatch.setattr(runner, "smoke_cases", lambda: cases)
    monkeypatch.setattr(runner, "APIEvidenceAssessor", FakeAssessor)
    monkeypatch.setattr(runner, "evaluate_case", lambda *_: {"semantic_pass": semantic_pass})
    return SimpleNamespace(calls=[], block_reason=None, fail=fail, attempts=0)


@pytest.mark.parametrize("semantic_pass,fail", [(False, False), (True, True)])
def test_failed_gate_prevents_any_hotpot_calls(monkeypatch, tmp_path, semantic_pass, fail):
    client = setup_gate(monkeypatch, semantic_pass=semantic_pass, fail=fail)

    def forbidden(*args, **kwargs):
        raise AssertionError("Hotpot must not execute when synthetic gate fails")

    monkeypatch.setattr(runner, "run_question", forbidden)
    result = runner.execute(client, [object()], (), tmp_path)
    assert result["smoke_passed"] is False
    assert client.block_reason
    assert len(client.calls) == (1 if fail else 2)
    assert (tmp_path / "smoke_gate.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert not (tmp_path / "questions").exists()


def test_passing_gate_enables_only_explicit_examples_with_shared_execution(monkeypatch, tmp_path):
    client = setup_gate(monkeypatch)
    invoked = []

    def run_question(example, views, current_client, path, *, share_execution):
        assert len(client.calls) == 2 and current_client is client
        assert views == ("frozen-card",) and share_execution is True
        invoked.append((example, path))
        return {"mock": example}

    monkeypatch.setattr(runner, "run_question", run_question)
    monkeypatch.setattr(
        runner,
        "summarize",
        lambda reports, **kw: {"cost_notice": "", "reported": len(reports), **kw},
    )
    result = runner.execute(client, ("q4", "q5"), ("frozen-card",), tmp_path)
    assert result == {"smoke_passed": True, "reported_questions": 2}
    assert [q for q, _ in invoked] == ["q4", "q5"]
    assert not client.block_reason


def test_interruption_preserves_started_count_and_never_retries(monkeypatch, tmp_path):
    client = setup_gate(monkeypatch)
    invocations = []

    def interrupted(*args, **kwargs):
        invocations.append(1)
        raise RuntimeError("local interruption")

    monkeypatch.setattr(runner, "run_question", interrupted)
    with pytest.raises(RuntimeError, match="local interruption"):
        runner.execute(client, ("q4", "q5"), (), tmp_path)
    import json

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["started"] == 1 and summary["reported"] == 0
    assert len(invocations) == 1


def test_one_use_output_cannot_escape_run_root(tmp_path):
    with pytest.raises(ValueError, match="strictly inside"):
        runner.main(
            [
                "--manifest",
                "not-read.json",
                "--source-dir",
                "not-read",
                "--runs-root",
                str(tmp_path),
                "--output",
                str(tmp_path.parent / "outside"),
            ]
        )
