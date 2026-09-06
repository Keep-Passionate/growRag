from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from growrag.experiments.mock import (
    MOCK_USAGE,
    MockReader,
    MockRetriever,
    MockRewriter,
    demo_inputs,
    run_demo,
)
from growrag.experiments.paired_runner import PairedRunner, load_run, write_run
from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    BranchSpec,
    CallResult,
    DecisionState,
    Evidence,
    ExecutionKind,
    MemoryView,
    RuntimeQuestion,
    Usage,
)


def make_runner(**overrides: object) -> PairedRunner:
    arguments = {
        "retriever": MockRetriever(),
        "rewriter": MockRewriter(),
        "reader": MockReader(),
        "execution_kind": ExecutionKind.MOCK,
        "top_k": 1,
        **overrides,
    }
    return PairedRunner(**arguments)  # type: ignore[arg-type]


def test_demo_is_explicitly_mock_with_zero_api_requests_and_no_fake_tokens() -> None:
    run = run_demo()
    report = run.to_dict()
    assert report["execution_kind"] == "mock"
    assert report["synthetic_demo"] is True
    assert "not HotpotQA" in report["notice"]
    assert "NOT included" in report["cost_scope"]
    assert [branch.retrieval_calls for branch in run.branches] == [0, 1, 1]
    assert [branch.model_component_calls for branch in run.branches] == [1, 2, 2]
    assert all(branch.api_requests == 0 for branch in run.branches)
    assert all(call.usage.input_tokens is None for branch in run.branches for call in branch.calls)
    assert all(call.usage.output_tokens is None for branch in run.branches for call in branch.calls)


def test_branches_share_only_initial_state_and_keep_original_answer_target() -> None:
    observed_memories = []
    observed_questions = []
    observed_evidence = []

    class RecordingRewriter(MockRewriter):
        def rewrite(self, state: DecisionState, *, memory: MemoryView | None) -> CallResult[str]:
            observed_memories.append(memory)
            assert [item.evidence_id for item in state.evidence] == ["aster:0"]
            return super().rewrite(state, memory=memory)

    class RecordingReader(MockReader):
        def answer(
            self, question: RuntimeQuestion, evidence: tuple[Evidence, ...]
        ) -> CallResult[Answer]:
            observed_questions.append(question)
            observed_evidence.append(tuple(item.evidence_id for item in evidence))
            return super().answer(question, evidence)

    state, specs = demo_inputs()
    run = make_runner(rewriter=RecordingRewriter(), reader=RecordingReader()).run(state, specs)
    assert observed_memories == [None, specs[-1].memory]
    assert observed_questions == [state.question] * 3
    assert observed_evidence == [("aster:0",), ("aster:0", "aster:1"), ("aster:0", "mira:0")]
    assert {branch.state_fingerprint for branch in run.branches} == {state.fingerprint}
    assert state.evidence == (Evidence("aster:0", "Aster", 0, "Aster was written by Mira."),)


def test_branch_order_does_not_change_observed_results() -> None:
    state, specs = demo_inputs()
    forward = make_runner().run(state, specs)
    backward = make_runner().run(state, tuple(reversed(specs)))

    def observations(run: object) -> dict[str, object]:
        return {
            branch.branch_id: (
                branch.query,
                branch.answer,
                branch.new_evidence,
                branch.cumulative_evidence,
                branch.status,
            )
            for branch in run.branches
        }

    assert observations(forward) == observations(backward)


def test_repeat_query_stops_before_retrieval_and_reader() -> None:
    class RepeatingRewriter(MockRewriter):
        def rewrite(self, state: DecisionState, *, memory: MemoryView | None) -> CallResult[str]:
            return CallResult("  " + state.question.text.upper() + "  ", MOCK_USAGE)

    state, specs = demo_inputs()
    branch = make_runner(rewriter=RepeatingRewriter()).run(state, (specs[1],)).branches[0]
    assert branch.status == "stopped"
    assert branch.stop_reason == "duplicate_query"
    assert branch.retrieval_calls == 0
    assert branch.model_component_calls == 1
    assert branch.answer is None


def test_retrieval_failure_is_logged_and_does_not_cancel_other_branches() -> None:
    class FailingRetriever(MockRetriever):
        def retrieve(self, query: str, *, top_k: int) -> CallResult[tuple[Evidence, ...]]:
            if "Mira" in query:
                raise BackendCallError("test retrieval failure", usage=Usage(api_requests=0))
            return super().retrieve(query, top_k=top_k)

    state, specs = demo_inputs()
    run = make_runner(retriever=FailingRetriever()).run(state, tuple(reversed(specs)))
    failed, fresh, base = run.branches
    assert failed.status == "error"
    assert failed.error_type == "BackendCallError"
    assert failed.retrieval_calls == 1
    assert failed.calls[-1].status == "error"
    assert failed.answer is None
    assert failed.cumulative_evidence == state.evidence
    assert fresh.status == base.status == "completed"


def test_real_failure_retains_backend_usage_and_never_falls_back_to_mock() -> None:
    class RealRetriever:
        execution_kind = ExecutionKind.REAL

        def retrieve(self, query: str, *, top_k: int) -> CallResult[tuple[Evidence, ...]]:
            raise AssertionError("must not run after rewriting fails")

    class RealRewriter:
        execution_kind = ExecutionKind.REAL

        def rewrite(self, state: DecisionState, *, memory: MemoryView | None) -> CallResult[str]:
            raise BackendCallError(
                "sanitized provider error",
                usage=Usage(input_tokens=10, api_requests=1),
                provider="test-provider",
                model="test-model",
                request_id="req-fixture",
                audit_path="audit/failed-request.json",
                transport_source="injected-test-transport",
            )

    class RealReader:
        execution_kind = ExecutionKind.REAL

        def answer(
            self, question: RuntimeQuestion, evidence: tuple[Evidence, ...]
        ) -> CallResult[Answer]:
            raise AssertionError("must not run after rewriting fails")

    state, specs = demo_inputs()
    branch = (
        make_runner(
            retriever=RealRetriever(),
            rewriter=RealRewriter(),
            reader=RealReader(),
            execution_kind=ExecutionKind.REAL,
        )
        .run(state, (specs[1],))
        .branches[0]
    )
    assert branch.status == "error"
    assert branch.execution_kind is ExecutionKind.REAL
    assert branch.api_requests == 1
    assert branch.calls[0].usage.input_tokens == 10
    assert branch.calls[0].usage.output_tokens is None
    assert branch.calls[0].request_id == "req-fixture"
    assert branch.calls[0].audit_path == "audit/failed-request.json"
    assert branch.calls[0].transport_source == "injected-test-transport"
    assert branch.retrieval_calls == 0


def test_successful_call_preserves_transport_audit_link() -> None:
    class AuditedReader(MockReader):
        def answer(
            self, question: RuntimeQuestion, evidence: tuple[Evidence, ...]
        ) -> CallResult[Answer]:
            response = super().answer(question, evidence)
            return replace(
                response,
                request_id="req-success",
                audit_path="audit/successful-request.json",
                transport_source="injected-test-transport",
            )

    state, specs = demo_inputs()
    run = make_runner(reader=AuditedReader()).run(state, (specs[0],))
    event = run.branches[0].calls[0]
    assert event.status == "completed"
    assert event.request_id == "req-success"
    assert event.audit_path == "audit/successful-request.json"
    assert event.transport_source == "injected-test-transport"
    serialized = run.to_dict()["branches"][0]["calls"][0]
    assert serialized["audit_path"] == event.audit_path


def test_inconsistent_backend_kinds_cannot_be_run() -> None:
    with pytest.raises(ValueError, match="no automatic mock fallback"):
        make_runner(execution_kind=ExecutionKind.REAL)


def test_target_query_cannot_be_its_own_source() -> None:
    state, specs = demo_inputs()
    self_memory = replace(specs[-1].memory, source_query_id=state.question.question_id)
    with pytest.raises(ValueError, match="target query"):
        make_runner().run(state, (BranchSpec("reuse", Action.REUSE, self_memory),))


def test_conflicting_evidence_id_is_not_silently_overwritten() -> None:
    class ConflictingRetriever(MockRetriever):
        def retrieve(self, query: str, *, top_k: int) -> CallResult[tuple[Evidence, ...]]:
            return CallResult((Evidence("aster:0", "Aster", 0, "tampered content"),), MOCK_USAGE)

    state, specs = demo_inputs()
    branch = make_runner(retriever=ConflictingRetriever()).run(state, (specs[1],)).branches[0]
    assert branch.status == "error"
    assert branch.cumulative_evidence == state.evidence
    assert branch.retrieval_calls == 1


def test_reader_cannot_cite_another_branchs_evidence() -> None:
    class InvalidReader(MockReader):
        def answer(
            self, question: RuntimeQuestion, evidence: tuple[Evidence, ...]
        ) -> CallResult[Answer]:
            return CallResult(Answer("invented answer", ("mira:0",)), MOCK_USAGE)

    state, specs = demo_inputs()
    branch = make_runner(reader=InvalidReader()).run(state, (specs[0],)).branches[0]
    assert branch.status == "error"
    assert branch.answer is None
    assert branch.error_type == "ValueError"


def test_write_and_replay_preserve_observations_without_overwrite(tmp_path: Path) -> None:
    run = run_demo()
    output = tmp_path / "new-result"
    path = write_run(run, output)
    replay = load_run(path)
    assert replay["state_fingerprint"] == run.state.fingerprint
    expected_answer = json.loads(json.dumps(asdict(run.branches[-1].answer)))
    assert replay["branches"][-1]["answer"] == expected_answer
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_run(run, output)
    assert path.read_bytes() == before


def test_replay_rejects_changed_shared_state(tmp_path: Path) -> None:
    path = write_run(run_demo(), tmp_path / "new-result")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["state"]["gap"] = "silently changed gap"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_run(path)


def test_cli_creates_explicit_mock_and_refuses_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "cli-result"
    args = [sys.executable, "-m", "growrag.experiments", "--output", str(output)]
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    assert completed.returncode == 0
    assert "MOCK ONLY" in completed.stdout
    assert (output / "run.json").is_file()
    second = subprocess.run(args, check=False, capture_output=True, text=True)
    assert second.returncode == 2


def test_cli_replay_has_no_real_execution_option(tmp_path: Path) -> None:
    path = write_run(run_demo(), tmp_path / "saved")
    completed = subprocess.run(
        [sys.executable, "-m", "growrag.experiments", "--replay", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["execution_kind"] == "mock"
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "growrag.experiments",
            "--output",
            str(tmp_path / "other"),
            "--backend",
            "real",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2


def test_unexpected_error_does_not_leak_raw_secret_text() -> None:
    class SecretErrorRewriter(MockRewriter):
        def rewrite(self, state: DecisionState, *, memory: MemoryView | None) -> CallResult[str]:
            raise RuntimeError("Authorization: Bearer CANARY_SECRET_NOT_FOR_LOGS")

    state, specs = demo_inputs()
    run = make_runner(rewriter=SecretErrorRewriter()).run(state, (specs[1],))
    assert run.branches[0].status == "error"
    assert run.branches[0].error_type == "RuntimeError"
    assert "CANARY_SECRET_NOT_FOR_LOGS" not in json.dumps(run.to_dict())


def test_retriever_cannot_exceed_top_k_evidence_budget() -> None:
    class TooManyRetriever(MockRetriever):
        def retrieve(self, query: str, *, top_k: int) -> CallResult[tuple[Evidence, ...]]:
            evidence = tuple(Evidence(f"doc:{index}", "doc", index, "text") for index in range(2))
            return CallResult(evidence, MOCK_USAGE)

    state, specs = demo_inputs()
    branch = make_runner(retriever=TooManyRetriever(), top_k=1).run(state, (specs[1],)).branches[0]
    assert branch.status == "error"
    assert branch.retrieval_calls == 1
    assert branch.cumulative_evidence == state.evidence
    assert branch.answer is None
