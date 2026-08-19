from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from growrag.evaluation.oracle import OracleInputRow, analyze_oracle, read_oracle_csv


def _row(
    target_query_id: str,
    experience_id: str,
    candidate_rank: int,
    direct_score: float,
    reuse_score: float,
    fresh_score: float | None = None,
    *,
    memory_snapshot_id: str = "memory-test-v1",
    retrieval_query_count: int = 1,
    requested_top_k: int = 2,
    context_token_budget: int = 2048,
) -> OracleInputRow:
    return OracleInputRow(
        target_query_id=target_query_id,
        experience_id=experience_id,
        memory_snapshot_id=memory_snapshot_id,
        candidate_rank=candidate_rank,
        direct_score=direct_score,
        reuse_score=reuse_score,
        fresh_score=fresh_score,
        retrieval_query_count=retrieval_query_count,
        requested_top_k=requested_top_k,
        context_token_budget=context_token_budget,
    )


def _pilot_rows() -> list[OracleInputRow]:
    return [
        _row("q1", "e1", 1, 0.4, 0.7, 0.6),
        _row("q1", "e2", 2, 0.4, 0.3, 0.6),
        _row("q2", "e1", 1, 0.8, 0.6, 0.9),
        _row("q2", "e3", 2, 0.8, 0.8, 0.9),
        _row("q3", "e2", 1, 0.6, 0.6, 0.6),
        _row("q3", "e3", 2, 0.6, 0.2, 0.6),
    ]


def test_candidate_set_oracle_summary_and_offline_policy_ceiling() -> None:
    result = analyze_oracle(_pilot_rows(), good_threshold=0.5)

    assert result.memory_snapshot_id == "memory-test-v1"
    assert result.target_count == 3
    assert result.candidate_count == 6
    assert result.mean_candidate_count == pytest.approx(2.0)
    assert result.helpful_coverage == pytest.approx(1 / 3)
    assert result.pair_harm_count == 1
    assert result.pair_harm_rate == pytest.approx(1 / 6)
    assert result.direct_mean == pytest.approx(0.6)
    assert result.candidate_set_oracle_reuse_mean == pytest.approx(0.7)
    assert result.offline_policy_oracle_mean == pytest.approx(2.2 / 3)
    assert result.candidate_set_oracle_gain_over_direct == pytest.approx(0.1)
    assert result.offline_policy_oracle_gain_over_direct == pytest.approx(2 / 15)
    assert result.offline_policy_oracle_gain_over_candidate_set_oracle == pytest.approx(1 / 30)
    assert result.offline_policy_oracle_action_counts == {
        "direct": 1,
        "reuse": 1,
        "fresh": 1,
    }


def test_serialized_targets_contain_budget_but_no_hindsight_routing_labels() -> None:
    payload = analyze_oracle(_pilot_rows()).to_dict()

    assert payload["offline_diagnostic_only"] is True
    assert payload["deployable_routing_labels_emitted"] is False
    first_target = payload["targets"][0]
    assert first_target["memory_snapshot_id"] == "memory-test-v1"
    assert first_target["candidate_ranks_evaluated"] == [1, 2]
    assert first_target["candidates_evaluated"] == [
        {"candidate_rank": 1},
        {"candidate_rank": 2},
    ]
    assert first_target["retrieval_query_count"] == 1
    assert first_target["requested_top_k"] == 2
    assert first_target["context_token_budget"] == 2048
    assert "policy_action" not in first_target
    assert "best_experience_id" not in first_target
    assert "best_candidate_rank" not in first_target


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "at least one row"),
        ([_row("q", "e", 1, 0.4, 0.5), _row("q", "e", 2, 0.4, 0.6)], "duplicate"),
        (
            [_row("q", "e1", 1, 0.4, 0.5), _row("q", "e2", 2, 0.3, 0.6)],
            "DIRECT score is inconsistent",
        ),
        (
            [_row("q", "e1", 1, 0.4, 0.5, 0.7), _row("q", "e2", 2, 0.4, 0.6)],
            "FRESH score/presence is inconsistent",
        ),
        (
            [
                _row("q", "e1", 1, 0.4, 0.5),
                _row("q", "e2", 2, 0.4, 0.6, retrieval_query_count=2),
            ],
            "execution budget is inconsistent",
        ),
        (
            [_row("q", "e1", 1, 0.4, 0.5), _row("q", "e2", 1, 0.4, 0.6)],
            "candidate_rank is duplicated",
        ),
        (
            [
                _row("q1", "e1", 1, 0.4, 0.5),
                _row("q2", "e2", 1, 0.4, 0.6, memory_snapshot_id="memory-test-v2"),
            ],
            "exactly one memory_snapshot_id",
        ),
    ],
)
def test_oracle_rejects_invalid_collections(rows: list[OracleInputRow], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_oracle(rows)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_query_id", " "),
        ("memory_snapshot_id", ""),
        ("candidate_rank", 0),
        ("retrieval_query_count", 0),
        ("requested_top_k", 0),
        ("context_token_budget", -1),
        ("direct_score", -0.1),
        ("reuse_score", 1.1),
        ("direct_score", float("nan")),
    ],
)
def test_input_row_rejects_invalid_identifiers_scores_and_budgets(
    field: str, value: object
) -> None:
    kwargs: dict[str, object] = {
        "target_query_id": "q",
        "experience_id": "e",
        "memory_snapshot_id": "memory-test-v1",
        "candidate_rank": 1,
        "direct_score": 0.2,
        "reuse_score": 0.3,
        "retrieval_query_count": 1,
        "requested_top_k": 2,
        "context_token_budget": 2048,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        OracleInputRow(**kwargs)  # type: ignore[arg-type]


def test_csv_reader_and_cli_write_explicit_offline_json(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "examples" / "oracle_pilot.csv"
    rows = read_oracle_csv(source)
    assert len(rows) == 6

    output = tmp_path / "nested" / "oracle.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_oracle_pilot.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--good-threshold",
            "0.5",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "offline_candidate_set_oracle_report"
    assert payload["memory_snapshot_id"] == "memory-demo-v1"
    assert payload["candidate_count"] == 6
    assert payload["offline_policy_oracle_action_counts"] == {
        "direct": 1,
        "reuse": 1,
        "fresh": 1,
    }
    assert payload["source_csv"] == str(source.resolve())


def test_csv_reader_rejects_unexpected_schema(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(
        "target_query_id,experience_id,memory_snapshot_id,candidate_rank,direct_score,"
        "reuse_score,retrieval_query_count,requested_top_k,context_token_budget,notes\n"
        "q,e,memory-v1,1,0.2,0.3,1,1,1024,unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected columns"):
        read_oracle_csv(source)
