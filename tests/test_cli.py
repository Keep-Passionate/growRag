from __future__ import annotations

import json
from pathlib import Path

from growrag.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_DEMO = PROJECT_ROOT / "examples" / "trusted_reuse"


def test_demo_command_emits_a_reuse_decision(capsys) -> None:
    assert main(["demo"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["executed_action"] == "reuse"
    assert payload["executed_query"] == "Carol birth date; David birth date"


def test_oracle_command_writes_a_marked_offline_report(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "pairs.csv"
    source.write_text(
        "target_query_id,experience_id,memory_snapshot_id,candidate_rank,"
        "retrieval_query_count,requested_top_k,context_token_budget,"
        "direct_score,reuse_score,fresh_score\n"
        "q1,e1,snapshot-v1,1,1,5,2048,0.2,0.8,0.7\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    assert main(["oracle", "--input", str(source), "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["offline_diagnostic_only"] is True
    assert payload["helpful_coverage"] == 1.0
    assert str(output.resolve()) in capsys.readouterr().out


def test_file_demo_validates_and_emits_reuse_and_direct(tmp_path: Path, capsys) -> None:
    config = FILE_DEMO / "config.toml"
    memory = FILE_DEMO / "memory.json"
    output = tmp_path / "decisions.jsonl"

    assert main(["config-validate", "--config", str(config)]) == 0
    assert main(["memory-validate", "--memory", str(memory)]) == 0
    assert (
        main(
            [
                "decide",
                "--config",
                str(config),
                "--memory",
                str(memory),
                "--requests",
                str(FILE_DEMO / "requests.jsonl"),
                "--plans",
                str(FILE_DEMO / "plans.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()

    artifacts = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    actions = {
        artifact["request"]["target_query_id"]: artifact["decision"]["executed_action"]
        for artifact in artifacts
    }
    assert actions == {
        "target-reuse": "reuse",
        "target-not-applicable": "direct",
        "target-missing-plan": "direct",
    }
    assert artifacts[2]["decision"]["reason"] == "invalid_query_plan_fallback"
    forbidden = {"gold", "answer", "retrieval_results", "paired_outcome"}
    assert all(_nested_keys(artifact).isdisjoint(forbidden) for artifact in artifacts)


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _nested_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()
