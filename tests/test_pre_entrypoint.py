"""Launch safety and new comparison states, all without external network."""

import json
from dataclasses import replace

import pytest
from test_experiment_api_client import completion, config, fake_provider, messages
from test_query_comparison import Factory, make_spec

from growrag.experiments.api_client import LiveChatClient
from growrag.experiments.comparison_report import comparison_report
from growrag.experiments.protocol import Action, ExecutionKind
from growrag.experiments.query_comparison import run_query_comparison
from growrag.experiments.run_pre_pilot import main


def test_missing_memory_has_no_reuse_factory_call_or_fabricated_result():
    factory = Factory()
    persisted = []
    run = run_query_comparison(
        replace(make_spec(), memory=None),
        factory,
        execution_kind=ExecutionKind.MOCK,
        on_arm=persisted.append,
    )
    assert len(run.arms) == len(persisted) == 3
    assert Action.REUSE not in factory.bundles
    row = comparison_report(run)["arms"]["REUSE"]
    assert row["stop_reason"] == "no_eligible_memory"
    assert row["completed"] is False
    assert row["answer"] is None
    assert row["usage"]["api_requests"] == 0
    assert run.to_dict()["memory_available"] is False


def test_arm_persistence_failure_prevents_further_execution():
    factory = Factory()

    def broken_save(arm):
        raise OSError("fixture disk full")

    with pytest.raises(OSError):
        run_query_comparison(
            make_spec(), factory, execution_kind=ExecutionKind.MOCK, on_arm=broken_save
        )
    assert sum(len(bundle.backend.requests) for bundle in factory.bundles.values()) == 1


def test_dry_launch_does_not_read_secret_or_create_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "growrag.experiments.run_pre_pilot.load_pre_examples", lambda *a: ((), (), {})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("dry launch must not read credentials")

    monkeypatch.setattr("growrag.experiments.run_pre_pilot.read_local_bailian_settings", forbidden)
    output = tmp_path / "dry"
    assert main(["--manifest", "fixture", "--api-config", "secret", "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["limits"]["budget_cny"] == 5
    assert plan["max_api_calls"] == 256
    assert plan["temperature"] == 0
    assert not output.exists()


def test_dirty_live_launch_fails_before_reading_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "growrag.experiments.run_pre_pilot.load_pre_examples", lambda *a: ((), (), {})
    )
    monkeypatch.setattr(
        "growrag.experiments.run_pre_pilot._git_state",
        lambda: {"commit": "x", "worktree_dirty": True},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("dirty launch must not read credentials")

    monkeypatch.setattr("growrag.experiments.run_pre_pilot.read_local_bailian_settings", forbidden)
    with pytest.raises(ValueError, match="commit reviewed"):
        main(
            [
                "--manifest",
                "fixture",
                "--api-config",
                "secret",
                "--output",
                str(tmp_path / "run"),
                "--allow-network",
            ]
        )


@pytest.mark.parametrize("temperature", [True, -1, 2, float("nan"), "0"])
def test_temperature_rejects_invalid_values(temperature):
    with pytest.raises(ValueError):
        config(temperature=temperature)


def test_fixed_temperature_is_sent_and_audited(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, completion())
    client = LiveChatClient(config(temperature=0.0), tmp_path, allow_network=True)
    response = client.complete(messages(), trace_id="temperature", prompt_version="test")
    assert json.loads(calls[0][0].data)["temperature"] == 0.0
    audit = json.loads(response.audit_path.read_text(encoding="utf-8"))
    assert audit["request"]["temperature"] == 0.0
