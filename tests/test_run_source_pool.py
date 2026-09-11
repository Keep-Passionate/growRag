"""Source launcher safety; synthetic fixtures and stubs only, never live APIs."""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest
from test_source_pool_pilot import population as population
from test_source_pool_pilot import source_batch

from growrag.experiments import run_source_pool as launch

SECRET = "synthetic-key-not-a-real-credential"


def _raw(example):
    return {
        "_id": example.question.question_id,
        "question": example.question.text,
        "answer": example.gold.answers[0],
        "supporting_facts": [list(fact) for fact in example.gold.supporting_facts],
        "type": example.question_type,
        "context": [[item.title, [item.text]] for item in example.candidate_context],
    }


@pytest.fixture
def files(tmp_path, population):
    sources, manifest = source_batch(population, 64)
    raw_path = tmp_path / "hotpot_train_v1.1.json"
    raw_path.write_text(json.dumps([_raw(e) for e in population]), encoding="utf-8")
    selected_ids = manifest["source_expansion_order"] + manifest["selected"]["target"]
    by_id = {e.question.question_id: e for e in population}
    selected_path = tmp_path / "selected_records.json"
    selected_path.write_text(
        json.dumps([_raw(by_id[qid]) for qid in selected_ids]), encoding="utf-8"
    )
    manifest.update(
        input_path=str(raw_path),
        source_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        synthetic_data=True,
        source_bytes_verified_by_builder=True,
        selected_records_file=selected_path.name,
        selected_records_count=len(selected_ids),
        selected_records_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return sources, manifest, manifest_path, selected_path, raw_path


def test_offline_loader_checks_both_hashes_and_only_parses_sources(files, monkeypatch):
    sources, manifest, manifest_path, selected_path, _ = files
    records = json.loads(selected_path.read_text(encoding="utf-8"))
    # Gold/contexts of reserved targets deliberately cannot be parsed as Hotpot.
    # Their IDs remain present for artifact-order checks, not target execution.
    for record in records[64:]:
        record["answer"] = ["TARGET_GOLD_MUST_NOT_BE_PARSED"]
        record["supporting_facts"] = "INVALID_TARGET_GOLD"
        record["context"] = None
    selected_path.write_text(json.dumps(records), encoding="utf-8")
    manifest["selected_records_sha256"] = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    called = []
    original = launch.parse_hotpot_example

    def spy(record, **kwargs):
        called.append(record["_id"])
        return original(record, **kwargs)

    monkeypatch.setattr(launch, "parse_hotpot_example", spy)
    loaded, actual, verified = launch.load_source_pool_examples(
        manifest_path, selected_path, allow_synthetic=True
    )
    assert len(loaded) == 64
    assert called == [source.question.question_id for source in sources]
    assert actual == manifest
    assert verified["original_bytes_reverified"]
    assert verified["selected_bytes_reverified"]
    assert verified["target_gold_parsed"] is False
    assert verified["official_authenticity_independently_verified"] is False


@pytest.mark.parametrize("changed", ["original", "selected"])
def test_tampered_original_or_selected_bytes_are_rejected(files, changed):
    _, _, manifest_path, selected_path, raw_path = files
    path = raw_path if changed == "original" else selected_path
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        launch.load_source_pool_examples(manifest_path, selected_path, allow_synthetic=True)


@pytest.mark.parametrize("problem", ["count", "source_order", "selected_order", "role"])
def test_selection_tampering_fails_before_calls(files, problem):
    _, manifest, manifest_path, selected_path, _ = files
    if problem == "count":
        manifest["selected"]["source"] = manifest["source_expansion_order"][:32]
    elif problem == "source_order":
        manifest["selected"]["source"].reverse()
    elif problem == "selected_order":
        records = json.loads(selected_path.read_text(encoding="utf-8"))
        records.reverse()
        selected_path.write_text(json.dumps(records), encoding="utf-8")
        manifest["selected_records_sha256"] = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    else:
        manifest["selected_roles"]["source"] = "calibration_dev"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        launch.load_source_pool_examples(manifest_path, selected_path, allow_synthetic=True)


def test_production_loader_rejects_synthetic_files(files):
    _, _, manifest_path, selected_path, _ = files
    with pytest.raises(ValueError, match="synthetic"):
        launch.load_source_pool_examples(manifest_path, selected_path)


def _stub_loader(monkeypatch, files):
    sources, manifest, *_ = files
    monkeypatch.setattr(launch, "load_source_pool_examples", lambda *args: (sources, manifest, {}))
    monkeypatch.setattr(launch, "_git_state", lambda: {"commit": "f" * 40, "worktree_dirty": False})


def _args(output, *, live=False):
    args = [
        "--manifest",
        "FIXTURE_MANIFEST",
        "--data",
        "FIXTURE_SELECTED",
        "--output",
        str(output),
        "--api-config",
        "DO_NOT_READ_REAL_CONFIG",
    ]
    return args + ["--allow-network"] if live else args


def _forbidden(*args, **kwargs):
    pytest.fail("credentials or network must not be accessed here")


def test_dry_plan_has_fixed_limits_and_neither_reads_key_nor_writes(
    files, tmp_path, monkeypatch, capsys
):
    _stub_loader(monkeypatch, files)
    monkeypatch.setattr(launch, "read_local_bailian_settings", _forbidden)
    monkeypatch.setattr(launch, "LiveChatClient", _forbidden)
    output = tmp_path / "dry"
    assert launch.main(_args(output)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["source_count"] == 64 and plan["target_count"] == 0
    assert plan["authorized_batch_budget_cny"] == 5
    assert plan["automatic_expansion"] is False
    assert plan["model"] == "qwen3.7-flash-2026-07-15"
    assert plan["max_api_calls"] == 256 and plan["max_output_tokens"] == 768
    assert plan["limits"]["input_per_million_cny"] == 0.2
    assert plan["limits"]["output_per_million_cny"] == 0.8
    assert plan["limits"]["max_prompt_bytes"] == 24000
    assert plan["enable_thinking"] is False and plan["temperature"] == 0
    assert plan["pricing_verified_date"] == "2026-09-11"
    assert not output.exists()


@pytest.mark.parametrize(
    "git_state",
    [
        {"commit": None, "worktree_dirty": False},
        {"commit": "f" * 40, "worktree_dirty": True},
    ],
)
def test_unfrozen_code_is_rejected_before_credentials(files, tmp_path, monkeypatch, git_state):
    _stub_loader(monkeypatch, files)
    monkeypatch.setattr(launch, "_git_state", lambda: git_state)
    monkeypatch.setattr(launch, "read_local_bailian_settings", _forbidden)
    with pytest.raises(ValueError, match="commit reviewed"):
        launch.main(_args(tmp_path / "live", live=True))


def test_existing_output_rejected_before_credentials(files, tmp_path, monkeypatch):
    _stub_loader(monkeypatch, files)
    monkeypatch.setattr(launch, "read_local_bailian_settings", _forbidden)
    with pytest.raises(FileExistsError):
        launch.main(_args(tmp_path, live=True))


def test_non_beijing_endpoint_rejected_before_client_creation(files, tmp_path, monkeypatch):
    _stub_loader(monkeypatch, files)
    monkeypatch.setattr(
        launch,
        "read_local_bailian_settings",
        lambda *args: SimpleNamespace(
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            api_key=SECRET,
        ),
    )
    monkeypatch.setattr(launch, "LiveChatClient", _forbidden)
    with pytest.raises(ValueError, match="Beijing"):
        launch.main(_args(tmp_path / "live", live=True))


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("previous", [None, "previous-test-value"])
def test_stubbed_launch_restores_secret_and_sanitizes_failures(
    files, tmp_path, monkeypatch, capsys, failure, previous
):
    _stub_loader(monkeypatch, files)
    if previous is None:
        monkeypatch.delenv(launch.KEY_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(launch.KEY_VARIABLE, previous)
    monkeypatch.setattr(
        launch,
        "read_local_bailian_settings",
        lambda *args: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=SECRET,
        ),
    )
    transports = []

    def transport(config, directory, *, allow_network):
        assert allow_network is True
        assert config.enable_thinking is False and config.temperature == 0
        assert config.key_environment_variable == launch.KEY_VARIABLE
        value = SimpleNamespace(config=config, transport_source="live_api", attempts=0)
        transports.append(value)
        return value

    monkeypatch.setattr(launch, "LiveChatClient", transport)

    def runner(sources, client, directory, *, manifest, allow_real):
        assert os.environ[launch.KEY_VARIABLE] == SECRET
        assert len(sources) == 64 and allow_real is True
        assert client.limits.budget_cny == 5
        assert client.delegate is transports[0]
        if failure:
            raise RuntimeError(SECRET)
        return SimpleNamespace(
            summary={
                "status": "completed",
                "processed_source_count": 64,
                "candidate_count": 0,
            }
        )

    monkeypatch.setattr(launch, "run_source_pool", runner)
    output = tmp_path / "stubbed"
    assert launch.main(_args(output, live=True)) == (1 if failure else 0)
    assert os.environ.get(launch.KEY_VARIABLE) == previous
    assert (output / "final_budget.json").exists()
    text = capsys.readouterr().out
    assert SECRET not in text
    for path in output.rglob("*.json"):
        assert SECRET not in path.read_text(encoding="utf-8")
    if failure:
        error = json.loads((output / "interrupted.json").read_text(encoding="utf-8"))
        assert error["error_type"] == "RuntimeError"


def test_settings_error_body_not_printed(files, tmp_path, monkeypatch, capsys):
    _stub_loader(monkeypatch, files)

    def broken(*args):
        raise ValueError(SECRET)

    monkeypatch.setattr(launch, "read_local_bailian_settings", broken)
    assert launch.main(_args(tmp_path / "live", live=True)) == 2
    assert SECRET not in capsys.readouterr().out
