"""Archive safety and temporal evidence checks using synthetic files only."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/audits/export_s2g_development_handoff.py"
SPEC = importlib.util.spec_from_file_location("s2g_handoff", SCRIPT)
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_project(tmp_path):
    project = tmp_path / "project"
    folder = project / Path(handoff.MANIFEST).parent
    folder.mkdir(parents=True)
    artifacts = {}
    for name in ("corpus.jsonl", "runtime_questions.jsonl"):
        path = folder / name
        path.write_text("{}\n", encoding="utf-8")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": handoff.digest(path),
            "runtime_safe": True,
            "contains_gold": False,
        }
    index = project / handoff.INDEX
    index.parent.mkdir(parents=True)
    index.write_bytes(b"synthetic index, never opened as database")
    ids = [f"q{i:03d}" for i in range(500)]
    manifest = {
        "official_split": "train",
        "role": "development",
        "official_dev_test_used": False,
        "question_ids": ids,
        "artifacts": artifacts,
        "runtime_input_allowlist": list(artifacts),
    }
    write_json(project / handoff.MANIFEST, manifest)
    run_id = handoff.SERIES + "_0000_0500"
    batch = project / "runs" / run_id
    source_files = {
        name: {"text": "# frozen\n", "sha256": hashlib.sha256(b"# frozen\n").hexdigest()}
        for name in handoff.MODULES
    }
    snapshot = {"files": source_files, "sha256": handoff.fingerprint(source_files)}
    write_json(batch / "source_snapshot.json", snapshot)
    plan = {
        "run_id": run_id,
        "manifest_path": str(project / handoff.MANIFEST),
        "manifest_sha256": handoff.digest(project / handoff.MANIFEST),
        "start": 0,
        "count": 500,
        "question_ids": ids,
        "index_path": str(index),
        "runtime_metadata": {
            "manifest_sha256": handoff.digest(project / handoff.MANIFEST),
            "corpus_sha256": artifacts["corpus.jsonl"]["sha256"],
        },
        "source_sha256": snapshot["sha256"],
        "author": {},
        "historical_budget": {"secret": "do not export"},
    }
    write_json(batch / "launch_plan.json", plan)

    def event(kind, **payload):
        qid = (
            "q000"
            if kind in {"arm_start", "question_complete"}
            else run_id + "/q000/" + handoff.ARM
        )
        return {"arm": handoff.ARM, "question_id": qid, "kind": kind, **payload}

    doc = {
        "doc_id": "d1",
        "title": "Title",
        "text": "A public passage.",
        "extra_private_field": "excluded",
    }
    events = [
        event("arm_start"),
        event("start", question="Who founded A?"),
        event(
            "judge",
            evidence_contexts=[""],
            verdicts=[False],
            gap_items=[[{"target": "A", "slot": "founder", "description": "Need founder"}]],
        ),
        event(
            "query",
            query="A founder",
            gap_items=[{"target": "A", "slot": "founder"}],
            gap_profile="paper_k1",
        ),
        event("retrieval", round=1, queries=["A founder"], documents=[[doc]]),
        event(
            "extraction",
            round=1,
            pointers=[[[1]]],
            sources=[
                {"doc_id": "d1", "title": "Title", "sentence_id": 1, "text": "A public passage."}
            ],
        ),
        event(
            "judge",
            evidence_contexts=["Only selected sentence, not full docs"],
            verdicts=[True],
            gap_items=[[]],
        ),
        event("finish", answer="DO NOT EXPORT ANSWER", stop_reason="sufficient", api_calls=4),
        event("question_complete", status="completed", requests=4),
    ]
    # Invalid JSON after API kind proves its payload is never deserialized.
    api = '{"arm":"S2G_AUTHOR_API4","kind":"api_request","messages":DO_NOT_PARSE}\n'
    (batch / "events.jsonl").write_text(
        api + "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    return project, batch, plan


def test_skip_raw_api_payload_before_parse():
    assert handoff.selected_event('{"kind":"api_response","content": INVALID JSON}') is None
    with pytest.raises(ValueError, match="before event kind"):
        handoff.selected_event('{"content":"nested kind confusion","kind":"judge"}')


def test_finish_answer_is_never_decoded():
    event = handoff.selected_event(
        '{"arm":"S2G_AUTHOR_API4","kind":"finish",'
        '"question_id":"run/q/arm","answer":INVALID_JSON_VALUE,'
        '"stop_reason":"sufficient","api_calls":4}'
    )
    assert event["stop_reason"] == "sufficient"
    assert "answer" not in event


def test_fixed_selection_temporal_evidence_and_completeness(tmp_path):
    project, _, _ = fixture_project(tmp_path)
    summary, inventory, traces = handoff.build_package(project, 10)
    assert summary["selected_question_ids"] == [f"q{i:03d}" for i in range(10)]
    assert len(traces) == 10  # Includes the nine unstarted selections, no success filter.
    assert summary["counts"]["started_questions"] == 1
    assert summary["counts"]["not_started_questions"] == 499
    first = traces[0]
    judges = [e for e in first["events"] if e["kind"] == "judge"]
    assert judges[0]["evidence_contexts"] == [""]
    assert judges[0]["prior_retrieved_doc_ids"] == []
    assert judges[1]["evidence_contexts"] == ["Only selected sentence, not full docs"]
    assert judges[1]["prior_retrieved_doc_ids"] == ["d1"]
    retrieval = next(e for e in first["events"] if e["kind"] == "retrieval")
    assert retrieval["prior_retrieved_doc_ids"] == []
    assert "extra_private_field" not in retrieval["documents"][0][0]
    assert inventory[0]["complete_trajectory_present"]
    assert not inventory[1]["complete_trajectory_present"]
    text = json.dumps((summary, inventory, traces))
    assert "DO NOT EXPORT ANSWER" not in text
    assert "historical_budget" not in text
    assert "do not export" not in text


def test_wrong_manifest_slice_rejected_before_output(tmp_path):
    project, batch, plan = fixture_project(tmp_path)
    plan["question_ids"][0:2] = reversed(plan["question_ids"][0:2])
    write_json(batch / "launch_plan.json", plan)
    output = project / "runs" / "export"
    with pytest.raises(ValueError, match="manifest slice"):
        handoff.export_package(project, output)
    assert not output.exists()


def test_wrong_trace_id_not_normalized_generically(tmp_path):
    project, batch, _ = fixture_project(tmp_path)
    with (batch / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "arm": handoff.ARM,
                    "kind": "judge",
                    "question_id": "another-run/q000/" + handoff.ARM,
                }
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="trace ID"):
        handoff.build_package(project)


def test_snapshot_content_must_match_individual_and_aggregate_hashes(tmp_path):
    project, batch, plan = fixture_project(tmp_path)
    snapshot = handoff.read_json(batch / "source_snapshot.json")
    snapshot["files"][handoff.MODULES[0]]["text"] = "# changed"
    snapshot["sha256"] = handoff.fingerprint(snapshot["files"])
    plan["source_sha256"] = snapshot["sha256"]
    write_json(batch / "source_snapshot.json", snapshot)
    write_json(batch / "launch_plan.json", plan)
    with pytest.raises(ValueError, match="module mismatch"):
        handoff.build_package(project)


def test_export_is_exclusive_local_and_hashes_artifacts(tmp_path):
    project, _, _ = fixture_project(tmp_path)
    output = project / "runs" / "export"
    result = handoff.export_package(project, output)
    manifest = handoff.read_json(output / "package_manifest.json")
    assert result["sample_count"] == 10
    assert manifest["artifacts"]["selected_traces.jsonl"]["sha256"] == handoff.digest(
        output / "selected_traces.jsonl"
    )
    with pytest.raises(ValueError, match="already exists"):
        handoff.export_package(project, output)
    with pytest.raises(ValueError, match="runs subdirectory"):
        handoff.export_package(project, project / "public_export")


def test_never_open_disallowed_sources(tmp_path, monkeypatch):
    project, _, _ = fixture_project(tmp_path)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name not in {"gold.jsonl", "qwenAPI.md"}
        assert not ({"api_audit", "request_journal"} & set(path.parts))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    handoff.build_package(project)


def test_official_dev_role_rejected(tmp_path):
    project, _, _ = fixture_project(tmp_path)
    path = project / handoff.MANIFEST
    manifest = handoff.read_json(path)
    manifest["official_split"] = "dev"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="train/development"):
        handoff.build_package(project)
