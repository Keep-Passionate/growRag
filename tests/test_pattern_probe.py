"""Offline diagnostic contracts; fixtures do not measure real RAG quality."""

import json
from dataclasses import asdict

import pytest
from test_pattern_matcher import make_view, question, scope
from test_pre_manifest import _records, _write_original

from growrag.experiments.pattern_probe import diagnose, load_views, main


def library(tmp_path, views=None):
    path = tmp_path / "library.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "growrag-pre-frozen-library-v1",
                "views": [asdict(view) for view in (views or [make_view()])],
            }
        ),
        encoding="utf-8",
    )
    return path


def arguments(path, output):
    return ["--library", str(path), "--declare-age-scope", "age-card@v1", "--output", str(output)]


def test_compiled_view_roundtrip_and_hash(tmp_path):
    views, fingerprint = load_views(library(tmp_path))
    assert views == (make_view(),)
    assert len(fingerprint) == 64


def test_duplicate_identity_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate compiled"):
        load_views(library(tmp_path, [make_view(), make_view()]))


def test_duplicate_json_field_refused(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"views":[],"views":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate library JSON"):
        load_views(path)


@pytest.mark.parametrize("field,value", [("schema_version", "unknown"), ("views", {})])
def test_invalid_library_refused(tmp_path, field, value):
    path = library(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_views(path)


def test_diagnostic_does_not_upgrade_candidate_to_reuse():
    view = make_view()
    result = diagnose((question(),), ((view, scope(view)),), origin="synthetic")
    assert result["candidate_count"] == 1
    assert result["rows"][0]["route"] == "CANDIDATE"
    assert "answer" not in result["rows"][0]


def test_cli_has_no_model_calls_and_does_not_change_library(tmp_path, monkeypatch):
    from growrag.experiments import api_client, api_preflight

    def forbidden(*args, **kwargs):
        raise AssertionError("offline probe must not read keys or invoke the API")

    monkeypatch.setattr(api_client.LiveChatClient, "complete", forbidden)
    monkeypatch.setattr(api_preflight, "read_local_bailian_settings", forbidden)
    path, output = library(tmp_path), tmp_path / "report.json"
    original = path.read_bytes()
    assert main(arguments(path, output)) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["api_requests"] == report["rag_calls"] == 0
    assert report["answer_effect"] == "unknown"
    assert report["scope"]["review_origin"] == "developer_provisional"
    assert report["groups"][0]["question_count"] == 9
    assert report["groups"][0]["candidate_count"] == 3
    assert report["groups"][0]["base_count"] == 6
    assert path.read_bytes() == original


def test_manifest_targets_separate_from_synthetic_without_answers(tmp_path):
    records = _records()
    for record in records:
        record["answer"] = "UNIQUE_GOLD_NEVER_TO_PROBE"
    manifest, _, _ = _write_original(tmp_path, records)
    output = tmp_path / "report.json"
    assert main(arguments(library(tmp_path), output) + ["--manifest", str(manifest)]) == 0
    encoded = output.read_text(encoding="utf-8")
    report = json.loads(encoded)
    assert "UNIQUE_GOLD_NEVER_TO_PROBE" not in encoded
    assert len(report["groups"]) == 2
    assert report["groups"][1]["question_count"] == 16
    assert report["groups"][1]["base_count"] == 16
    assert report["groups"][1]["origin"].endswith("query_only_posthoc")


def test_nonexistent_selected_memory_stops_before_output(tmp_path):
    output = tmp_path / "report.json"
    args = arguments(library(tmp_path), output)
    args[3] = "not-in-library"
    with pytest.raises(ValueError, match="not in this library"):
        main(args)
    assert not output.exists()


def test_existing_output_stops_before_reading_library(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(arguments(tmp_path / "missing-library.json", output))
    assert output.read_text(encoding="utf-8") == "preserve"


def test_scope_declaration_must_be_explicit(tmp_path):
    with pytest.raises(SystemExit):
        main(["--library", str(library(tmp_path)), "--output", str(tmp_path / "report.json")])
