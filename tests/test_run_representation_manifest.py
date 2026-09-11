import hashlib
import json

import pytest

from growrag.experiments import run_representation_manifest as cli


def raw_record(index, kind):
    return {
        "_id": f"fixture-{kind}-{index}",
        "question": f"Where does fixture person {index} of type {kind} work?",
        "answer": "PRIVATE GOLD",
        "supporting_facts": [["Evidence", 0]],
        "context": [["Evidence", ["PRIVATE CONTEXT"]]],
        "type": kind,
        "level": "hard",
    }


@pytest.fixture
def files(tmp_path):
    records = [raw_record(index, kind) for index in range(150) for kind in ("bridge", "comparison")]
    data = tmp_path / "train.json"
    preview = tmp_path / "preview.json"
    data.write_text(json.dumps(records), encoding="utf-8")
    preview.write_text(json.dumps(records[:8]), encoding="utf-8")
    return data, preview, records


def run(files, output, **kwargs):
    data, preview, records = files
    options = {
        "expected_row_count": len(records),
        "synthetic": True,
        "source_count": 4,
        "source_cap": 8,
        "target_count": 4,
        "debug_count": 2,
    }
    options.update(kwargs)
    return cli.write_representation_manifest(data, output, preview, **options)


def test_writes_bound_full_source_and_selected_hashes(files, tmp_path):
    output = tmp_path / "new-artifacts"
    result = run(files, output)
    data, _, records = files
    selected_bytes = (output / "selected_records.json").read_bytes()
    selected = json.loads(selected_bytes)
    assert result["source_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert result["selected_records_sha256"] == hashlib.sha256(selected_bytes).hexdigest()
    assert result["source_sha256"] != result["selected_records_sha256"]
    assert result["selected_records_count"] == 12
    assert [record["_id"] for record in selected] == result["source_expansion_order"] + result[
        "selected"
    ]["target"]
    assert all(record in records for record in selected)
    assert result["selected_records_contains_gold"] is True
    assert result["selected_records_runtime_safe"] is False
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == result
    assert "PRIVATE GOLD" not in json.dumps(result)
    assert "PRIVATE CONTEXT" not in json.dumps(result)
    assert result["synthetic_data"] is True
    assert result["source_url"] is None


def test_all_preview_questions_excluded_and_parser_sees_empty_context(files, tmp_path, monkeypatch):
    original_parser = cli.parse_hotpot_example
    seen = []

    def metadata_parser(record, **kwargs):
        seen.append(record["context"])
        assert record["context"] == []
        return original_parser(record, **kwargs)

    monkeypatch.setattr(cli, "parse_hotpot_example", metadata_parser)
    result = run(files, tmp_path / "out")
    assert len(seen) == len(files[2])
    excluded_ids = {row["_id"] for row in files[2][:8]}
    assert set(result["exclusions"]["excluded_union_question_ids"]) == excluded_ids
    assert excluded_ids.isdisjoint(result["source_expansion_order"] + result["selected"]["target"])
    assert result["prior_exposure_exclusion"]["preview_record_count"] == 8


def test_default_rejects_partial_file_without_writing(files, tmp_path):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="expected 90447"):
        cli.write_representation_manifest(files[0], output, files[1])
    assert not output.exists()


def test_small_count_requires_explicit_synthetic_mode(files, tmp_path):
    with pytest.raises(ValueError, match="production requires expected_row_count=90447"):
        run(files, tmp_path / "out", synthetic=False)
    assert not (tmp_path / "out").exists()


def test_row_count_mismatch_rejected_even_in_synthetic_mode(files, tmp_path):
    with pytest.raises(ValueError, match="expected 301"):
        run(files, tmp_path / "out", expected_row_count=301)
    assert not (tmp_path / "out").exists()


def test_existing_directory_not_overwritten(files, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "manifest.json"
    marker.write_text("DO NOT OVERWRITE", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(files, output)
    assert marker.read_text(encoding="utf-8") == "DO NOT OVERWRITE"


@pytest.mark.parametrize(
    "bad_context", [[], None, [["Evidence", [3]]], [["Evidence", []], ["Evidence", []]]]
)
def test_invalid_raw_context_rejected_before_projection(files, tmp_path, bad_context):
    data, _, records = files
    records[0]["context"] = bad_context
    data.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError):
        run(files, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_production_checks_all_preview_rows(files, tmp_path, monkeypatch):
    # Exercise the production branch with a declared synthetic fixture count,
    # not by pretending this helper test downloaded real official records.
    monkeypatch.setattr(cli, "EXPECTED_TRAIN_ROWS", len(files[2]))
    with pytest.raises(ValueError, match="all 200"):
        run(files, tmp_path / "out", synthetic=False)
    assert not (tmp_path / "out").exists()


def test_duplicate_normalized_groups_are_all_excluded(files, tmp_path):
    data, _, records = files
    duplicate = dict(records[20], _id="duplicate-question-new-id")
    duplicate["question"] = duplicate["question"].upper()
    records.append(duplicate)
    data.write_text(json.dumps(records), encoding="utf-8")
    result = run(files, tmp_path / "out")
    assert result["exclusions"]["duplicate_policy"] == "exclude_all"
    assert result["exclusions"]["duplicate_normalized_member_count"] == 2
    assert {records[20]["_id"], duplicate["_id"]} <= set(
        result["exclusions"]["excluded_union_question_ids"]
    )


def test_cli_runs_on_explicit_local_synthetic_fixture(files, tmp_path, monkeypatch, capsys):
    output = tmp_path / "cli-out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "manifest",
            "--data",
            str(files[0]),
            "--output",
            str(output),
            "--exclude-preview",
            str(files[1]),
            "--synthetic",
            "--expected-row-count",
            str(len(files[2])),
            "--source-count",
            "4",
            "--source-cap",
            "8",
            "--target-count",
            "4",
            "--debug-count",
            "2",
        ],
    )
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["synthetic_data"] is True
    assert (output / "manifest.json").is_file()
    assert "PRIVATE GOLD" not in json.dumps(printed)


def mirror_metadata(files, tmp_path):
    metadata = {
        "schema_version": "growrag-hotpot-mirror-provenance-v1",
        "dataset": "hotpotqa/hotpot_qa",
        "configuration": "distractor",
        "official_split": "train",
        "row_count": len(files[2]),
        "output_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "original_cmu_json_bytes": False,
        "source_url": (
            "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/train"
        ),
    }
    path = tmp_path / "mirror_provenance.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path, metadata


def test_mirror_provenance_binds_actual_source_without_changing_split_label(
    files, tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "EXPECTED_TRAIN_ROWS", len(files[2]))
    monkeypatch.setattr(cli, "EXPECTED_PREVIEW_ROWS", 8)
    path, metadata = mirror_metadata(files, tmp_path)
    result = run(files, tmp_path / "out", synthetic=False, source_provenance_path=path)
    assert result["input_dataset_label"] == "hotpotqa-distractor-train-v1.1"
    assert result["data_version"] == "HotpotQA v1.1 train"
    assert result["source_url"] == metadata["source_url"]
    assert result["original_cmu_json_bytes"] is False
    assert result["source_provenance_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["source_provenance"] == metadata
    assert "NOT original CMU JSON bytes" in result["provenance_notice"]


@pytest.mark.parametrize("field,bad", [("output_sha256", "0" * 64), ("official_split", "dev")])
def test_mismatched_mirror_provenance_rejected(files, tmp_path, monkeypatch, field, bad):
    monkeypatch.setattr(cli, "EXPECTED_TRAIN_ROWS", len(files[2]))
    monkeypatch.setattr(cli, "EXPECTED_PREVIEW_ROWS", 8)
    path, metadata = mirror_metadata(files, tmp_path)
    metadata[field] = bad
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="must match the complete train"):
        run(files, tmp_path / "out", synthetic=False, source_provenance_path=path)
    assert not (tmp_path / "out").exists()


def test_synthetic_cannot_claim_real_mirror_provenance(files, tmp_path):
    path, _ = mirror_metadata(files, tmp_path)
    with pytest.raises(ValueError, match="synthetic fixtures cannot claim"):
        run(files, tmp_path / "out", source_provenance_path=path)
