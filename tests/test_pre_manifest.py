"""Synthetic local manifest tests; no download or experiment API calls."""

import hashlib
import json

import pytest

from growrag.experiments.data_protocol import build_manifest, normalize_question, role_for_question
from growrag.experiments.pre_manifest import SCHEMA_VERSION, load_pre_examples


def _records():
    quotas = {"memory_seed": 163, "selector_train": 21, "calibration_dev": 16}
    counts = dict.fromkeys(quotas, 0)
    rows, index = [], 0
    while len(rows) < sum(quotas.values()):
        question = f"Which fictional event happened in year {index}?"
        role = role_for_question(question)
        if counts[role] < quotas[role]:
            rows.append(
                {
                    "_id": f"q-{index}",
                    "question": question,
                    "answer": "event",
                    "context": [["Synthetic", ["A fictional event."]]],
                    "supporting_facts": [["Synthetic", 0]],
                }
            )
            counts[role] += 1
        index += 1
    return rows


def _write_original(tmp_path, records=None):
    records = _records() if records is None else records
    raw = json.dumps(records).encode("utf-8")
    (tmp_path / "data.json").write_bytes(raw)
    original = build_manifest(records)
    original.update(data_file="data.json", data_sha256=hashlib.sha256(raw).hexdigest())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    return path, original, records


def test_default_selects_32_and_16_using_original_stable_role_order(tmp_path):
    path, original, _ = _write_original(tmp_path)
    sources, targets, manifest = load_pre_examples(path)
    assert [row.question.question_id for row in sources] == original["roles"]["memory_seed"][:32]
    assert [row.question.question_id for row in targets] == original["roles"]["calibration_dev"][
        :16
    ]
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["data_sha256"] == original["data_sha256"]
    assert manifest["source_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["record_count"] == 200
    assert manifest["role_counts"] == {
        "memory_seed": 163,
        "selector_train": 21,
        "calibration_dev": 16,
    }
    assert manifest["selected_counts"] == {"memory_seed": 32, "calibration_dev": 16}
    assert manifest["authorized_batch_budget_cny"] == 5.0
    assert manifest["official_validation_used"] is False
    assert manifest["official_test_used"] is False
    assert manifest["selector_train_used"] is False
    assert manifest["calibration_previously_exposed"] is True
    assert "not a held-out" in manifest["exposure_notice"]
    assert "convenience" in manifest["sampling_notice"]
    assert "SHA256(selection:" in manifest["selection_order"]
    source_ids = {row.question.question_id for row in sources}
    target_ids = {row.question.question_id for row in targets}
    assert not source_ids & target_ids
    assert not (source_ids | target_ids) & set(original["roles"]["selector_train"])
    for role, examples in (("memory_seed", sources), ("calibration_dev", targets)):
        for selected, example in zip(manifest["selected_queries"][role], examples, strict=True):
            assert selected["question_id"] == example.question.question_id
            assert (
                selected["normalized_question_sha256"]
                == hashlib.sha256(
                    normalize_question(example.question.text).encode("utf-8")
                ).hexdigest()
            )


def test_original_files_are_unchanged_and_repeated_selection_is_identical(tmp_path):
    path, _, _ = _write_original(tmp_path)
    before = {file.name: file.read_bytes() for file in tmp_path.iterdir()}
    first = load_pre_examples(path, source_count=3, target_count=2)
    second = load_pre_examples(path, source_count=3, target_count=2)
    assert first == second
    assert len(first[0]) == 3 and len(first[1]) == 2
    assert before == {file.name: file.read_bytes() for file in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "source_count,target_count",
    [(33, 16), (32, 17), (0, 1), (1, 0), (True, 1), (1, False), (1.5, 1), (1, -1)],
)
def test_requested_counts_are_strictly_positive_and_bounded(tmp_path, source_count, target_count):
    # Invalid counts must fail even before opening a nonexistent manifest.
    with pytest.raises(ValueError, match="integer between"):
        load_pre_examples(tmp_path / "missing.json", source_count, target_count)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "other"),
        ("dataset", "other/dataset"),
        ("official_split", "validation"),
        ("official_split", "test"),
        ("configuration", "fullwiki"),
        ("split_version", "other-version"),
        ("data_sha256", "changed"),
        ("record_count", 201),
        ("official_validation_used", True),
        ("official_test_used", True),
    ],
)
def test_corrupt_original_metadata_is_rejected(tmp_path, key, value):
    path, original, _ = _write_original(tmp_path)
    original[key] = value
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        load_pre_examples(path)


@pytest.mark.parametrize("change", ["roles", "selected", "counts", "weights", "seed"])
def test_all_original_split_metadata_is_recomputed(tmp_path, change):
    path, original, _ = _write_original(tmp_path)
    if change == "roles":
        original["roles"]["calibration_dev"][0] = original["roles"]["memory_seed"][0]
    elif change == "selected":
        original["selected"]["memory_seed"].reverse()
    elif change == "counts":
        original["role_counts"]["memory_seed"] = 162
    elif change == "weights":
        original["role_weights"]["memory_seed"] = 0.7
    else:
        original["seed"] = 43
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        load_pre_examples(path)


@pytest.mark.parametrize("filename", ["../outside.json", "subdir/data.json", ""])
def test_manifest_data_path_cannot_escape_or_select_another_directory(tmp_path, filename):
    path, original, _ = _write_original(tmp_path)
    original["data_file"] = filename
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="filename"):
        load_pre_examples(path)


def test_absolute_data_path_is_rejected_even_for_the_same_directory(tmp_path):
    path, original, _ = _write_original(tmp_path)
    original["data_file"] = str((tmp_path / "data.json").resolve())
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="filename"):
        load_pre_examples(path)


def test_duplicate_source_text_is_skipped_deterministically_without_using_another_role(tmp_path):
    path, original, records = _write_original(tmp_path)
    first, duplicate = original["roles"]["memory_seed"][:2]
    by_id = {row["_id"]: row for row in records}
    by_id[duplicate]["question"] = by_id[first]["question"].upper()
    path, original, _ = _write_original(tmp_path, records)
    sources, _, manifest = load_pre_examples(path)
    assert len(sources) == 32
    assert manifest["skipped_duplicate_question_ids"]["memory_seed"] == [duplicate]
    assert duplicate not in manifest["selected"]["memory_seed"]
    assert len({normalize_question(row.question.text) for row in sources}) == 32
    assert set(manifest["selected"]["memory_seed"]) <= set(original["roles"]["memory_seed"])


def test_insufficient_unique_targets_fail_instead_of_borrowing_other_roles(tmp_path):
    _, original, records = _write_original(tmp_path)
    first, duplicate = original["roles"]["calibration_dev"][:2]
    by_id = {row["_id"]: row for row in records}
    by_id[duplicate]["question"] = by_id[first]["question"]
    path, _, _ = _write_original(tmp_path, records)
    with pytest.raises(ValueError, match="calibration_dev lacks 16 unique"):
        load_pre_examples(path)


def test_changed_gold_does_not_change_selected_questions_or_order(tmp_path):
    path, _, records = _write_original(tmp_path)
    first = load_pre_examples(path)[2]
    for row in records:
        row["answer"] = "entirely different gold answer"
        row["supporting_facts"] = [["Different gold document", 5]]
    path, _, _ = _write_original(tmp_path, records)
    second = load_pre_examples(path)[2]
    assert first["selected"] == second["selected"]
    assert first["selected_queries"] == second["selected_queries"]
    assert first["data_sha256"] != second["data_sha256"]
    assert second["selection_uses_gold_outcomes"] is False


def test_duplicate_question_ids_are_rejected_even_after_rehashing_data(tmp_path):
    path, original, records = _write_original(tmp_path)
    records[1]["_id"] = records[0]["_id"]
    raw = json.dumps(records).encode()
    (tmp_path / "data.json").write_bytes(raw)
    original["data_sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="unique question IDs"):
        load_pre_examples(path)
