import pytest

from growrag.experiments.data_protocol import build_manifest, from_hf_row, role_for_question


def row(index):
    return {
        "id": f"q{index}",
        "question": f"Where did author {index} live?",
        "answer": "Place",
        "type": "bridge",
        "level": "hard",
        "context": {"title": ["A"], "sentences": [["Text"]]},
        "supporting_facts": {"title": ["A"], "sent_id": [0]},
    }


def test_normalized_duplicates_cannot_cross_roles():
    assert role_for_question(" WHO wrote Hamlet? ") == role_for_question("who WROTE Hamlet!")


def test_train_roles_and_fixed_selection_do_not_overlap():
    records = [from_hf_row(row(i)) for i in range(200)]
    manifest = build_manifest(records)
    assert manifest == build_manifest(records[::-1])
    roles = [set(ids) for ids in manifest["roles"].values()]
    assert sum(map(len, roles)) == 200
    assert not roles[0] & roles[1] and not roles[0] & roles[2] and not roles[1] & roles[2]
    assert not manifest["official_validation_used"]
    assert len(manifest["selected"]["memory_seed"]) == 8
    assert len(manifest["selected"]["calibration_dev"]) == 8


def test_column_conversion_preserves_sentence_positions():
    sample = row(0)
    sample["context"]["sentences"] = [["", "Second sentence"]]
    sample["supporting_facts"]["sent_id"] = [1]
    converted = from_hf_row(sample)
    assert converted["context"] == [["A", ["", "Second sentence"]]]
    assert converted["supporting_facts"] == [["A", 1]]


def test_inconsistent_columns_rejected():
    sample = row(0)
    sample["context"]["title"].append("B")
    with pytest.raises(ValueError):
        from_hf_row(sample)


def test_no_cherry_picking_extra_rows_when_frozen_pool_is_too_small():
    with pytest.raises(ValueError):
        build_manifest([from_hf_row(row(0))])
