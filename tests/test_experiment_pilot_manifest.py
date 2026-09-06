import hashlib
import json

import pytest

from growrag.experiments.data_protocol import build_manifest
from growrag.experiments.run_pilot import load_selected_examples


def create_manifest(tmp_path):
    records = [
        {
            "_id": f"q{i}",
            "question": f"What happened in year {i}?",
            "answer": "Answer",
            "type": "bridge",
            "level": "hard",
            "context": [["Doc", ["Sentence."]]],
            "supporting_facts": [["Doc", 0]],
        }
        for i in range(200)
    ]
    raw = json.dumps(records).encode()
    (tmp_path / "data.json").write_bytes(raw)
    manifest = build_manifest(records)
    manifest.update(data_file="data.json", data_sha256=hashlib.sha256(raw).hexdigest())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_manifest_validated_before_any_network(tmp_path):
    path, _ = create_manifest(tmp_path)
    sources, targets, _ = load_selected_examples(path)
    assert len(sources) == len(targets) == 8
    assert not {e.question.question_id for e in sources} & {e.question.question_id for e in targets}


@pytest.mark.parametrize("change", ["test_split", "wrong_roles", "changed_data", "path_escape"])
def test_corrupt_manifest_or_split_rejected(tmp_path, change):
    path, manifest = create_manifest(tmp_path)
    if change == "test_split":
        manifest["official_split"] = "validation"
    elif change == "wrong_roles":
        manifest["selected"]["calibration_dev"] = manifest["selected"]["memory_seed"]
    elif change == "changed_data":
        (tmp_path / "data.json").write_text("[]", encoding="utf-8")
    else:
        manifest["data_file"] = "../outside.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_selected_examples(path)
