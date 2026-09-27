"""Small synthetic streaming contracts; production counts are tested separately."""

import hashlib
import json
from pathlib import Path

import pytest
from test_shared_hotpot_dev import inputs, read_jsonl  # noqa: F401 -- reused synthetic fixture

from growrag.experiments import shared_hotpot_scale as scale


@pytest.fixture
def scale_inputs(inputs, monkeypatch):  # noqa: F811 -- pytest resolves the imported fixture
    frozen_dir = inputs["root"] / "frozen32"
    result = scale.base.prepare_shared_dev(inputs["root"], frozen_dir, prepare_new=True)
    monkeypatch.setattr(scale, "FROZEN32", "frozen32/manifest.json")
    monkeypatch.setattr(scale, "FROZEN32_SHA", result["manifest_sha256"])
    monkeypatch.setattr(scale, "TARGET_COUNT", 10)
    monkeypatch.setattr(scale, "CORPUS_SOURCE_COUNT", 64)
    inputs["frozen_dir"] = frozen_dir
    inputs["output"] = inputs["root"] / "scaled"
    return inputs


def test_production_counts_and_proportional_quotas():
    assert (scale.TARGET_COUNT, scale.CORPUS_SOURCE_COUNT, scale.SEED) == (500, 8192, 2026092702)
    assert scale._quotas({"bridge": 72630, "comparison": 17239}) == {
        "bridge": 404,
        "comparison": 96,
    }


def test_plan_metadata_only_excludes_frozen32_and_uses_independent_order(scale_inputs, monkeypatch):
    original = scale.base.legacy._project

    def guarded(raw, wanted):
        assert wanted == {"_id", "type"}
        return original(raw, wanted)

    monkeypatch.setattr(scale.base.legacy, "_project", guarded)
    plan = scale.prepare_shared_scale(scale_inputs["root"], scale_inputs["output"])
    frozen = json.loads((scale_inputs["frozen_dir"] / "manifest.json").read_bytes())
    assert not scale_inputs["output"].exists()
    assert plan["excluded_id_count"] == 578
    assert set(frozen["question_ids"]) <= set(plan["excluded_question_ids"])
    assert not set(plan["corpus_source_question_ids"]) & set(plan["excluded_question_ids"])
    assert plan["question_type_counts"] == {"bridge": 5, "comparison": 5}
    assert plan["question_ids"] == scale._ordered(plan["question_ids"], "shared-scale-order")
    assert len(plan["background_question_ids"]) == 54
    assert not set(plan["background_question_ids"]) & set(plan["question_ids"])
    assert plan["background_is_memory_training"] is False
    assert plan["training_exclusion_enforced_globally"] is False
    assert plan["role"] == "development" and plan["official_split"] == "train"


def test_streamed_bundle_keeps_runtime_gold_and_ownership_separate(scale_inputs, monkeypatch):
    expected = scale.plan_shared_scale(scale_inputs["root"])
    targets, projected = set(expected["question_ids"]), set()
    original = scale.base.legacy._project

    def guarded(raw, wanted):
        if "answer" in wanted:
            qid = original(raw, {"_id"})["_id"]
            assert qid in targets
            projected.add(qid)
        elif "context" in wanted:
            assert wanted == {"context"}
        return original(raw, wanted)

    monkeypatch.setattr(scale.base.legacy, "_project", guarded)
    result = scale.prepare_shared_scale(
        scale_inputs["root"], scale_inputs["output"], prepare_new=True
    )
    assert projected == targets
    folder = scale_inputs["output"]
    manifest = json.loads((folder / "manifest.json").read_bytes())
    runtime, gold = (
        read_jsonl(folder / "runtime_questions.jsonl"),
        read_jsonl(folder / "gold.jsonl"),
    )
    corpus = {row["doc_id"]: row for row in read_jsonl(folder / "corpus.jsonl")}
    assert len(runtime) == len(gold) == 10 and len(corpus) == 67
    assert [row["question_id"] for row in runtime] == expected["question_ids"]
    assert all(set(row) == {"question_id", "text", "dataset"} for row in runtime)
    assert all(set(row) == {"doc_id", "title", "sentences"} for row in corpus.values())
    assert len([row for row in corpus.values() if row["title"] == "Shared"]) == 2
    for label in gold:
        assert label["annotation_status"] == "valid"
        for support in label["exact_support"]:
            doc = corpus[support["doc_id"]]
            assert support["doc_id"] == scale.base.document_id(doc["title"], doc["sentences"])
            assert (
                support["text_sha256"]
                == hashlib.sha256(doc["sentences"][support["sentence_index"]].encode()).hexdigest()
            )
    assert manifest["runtime_input_allowlist"] == ["runtime_questions.jsonl", "corpus.jsonl"]
    for name, info in manifest["artifacts"].items():
        path = folder / name
        assert scale.base.legacy._sha(path) == info["sha256"]
        assert path.stat().st_size == info["bytes"]
        assert len(read_jsonl(path)) == info["rows"]
        assert info["runtime_safe"] == (name in manifest["runtime_input_allowlist"])
    assert scale.base.legacy._sha(folder / "manifest.json") == result["manifest_sha256"]
    assert (folder / "manifest.sha256").read_text().split()[0] == result["manifest_sha256"]


def test_invalid_annotation_is_marked_and_never_replaced(scale_inputs, monkeypatch):
    expected = scale.plan_shared_scale(scale_inputs["root"])["question_ids"]
    bad_id = expected[0]
    original = scale.base.legacy._project

    def changed_label(raw, wanted):
        row = original(raw, wanted)
        if "answer" in wanted and row.get("_id") == bad_id:
            row["supporting_facts"] = [["Missing supporting title", 0]]
        return row

    monkeypatch.setattr(scale.base.legacy, "_project", changed_label)
    result = scale.prepare_shared_scale(
        scale_inputs["root"], scale_inputs["output"], prepare_new=True
    )
    manifest = json.loads(Path(result["manifest_path"]).read_bytes())
    assert manifest["question_ids"] == expected
    assert manifest["annotation_summary"] == {
        "valid": 9,
        "invalid": 1,
        "invalid_question_ids": [bad_id],
        "replacement_count": 0,
    }
    labels = read_jsonl(scale_inputs["output"] / "gold.jsonl")
    assert labels[0]["question_id"] == bad_id and labels[0]["annotation_status"] == "invalid"
    assert labels[0]["answers"] == [] and labels[0]["exact_support"] == []


def test_deterministic_streamed_bytes_and_refuses_overwrite(scale_inputs):
    first = scale.prepare_shared_scale(
        scale_inputs["root"], scale_inputs["output"], prepare_new=True
    )
    other = scale_inputs["root"] / "second"
    second = scale.prepare_shared_scale(scale_inputs["root"], other, prepare_new=True)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    for path in scale_inputs["output"].iterdir():
        assert path.read_bytes() == (other / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        scale.prepare_shared_scale(scale_inputs["root"], scale_inputs["output"], prepare_new=True)


def test_frozen32_sha_tamper_fails_before_writing(scale_inputs):
    with (scale_inputs["frozen_dir"] / "manifest.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="frozen32 manifest SHA"):
        scale.prepare_shared_scale(scale_inputs["root"], scale_inputs["output"], prepare_new=True)
    assert not scale_inputs["output"].exists()


def test_interruption_never_publishes_manifest_or_overwrites_partial(scale_inputs, monkeypatch):
    def interrupted(_row):
        raise OSError("synthetic interruption")

    monkeypatch.setattr(scale, "_gold", interrupted)
    with pytest.raises(OSError, match="synthetic interruption"):
        scale.prepare_shared_scale(scale_inputs["root"], scale_inputs["output"], prepare_new=True)
    assert scale_inputs["output"].exists()
    assert not (scale_inputs["output"] / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        scale.prepare_shared_scale(scale_inputs["root"], scale_inputs["output"], prepare_new=True)
