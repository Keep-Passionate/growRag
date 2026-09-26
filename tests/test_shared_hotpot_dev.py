"""Synthetic-only tests: no paid calls, old check labels, or real data export."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import shared_hotpot_dev as shared


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    ids = [f"{i:024x}" for i in range(1600)]
    records = [
        {
            "_id": qid,
            "type": shared.legacy.KINDS[i % 2],
            "question": f"When was synthetic town {i} founded?",
            "answer": f"PRIVATE_GOLD_{i}",
            "supporting_facts": [["Shared", 0]],
            "context": [
                ["Shared", [f"Version {i % 2}.", "", 'A quote: "} [, \\".']],
                ["Fixed", ["Common passage."]],
                [f"Extra {i}", ["Unsupported distractor."]],
            ],
        }
        for i, qid in enumerate(ids)
    ]
    for record in records[464:488]:
        record["question"] = "DO_NOT_PROJECT_OLD_CHECK_QUESTION"
        record["answer"] = "DO_NOT_PROJECT_OLD_CHECK_GOLD"
    source = tmp_path / shared.DATA_ROOT / "source.json"
    write_json(source, records)
    source_sha = shared.legacy._sha(source)
    monkeypatch.setattr(shared, "SOURCE_SHA", source_sha)
    shards = []
    for index in range(2):
        path = source.parent / f"train-{index}.parquet"
        path.write_bytes(f"synthetic-shard-{index}".encode())
        shards.append({"file": path.name, "sha256": shared.legacy._sha(path), "row_count": 800})
    mirror = source.parent / "mirror_provenance.json"
    write_json(
        mirror,
        {
            "official_split": "train",
            "output_file": source.name,
            "output_sha256": source_sha,
            "original_cmu_json_bytes": False,
            "shards": shards,
            "row_count": len(records),
        },
    )
    old = tmp_path / shared.REPRESENTATION
    groups = [
        {
            "normalized_question_sha256": hashlib.sha256(f"dup{i}".encode()).hexdigest(),
            "question_ids": ids[488 + 2 * i : 490 + 2 * i],
        }
        for i in range(13)
    ]
    write_json(
        old,
        {
            "schema_version": "growrag-representation-manifest-v1",
            "official_split": "train",
            "source_sha256": source_sha,
            "complete_train_declared": True,
            "checks": {"normalized_duplicates_handled": True},
            "source_expansion_order": ids[200:456],
            "selected": {
                "source": ids[200:264],
                "target": ids[456:488],
                "debug": ids[456:464],
                "check": ids[464:488],
            },
            "exclusions": {
                "duplicate_policy": "exclude_all",
                "duplicate_normalized_groups": groups,
                "duplicate_normalized_member_count": 26,
                "excluded_union_question_ids": ids[:200] + ids[488:514],
                "requested_normalized_question_sha256s": [
                    hashlib.sha256(shared.normalize_question(r["question"]).encode()).hexdigest()
                    for r in records[:200]
                ],
            },
        },
    )
    preview = tmp_path / shared.PREVIEW
    write_json(
        preview,
        {"official_split": "train", "record_count": 200, "roles": {"memory_seed": ids[:200]}},
    )
    launch = tmp_path / "runs/2026-09-14_fresh_baseline_32_v1/launch_plan.json"
    write_json(launch, {"data": {"question_ids": ids[514:546]}})
    pattern = tmp_path / "runs/2026-09-07_pattern_probe_v1/report.json"
    # Behavior `selected` is not a question ID and must never enter the ID parser.
    write_json(pattern, {"manifest": {"question_ids": ids[:8]}, "selected": "FRESH"})
    probe = tmp_path / "runs/2026-09-07_pre_source_retrieval_probe_v1/probe.json"
    write_json(probe, {"question_id": ids[0]})
    pinned = {
        p.relative_to(tmp_path).as_posix(): shared.legacy._sha(p)
        for p in [old, preview, launch, pattern, probe]
    }
    monkeypatch.setattr(shared, "EXPOSURE_SHA256", pinned)
    return {
        "root": tmp_path,
        "output": tmp_path / "output",
        "source": source,
        "mirror": mirror,
        "old": old,
        "preview": preview,
        "launch": launch,
        "shard": source.parent / shards[0]["file"],
        "records": records,
        "ids": ids,
    }


def test_default_plan_decodes_metadata_only_and_writes_nothing(inputs, monkeypatch):
    original = shared.legacy._project

    def guarded(raw, wanted):
        assert wanted == {"_id", "type"}
        return original(raw, wanted)

    monkeypatch.setattr(shared.legacy, "_project", guarded)
    plan = shared.prepare_shared_dev(inputs["root"], inputs["output"])
    assert not inputs["output"].exists()
    assert plan["role"] == "development" and plan["official_split"] == "train"
    assert plan["seed"] == 20260927
    assert plan["excluded_id_count"] == 546 and len(plan["exclusion_sources"]) == 5
    assert plan["question_type_counts"] == {"bridge": 16, "comparison": 16}
    assert plan["eligible_type_counts"] == {"bridge": 527, "comparison": 527}
    assert plan["selection_uses_gold"] is False
    assert plan["official_dev_test_used"] is False
    assert plan["training_exclusion_enforced_globally"] is False
    assert len(plan["corpus_source_question_ids"]) == 1024
    assert len(set(plan["corpus_source_question_ids"])) == 1024
    assert not set(plan["question_ids"]) & set(plan["background_question_ids"])
    assert not set(plan["corpus_source_question_ids"]) & set(plan["excluded_question_ids"])
    expected = []
    for kind in shared.legacy.KINDS:
        eligible = [r["_id"] for r in inputs["records"][546:] if r["type"] == kind]
        expected.append(shared._ordered(eligible, "shared-dev")[:16])
    assert plan["question_ids"] == [q for pair in zip(*expected, strict=True) for q in pair]


def test_export_projects_gold_for_only_new32_and_preserves_exact_versions(inputs, monkeypatch):
    plan = shared.plan_shared_dev(inputs["root"])
    target_ids = set(plan["question_ids"])
    original_project, original_loads = shared.legacy._project, json.loads
    gold_seen, backgrounds_seen = set(), set()

    def guarded_project(raw, wanted):
        qid = original_project(raw, {"_id"})["_id"]
        if "answer" in wanted:
            assert qid in target_ids
            gold_seen.add(qid)
        elif "context" in wanted:
            assert wanted == {"context"} and qid not in target_ids
            backgrounds_seen.add(qid)
        return original_project(raw, wanted)

    def guarded_loads(value, *args, **kwargs):
        text = value.decode() if isinstance(value, bytes) else value
        assert "DO_NOT_PROJECT_OLD_CHECK" not in text
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(shared.legacy, "_project", guarded_project)
    monkeypatch.setattr(shared.json, "loads", guarded_loads)
    result = shared.prepare_shared_dev(inputs["root"], inputs["output"], prepare_new=True)
    assert gold_seen == target_ids and len(backgrounds_seen) == 992
    manifest = json.loads(Path(result["manifest_path"]).read_bytes())
    runtime = read_jsonl(inputs["output"] / "runtime_questions.jsonl")
    gold = read_jsonl(inputs["output"] / "gold.jsonl")
    corpus = read_jsonl(inputs["output"] / "corpus.jsonl")
    owners = read_jsonl(inputs["output"] / "corpus_sources.jsonl")
    assert len(runtime) == len(gold) == 32
    assert all(set(row) == {"question_id", "text", "dataset"} for row in runtime)
    assert all(set(row) == {"doc_id", "title", "sentences"} for row in corpus)
    assert len(corpus) == 1027  # 1024 distractors + one fixed + two Shared versions.
    assert manifest["corpus_statistics"]["titles_with_multiple_versions"] == 1
    docs = {row["doc_id"]: row for row in corpus}
    assert all(
        key == shared.document_id(row["title"], row["sentences"]) for key, row in docs.items()
    )
    assert sum(row["title"].startswith("Extra ") for row in corpus) == 1024
    fixed_id = shared.document_id("Fixed", ["Common passage."])
    assert (
        len(next(row for row in owners if row["doc_id"] == fixed_id)["source_question_ids"]) == 1024
    )
    for record in gold:
        assert len(record["exact_support"]) == 1
        support = record["exact_support"][0]
        text = docs[support["doc_id"]]["sentences"][support["sentence_index"]]
        assert support["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        source = inputs["records"][int(record["question_id"], 16)]
        assert support["doc_id"] == shared.document_id(*source["context"][0])
    assert manifest["runtime_input_allowlist"] == ["runtime_questions.jsonl", "corpus.jsonl"]
    for name, item in manifest["artifacts"].items():
        path = inputs["output"] / name
        assert shared.legacy._sha(path) == item["sha256"]
        assert path.stat().st_size == item["bytes"]
        assert item["rows"] == len(read_jsonl(path))
        assert item["runtime_safe"] == (name in manifest["runtime_input_allowlist"])
    assert (inputs["output"] / "manifest.sha256").read_text().split()[0] == result[
        "manifest_sha256"
    ]
    assert shared.legacy._sha(Path(result["manifest_path"])) == result["manifest_sha256"]


def test_deterministic_outputs_and_no_overwrite(inputs):
    first = shared.prepare_shared_dev(inputs["root"], inputs["output"], prepare_new=True)
    another = inputs["root"] / "another"
    second = shared.prepare_shared_dev(inputs["root"], another, prepare_new=True)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    for path in inputs["output"].iterdir():
        assert path.read_bytes() == (another / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        shared.prepare_shared_dev(inputs["root"], inputs["output"], prepare_new=True)


@pytest.mark.parametrize("target", ["source", "old", "preview", "launch", "shard"])
def test_mutated_input_bytes_fail_before_creating_output(inputs, target):
    with inputs[target].open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="SHA"):
        shared.prepare_shared_dev(inputs["root"], inputs["output"], prepare_new=True)
    assert not inputs["output"].exists()


def test_unknown_historical_plan_fails_closed(inputs):
    path = inputs["root"] / "runs/2026-09-27_unreviewed/plan.json"
    write_json(path, {"question_ids": [inputs["ids"][-1]]})
    with pytest.raises(ValueError, match="unreviewed historical plan"):
        shared.plan_shared_dev(inputs["root"])


@pytest.mark.parametrize("defect", ["policy", "missing_duplicates", "overlapping_reservation"])
def test_incomplete_duplicate_or_reservation_audit_fails_closed(inputs, monkeypatch, defect):
    value = json.loads(inputs["old"].read_bytes())
    if defect == "policy":
        value["exclusions"]["duplicate_policy"] = "keep"
    elif defect == "missing_duplicates":
        value["exclusions"]["excluded_union_question_ids"] = inputs["ids"][:200]
    else:
        value["source_expansion_order"][0] = value["selected"]["target"][0]
    write_json(inputs["old"], value)
    monkeypatch.setitem(
        shared.EXPOSURE_SHA256, shared.REPRESENTATION, shared.legacy._sha(inputs["old"])
    )
    with pytest.raises(ValueError, match="audit"):
        shared.plan_shared_dev(inputs["root"])


@pytest.mark.parametrize(
    "context,facts",
    [
        ([["Other", ["Sentence."]]], (("Title", 0),)),
        ([["Title", ["Sentence."]]], (("Title", 1),)),
        ([["Title", [""]]], (("Title", 0),)),
        ([["Title", ["One."]], ["Title", ["Two."]]], (("Title", 0),)),
    ],
)
def test_exact_gold_support_must_resolve_without_ambiguity(context, facts):
    with pytest.raises(ValueError, match="exact gold support"):
        shared._exact_support(context, facts)


def test_document_hash_keeps_order_empty_sentences_and_title():
    identities = [
        shared.document_id("Title", ["a", "b"]),
        shared.document_id("Title", ["b", "a"]),
        shared.document_id("Title", ["a", "", "b"]),
        shared.document_id("Other", ["a", "b"]),
    ]
    assert len(set(identities)) == 4


def test_insufficient_background_fails_without_resampling(inputs):
    records = inputs["records"][:1550]  # 1004 unused: cannot supply 32 + 992.
    write_json(inputs["source"], records)
    new_sha = shared.legacy._sha(inputs["source"])
    # Re-freeze synthetic provenance to test capacity, not a byte-tamper rejection.
    shared.SOURCE_SHA = new_sha
    old = json.loads(inputs["old"].read_bytes())
    old["source_sha256"] = new_sha
    write_json(inputs["old"], old)
    shared.EXPOSURE_SHA256[shared.REPRESENTATION] = shared.legacy._sha(inputs["old"])
    mirror = json.loads(inputs["mirror"].read_bytes())
    mirror["output_sha256"], mirror["row_count"] = new_sha, len(records)
    mirror["shards"][1]["row_count"] = 750
    write_json(inputs["mirror"], mirror)
    with pytest.raises(ValueError, match="insufficient unused background"):
        shared.prepare_shared_dev(inputs["root"], inputs["output"], prepare_new=True)
    assert not inputs["output"].exists()
