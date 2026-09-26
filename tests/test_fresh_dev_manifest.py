"""Small synthetic-only protocol tests; no old check text or API access."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import fresh_dev_manifest as fresh


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    ids = [f"{index:024x}" for index in range(610)]
    records = [
        {
            "_id": qid,
            "question": f"When was synthetic town {index} founded?",
            "answer": "1901",
            "supporting_facts": [["Town", 0]],
            "context": [["Town", ['The town was founded in 1901. A quote: "} [, \\".']]],
            "type": fresh.KINDS[index % 2],
            "level": "easy",
        }
        for index, qid in enumerate(ids)
    ]
    # Protected records contain sentinels that must never reach json.loads.
    for record in records[464:488]:
        record["question"] = "DO_NOT_PROJECT_OLD_CHECK_QUESTION"
        record["answer"] = "DO_NOT_PROJECT_OLD_CHECK_GOLD"
    source = tmp_path / "source.json"
    write_json(source, records)
    source_sha = fresh._sha(source)
    monkeypatch.setattr(fresh, "SOURCE_SHA", source_sha)
    mirror = tmp_path / "mirror.json"
    write_json(
        mirror,
        {
            "official_split": "train",
            "output_sha256": source_sha,
            "row_count": len(ids),
            "original_cmu_json_bytes": False,
        },
    )
    old = tmp_path / "old.json"
    old_hashes = [
        hashlib.sha256(fresh.normalize_question(r["question"]).encode()).hexdigest()
        for r in records[:200]
    ]
    groups = [
        {
            "normalized_question_sha256": hashlib.sha256(f"dup{i}".encode()).hexdigest(),
            "question_ids": ids[488 + i * 2 : 490 + i * 2],
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
                "requested_normalized_question_sha256s": old_hashes,
                "excluded_union_question_ids": ids[:200] + ids[488:514],
            },
        },
    )
    preview = tmp_path / "preview.json"
    write_json(
        preview,
        {"record_count": 200, "official_split": "train", "roles": {"memory_seed": ids[:200]}},
    )
    runs = tmp_path / "runs"
    write_json(
        runs / "2026-09-14_fresh_baseline_32_v1/launch_plan.json",
        {"data": {"question_ids": ids[514:546]}},
    )
    for probe in fresh.EARLY_PROBES:
        write_json(runs / probe, {"question_id": ids[0]})
    return source, mirror, old, preview, runs, tmp_path / "output"


def test_prepare_and_load_only_new_records_with_frozen_roles(inputs, monkeypatch):
    original_loads = json.loads

    def guarded_loads(value, *args, **kwargs):
        text = value.decode() if isinstance(value, bytes) else value
        assert "DO_NOT_PROJECT_OLD_CHECK" not in text
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(fresh.json, "loads", guarded_loads)
    manifest_path = fresh.prepare_fresh_dev(*inputs)
    examples, provenance = fresh.load_fresh_dev(manifest_path)
    assert len(examples) == 12
    assert provenance["role"] == "development"
    assert provenance["official_split"] == "train" and provenance["subset"] == "new12"
    assert provenance["question_type_counts"] == {"bridge": 6, "comparison": 6}
    assert provenance["seed"] == 20260926
    assert provenance["excluded_id_count"] == 546
    assert len(provenance["exclusion_sources"]) == 5
    assert provenance["all_inputs_verified"] is True
    assert provenance["old_check_records_projected"] is False
    assert provenance["selected_records_contains_gold"] is True
    assert all(example.gold is not None for example in examples)
    assert [e.question_type for e in examples] == ["bridge", "comparison"] * 6
    assert not set(provenance["question_ids"]) & set(provenance["excluded_question_ids"])
    assert "DO_NOT_PROJECT_OLD_CHECK" not in manifest_path.read_text()


def test_selection_is_independent_of_input_order_and_uses_only_metadata():
    meta = {f"{i:024x}": (fresh.KINDS[i % 2], i, i + 1) for i in range(30)}
    expected = fresh.select_fresh_ids(meta, set())
    assert fresh.select_fresh_ids(dict(reversed(list(meta.items()))), set()) == expected
    assert len(expected) == 12
    with pytest.raises(ValueError, match="historical IDs"):
        fresh.select_fresh_ids(meta, {"f" * 24})
    with pytest.raises(ValueError, match="not enough"):
        fresh.select_fresh_ids(dict(list(meta.items())[:10]), set())


def test_project_skips_all_unrequested_values_and_handles_escaped_nested_data():
    raw = json.dumps(
        {
            "context": [["{title}", ['x \\" } ]']]],
            "answer": "NO_DECODE",
            "_id": "0" * 24,
            "type": "bridge",
            "other": [1, None, False],
        }
    ).encode()
    assert fresh._project(raw, {"_id", "type"}) == {"_id": "0" * 24, "type": "bridge"}
    spans = list(fresh._record_spans(b" [\n" + raw + b",\n" + raw + b"\n] "))
    assert len(spans) == 2


@pytest.mark.parametrize(
    "value", [b'[{"_id":"a","_id":"b"}]', b"[42]", b'[{"x":[1}]', b'[{"x":1}] trailing']
)
def test_invalid_source_lexical_forms_are_rejected(value):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        for start, end in fresh._record_spans(value):
            fresh._project(value[start:end], {"_id", "type"})


def test_never_overwrites_existing_output(inputs):
    inputs[-1].mkdir()
    with pytest.raises(FileExistsError):
        fresh.prepare_fresh_dev(*inputs)
    assert not list(inputs[-1].iterdir())


@pytest.mark.parametrize(
    "field", ["seed", "role", "official_split", "question_ids", "selected_records_file"]
)
def test_manifest_tampering_is_rejected(inputs, field):
    manifest_path = fresh.prepare_fresh_dev(*inputs)
    value = json.loads(manifest_path.read_bytes())
    value[field] = "tampered"
    write_json(manifest_path, value)
    with pytest.raises(ValueError):
        fresh.load_fresh_dev(manifest_path)


@pytest.mark.parametrize("target", ["source", "mirror", "old", "preview", "historical", "subset"])
def test_changed_source_or_exclusion_bytes_are_rejected(inputs, target):
    manifest_path = fresh.prepare_fresh_dev(*inputs)
    paths = {
        "source": inputs[0],
        "mirror": inputs[1],
        "old": inputs[2],
        "preview": inputs[3],
        "historical": inputs[4] / "2026-09-14_fresh_baseline_32_v1/launch_plan.json",
        "subset": manifest_path.parent / "selected_records.json",
    }
    with paths[target].open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError):
        fresh.load_fresh_dev(manifest_path)


def test_new_historical_pool_is_not_silently_ignored(inputs):
    manifest_path = fresh.prepare_fresh_dev(*inputs)
    write_json(
        inputs[4] / "2026-09-22_extra/plan.json", {"data": {"question_ids": [f"{550:024x}"]}}
    )
    with pytest.raises(ValueError):
        fresh.load_fresh_dev(manifest_path)


def test_missing_duplicate_audit_fails_closed(inputs):
    old = json.loads(inputs[2].read_bytes())
    old["exclusions"]["duplicate_policy"] = "reject"
    write_json(inputs[2], old)
    with pytest.raises(ValueError, match="audit"):
        fresh.prepare_fresh_dev(*inputs)
    assert not inputs[-1].exists()
