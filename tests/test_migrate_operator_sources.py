"""Synthetic source-only migration tests; no production data, labels or API."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import migrate_operator_sources as module
from growrag.experiments.operator_data_plan import _digest, question_metadata
from growrag.experiments.run_operator_study import load_inputs
from growrag.experiments.shared_hotpot_dev import document_id


def _id(number):
    return f"{number:024x}"


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(text):
    return question_metadata(_id(1), text, official_split="train").normalized_question_sha256


def _seal(path, value):
    _json(path, value)
    path.with_suffix(".sha256").write_text(f"{module._sha(path)}  manifest.json\n")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    previous = module.previous
    shared_path = tmp_path / previous.SHARED
    shared_corpus = shared_path.parent / "corpus.jsonl"
    shared_corpus.parent.mkdir(parents=True)
    doc = {"title": "Shared", "sentences": ["An existing background sentence."]}
    doc["doc_id"] = document_id(doc["title"], doc["sentences"])
    shared_corpus.write_bytes(json.dumps(doc).encode() + b"\r\n")
    _json(
        shared_path,
        {
            "official_split": "train",
            "question_ids": [_id(1800)],
            "background_question_ids": [_id(i) for i in range(1, 1801)],
            "corpus_source_question_ids": [_id(i) for i in range(1, 1801)],
            "artifacts": {"corpus.jsonl": {"sha256": module._sha(shared_corpus), "rows": 1}},
        },
    )
    monkeypatch.setattr(previous, "RESERVATIONS", (previous.SHARED,))
    train_dir = tmp_path / previous.TRAIN_DIR
    train_dir.mkdir(parents=True)
    rows = [
        {
            "id": _id(i),
            "question": f"Synthetic training question {i}",
            "answer": "FORBIDDEN_ANSWER",
            "supporting_facts": "FORBIDDEN_SUPPORT",
            "type": "FORBIDDEN_TYPE",
        }
        for i in range(1, 1802)
    ]
    # Same normalized text as a sealed calibration item must be excluded by hash.
    rows[607]["question"] = "Synthetic training question 501"
    rows[609]["question"] = rows[610]["question"] = "Duplicated candidate question"
    shard = train_dir / "part.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    raw = train_dir / "raw.json"
    raw.write_bytes(b'{"answer":"FORBIDDEN_RAW_GOLD"}')
    _json(
        train_dir / "mirror_provenance.json",
        {
            "official_split": "train",
            "output_file": raw.name,
            "output_sha256": module._sha(raw),
            "row_count": len(rows),
            "shards": [{"file": shard.name, "sha256": module._sha(shard)}],
        },
    )
    roles = {
        "source": [_id(i) for i in range(1, 501)],
        "calibration": [_id(i) for i in range(501, 601)],
        "evaluation": [_id(i) for i in range(10001, 10501)],
    }
    parent_path = tmp_path / module.DEFAULT_PARENT
    parent_path.parent.mkdir(parents=True)
    artifacts = {}
    for role, ids in roles.items():
        path = parent_path.parent / f"{role}_runtime_questions.jsonl"
        path.write_bytes(
            b"".join(
                json.dumps(
                    {
                        "question_id": qid,
                        "text": f"SEALED_{role.upper()}_{qid}",
                        "dataset": "synthetic",
                    },
                    separators=(", ", ": "),
                ).encode()
                + b"\r\n"
                for qid in ids
            )
        )
        artifacts[path.name] = {
            **previous._artifact(path, rows=len(ids)),
            "contains_gold": False,
            "runtime_safe": True,
        }
    corpus = parent_path.parent / "corpus.jsonl"
    corpus.write_bytes(shared_corpus.read_bytes())
    artifacts[corpus.name] = {
        **previous._artifact(corpus, rows=1),
        "contains_gold": False,
        "runtime_safe": True,
    }
    parent = {
        "schema_version": module.SCHEMA,
        "source_sizes": list(module.SOURCE_SIZES),
        "counts": module.COUNTS,
        "roles": roles,
        "official_splits": {"source": "train", "calibration": "train", "evaluation": "dev"},
        "selected_normalized_question_sha256s": {
            role: [_hash(f"Synthetic training question {int(qid, 16)}") for qid in ids]
            for role, ids in roles.items()
        },
        "role_ids_sha256": {role: _digest(ids) for role, ids in roles.items()},
        "nested_source_ids": {str(n): roles["source"][:n] for n in module.SOURCE_SIZES},
        "input_files": [],
        "artifacts": artifacts,
        "gold_projected": False,
        "official_test_used": False,
        "evaluation_memory_updates_allowed": False,
        "calibration_memory_updates_allowed": False,
        "eligibility_counts": {"source": 1200, "calibration": 1200, "evaluation": 500},
        "eligibility_ids_sha256": {role: _digest(ids) for role, ids in roles.items()},
        "raw_source_rows": {"train": len(rows), "dev": 500},
    }
    _seal(parent_path, parent)
    _json(tmp_path / "data/hotpotqa/prior/manifest.json", {"roles": {"probe": [_id(601)]}})
    _json(tmp_path / "runs/prior/reservation.claim.json", {"question_ids": [_id(602)]})
    _json(
        tmp_path / "runs/prior/launch_plan.json",
        {
            "question_ids": [_id(603)],
            "selected_normalized_question_sha256s": [_hash("Synthetic training question 609")],
        },
    )
    return tmp_path


def _migrate(root, **kwargs):
    return module.migrate_operator_sources(root, **kwargs)


def test_dry_run_preserves_roles_excludes_history_and_writes_nothing(bundle):
    result = _migrate(bundle)
    parent = module._read_json(bundle / module.DEFAULT_PARENT)
    assert not (bundle / module.DEFAULT_OUTPUT).exists()
    assert len(result["roles"]["source"]) == 500
    for role in ("calibration", "evaluation"):
        assert result["roles"][role] == parent["roles"][role]
        assert result["role_ids_sha256"][role] == parent["role_ids_sha256"][role]
    assert not set(result["roles"]["source"]) & {
        qid for ids in parent["roles"].values() for qid in ids
    }
    assert {_id(i) for i in (601, 602, 603, 608, 609, 610, 611, 1800)} <= set(
        result["excluded_question_ids"]
    )
    assert _id(1801) not in result["roles"]["source"]
    assert result["api_calls"] == 0
    for field in (
        "gold_projected",
        "old_reserved_text_projected",
        "corpus_rebuilt",
        "calibration_evaluation_runtime_decoded",
        "old_claims_released",
    ):
        assert result[field] is False


def test_write_preserves_exact_bytes_and_loads_source(bundle):
    parent = bundle / Path(module.DEFAULT_PARENT).parent
    old_bytes = {p.name: p.read_bytes() for p in parent.iterdir()}
    result = _migrate(bundle, prepare_new=True)
    output = bundle / module.DEFAULT_OUTPUT
    for name in module.PRESERVED:
        assert (output / name).read_bytes() == old_bytes[name]
        assert result["artifacts"][name]["sha256"] == hashlib.sha256(old_bytes[name]).hexdigest()
    assert {p.name: p.read_bytes() for p in parent.iterdir()} == old_bytes
    manifest, questions = load_inputs(output / "manifest.json", "source")
    assert len(questions) == 500
    assert [question.question_id for question in questions] == result["roles"]["source"]
    for n in module.SOURCE_SIZES:
        assert manifest["nested_source_ids"][str(n)] == result["roles"]["source"][:n]
    source_bytes = (output / "source_runtime_questions.jsonl").read_bytes()
    assert b"FORBIDDEN" not in source_bytes
    assert all(
        set(json.loads(line)) == {"question_id", "text", "dataset"}
        for line in source_bytes.splitlines()
    )
    assert any(item["path"].endswith("mirror_provenance.json") for item in result["input_files"])


def test_arrow_filters_old_ids_before_text_materialization_without_labels(bundle, monkeypatch):
    ds = pytest.importorskip("pyarrow.dataset")
    actual = ds.dataset
    calls = []

    class CheckedTable:
        def __init__(self, table):
            self.table = table

        def to_pylist(self):
            ids = self.table["id"].to_pylist()
            assert not set(ids) & {_id(i) for i in range(1, 604)}
            assert _id(1800) not in ids and _id(1801) not in ids
            return self.table.to_pylist()

    class CheckedDataset:
        def __init__(self, paths, **kwargs):
            assert all(
                module.previous.TRAIN_DIR.replace("/", "\\") in str(p).replace("/", "\\")
                for p in paths
            )
            self.dataset = actual(paths, **kwargs)

        def to_table(self, *, columns, **kwargs):
            calls.append(columns)
            assert columns in (["id"], ["id", "question"])
            table = self.dataset.to_table(columns=columns, **kwargs)
            return CheckedTable(table) if "question" in columns else table

    monkeypatch.setattr(ds, "dataset", CheckedDataset)
    _migrate(bundle)
    assert calls == [["id"], ["id", "question"]]


def test_preserved_question_text_and_raw_labels_never_decoded(bundle, monkeypatch):
    actual = json.loads

    def checked(value, *args, **kwargs):
        encoded = value.encode() if isinstance(value, str) else bytes(value)
        assert b"SEALED_" not in encoded
        assert b"FORBIDDEN_RAW_GOLD" not in encoded
        return actual(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", checked)
    _migrate(bundle, prepare_new=True)


def test_seed_is_deterministic_but_only_source_can_change(bundle):
    first = _migrate(bundle)
    assert _migrate(bundle)["roles"] == first["roles"]
    other = _migrate(bundle, seed="another-fixed-seed")
    assert other["roles"]["source"] != first["roles"]["source"]
    for role in ("calibration", "evaluation"):
        assert other["roles"][role] == first["roles"][role]


@pytest.mark.parametrize("kind", ["ids", "hashes"])
def test_parent_overlap_is_rejected_before_output(bundle, kind):
    path = bundle / module.DEFAULT_PARENT
    parent = module._read_json(path)
    if kind == "ids":
        parent["roles"]["evaluation"][0] = parent["roles"]["calibration"][0]
        parent["role_ids_sha256"]["evaluation"] = _digest(parent["roles"]["evaluation"])
    else:
        registry = parent["selected_normalized_question_sha256s"]
        registry["evaluation"][0] = registry["calibration"][0]
    _seal(path, parent)
    with pytest.raises(ValueError, match="overlap"):
        _migrate(bundle, prepare_new=True)
    assert not (bundle / module.DEFAULT_OUTPUT).exists()


@pytest.mark.parametrize("filename", ["manifest.sha256", "evaluation_runtime_questions.jsonl"])
def test_parent_tampering_fails_before_output(bundle, filename):
    path = bundle / Path(module.DEFAULT_PARENT).parent / filename
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        _migrate(bundle, prepare_new=True)
    assert not (bundle / module.DEFAULT_OUTPUT).exists()


def test_not_enough_candidates_never_relaxes_exclusions(bundle):
    _json(bundle / "runs/all.claim.json", {"question_ids": [_id(i) for i in range(704, 1801)]})
    with pytest.raises(ValueError, match="not enough fresh"):
        _migrate(bundle, prepare_new=True)
    assert not (bundle / module.DEFAULT_OUTPUT).exists()


def test_empty_candidates_fail_closed_before_output(bundle):
    pa = pytest.importorskip("pyarrow")
    _json(bundle / "runs/all.claim.json", {"question_ids": [_id(i) for i in range(604, 1801)]})
    # Empty isin currently fails at Arrow's typed-set validation, still before writes.
    with pytest.raises((ValueError, pa.ArrowTypeError)):
        _migrate(bundle, prepare_new=True)
    assert not (bundle / module.DEFAULT_OUTPUT).exists()


def test_existing_partial_never_overwritten(bundle):
    output = bundle / module.DEFAULT_OUTPUT
    output.mkdir()
    sentinel = output / "partial.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="never overwrite"):
        _migrate(bundle, prepare_new=True)
    assert sentinel.read_bytes() == b"keep"


@pytest.mark.parametrize("new_file", [False, True])
def test_reservation_race_fails_before_output(bundle, monkeypatch, new_file):
    actual = module._project_candidates

    def changed(*args):
        result = actual(*args)
        path = "runs/new.claim.json" if new_file else "runs/prior/reservation.claim.json"
        _json(bundle / path, {"question_ids": [_id(700)]})
        return result

    monkeypatch.setattr(module, "_project_candidates", changed)
    with pytest.raises(ValueError, match="changed during migration"):
        _migrate(bundle, prepare_new=True)
    assert not (bundle / module.DEFAULT_OUTPUT).exists()


def test_failed_copy_is_retained_unsealed_and_never_overwritten(bundle, monkeypatch):
    actual = module._copy_preserved

    def changed(source, target, expected):
        actual(source, target, expected)
        raise ValueError("synthetic copy failure")

    monkeypatch.setattr(module, "_copy_preserved", changed)
    with pytest.raises(ValueError, match="synthetic copy failure"):
        _migrate(bundle, prepare_new=True)
    output = bundle / module.DEFAULT_OUTPUT
    assert (output / module.PRESERVED[0]).exists()
    assert not (output / "manifest.sha256").exists()
    with pytest.raises(FileExistsError):
        _migrate(bundle, prepare_new=True)


def test_running_study_blocks_migration(bundle):
    (bundle / "runs/.operator-study.lock").write_text("running")
    with pytest.raises(ValueError, match="study is running"):
        _migrate(bundle, prepare_new=True)
    assert not (bundle / module.DEFAULT_OUTPUT).exists()


def test_cli_is_read_only_and_does_not_print_question_text_or_ids(bundle, capsys):
    assert module.main(["--root", str(bundle)]) == 0
    text = capsys.readouterr().out
    assert "Synthetic training" not in text and _id(700) not in text
    assert json.loads(text)["prepared"] is False
    assert not (bundle / module.DEFAULT_OUTPUT).exists()
