"""Synthetic export tests: no real Hotpot records, gold, secrets, or network."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import prepare_operator_data as module
from growrag.experiments.shared_hotpot_dev import document_id


def _id(number):
    return f"{number:024x}"


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mirror(root, directory, split, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    folder = root / directory
    folder.mkdir(parents=True)
    shard = folder / "part.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    raw = folder / "raw.json"
    # The implementation hashes this opaque file, never parses its labels.
    raw.write_text(json.dumps({"answer": "DO_NOT_OPEN_RAW_GOLD"}), encoding="utf-8")
    _json(
        folder / "mirror_provenance.json",
        {
            "official_split": split,
            "output_file": raw.name,
            "output_sha256": _sha(raw),
            "row_count": len(rows),
            "shards": [{"file": shard.name, "sha256": _sha(shard)}],
        },
    )


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    shared_path = tmp_path / module.SHARED
    corpus = shared_path.parent / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    doc = {"title": "Shared", "sentences": ["An old sentence."]}
    doc["doc_id"] = document_id(doc["title"], doc["sentences"])
    corpus.write_bytes(json.dumps(doc, sort_keys=True).encode() + b"\n")
    _json(
        shared_path,
        {
            "official_split": "train",
            "question_ids": [_id(1)],
            "excluded_question_ids": [],
            "background_question_ids": [_id(i) for i in range(2, 30)],
            "corpus_source_question_ids": [_id(i) for i in range(1, 30)],
            "artifacts": {"corpus.jsonl": {"sha256": _sha(corpus), "rows": 1}},
        },
    )
    memory = "data/hotpotqa/memory_sources_sep27_v1/manifest.json"
    _json(
        tmp_path / memory,
        {"roles": {"source": [_id(2)], "calibration": [_id(3)], "probe": [_id(4)]}},
    )
    monkeypatch.setattr(module, "RESERVATIONS", (module.SHARED, memory))
    train = [
        {
            "id": _id(i),
            "question": f"Private synthetic train {i}",
            "answer": "FORBIDDEN_ANSWER",
            "supporting_facts": "FORBIDDEN_SUPPORT",
        }
        for i in range(1, 31)
    ]
    dev = [
        {
            "id": _id(i),
            "question": f"Private synthetic dev {i}",
            "answer": "FORBIDDEN_DEV_ANSWER",
            "supporting_facts": "FORBIDDEN_SUPPORT",
            "context": {
                "title": ["Shared", "Same title"],
                "sentences": [["An old sentence."], [f"Version from {i}."]],
            },
        }
        for i in range(100, 120)
    ]
    _mirror(tmp_path, module.TRAIN_DIR, "train", train)
    _mirror(tmp_path, module.DEV_DIR, "dev", dev)
    return tmp_path


def _prepare(root, **kwargs):
    return module.prepare_operator_data(
        root,
        Path("data/new"),
        source_sizes=(2, 4),
        calibration_count=2,
        evaluation_count=3,
        **kwargs,
    )


def test_dry_run_writes_nothing_and_all_prior_roles_are_excluded(bundle):
    result = _prepare(bundle)
    assert not (bundle / "data/new").exists()
    selected = {qid for values in result["roles"].values() for qid in values}
    assert not selected & {_id(i) for i in (1, 2, 3, 4)}
    assert result["counts"] == {"source": 4, "calibration": 2, "evaluation": 3}
    assert result["prior_reservation_count"] == 4
    assert result["api_calls"] == 0
    assert result["gold_projected"] is False
    assert result["old_reserved_text_projected"] is False
    assert "cannot be certified" in result["normalized_exclusion_coverage"]


def test_export_runtime_only_with_fixed_corpus_and_no_labels(bundle):
    result = _prepare(bundle, prepare_new=True)
    output = bundle / "data/new"
    for role, ids in result["roles"].items():
        rows = [
            json.loads(line)
            for line in (output / f"{role}_runtime_questions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [row["question_id"] for row in rows] == ids
        assert all(set(row) == {"question_id", "text", "dataset"} for row in rows)
    assert not list(output.glob("*gold*"))
    assert result["corpus_statistics"] == {
        "old_documents": 1,
        "added_documents": 3,
        "total_documents": 4,
        "unique_titles": 2,
    }
    new_bytes = (output / "corpus.jsonl").read_bytes()
    assert new_bytes.startswith((bundle / Path(module.SHARED).parent / "corpus.jsonl").read_bytes())
    documents = [json.loads(line) for line in new_bytes.splitlines()]
    assert len([doc for doc in documents if doc["title"] == "Same title"]) == 3
    assert all(item["contains_gold"] is False for item in result["artifacts"].values())
    assert (output / "manifest.sha256").read_text().startswith(_sha(output / "manifest.json"))
    assert b"FORBIDDEN" not in b"".join(path.read_bytes() for path in output.iterdir())


def test_no_overwrite_even_when_existing_directory_is_partial(bundle):
    output = bundle / "data/new"
    output.mkdir()
    (output / "partial.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _prepare(bundle, prepare_new=True)
    assert (output / "partial.txt").read_text() == "keep"


def test_old_reserved_text_is_not_passed_to_normalization(bundle, monkeypatch):
    original = module.question_metadata
    seen = []

    def checked(qid, question, *, official_split):
        assert qid not in {_id(i) for i in (1, 2, 3, 4)}
        seen.append(qid)
        return original(qid, question, official_split=official_split)

    monkeypatch.setattr(module, "question_metadata", checked)
    _prepare(bundle)
    assert len(seen) == 46


def test_run_launch_and_claim_metadata_extend_reservations(bundle):
    _json(bundle / "runs/old/launch_plan.json", {"question_ids": [_id(5)]})
    _json(bundle / "runs/reserve.claim.json", {"roles": {"probe": [_id(6)]}})
    result = _prepare(bundle)
    assert result["prior_reservation_count"] == 6
    assert {_id(5), _id(6)} <= set(result["forbidden_question_ids"])


def test_registered_normalized_hash_is_used_without_opening_old_question(bundle):
    fingerprint = hashlib.sha256(b"private synthetic train 7").hexdigest()
    _json(
        bundle / "runs/old/launch_plan.json",
        {"question_ids": [_id(999)], "selected_normalized_question_sha256s": [fingerprint]},
    )
    result = _prepare(bundle)
    assert result["prior_normalized_hash_count"] == 1
    assert _id(7) in result["excluded_question_ids"]


def test_input_sha_mismatch_fails_before_output_creation(bundle):
    raw = bundle / module.DEV_DIR / "raw.json"
    raw.write_bytes(b"modified")
    with pytest.raises(ValueError, match="raw mirror SHA mismatch"):
        _prepare(bundle, prepare_new=True)
    assert not (bundle / "data/new").exists()


def test_existing_corpus_hash_mismatch_stops_before_any_output(bundle):
    corpus = bundle / Path(module.SHARED).parent / "corpus.jsonl"
    corpus.write_bytes(corpus.read_bytes() + b" ")
    with pytest.raises(ValueError, match="corpus SHA mismatch"):
        _prepare(bundle)


def test_missing_dev_does_not_create_train_heldout_fallback(bundle, monkeypatch):
    monkeypatch.setattr(module, "DEV_DIR", "data/missing_dev")
    with pytest.raises(FileNotFoundError):
        _prepare(bundle)


def test_output_cannot_target_project_root_or_escape(bundle):
    with pytest.raises(ValueError, match="inside project"):
        module.prepare_operator_data(bundle, bundle)
    with pytest.raises(ValueError, match="inside project"):
        module.prepare_operator_data(bundle, bundle.parent / "outside")


def test_hash_registry_accepts_nested_roles_and_rejects_bad_hashes():
    assert module._hash_registry(
        {"selected_normalized_question_sha256s": {"source": ["a" * 64], "evaluation": ["b" * 64]}}
    ) == {"a" * 64, "b" * 64}
    with pytest.raises(ValueError, match="invalid registered"):
        module._hash_registry({"normalized_question_sha256": "bad"})


def test_background_ids_are_not_mistaken_for_supervised_exposure():
    assert module._reservation_ids(
        {
            "question_ids": [_id(1)],
            "background_question_ids": [_id(2)],
            "corpus_source_question_ids": [_id(1), _id(2)],
            "roles": {"probe": [_id(3)]},
        }
    ) == {_id(1), _id(3)}


def test_cli_prints_summary_not_questions_or_ids(bundle, monkeypatch, capsys):
    actual = module.prepare_operator_data

    def tiny(root, output, *, prepare_new):
        return actual(
            root,
            output,
            prepare_new=prepare_new,
            source_sizes=(2, 4),
            calibration_count=2,
            evaluation_count=3,
        )

    monkeypatch.setattr(module, "prepare_operator_data", tiny)
    module.main(["--root", str(bundle), "--output-dir", "data/new"])
    output = capsys.readouterr().out
    assert "Private synthetic" not in output
    assert _id(5) not in output
    assert json.loads(output)["counts"]["evaluation"] == 3


def test_no_supporting_facts_or_answers_projected_by_parquet(bundle, monkeypatch):
    import pyarrow.dataset as ds

    original = ds.dataset
    projected = []

    class CheckedDataset:
        def __init__(self, dataset):
            self.dataset = dataset

        def to_table(self, *, columns, **kwargs):
            assert set(columns) <= {"id", "question", "context"}
            projected.append(columns)
            return self.dataset.to_table(columns=columns, **kwargs)

    monkeypatch.setattr(
        ds, "dataset", lambda *args, **kwargs: CheckedDataset(original(*args, **kwargs))
    )
    _prepare(bundle, prepare_new=True)
    assert ["id", "context"] in projected
