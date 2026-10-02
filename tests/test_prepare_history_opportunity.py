"""Synthetic parquet only: no real train/dev labels, paid requests or real export."""

import json
from pathlib import Path

import pytest

from growrag.experiments import prepare_history_opportunity as prep
from growrag.history_library import FrozenHistoryLibrary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def sample(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ids = [f"{number:024x}" for number in range(7692)]
    root = tmp_path
    provenance = root / prep.PROVENANCE
    provenance.parent.mkdir(parents=True)
    shard = provenance.parent / "train-0.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": qid,
                    "question": f"Synthetic independent question {i}?",
                    "answer": "DO NOT PROJECT",
                    "supporting_facts": "DO NOT PROJECT",
                    "context": "DO NOT PROJECT",
                    "type": "DO NOT PROJECT",
                    "level": "DO NOT PROJECT",
                }
                for i, qid in enumerate(ids)
            ]
        ),
        shard,
    )
    write(
        provenance,
        {"official_split": "train", "shards": [{"file": shard.name, "sha256": prep._sha(shard)}]},
    )
    background = root / prep.BACKGROUND_MANIFEST
    write(
        background,
        {
            "official_split": "train",
            "background_question_ids": ids,
            "corpus_source_question_ids": ids,
            "question_ids": [ids[0]],
        },
    )
    corpus = root / prep.CORPUS
    corpus.parent.mkdir(parents=True)
    corpus.write_bytes(b'{"synthetic":"unchanged"}\n')
    (root / prep.INDEX).write_bytes(b"synthetic existing index, never opened or changed")
    source_ids = [f"{10000:024x}"]
    source = root / prep.SOURCE_MANIFEST
    write(
        source,
        {
            "roles": {
                "source": source_ids,
                "calibration": [ids[1]],
                "evaluation": [f"{20000:024x}"],
            },
            "input_files": [
                {"path": p.relative_to(root).as_posix(), "sha256": prep._sha(p)}
                for p in (provenance, background, shard)
            ],
            "artifacts": {"corpus.jsonl": {"sha256": prep._sha(corpus), "rows": 76691}},
        },
    )
    library = FrozenHistoryLibrary("synthetic", tuple(source_ids), ())
    library_path = root / prep.LIBRARY
    library_path.parent.mkdir(parents=True)
    library_path.write_text(library.to_json(), encoding="utf-8")
    write(root / "runs/pending.claim.json", {"question_ids": [ids[2]], "status": "pending"})
    write(root / "runs/prior/launch_plan.json", {"question_ids": [ids[3]]})
    write(root / "data/hotpotqa/old/manifest.json", {"roles": {"reserved": [ids[4]]}})
    # Ignore its synthetic schema, but conservatively protect a valid ID it mentions.
    write(
        root / "runs/history_tests_fixture/test_one/runs/fake.claim.json",
        {"roles": {"source": ["not-an-id", ids[5]]}},
    )
    write(
        root / "runs/history_tests_fixture/test_one/runs/fake/launch_plan.json",
        {"roles": {"source": ["not-an-id"]}},
    )
    monkeypatch.setattr(
        prep, "PINS", {relative: prep._sha(root / relative) for relative in prep.PINS}
    )
    monkeypatch.setattr(prep, "LIBRARY_FINGERPRINT", library.fingerprint)
    return root, ids


def test_default_is_read_only_and_exact_columns(sample, monkeypatch):
    import pyarrow.dataset as ds

    root, ids = sample
    original = ds.dataset
    observed = []

    class ReadOnlyProjection:
        def to_table(self, **kwargs):
            observed.append(kwargs["columns"])
            assert kwargs["columns"] == ["id", "question"]
            return original(
                [str(root / Path(prep.PROVENANCE).parent / "train-0.parquet")], format="parquet"
            ).to_table(**kwargs)

    monkeypatch.setattr(ds, "dataset", lambda *a, **kw: ReadOnlyProjection())
    before = sorted(str(p) for p in root.rglob("*") if p.is_file())
    result = prep.prepare(root)
    assert observed == [["id", "question"]]
    assert sorted(str(p) for p in root.rglob("*") if p.is_file()) == before
    assert len(result["question_ids"]) == 50 and not set(ids[:6]) & set(result["question_ids"])
    assert result["preflight"]["eligible_count"] == 7686
    assert result["preflight"]["old_role_overlap"] == {
        "source": 0,
        "calibration": 0,
        "evaluation": 0,
    }
    assert result["exclusion_registry"]["claim_files_checked"] == 2
    assert result["exclusion_registry"]["conservative_extra_claim_ids"] == [ids[5]]
    assert not (root / prep.OUTPUT).exists()


def test_exclusive_export_load_and_unchanged_corpus(sample):
    root, _ = sample
    corpus_before = (root / prep.CORPUS).read_bytes()
    first = prep.prepare(root)
    result = prep.prepare(root, prepare_new=True)
    assert first == result
    directory = root / prep.OUTPUT
    assert sorted(p.name for p in directory.iterdir()) == [
        "manifest.json",
        "manifest.sha256",
        "runtime_questions.jsonl",
    ]
    sha = prep._sha(directory / "manifest.json")
    manifest, questions, corpus = prep.load_bundle(root, expected_sha=sha)
    assert [q.question_id for q in questions] == manifest["question_ids"]
    assert len(questions) == 50 and corpus == root / prep.CORPUS
    assert corpus.read_bytes() == corpus_before
    assert (root / prep.INDEX).read_bytes() == b"synthetic existing index, never opened or changed"
    with pytest.raises(FileExistsError):
        prep.prepare(root)
    with pytest.raises(FileExistsError):
        prep.prepare(root, prepare_new=True)
    with pytest.raises(ValueError, match="externally pinned"):
        prep.load_bundle(root, expected_sha="0" * 64)


def test_hash_sampling_not_source_row_order_and_duplicate_groups_excluded(sample, monkeypatch):
    root, ids = sample
    project = prep._project
    first = prep.prepare(root)
    monkeypatch.setattr(
        prep, "_project", lambda shards, eligible: list(reversed(project(shards, eligible)))
    )
    assert prep.prepare(root)["question_ids"] == first["question_ids"]

    def duplicate(shards, eligible):
        rows = project(shards, eligible)
        for row in rows:
            if row["id"] in ids[42:44]:
                row["question"] = "Same duplicated synthetic query"
        return rows

    monkeypatch.setattr(prep, "_project", duplicate)
    result = prep.prepare(root)
    assert result["preflight"]["duplicate_hash_groups"] == 1
    assert result["preflight"]["eligible_count"] == 7684
    assert not set(ids[42:44]) & set(result["question_ids"])


def test_registered_normalized_hash_is_excluded(sample):
    import hashlib

    root, ids = sample
    digest = hashlib.sha256(
        prep.normalize_question("Synthetic independent question 42?").encode()
    ).hexdigest()
    write(root / "runs/hashonly/plan.json", {"normalized_question_sha256s": [digest]})
    result = prep.prepare(root)
    assert result["preflight"]["known_hash_matches"] == 1
    assert result["preflight"]["eligible_count"] == 7685
    assert ids[42] not in result["question_ids"]


@pytest.mark.parametrize("kind", ["missing_row", "extra_column", "changed_corpus", "changed_shard"])
def test_invalid_projection_or_anchor_fails_before_output(sample, monkeypatch, kind):
    root, _ = sample
    original = prep._project
    if kind == "missing_row":
        monkeypatch.setattr(prep, "_project", lambda shards, ids: original(shards, ids)[1:])
    elif kind == "extra_column":
        monkeypatch.setattr(
            prep,
            "_project",
            lambda shards, ids: [
                {**row, "answer": "not permitted"} for row in original(shards, ids)
            ],
        )
    else:
        path = root / (
            prep.CORPUS
            if kind == "changed_corpus"
            else str(Path(prep.PROVENANCE).parent / "train-0.parquet")
        )
        with path.open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


def test_new_pending_claim_during_prepare_aborts_without_resampling(sample, monkeypatch):
    root, _ = sample
    original = prep._project

    def project(shards, ids):
        rows = original(shards, ids)
        write(root / "runs/new_pending.claim.json", {"question_ids": [rows[0]["id"]]})
        return rows

    monkeypatch.setattr(prep, "_project", project)
    with pytest.raises(ValueError, match="registry changed"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


def test_runtime_tampering_and_protected_ids_rejected(sample):
    root, ids = sample
    prep.prepare(root, prepare_new=True)
    directory = root / prep.OUTPUT
    manifest = prep._read_json(directory / "manifest.json")
    manifest["question_ids"][0] = ids[2]  # A pending claim is still protected.
    manifest["question_ids_sha256"] = prep._digest(manifest["question_ids"])
    write(directory / "manifest.json", manifest)
    (directory / "manifest.sha256").write_text(
        f"{prep._sha(directory / 'manifest.json')}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="frozen protections"):
        prep.load_bundle(root)
