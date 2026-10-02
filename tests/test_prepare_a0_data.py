"""Synthetic-only A0 data tests: no real export, gold, key access or API calls."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import prepare_a0_data as prep
from growrag.experiments import prepare_history_opportunity as old


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prep._canonical(value) + b"\n")


def pin_manifest(directory, manifest):
    write(directory / "manifest.json", manifest)
    digest = prep._sha(directory / "manifest.json")
    (directory / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
    return digest


@pytest.fixture
def sample(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path
    ids = [f"{number:024x}" for number in range(7692)]
    source_id, calibration_id, dev_id = (f"{number:024x}" for number in (10000, 10001, 20000))
    provenance = root / prep.PROVENANCE
    provenance.parent.mkdir(parents=True)
    shard = provenance.parent / "train-0.parquet"
    text_overrides = {
        source_id: "Old SOURCE query!",
        calibration_id: "Old calibration query",
        ids[42]: "old source query",  # Different ID, same normalized source text.
        ids[43]: "OLD calibration query!",
        ids[44]: "Old dev query",
        ids[45]: "Synthetic question 2",  # Same normalized text as pending claim.
        ids[46]: "Duplicated new candidate",
        ids[47]: "DUPLICATED new candidate!",
    }
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": qid,
                    "question": text_overrides.get(qid, f"Synthetic question {int(qid, 16)}?"),
                    "answer": "FORBIDDEN",
                    "supporting_facts": "FORBIDDEN",
                    "context": "FORBIDDEN",
                    "type": "FORBIDDEN",
                    "level": "FORBIDDEN",
                }
                for qid in [*ids, source_id, calibration_id]
            ]
        ),
        shard,
    )
    write(
        provenance,
        {
            "official_split": "train",
            "shards": [{"file": shard.name, "sha256": prep._sha(shard)}],
        },
    )
    write(
        root / prep.BACKGROUND_MANIFEST,
        {
            "official_split": "train",
            "background_question_ids": ids,
            "corpus_source_question_ids": ids,
            "question_ids": [ids[0]],
        },
    )
    corpus = root / prep.CORPUS
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_bytes(b'{"synthetic":"unchanged corpus"}\n')
    (root / prep.INDEX).write_bytes(b"synthetic frozen existing index")
    source = {
        "roles": {
            "source": [source_id],
            "calibration": [calibration_id],
            "evaluation": [dev_id],
        },
    }
    write(root / prep.SOURCE_MANIFEST, source)
    runtime = (root / prep.SOURCE_MANIFEST).parent / "evaluation_runtime_questions.jsonl"
    write(runtime, {"question_id": dev_id, "text": "OLD dev query!", "dataset": "old-dev"})
    write(root / prep.LIBRARY, {"synthetic": "existing library unchanged"})
    write(root / "runs/pending.claim.json", {"question_ids": [ids[2]], "status": "pending"})
    write(root / "runs/prior/launch_plan.json", {"question_ids": [ids[3]]})
    write(root / "data/hotpotqa/old/manifest.json", {"roles": {"reserved": [ids[4]]}})
    write(
        root / "runs/history_tests_fixture/test_one/runs/fake.claim.json",
        {
            "roles": {"source": ["not-an-id", ids[5]]},
        },
    )
    write(
        root / old.OUTPUT / "manifest.json",
        {
            "schema_version": old.SCHEMA,
            "question_ids": ids[100:150],
            "normalized_question_sha256s": [
                prep._question_hash(f"Synthetic question {i}?") for i in range(100, 150)
            ],
        },
    )
    pins = {relative: prep._sha(root / relative) for relative in prep.PINS}
    monkeypatch.setattr(prep, "PINS", pins)
    monkeypatch.setattr(prep, "LIBRARY_FINGERPRINT", "synthetic-library-fingerprint")

    def anchors(project):
        # Production delegates to unchanged parent _anchors; synthetic tests replace
        # this boundary only, never parent constants or real-data identity pins.
        assert project == root
        for relative, expected in pins.items():
            prep._check(prep._sha(project / relative) == expected, "pinned input bytes changed")
        declared = prep._read_json(provenance)["shards"][0]
        prep._check(prep._sha(shard) == declared["sha256"], "pinned shard bytes changed")
        return (
            source,
            set(ids),
            [str(shard)],
            [
                {"path": shard.relative_to(root).as_posix(), "sha256": declared["sha256"]},
            ],
        )

    monkeypatch.setattr(prep, "_anchors", anchors)
    return root, ids


def test_read_only_exact_projection_all_old_roles_claims_and_previous_50(sample, monkeypatch):
    import pyarrow.dataset as ds

    root, ids = sample
    original = ds.dataset
    projections = []

    class ProjectOnly:
        def to_table(self, **kwargs):
            projections.append(kwargs["columns"])
            assert kwargs["columns"] == ["id", "question"]
            shards = [str(root / Path(prep.PROVENANCE).parent / "train-0.parquet")]
            return original(shards, format="parquet").to_table(**kwargs)

    monkeypatch.setattr(ds, "dataset", lambda *args, **kwargs: ProjectOnly())
    before = sorted(str(path) for path in root.rglob("*") if path.is_file())
    manifest = prep.prepare(root)
    assert projections == [["id", "question"]]
    assert sorted(str(path) for path in root.rglob("*") if path.is_file()) == before
    assert manifest["schema_version"] == "growrag-a0-data-v1"
    assert manifest["seed"] == "growrag-a0-query-construction-20261002-v1"
    assert len(manifest["question_ids"]) == 100
    assert not set(manifest["question_ids"]) & set(ids[:1] + ids[2:6] + ids[42:48] + ids[100:150])
    assert manifest["preflight"]["known_hash_matches"] == 4
    assert manifest["preflight"]["duplicate_hash_groups"] == 1
    assert manifest["preflight"]["eligible_count"] == 7692 - 55 - 6
    assert manifest["preflight"]["old_role_overlap"] == {
        "source": 0,
        "calibration": 0,
        "evaluation": 0,
    }
    assert manifest["preflight"]["blocked_train_texts_projected"] == 57
    assert manifest["preflight"]["unavailable_blocked_texts"] == 0
    assert not (root / prep.OUTPUT).exists()
    assert old.COUNT == 50 and old.OUTPUT != prep.OUTPUT and old.PINS is not prep.PINS


def test_selection_and_entire_manifest_independent_of_projection_order(sample, monkeypatch):
    root, _ = sample
    first = prep.prepare(root)
    project = prep._project
    monkeypatch.setattr(prep, "_project", lambda shards, ids: list(reversed(project(shards, ids))))
    assert prep.prepare(root) == first
    # Independent seed is not an accidental reuse of the old 50-question order.
    expected = sorted(
        first["question_ids"],
        key=lambda qid: (
            hashlib.sha256(f"{prep.SEED}:development:{qid}".encode()).hexdigest(),
            qid,
        ),
    )
    assert first["question_ids"] == expected


def test_exclusive_export_mandatory_external_pin_and_unchanged_anchors(sample):
    root, _ = sample
    before = {path: (root / path).read_bytes() for path in (*prep.PINS, prep.INDEX)}
    planned = prep.prepare(root)
    assert prep.prepare(root, prepare_new=True) == planned
    directory = root / prep.OUTPUT
    sha = prep._sha(directory / "manifest.json")
    manifest, questions, corpus = prep.load_bundle(root, expected_sha=sha)
    assert len(questions) == 100
    assert [question.question_id for question in questions] == manifest["question_ids"]
    assert corpus == root / prep.CORPUS
    assert all(question.dataset == prep.DATASET for question in questions)
    assert {path: (root / path).read_bytes() for path in before} == before
    assert set(manifest["question_ids"]) <= set(prep._registry(root)["question_ids"])
    with pytest.raises(TypeError, match="expected_sha"):
        prep.load_bundle(root)
    for invalid in (None, "", "0" * 64):
        with pytest.raises(ValueError, match="externally pinned"):
            prep.load_bundle(root, expected_sha=invalid)
    for prepare_new in (False, True):
        with pytest.raises(FileExistsError):
            prep.prepare(root, prepare_new=prepare_new)


def test_registered_hash_and_missing_claim_text_are_handled(sample):
    root, ids = sample
    missing_id = f"{30000:024x}"
    write(
        root / "runs/hashonly/plan.json",
        {
            "question_ids": [missing_id],
            "normalized_question_sha256s": [prep._question_hash("Synthetic question 60?")],
        },
    )
    manifest = prep.prepare(root)
    assert ids[60] not in manifest["question_ids"]
    assert manifest["preflight"]["known_hash_matches"] == 5
    assert manifest["exclusion_registry"]["unavailable_text_question_ids"] == [missing_id]


def test_nested_data_and_real_run_manifests_are_also_protected(sample):
    root, ids = sample
    write(root / "data/hotpotqa/nested/experiment/manifest.json", {"question_ids": [ids[60]]})
    write(
        root / "runs/previous_experiment/manifest.json",
        {
            "question_ids": [ids[61]],
            "normalized_question_sha256s": [prep._question_hash("Synthetic question 62?")],
        },
    )
    manifest = prep.prepare(root)
    assert not set(ids[60:63]) & set(manifest["question_ids"])
    assert set(ids[60:62]) <= set(manifest["exclusion_registry"]["question_ids"])
    assert manifest["preflight"]["known_hash_matches"] == 5


def test_insufficient_independent_pool_fails_without_export(sample):
    root, ids = sample
    write(root / "runs/huge_reservation.claim.json", {"question_ids": ids[:7600]})
    with pytest.raises(ValueError, match="not enough independent"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


@pytest.mark.parametrize(
    "kind", ["missing_row", "extra_column", "duplicate_id", "changed_corpus", "changed_shard"]
)
def test_projection_or_anchor_failure_leaves_no_output(sample, monkeypatch, kind):
    root, _ = sample
    original = prep._project
    if kind == "missing_row":
        monkeypatch.setattr(prep, "_project", lambda shards, ids: original(shards, ids)[1:])
    elif kind == "extra_column":
        monkeypatch.setattr(
            prep,
            "_project",
            lambda shards, ids: [{**row, "answer": "FORBIDDEN"} for row in original(shards, ids)],
        )
    elif kind == "duplicate_id":
        monkeypatch.setattr(
            prep, "_project", lambda shards, ids: original(shards, ids) + original(shards, ids)[:1]
        )
    else:
        path = root / (
            prep.CORPUS
            if kind == "changed_corpus"
            else str(Path(prep.PROVENANCE).parent / "train-0.parquet")
        )
        path.write_bytes(path.read_bytes() + b"CHANGED")
    with pytest.raises(ValueError):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


def test_pending_claim_during_projection_aborts_without_replacement(sample, monkeypatch):
    root, _ = sample
    original = prep._project

    def changed(shards, ids):
        rows = original(shards, ids)
        write(root / "runs/new_pending.claim.json", {"question_ids": [rows[70]["id"]]})
        return rows

    monkeypatch.setattr(prep, "_project", changed)
    with pytest.raises(ValueError, match="registry changed"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


@pytest.mark.parametrize("kind", ["index", "runtime", "manifest", "sidecar"])
def test_tampered_frozen_bytes_rejected(sample, kind):
    root, _ = sample
    prep.prepare(root, prepare_new=True)
    directory = root / prep.OUTPUT
    sha = prep._sha(directory / "manifest.json")
    target = {
        "index": root / prep.INDEX,
        "runtime": directory / "runtime_questions.jsonl",
        "manifest": directory / "manifest.json",
        "sidecar": directory / "manifest.sha256",
    }[kind]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        prep.load_bundle(root, expected_sha=sha)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_manifest",
        "false_as_zero",
        "dataset",
        "runtime_gold",
        "runtime_order",
        "protected_id",
    ],
)
def test_self_consistent_repin_does_not_bypass_strict_schema(sample, kind):
    root, ids = sample
    prep.prepare(root, prepare_new=True)
    directory = root / prep.OUTPUT
    manifest = prep._read_json(directory / "manifest.json")
    if kind == "unknown_manifest":
        manifest["answer"] = "FORBIDDEN"
    elif kind == "false_as_zero":
        manifest["gold_projected"] = 0
    elif kind == "protected_id":
        manifest["question_ids"][0] = ids[2]
        manifest["question_ids_sha256"] = prep._digest(manifest["question_ids"])
    else:
        runtime = directory / "runtime_questions.jsonl"
        rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines()]
        if kind == "dataset":
            rows[0]["dataset"] = "wrong-cohort"
        elif kind == "runtime_gold":
            rows[0]["answer"] = "FORBIDDEN"
        else:
            rows.reverse()
        runtime.write_bytes(b"".join(prep._canonical(row) + b"\n" for row in rows))
        manifest["runtime_artifact"].update(sha256=prep._sha(runtime), bytes=runtime.stat().st_size)
    sha = pin_manifest(directory, manifest)
    with pytest.raises(ValueError):
        prep.load_bundle(root, expected_sha=sha)


def test_cli_default_does_not_export_or_print_question_texts(sample, monkeypatch, capsys):
    root, _ = sample
    monkeypatch.chdir(root)
    assert prep.main([]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["prepared"] is False
    assert "Synthetic question" not in output and "FORBIDDEN" not in output
    assert not (root / prep.OUTPUT).exists()
