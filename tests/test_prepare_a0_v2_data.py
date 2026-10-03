"""Synthetic-only v2 isolation checks; no real exports, keys, gold or API calls."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import prepare_a0_data as a0
from growrag.experiments import prepare_a0_v2_data as prep


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
    source_id, calibration_id, eval_id = (f"{number:024x}" for number in (10000, 10001, 20000))
    source = {
        "roles": {"source": [source_id], "calibration": [calibration_id], "evaluation": [eval_id]}
    }
    overrides = {
        source_id: "Sealed source question!",
        calibration_id: "Sealed calibration question?",
        ids[40]: "SEALED source question?",
        ids[41]: "SEALED calibration question!",
        ids[42]: "Sealed evaluation question",
        ids[43]: "Synthetic question 2",  # Pending claim, different ID.
        ids[44]: "Synthetic question 151",  # Unattempted v1, different ID.
        ids[45]: "Synthetic question 148",  # Unpaired v1, different ID.
        ids[46]: "Duplicated candidate?",
        ids[47]: "DUPLICATED candidate!",
    }
    shard = root / Path(a0.PROVENANCE).parent / "train-0.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": qid,
                    "question": overrides.get(qid, f"Synthetic question {int(qid, 16)}?"),
                    **dict.fromkeys(
                        ("answer", "supporting_facts", "context", "type", "level"), "FORBIDDEN"
                    ),
                }
                for qid in [*ids, source_id, calibration_id]
            ]
        ),
        shard,
    )
    write(
        root / a0.PROVENANCE,
        {"official_split": "train", "shards": [{"file": shard.name, "sha256": prep._sha(shard)}]},
    )
    write(
        root / a0.BACKGROUND_MANIFEST,
        {
            "official_split": "train",
            "background_question_ids": ids,
            "corpus_source_question_ids": ids,
            "question_ids": ids[:1],
        },
    )
    write(root / a0.SOURCE_MANIFEST, source)
    write(
        (root / a0.SOURCE_MANIFEST).parent / "evaluation_runtime_questions.jsonl",
        {
            "question_id": eval_id,
            "text": "SEALED evaluation question!",
            "dataset": "old-evaluation",
        },
    )
    write(root / prep.CORPUS, {"synthetic": "unchanged corpus"})
    (root / prep.INDEX).write_bytes(b"unchanged synthetic index")
    write(root / a0.LIBRARY, {"synthetic": "unchanged library"})
    write(root / "runs/pending.claim.json", {"question_ids": ids[2:3], "status": "pending"})
    write(root / "runs/previous/launch_plan.json", {"question_ids": ids[3:4]})
    write(
        root / "data/hotpotqa/history_opportunity_20261002_v1/manifest.json",
        {"question_ids": ids[200:250]},
    )
    old_ids = ids[100:200]
    write(
        root / prep.OLD_A0_MANIFEST,
        {
            "schema_version": a0.SCHEMA,
            "question_ids": old_ids,
            "normalized_question_sha256s": [
                prep._question_hash(f"Synthetic question {i}?") for i in range(100, 200)
            ],
        },
    )
    # 48 paired + 3 unpaired were attempted, but all 100 must stay reserved.
    write(
        root / "runs/old_a0.claim.json", {"question_ids": ids[100:151], "status": "technical_stop"}
    )
    write(root / "runs/old_a0/plan.json", {"question_ids": old_ids})
    write(
        root / "runs/history_tests_fixture/test_synthetic/runs/extra.claim.json",
        {"question_ids": [ids[4], "not-an-id"]},
    )
    for relative in ("runs/old_a0/results.jsonl", "data/hotpotqa/old_sealed_gold.jsonl"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT JSON; MUST NEVER BE READ", encoding="utf-8")
    pins = {relative: prep._sha(root / relative) for relative in prep.PINS}
    # Only synthetic fixtures replace these I/O identities. Production never
    # assigns to v1 globals and always uses the original frozen anchor pins.
    monkeypatch.setattr(prep, "PINS", pins)
    monkeypatch.setattr(a0, "PINS", pins)
    monkeypatch.setattr(a0, "LIBRARY_FINGERPRINT", "synthetic-fingerprint")
    monkeypatch.setattr(prep, "OLD_A0_MANIFEST_SHA", prep._sha(root / prep.OLD_A0_MANIFEST))

    def anchors(project):
        assert project == root
        for relative, expected in pins.items():
            prep._check(prep._sha(project / relative) == expected, "pinned input bytes changed")
        declared = prep._read_json(root / a0.PROVENANCE)["shards"][0]
        prep._check(prep._sha(shard) == declared["sha256"], "pinned shard bytes changed")
        return (
            source,
            set(ids),
            [str(shard)],
            [{"path": shard.relative_to(root).as_posix(), "sha256": declared["sha256"]}],
        )

    monkeypatch.setattr(prep, "_anchors", anchors)
    return root, ids


def test_read_only_projection_excludes_entire_v1_and_normalized_old_text(sample, monkeypatch):
    import pyarrow.dataset as ds

    root, ids = sample
    dataset = ds.dataset
    projections = []

    class QuestionOnly:
        def __init__(self, *args, **kwargs):
            self.dataset = dataset(*args, **kwargs)

        def to_table(self, **kwargs):
            projections.append(kwargs["columns"])
            assert kwargs["columns"] == ["id", "question"]
            return self.dataset.to_table(**kwargs)

    monkeypatch.setattr(ds, "dataset", QuestionOnly)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original_constants = (a0.SCHEMA, a0.SEED, a0.COUNT, a0.OUTPUT, a0.DATASET, a0.ROLE)
    manifest = prep.prepare(root)
    assert projections == [["id", "question"]]
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert (a0.SCHEMA, a0.SEED, a0.COUNT, a0.OUTPUT, a0.DATASET, a0.ROLE) == original_constants
    assert manifest["schema_version"] == "growrag-a0-v2-data-v1"
    assert manifest["seed"] == "growrag-a0-wire-v2-20261003"
    assert manifest["role"] == prep.ROLE != a0.ROLE
    assert len(manifest["question_ids"]) == 100
    forbidden = ids[:1] + ids[2:5] + ids[40:48] + ids[100:250]
    assert not set(manifest["question_ids"]) & set(forbidden)
    registry = manifest["exclusion_registry"]
    assert set(ids[100:148]) <= set(registry["question_ids"])
    assert set(ids[148:151]) <= set(registry["question_ids"])
    assert set(ids[151:200]) <= set(registry["question_ids"])
    assert manifest["preflight"]["known_hash_matches"] == 6
    assert manifest["preflight"]["duplicate_hash_groups"] == 1
    assert manifest["preflight"]["eligible_count"] == 7692 - 154 - 8
    assert manifest["preflight"]["old_role_overlap"] == {
        "source": 0,
        "calibration": 0,
        "evaluation": 0,
    }
    assert manifest["preflight"]["unavailable_blocked_texts"] == 0
    assert not (root / prep.OUTPUT).exists()


def test_repeatable_independent_seed_order_without_gold_or_score_inputs(sample, monkeypatch):
    root, _ = sample
    first = prep.prepare(root)
    project = a0._project
    monkeypatch.setattr(a0, "_project", lambda shards, ids: list(reversed(project(shards, ids))))
    assert prep.prepare(root) == first
    expected = sorted(
        first["question_ids"],
        key=lambda qid: (
            hashlib.sha256(f"{prep.SEED}:development:{qid}".encode()).hexdigest(),
            qid,
        ),
    )
    assert first["question_ids"] == expected
    assert prep.SEED != a0.SEED and prep.DATASET != a0.DATASET
    assert first["gold_projected"] is False
    assert first["selection_uses_scores_or_difficulty"] is False


def test_exclusive_export_shared_contract_and_external_pin(sample):
    root, _ = sample
    before = {
        path: (root / path).read_bytes() for path in (*prep.PINS, prep.INDEX, prep.OLD_A0_MANIFEST)
    }
    planned = prep.prepare(root)
    assert prep.prepare(root, prepare_new=True) == planned
    directory = root / prep.OUTPUT
    sha = prep._sha(directory / "manifest.json")
    manifest, questions, corpus = prep.load_bundle(root, expected_sha=sha)
    assert manifest == planned
    assert prep._sha(directory / "manifest.json") == sha
    assert len(questions) == 100
    assert [row.question_id for row in questions] == manifest["question_ids"]
    assert all(row.dataset == prep.DATASET for row in questions)
    assert corpus == root / prep.CORPUS
    assert {path: (root / path).read_bytes() for path in before} == before
    assert set(manifest["question_ids"]) <= set(prep._registry_snapshot(root)["question_ids"])
    assert set(prep._fixed()) == set(a0._fixed())
    with pytest.raises(TypeError, match="expected_sha"):
        prep.load_bundle(root)
    for invalid in (None, "", "0" * 64):
        with pytest.raises(ValueError, match="externally pinned"):
            prep.load_bundle(root, expected_sha=invalid)
    for flag in (False, True):
        with pytest.raises(FileExistsError):
            prep.prepare(root, prepare_new=flag)


def test_nested_prior_manifests_claims_and_registered_hashes_are_excluded(sample):
    root, ids = sample
    missing = f"{30000:024x}"
    write(root / "data/hotpotqa/nested/cohort/manifest.json", {"question_ids": ids[60:61]})
    write(
        root / "runs/prior_nested/manifest.json",
        {
            "question_ids": ids[61:62],
            "normalized_question_sha256s": [prep._question_hash("Synthetic question 62")],
        },
    )
    write(
        root / "runs/hash_only/plan.json",
        {
            "question_ids": [missing],
            "normalized_question_sha256s": [prep._question_hash("Synthetic question 63")],
        },
    )
    manifest = prep.prepare(root)
    assert not set(ids[60:64]) & set(manifest["question_ids"])
    assert set(ids[60:62]) <= set(manifest["exclusion_registry"]["question_ids"])
    assert manifest["exclusion_registry"]["unavailable_text_question_ids"] == [missing]


def test_unregistered_or_modified_old_cohort_fails_before_export(sample, monkeypatch):
    root, ids = sample
    original = prep._registry_snapshot

    def incomplete(project):
        value = original(project)
        value["question_ids"].remove(ids[199])
        return value

    monkeypatch.setattr(prep, "_registry_snapshot", incomplete)
    with pytest.raises(ValueError, match="complete sealed old A0 cohort"):
        prep.prepare(root, prepare_new=True)
    monkeypatch.setattr(prep, "_registry_snapshot", original)
    old = root / prep.OLD_A0_MANIFEST
    old.write_bytes(old.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sealed old A0 manifest changed"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


def test_insufficient_pool_never_replaces_or_exports(sample):
    root, ids = sample
    write(root / "runs/large.claim.json", {"question_ids": ids[:7600]})
    with pytest.raises(ValueError, match="not enough independent"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


@pytest.mark.parametrize(
    "kind", ["missing_row", "extra_column", "duplicate_id", "changed_corpus", "changed_shard"]
)
def test_projection_or_anchor_failure_leaves_no_output(sample, monkeypatch, kind):
    root, _ = sample
    project = a0._project
    if kind == "missing_row":
        monkeypatch.setattr(a0, "_project", lambda shards, ids: project(shards, ids)[1:])
    elif kind == "extra_column":
        monkeypatch.setattr(
            a0,
            "_project",
            lambda shards, ids: [{**row, "answer": "FORBIDDEN"} for row in project(shards, ids)],
        )
    elif kind == "duplicate_id":
        monkeypatch.setattr(
            a0, "_project", lambda shards, ids: project(shards, ids) + project(shards, ids)[:1]
        )
    else:
        path = root / (
            prep.CORPUS
            if kind == "changed_corpus"
            else str(Path(a0.PROVENANCE).parent / "train-0.parquet")
        )
        path.write_bytes(path.read_bytes() + b"CHANGED")
    with pytest.raises(ValueError):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


def test_pending_claim_during_projection_aborts_without_replacement(sample, monkeypatch):
    root, _ = sample
    project = a0._project

    def changed(shards, ids):
        rows = project(shards, ids)
        write(root / "runs/new_pending.claim.json", {"question_ids": [rows[70]["id"]]})
        return rows

    monkeypatch.setattr(a0, "_project", changed)
    with pytest.raises(ValueError, match="registry changed"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


@pytest.mark.parametrize("kind", ["index", "runtime", "manifest", "sidecar", "old_a0"])
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
        "old_a0": root / prep.OLD_A0_MANIFEST,
    }[kind]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        prep.load_bundle(root, expected_sha=sha)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", a0.SCHEMA),
        ("seed", a0.SEED),
        ("role", a0.ROLE),
        ("count", 100.0),
        ("gold_projected", 0),
        ("api_calls", False),
        ("memory_updates_allowed", True),
        ("official_test_used", True),
        ("selection_uses_scores_or_difficulty", True),
        ("projected_columns", ["id", "question", "answer"]),
    ],
)
def test_fixed_v2_contract_rejects_self_consistent_repin(sample, field, value):
    root, _ = sample
    manifest = prep.prepare(root, prepare_new=True)
    manifest[field] = value
    sha = pin_manifest(root / prep.OUTPUT, manifest)
    with pytest.raises(ValueError, match="v2 runtime bundle fields/protocol"):
        prep.load_bundle(root, expected_sha=sha)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_manifest",
        "dataset",
        "runtime_gold",
        "runtime_order",
        "protected_id",
        "old_reservation",
        "gold_declaration",
        "runtime_rows_boolean",
    ],
)
def test_repin_does_not_bypass_strict_runtime_schema_or_exclusions(sample, kind):
    root, ids = sample
    manifest = prep.prepare(root, prepare_new=True)
    directory = root / prep.OUTPUT
    if kind == "unknown_manifest":
        manifest["answer"] = "FORBIDDEN"
    elif kind == "protected_id":
        manifest["question_ids"][0] = ids[199]
        manifest["question_ids_sha256"] = prep._digest(manifest["question_ids"])
    elif kind == "old_reservation":
        manifest["exclusion_registry"]["question_ids"].remove(ids[199])
    elif kind == "gold_declaration":
        manifest["runtime_artifact"]["contains_gold"] = 0
    elif kind == "runtime_rows_boolean":
        manifest["runtime_artifact"]["rows"] = True
    else:
        runtime = directory / "runtime_questions.jsonl"
        rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines()]
        if kind == "dataset":
            rows[0]["dataset"] = a0.DATASET
        elif kind == "runtime_gold":
            rows[0]["answer"] = "FORBIDDEN"
        else:
            rows.reverse()
        runtime.write_bytes(b"".join(prep._canonical(row) + b"\n" for row in rows))
        manifest["runtime_artifact"].update(sha256=prep._sha(runtime), bytes=runtime.stat().st_size)
    sha = pin_manifest(directory, manifest)
    with pytest.raises(ValueError):
        prep.load_bundle(root, expected_sha=sha)


def test_cli_default_never_exports_or_prints_question_texts(sample, monkeypatch, capsys):
    root, _ = sample
    monkeypatch.chdir(root)
    assert prep.main([]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["prepared"] is False
    assert "Synthetic question" not in output and "FORBIDDEN" not in output
    assert not (root / prep.OUTPUT).exists()
