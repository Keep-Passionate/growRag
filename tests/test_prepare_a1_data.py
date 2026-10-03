"""Independent synthetic A1 cohort tests: never sample real train or read gold/API keys."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import prepare_a0_data as a0
from growrag.experiments import prepare_a0_v2_data as a0v2
from growrag.experiments import prepare_a1_data as prep


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prep._canonical(value) + b"\n")


def repin(directory, manifest):
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
    source_id, calibration_id, evaluation_id = (f"{n:024x}" for n in (10000, 10001, 20000))
    overrides = {
        source_id: "OLD source question!",
        calibration_id: "OLD calibration question?",
        ids[41]: "old SOURCE question",
        ids[42]: "old calibration QUESTION!",
        ids[43]: "Old evaluation question",
        ids[44]: "SYNTHETIC question 2!",  # Pending claim, distinct ID.
        ids[45]: "Synthetic question 199",  # v1 never-attempted, distinct ID.
        ids[46]: "Synthetic question 399",  # v2 never-attempted, distinct ID.
        ids[48]: "Duplicate new question",
        ids[49]: "DUPLICATE new question!",
    }
    shard = root / Path(prep.PROVENANCE).parent / "train-0.parquet"
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
        root / prep.PROVENANCE,
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
            "question_ids": ids[:1],
        },
    )
    write(root / prep.CORPUS, {"opaque": "synthetic frozen corpus"})
    (root / prep.INDEX).write_bytes(b"opaque frozen index")
    library = root / prep.LIBRARY
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_bytes(b"NOT JSON: library must be hashed, never deserialized for examples")
    source = {
        "roles": {
            "source": [source_id],
            "calibration": [calibration_id],
            "evaluation": [evaluation_id],
        },
        "input_files": [
            {"path": relative, "sha256": prep._sha(root / relative)}
            for relative in (
                prep.BACKGROUND_MANIFEST,
                prep.PROVENANCE,
                shard.relative_to(root).as_posix(),
            )
        ],
        "artifacts": {"corpus.jsonl": {"sha256": prep._sha(root / prep.CORPUS), "rows": 76691}},
    }
    write(root / prep.SOURCE_MANIFEST, source)
    write(
        (root / prep.SOURCE_MANIFEST).parent / "evaluation_runtime_questions.jsonl",
        {
            "question_id": evaluation_id,
            "text": "OLD evaluation question!",
            "dataset": "old-eval",
        },
    )
    write(root / "runs/pending.claim.json", {"question_ids": ids[2:3], "status": "pending"})
    write(root / "runs/prior/launch_plan.json", {"question_ids": ids[3:4]})
    write(root / "data/hotpotqa/prior50/manifest.json", {"question_ids": ids[200:250]})
    write(root / "data/hotpotqa/nested/old/manifest.json", {"question_ids": ids[6:7]})
    write(
        root / "runs/history_tests_fixture/test_synthetic/runs/example.claim.json",
        {
            "question_ids": [ids[5], "synthetic-not-real-id"],
        },
    )
    for relative, cohort in zip(prep.A0_COHORTS, (ids[100:200], ids[300:400]), strict=True):
        write(
            root / relative,
            {
                "question_ids": cohort,
                "normalized_question_sha256s": [
                    prep._question_hash(f"Synthetic question {int(qid, 16)}?") for qid in cohort
                ],
            },
        )
    for relative in ("runs/prior/results.jsonl", "data/hotpotqa/sealed_gold.jsonl"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"NOT JSON: results/gold are forbidden")
    monkeypatch.setattr(
        prep, "PINS", {relative: prep._sha(root / relative) for relative in prep.PINS}
    )
    monkeypatch.setattr(
        prep, "A0_COHORTS", {relative: prep._sha(root / relative) for relative in prep.A0_COHORTS}
    )
    return root, ids


def test_read_only_projection_all_old_roles_claims_and_both_entire_a0_cohorts(sample, monkeypatch):
    import pyarrow.dataset as ds

    root, ids = sample
    original = ds.dataset
    projected = []

    class QuestionOnly:
        def __init__(self, *args, **kwargs):
            self.dataset = original(*args, **kwargs)

        def to_table(self, **kwargs):
            projected.append(kwargs["columns"])
            assert kwargs["columns"] == ["id", "question"]
            return self.dataset.to_table(**kwargs)

    monkeypatch.setattr(ds, "dataset", QuestionOnly)
    files = {str(path): prep._sha(path) for path in root.rglob("*") if path.is_file()}
    value = prep.prepare(root)
    assert projected == [["id", "question"]]
    assert {str(path): prep._sha(path) for path in root.rglob("*") if path.is_file()} == files
    assert value["count"] == 40 and len(value["question_ids"]) == 40
    assert value["schema_version"] == "growrag-a1-data-v1"
    assert value["seed"] == "growrag-a1-applicability-shadow-20261003-v1"
    assert value["role"] == "development_applicability_shadow_not_training"
    assert not set(value["question_ids"]) & set(
        ids[100:250] + ids[300:400] + ids[41:47] + ids[48:50]
    )
    registry = value["exclusion_registry"]
    assert set(ids[100:200] + ids[300:400]) <= set(registry["question_ids"])
    assert set(ids[:1] + ids[2:4] + ids[5:7]) <= set(registry["question_ids"])
    assert value["preflight"]["known_hash_matches"] == 6
    assert value["preflight"]["duplicate_hash_groups"] == 1
    assert value["preflight"]["old_role_overlap"] == {
        "source": 0,
        "calibration": 0,
        "evaluation": 0,
    }
    assert value["preflight"]["unavailable_blocked_texts"] == 0
    assert not (root / prep.OUTPUT).exists()
    assert a0.COUNT == a0v2.COUNT == 100 and prep.COUNT == 40


def test_seed_and_manifest_ignore_projection_order(sample, monkeypatch):
    root, _ = sample
    before = prep.prepare(root)
    original = a0._project
    monkeypatch.setattr(a0, "_project", lambda *args: list(reversed(original(*args))))
    assert prep.prepare(root) == before
    assert before["question_ids"] == sorted(
        before["question_ids"],
        key=lambda qid: (
            hashlib.sha256(f"{prep.SEED}:development:{qid}".encode()).hexdigest(),
            qid,
        ),
    )


def test_exclusive_40_bundle_load_has_no_parquet_or_100_row_validator(sample, monkeypatch):
    root, _ = sample
    before = {relative: (root / relative).read_bytes() for relative in (*prep.PINS, prep.INDEX)}
    planned = prep.prepare(root)
    assert prep.prepare(root, prepare_new=True) == planned
    directory = root / prep.OUTPUT
    digest = prep._sha(directory / "manifest.json")

    def forbidden(*args, **kwargs):
        pytest.fail("runtime loading must not project train or invoke a 100-row validator")

    monkeypatch.setattr(a0, "_project", forbidden)
    monkeypatch.setattr(a0, "_validate_manifest", forbidden)
    monkeypatch.setattr(a0v2, "_validate_manifest", forbidden)
    manifest, questions, corpus = prep.load_bundle(root, expected_sha=digest)
    assert len(questions) == 40
    assert [question.question_id for question in questions] == manifest["question_ids"]
    assert all(question.dataset == prep.DATASET for question in questions)
    assert corpus == root / prep.CORPUS
    assert {relative: (root / relative).read_bytes() for relative in before} == before
    assert set(manifest["question_ids"]) <= set(prep._registry_snapshot(root)["question_ids"])
    with pytest.raises(TypeError, match="expected_sha"):
        prep.load_bundle(root)
    for invalid in (None, "", "0" * 64):
        with pytest.raises(ValueError, match="externally pinned"):
            prep.load_bundle(root, expected_sha=invalid)
    for prepare_new in (False, True):
        with pytest.raises(FileExistsError):
            prep.prepare(root, prepare_new=prepare_new)


def test_registered_hash_only_and_unavailable_old_text_are_explicit(sample):
    root, ids = sample
    missing = f"{30000:024x}"
    write(
        root / "runs/hashonly/plan.json",
        {
            "question_ids": [missing],
            "normalized_question_sha256s": [prep._question_hash("Synthetic question 60?")],
        },
    )
    manifest = prep.prepare(root)
    assert ids[60] not in manifest["question_ids"]
    assert manifest["preflight"]["known_hash_matches"] == 7
    assert manifest["exclusion_registry"]["unavailable_text_question_ids"] == [missing]


def test_registry_cannot_smuggle_gold_file_into_runtime_validation(sample, monkeypatch):
    root, _ = sample
    registry = prep.prepare(root)["exclusion_registry"]
    forbidden = root / "data/hotpotqa/sealed_gold.jsonl"
    registry["inputs"].append({"path": forbidden.relative_to(root).as_posix(), "sha256": "a" * 64})
    registry["sha256"] = prep._digest(registry["inputs"])
    original = prep._sha

    def checked(path):
        if path == forbidden:
            pytest.fail("even hashing an injected gold file must be refused before opening it")
        return original(path)

    monkeypatch.setattr(prep, "_sha", checked)
    with pytest.raises(ValueError, match="unregistered exclusion metadata filename"):
        prep._validate_registry(root, registry)


@pytest.mark.parametrize("filename", ["runtime_questions.jsonl", "plan.json"])
def test_registry_redirect_is_rejected_before_the_old_scanner_reads(sample, monkeypatch, filename):
    root, _ = sample
    forbidden = root / "data/hotpotqa/sealed_gold.jsonl"
    candidate = root / "runs/prior" / filename
    try:
        candidate.symlink_to(forbidden)
    except OSError:
        # Windows developer-mode privileges may be absent. Model the same resolved
        # target instead of skipping this before-read safety assertion.
        candidate.write_bytes(b"{}\n")
        original_resolve = Path.resolve

        def redirected(path, *args, **kwargs):
            if path == candidate:
                return forbidden
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", redirected)

    def scanner_must_not_run(*args, **kwargs):
        pytest.fail("redirect target must be rejected before any old-scanner reads")

    monkeypatch.setattr(a0, "_registry_snapshot", scanner_must_not_run)
    with pytest.raises(ValueError, match="redirected"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


def test_registry_change_during_projection_aborts_without_output(sample, monkeypatch):
    root, _ = sample
    original = a0._project

    def changed(*args):
        rows = original(*args)
        write(root / "runs/new_pending.claim.json", {"question_ids": [rows[70]["id"]]})
        return rows

    monkeypatch.setattr(a0, "_project", changed)
    with pytest.raises(ValueError, match="registry changed"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


def test_insufficient_pool_never_resamples_or_exports(sample):
    root, ids = sample
    write(root / "runs/reserved.claim.json", {"question_ids": ids[:7680]})
    with pytest.raises(ValueError, match="not enough independent"):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


@pytest.mark.parametrize("kind", ["missing", "extra_column", "duplicate_id"])
def test_projection_boundary_rejects_incomplete_or_non_question_data(sample, monkeypatch, kind):
    root, _ = sample
    original = a0._project

    def changed(*args):
        rows = original(*args)
        if kind == "missing":
            return rows[1:]
        if kind == "extra_column":
            return [{**row, "answer": "FORBIDDEN"} for row in rows]
        return rows + rows[:1]

    monkeypatch.setattr(a0, "_project", changed)
    with pytest.raises(ValueError):
        prep.prepare(root, prepare_new=True)
    assert not (root / prep.OUTPUT).exists()


@pytest.mark.parametrize("kind", ["index", "runtime", "manifest", "sidecar", "v1", "v2", "claim"])
def test_tampered_frozen_artifacts_fail_without_labels(sample, kind):
    root, _ = sample
    prep.prepare(root, prepare_new=True)
    directory = root / prep.OUTPUT
    digest = prep._sha(directory / "manifest.json")
    target = {
        "index": root / prep.INDEX,
        "runtime": directory / "runtime_questions.jsonl",
        "manifest": directory / "manifest.json",
        "sidecar": directory / "manifest.sha256",
        "v1": root / next(iter(prep.A0_COHORTS)),
        "v2": root / list(prep.A0_COHORTS)[1],
        "claim": root / "runs/pending.claim.json",
    }[kind]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        prep.load_bundle(root, expected_sha=digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", a0.SCHEMA),
        ("seed", a0.SEED),
        ("count", 100),
        ("count", 40.0),
        ("role", a0.ROLE),
        ("gold_projected", 0),
        ("api_calls", False),
        ("projected_columns", ["id", "question", "answer"]),
        ("selection_uses_scores_or_difficulty", True),
        ("official_test_used", True),
    ],
)
def test_fixed_contract_cannot_be_bypassed_by_self_consistent_repin(sample, field, value):
    root, _ = sample
    manifest = prep.prepare(root, prepare_new=True)
    manifest[field] = value
    digest = repin(root / prep.OUTPUT, manifest)
    with pytest.raises(ValueError, match="A1 runtime bundle fields/protocol"):
        prep.load_bundle(root, expected_sha=digest)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_manifest",
        "dataset",
        "runtime_gold",
        "order",
        "protected_id",
        "removed_a0",
        "removed_claim",
        "gold_declaration",
        "rows_bool",
        "runtime_count",
    ],
)
def test_repin_does_not_bypass_strict_40_schema_and_exclusion_provenance(sample, kind):
    root, ids = sample
    manifest = prep.prepare(root, prepare_new=True)
    directory = root / prep.OUTPUT
    if kind == "unknown_manifest":
        manifest["answer"] = "FORBIDDEN"
    elif kind == "protected_id":
        manifest["question_ids"][0] = ids[399]
        manifest["question_ids_sha256"] = prep._digest(manifest["question_ids"])
    elif kind in {"removed_a0", "removed_claim"}:
        manifest["exclusion_registry"]["question_ids"].remove(
            ids[399 if kind == "removed_a0" else 2]
        )
    elif kind == "gold_declaration":
        manifest["runtime_artifact"]["contains_gold"] = 0
    elif kind == "rows_bool":
        manifest["runtime_artifact"]["rows"] = True
    else:
        runtime = directory / "runtime_questions.jsonl"
        rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines()]
        if kind == "dataset":
            rows[0]["dataset"] = a0.DATASET
        elif kind == "runtime_gold":
            rows[0]["answer"] = "FORBIDDEN"
        elif kind == "runtime_count":
            rows = rows[:-1]
        else:
            rows.reverse()
        runtime.write_bytes(b"".join(prep._canonical(row) + b"\n" for row in rows))
        manifest["runtime_artifact"].update(sha256=prep._sha(runtime), bytes=runtime.stat().st_size)
    digest = repin(directory, manifest)
    with pytest.raises(ValueError):
        prep.load_bundle(root, expected_sha=digest)


def test_cli_default_never_writes_or_prints_question_text(sample, monkeypatch, capsys):
    root, _ = sample
    monkeypatch.chdir(root)
    assert prep.main([]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["prepared"] is False
    assert "Synthetic question" not in output and "FORBIDDEN" not in output
    assert not (root / prep.OUTPUT).exists()
