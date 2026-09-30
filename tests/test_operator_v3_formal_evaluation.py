"""Formal v3 provenance tests use synthetic files only: no API or real gold."""

from copy import deepcopy

import pytest
from test_operator_evaluation_freeze import _setup_frozen
from test_operator_resume import (
    _eval_context,
    closed_run,
    dump,
    eval_run,
    read,
    signature,
    verify_eval,
)
from test_score_operator_evaluation import _gold, _replace_episode, _reuse_episode, _setup

from growrag.experiments import operator_evaluation_freeze as freeze
from growrag.experiments import operator_resume as resume
from growrag.experiments import score_operator_evaluation as scoring
from growrag.experiments.operator_profiles import ACTION_LIST, LEGACY, STRUCTURED
from growrag.experiments.representation_runner import fingerprint

V3 = ACTION_LIST.name


def _collect(data, *, profile=V3):
    return scoring.collect_frozen_evaluation(
        data["root"],
        data["manifest"],
        data["freeze"],
        data["runs"],
        data["certificate"],
        data["sha"],
        profile=profile,
    )


def _score(data, *, write=False, profile=V3):
    return scoring.score_operator_evaluation(
        data["root"],
        data["certificate"],
        data["runs"],
        data["root"] / "feedback",
        expected_certificate_sha256=data["sha"],
        write=write,
        profile=profile,
    )


@pytest.fixture
def v3_data(tmp_path, monkeypatch):
    return _setup(tmp_path, monkeypatch, profile=V3)


def test_v3_evaluation500_seven_arms_collect_without_gold(tmp_path, monkeypatch):
    data = _setup(tmp_path, monkeypatch, count=500, profile=V3)
    for offset in range(0, 500, 25):
        data["launch"](f"batch_{offset:04d}", data["ids"][offset : offset + 25])
    monkeypatch.setattr(scoring, "_load_dev_gold", lambda *a: pytest.fail("gold opened"))
    reports, _ = _collect(data)
    assert len(reports) == 500 and sum(len(row["arms"]) for row in reports) == 3500
    assert {row["protocol"] for row in reports} == {ACTION_LIST.protocol}
    audit = _score(data)
    assert audit["protocol"] == ACTION_LIST.protocol
    assert audit["all_questions_terminal"] and audit["all_arm_predictions_completed"]
    assert not audit["gold_loaded"] and audit["api_calls"] == 0
    assert not (data["root"] / "feedback").exists()


def test_v3_incomplete_evaluation_never_opens_gold(v3_data, monkeypatch):
    v3_data["launch"]("partial", v3_data["ids"], omit_last=True)
    monkeypatch.setattr(scoring, "_load_dev_gold", lambda *a: pytest.fail("gold opened"))
    with pytest.raises(ValueError, match="need terminal reports"):
        _score(v3_data, write=True)


def test_v3_failed_arm_and_unstarted_arms_are_terminal_unknown_not_retried(v3_data):
    ids = v3_data["ids"]
    v3_data["launch"]("first", ids[:1], fail_at=(ids[0], "fresh"))
    v3_data["launch"]("rest", ids[1:])
    reports, _ = _collect(v3_data)
    assert reports[0]["arms"]["fresh"]["status"] == "failed"
    assert reports[0]["arms"]["static"]["status"] == "failure_induced_unstarted"
    assert reports[0]["arms"]["memory500"]["reader"] is None
    audit = _score(v3_data)
    assert audit["all_questions_terminal"] and not audit["all_arm_predictions_completed"]
    assert not audit["gold_loaded"]


def test_v3_explicit_scoring_uses_only_synthetic_gold_after_terminal_gate(v3_data, monkeypatch):
    ids = v3_data["ids"]
    v3_data["launch"]("whole", ids, fail_at=(ids[-1], "fresh"))
    opened = []

    def synthetic_gold(root, manifest):
        opened.append(True)
        # Rechecking the gate here ensures every fixed question is terminal first.
        assert len(_collect(v3_data)[0]) == len(ids)
        return {qid: _gold(qid) for qid in ids}, []

    monkeypatch.setattr(scoring, "_load_dev_gold", synthetic_gold)
    audit = _score(v3_data, write=True)
    assert opened == [True] and audit["gold_loaded"]
    assert audit["protocol"] == ACTION_LIST.protocol and not audit["memory_updated"]
    feedback = read(v3_data["root"] / "feedback/feedback.json")
    assert feedback[ids[-1]]["static"]["status"] == "failure_induced_unstarted"
    assert feedback[ids[-1]]["static"]["answer_em"] is None
    with pytest.raises(FileExistsError):
        _score(v3_data, write=True)
    assert opened == [True]


@pytest.mark.parametrize("change", [False, True])
def test_v3_executed_memory_reuse_must_match_the_frozen_published_spec(v3_data, change):
    folder = v3_data["launch"]("whole", v3_data["ids"])
    episode = _reuse_episode(v3_data)
    if change:
        episode["proposals"][0]["spec"]["steps"][0]["template"] = "{term} changed"
    _replace_episode(folder, v3_data, "memory50", episode)
    if change:
        with pytest.raises(ValueError, match="executed reuse spec"):
            _collect(v3_data)
    else:
        assert _collect(v3_data)[0][0]["arms"]["memory50"]["episode"] == episode


@pytest.mark.parametrize("other", [LEGACY, STRUCTURED])
@pytest.mark.parametrize("kind", ["claim", "launch"])
def test_v3_foreign_evaluation_claims_cannot_hide_under_another_prefix(v3_data, other, kind):
    v3_data["launch"]("whole", v3_data["ids"])
    path = v3_data["runs"] / (
        f"{other.prefix}other.claim.json"
        if kind == "claim"
        else f"{other.prefix}other/launch_plan.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, {"protocol": other.protocol, "phase": "evaluation"})
    with pytest.raises(ValueError, match="mixed-profile"):
        _collect(v3_data)


@pytest.mark.parametrize(
    "change", ["legacy_signature", "different_signature", "protocol", "profile", "missing_profile"]
)
def test_v3_mixed_method_or_protocol_evaluation_is_rejected(v3_data, change):
    changed = deepcopy(v3_data["freeze"]["execution_signature"])
    changed["files"][next(iter(changed["files"]))] = "b" * 64
    changed["sha256"] = fingerprint({k: v for k, v in changed.items() if k != "sha256"})
    options = {
        "legacy_signature": {"execution_signature": signature()},
        "different_signature": {"execution_signature": changed},
        "protocol": {"protocol": LEGACY.protocol},
        "profile": {"profile": LEGACY.name},
        "missing_profile": {"profile": None},
    }[change]
    v3_data["launch"]("whole", v3_data["ids"], plan_change=options)
    with pytest.raises(ValueError, match="frozen protocol|mixed-profile"):
        _collect(v3_data)


@pytest.mark.parametrize("requested", [LEGACY.name, STRUCTURED.name])
def test_v3_collection_rejects_wrong_requested_profile(v3_data, requested):
    with pytest.raises(ValueError, match="profile"):
        _collect(v3_data, profile=requested)


@pytest.mark.parametrize("phase", ["source", "calibration"])
def test_v3_resume_requires_same_profile_signature_and_untouched_questions(tmp_path, phase):
    method = signature(profile=V3)
    directory, ids = closed_run(tmp_path, phase=phase, sig=method, profile=V3)
    cert = resume.build_certificate(tmp_path, directory.name, profile=V3)
    assert cert["protocol"] == ACTION_LIST.protocol
    assert cert["proof"]["source_execution_sha256"] == method["sha256"]
    path = tmp_path / "certificate.json"
    dump(path, cert)
    options = dict(
        phase=phase,
        question_ids=ids[1:],
        arms=["base"],
        manifest_sha256="c" * 64,
        model="fixture-model",
        signature=method,
    )
    # Omitted profile is inferred from a checked exact certificate protocol.
    assert resume.verify_certificate(tmp_path, path, **options) == cert
    for edits in (
        {"profile": LEGACY.name},
        {"signature": signature("b", profile=V3)},
        {"signature": signature()},
        {"question_ids": ids[:1]},
    ):
        with pytest.raises(ValueError):
            resume.verify_certificate(tmp_path, path, **{**options, **edits})
    with pytest.raises(ValueError, match="unsafe parent"):
        resume.build_certificate(tmp_path, directory.name)
    assert (tmp_path / f"{directory.name}.claim.json").exists()


def test_v3_resume_rejects_legacy_ancestor_even_when_ids_overlap(tmp_path):
    first, ids = closed_run(tmp_path, phase="source", sig=signature())
    legacy = resume.build_certificate(tmp_path, first.name)
    second, _ = closed_run(
        tmp_path,
        phase="source",
        ids=ids[1:],
        sig=signature(profile=V3),
        profile=V3,
        parent=legacy,
    )
    with pytest.raises(ValueError, match="profile"):
        resume.build_certificate(tmp_path, second.name, profile=V3)


def test_v3_evaluation_resume_binds_whole_suffix_and_original_freeze(tmp_path, monkeypatch):
    context = _eval_context(tmp_path, monkeypatch, profile=V3)
    first, ids = eval_run(context)
    cert = resume.build_certificate(context.runs, first.name, profile=V3)
    path = tmp_path / "resume.json"
    dump(path, cert)
    assert verify_eval(context, path, ids[1:], profile=V3) == cert
    with pytest.raises(ValueError, match="untouched suffix"):
        verify_eval(context, path, ids[2:], profile=V3)
    with pytest.raises(ValueError, match="profile"):
        verify_eval(context, path, ids[1:], profile=LEGACY.name)
    second, _ = eval_run(context, start=1, parent=cert)
    chained = resume.build_certificate(context.runs, second.name, profile=V3)
    assert chained["proof"]["ancestor_run_ids"] == [second.name, first.name]
    assert chained["proof"]["question_ids"] == ids[2:]


def test_v3_relabelled_resume_protocol_cannot_release_old_claim(tmp_path):
    directory, _ = closed_run(tmp_path, phase="source", sig=signature())
    certificate = resume.build_certificate(tmp_path, directory.name)
    certificate["protocol"] = ACTION_LIST.protocol
    certificate["sha256"] = fingerprint(
        {k: certificate[k] for k in ("schema", "protocol", "proof")}
    )
    with pytest.raises(ValueError, match="unsafe parent"):
        resume._validate(tmp_path, certificate)


@pytest.mark.parametrize("profile", [None, LEGACY.name, STRUCTURED.name])
def test_v3_resume_launch_profile_must_match_protocol(tmp_path, profile):
    directory, _ = closed_run(tmp_path, phase="source", sig=signature(profile=V3), profile=V3)
    launch = read(directory / "launch_plan.json")
    launch["profile"] = profile
    dump(directory / "launch_plan.json", launch)
    dump(tmp_path / f"{directory.name}.claim.json", {**launch, "plan_sha256": fingerprint(launch)})
    with pytest.raises(ValueError, match="launch/claim identity"):
        resume.build_certificate(tmp_path, directory.name, profile=V3)


@pytest.fixture
def v3_frozen(tmp_path, monkeypatch):
    return _setup_frozen(tmp_path, monkeypatch, profile=V3)


def test_v3_source500_four_banks_freeze_infers_profile_and_keeps_schema(v3_frozen):
    certificate = freeze.create_evaluation_freeze(**v3_frozen, write=True)
    body = certificate["freeze"]
    assert body["schema_version"] == freeze.SCHEMA and body["protocol"] == ACTION_LIST.protocol
    assert len(body["execution_signature"]["files"]) == 15
    assert len(body["evaluation_ids"]) == 500 and len(body["banks"]) == 4
    assert body["source_terminal_counts"] == {"failure_preserved": 500}
    assert not body["new_gold_decoded"] and not body["evaluation_runtime_decoded"]
    arguments = dict(expected_certificate_sha256=freeze._sha(v3_frozen["certificate_path"]))
    assert (
        freeze.validate_evaluation_freeze(
            v3_frozen["root"], v3_frozen["certificate_path"], **arguments
        )
        == body
    )
    with pytest.raises(ValueError, match="requested profile"):
        freeze.validate_evaluation_freeze(
            v3_frozen["root"],
            v3_frozen["certificate_path"],
            **arguments,
            profile=LEGACY.name,
        )


@pytest.mark.parametrize("other", [LEGACY, STRUCTURED, ACTION_LIST])
def test_v3_freeze_cannot_be_issued_after_any_profile_evaluation_claim(v3_frozen, other):
    dump(
        v3_frozen["runs_root"] / f"{other.prefix}evaluation.claim.json",
        {"phase": "evaluation", "protocol": other.protocol},
    )
    with pytest.raises(ValueError, match="already claimed"):
        freeze.create_evaluation_freeze(**v3_frozen)


def test_v3_changed_method_cannot_revalidate_old_freeze(v3_frozen):
    freeze.create_evaluation_freeze(**v3_frozen, write=True)
    path = v3_frozen["root"] / "src/growrag/experiments/operator_model_v3.py"
    path.write_text("changed inert test code", encoding="utf-8")
    with pytest.raises(ValueError, match="current method differs"):
        freeze.validate_evaluation_freeze(
            v3_frozen["root"],
            v3_frozen["certificate_path"],
            expected_certificate_sha256=freeze._sha(v3_frozen["certificate_path"]),
        )


def test_structured_v2_stays_outside_formal_pipeline_before_inputs(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="calibration-only"):
        resume.build_certificate(missing, "not-a-run", profile=STRUCTURED.name)
    with pytest.raises(ValueError, match="calibration-only"):
        scoring.score_operator_evaluation(
            missing,
            missing,
            missing,
            missing,
            expected_certificate_sha256="a" * 64,
            profile=STRUCTURED.name,
        )
    with pytest.raises(ValueError, match="calibration-only"):
        freeze.create_evaluation_freeze(
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
            expected_manifest_sha256="a" * 64,
            expected_scoring_audit_sha256="b" * 64,
            expected_bank_bundle_sha256="c" * 64,
            expected_execution_sha256="d" * 64,
            profile=STRUCTURED.name,
        )
