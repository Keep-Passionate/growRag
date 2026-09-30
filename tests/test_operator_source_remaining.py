"""Synthetic launcher checks: no network, model client, gold or real run writes."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_operator_source_remaining.py"
SPEC = importlib.util.spec_from_file_location("source_remaining", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
IDS = [f"{number:024x}" for number in range(500)]


def args(*extra):
    return launcher.parser().parse_args(
        [
            "--manifest",
            "data/manifest.json",
            "--expected-manifest-sha256",
            "a" * 64,
            "--expected-execution-sha256",
            "b" * 64,
            "--start",
            "0",
            "--end",
            "5",
            *extra,
        ]
    )


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def interval(argv):
    start = int(argv[argv.index("--start") + 1])
    return start, start + int(argv[argv.index("--count") + 1])


def seal_success(project, start, end):
    directory = project / "runs" / launcher.run_name(start, end)
    reports = [
        {"question_id": qid, "arms": {arm: {"status": "completed"} for arm in launcher.ARMS}}
        for qid in IDS[start:end]
    ]
    dump(directory / "predictions.json", reports)
    dump(
        directory / "predictions_frozen.json",
        {
            "status": "completed",
            "cleanup_errors": [],
            "phase": "source",
            "gold_loaded": False,
            "question_ids": IDS[start:end],
            "sha256": launcher.fingerprint(reports),
        },
    )


def failed_fixture(project, start, end, used=1, error="TimeoutError"):
    name = launcher.run_name(start, end)
    directory = project / "runs" / name
    trace = f"{name}/{IDS[start + used - 1]}/fresh/plan/1"
    audit_path = directory / "api_audit/failed.json"
    call = {"status": "failed", "trace_id": trace, "audit_path": str(audit_path)}
    reports = [
        {"question_id": qid, "arms": {arm: {"status": "completed"} for arm in launcher.ARMS}}
        for qid in IDS[start : start + used - 1]
    ]
    reports.append(
        {
            "question_id": IDS[start + used - 1],
            "arms": {
                "base": {"status": "completed"},
                "fresh": {
                    "status": "failed",
                    "error_type": "APIRequestError"
                    if error in launcher.TRANSPORT_ERRORS
                    else error,
                    "calls": [call],
                },
            },
        }
    )
    dump(directory / "predictions.json", reports)
    dump(directory / "final_budget.json", {"block_reason": "transport_failure", "calls": [call]})
    dump(
        directory / "predictions_frozen.json", {"question_ids": [r["question_id"] for r in reports]}
    )
    dump(
        audit_path,
        {
            "status": "failed",
            "error_type": error,
            "trace_id": trace,
            "network_attempted": True,
            "retry_count": 0,
            "api_requests": 1,
            "transport_source": "live_api",
        },
    )
    return {
        "proof": {
            "phase": "source",
            "parent_run_id": name,
            "arms": launcher.ARMS,
            "manifest_sha256": "a" * 64,
            "source_execution_sha256": "b" * 64,
            "started_question_ids": IDS[start : start + used],
            "question_ids": IDS[start + used : end],
        }
    }


def test_default_dry_run_never_invokes_child_or_writes(tmp_path, capsys):
    def forbidden(*a, **kw):
        pytest.fail("dry-run must not spawn a child")

    assert launcher.collect(args("--batch-size", "2"), tmp_path, IDS, execute=forbidden) == 0
    text = capsys.readouterr().out
    assert text.count("--phase source") == 3
    assert "--allow-network" not in text
    assert not list(tmp_path.iterdir())


def test_successful_children_finish_serially_before_next_batch(tmp_path):
    seen = []

    def execute(argv, **kwargs):
        start, end = interval(argv)
        assert kwargs["check"] is False and kwargs["cwd"] == tmp_path
        assert "--allow-network" in argv
        assert "--budget-cny" in argv and "--project-cap-cny" in argv
        seen.append((start, end))
        seal_success(tmp_path, start, end)
        return SimpleNamespace(returncode=0)

    assert (
        launcher.collect(
            args("--allow-network", "--batch-size", "2"), tmp_path, IDS, execute=execute
        )
        == 0
    )
    assert seen == [(0, 2), (2, 4), (4, 5)]


def test_nonzero_exit_stops_without_explicit_transport_option(tmp_path):
    seen = []

    def execute(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=7)

    assert launcher.collect(args("--allow-network"), tmp_path, IDS, execute=execute) == 7
    assert len(seen) == 1


def test_transport_failure_only_continues_certified_complete_suffix(tmp_path):
    seen, certificates = [], {}

    def execute(argv, **kwargs):
        start, end = interval(argv)
        seen.append((start, end))
        if start == 0:
            certificates[launcher.run_name(start, end)] = failed_fixture(tmp_path, start, end, 2)
            return SimpleNamespace(returncode=1)
        assert "--resume-certificate" in argv
        seal_success(tmp_path, start, end)
        return SimpleNamespace(returncode=0)

    def certify(runs, name, *, profile):
        assert profile == "action-list-v3"
        return certificates[name]

    assert (
        launcher.collect(
            args("--allow-network", "--continue-untouched-transport"),
            tmp_path,
            IDS,
            execute=execute,
            certifier=certify,
        )
        == 0
    )
    assert seen == [(0, 5), (2, 5)]
    assert len(list((tmp_path / "runs").glob("*_untouched_certificate.json"))) == 1


@pytest.mark.parametrize("remaining", [IDS[2:5], IDS[1:4], IDS[1:5][::-1], [], IDS[:5]])
def test_certificate_must_match_current_batch_exact_suffix(tmp_path, remaining):
    cert = failed_fixture(tmp_path, 0, 5)
    cert["proof"]["question_ids"] = remaining
    with pytest.raises(ValueError, match="exact strictly advancing"):
        launcher.audited_suffix(
            args(),
            tmp_path / "runs",
            launcher.run_name(0, 5),
            IDS[:5],
            certifier=lambda *a, **kw: cert,
        )


@pytest.mark.parametrize("error", ["ValueError", "HTTPError", "BudgetExceeded", "UnknownError"])
def test_schema_http_budget_and_unknown_errors_never_continue(tmp_path, error):
    cert = failed_fixture(tmp_path, 0, 5, error=error)
    with pytest.raises(ValueError, match="not an isolated"):
        launcher.audited_suffix(
            args(),
            tmp_path / "runs",
            launcher.run_name(0, 5),
            IDS[:5],
            certifier=lambda *a, **kw: cert,
        )


def test_api_audit_mismatch_cannot_be_called_transport_failure(tmp_path):
    failed_fixture(tmp_path, 0, 5)
    path = tmp_path / "runs" / launcher.run_name(0, 5) / "api_audit/failed.json"
    row = json.loads(path.read_text())
    row["error_type"] = "HTTPError"
    dump(path, row)
    with pytest.raises(ValueError, match="same transport failure"):
        launcher.transport_progress(path.parent.parent)


def test_three_transport_failures_without_full_question_stop(tmp_path):
    seen, certs = [], {}

    def execute(argv, **kwargs):
        start, end = interval(argv)
        seen.append((start, end))
        certs[launcher.run_name(start, end)] = failed_fixture(tmp_path, start, end)
        return SimpleNamespace(returncode=1)

    assert (
        launcher.collect(
            args("--allow-network", "--continue-untouched-transport"),
            tmp_path,
            IDS,
            execute=execute,
            certifier=lambda r, n, **kw: certs[n],
        )
        == 1
    )
    assert seen == [(0, 5), (1, 5), (2, 5)]
    assert len(list((tmp_path / "runs").glob("*_untouched_certificate.json"))) == 2


def test_full_audit_rejection_stops_before_any_certificate_or_second_child(tmp_path):
    seen = []

    def execute(argv, **kw):
        seen.append(interval(argv))
        failed_fixture(tmp_path, 0, 5)
        return SimpleNamespace(returncode=1)

    def certify(*a, **kw):
        raise ValueError("unclean seal or incomplete journal")

    with pytest.raises(ValueError, match="unclean seal"):
        launcher.collect(
            args("--allow-network", "--continue-untouched-transport"),
            tmp_path,
            IDS,
            execute=execute,
            certifier=certify,
        )
    assert seen == [(0, 5)]
    assert not list((tmp_path / "runs").glob("*_untouched_certificate.json"))


def test_existing_certificate_is_never_overwritten(tmp_path):
    name = launcher.run_name(0, 5)
    path = tmp_path / "runs" / f"{name}_untouched_certificate.json"
    dump(path, {"preserve": True})
    seen, certificates = [], {}

    def execute(argv, **kw):
        seen.append(interval(argv))
        certificates[name] = failed_fixture(tmp_path, 0, 5)
        return SimpleNamespace(returncode=1)

    with pytest.raises(FileExistsError):
        launcher.collect(
            args("--allow-network", "--continue-untouched-transport"),
            tmp_path,
            IDS,
            execute=execute,
            certifier=lambda r, n, **kw: certificates[n],
        )
    assert seen == [(0, 5)]
    assert json.loads(path.read_text()) == {"preserve": True}


def test_unknown_child_status_and_existing_claim_never_authorize_restart(tmp_path):
    with pytest.raises(ValueError, match="exit status unavailable"):
        launcher.collect(
            args("--allow-network"),
            tmp_path,
            IDS,
            execute=lambda *a, **kw: SimpleNamespace(returncode=None),
        )
    dump(tmp_path / "runs" / f"{launcher.run_name(0, 5)}.claim.json", {})
    with pytest.raises(ValueError, match="already exists"):
        launcher.collect(
            args("--allow-network"),
            tmp_path,
            IDS,
            execute=lambda *a, **kw: pytest.fail("must not replay"),
        )


def test_success_exit_requires_clean_complete_seal(tmp_path):
    def execute(argv, **kw):
        seal_success(tmp_path, 0, 4)  # A complete but shorter batch is not success for 0:5.
        (tmp_path / "runs" / launcher.run_name(0, 4)).rename(
            tmp_path / "runs" / launcher.run_name(0, 5)
        )
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="complete clean predictions"):
        launcher.collect(args("--allow-network"), tmp_path, IDS, execute=execute)


@pytest.mark.parametrize("cap", ["0", "201", "nan", "inf"])
def test_budget_cap_cannot_be_bypassed(tmp_path, cap):
    with pytest.raises(ValueError, match="project cap"):
        launcher.source_ids(args("--project-cap-cny", cap), tmp_path)


def initial_fixture(tmp_path, monkeypatch, wanted=None):
    path = tmp_path / "runs/explicit_certificate.json"
    dump(path, {"proof": {"question_ids": IDS[77:100] if wanted is None else wanted}})
    signature = {"sha256": "b" * 64, "configuration": {"model": "fixture-model"}}
    monkeypatch.setattr(launcher, "action_list_execution_signature", lambda p: signature)
    options = args(
        "--start", "77", "--end", "135", "--allow-network", "--resume-certificate", str(path)
    )
    return options, path, signature


def test_explicit_suffix_shortens_first_batch_only_and_reaudits_all_bindings(tmp_path, monkeypatch):
    options, path, signature = initial_fixture(tmp_path, monkeypatch)
    checked, seen = [], []

    def verify(runs, certificate, **kw):
        assert runs == tmp_path / "runs" and certificate == path
        assert kw == {
            "phase": "source",
            "question_ids": IDS[77:100],
            "arms": launcher.ARMS,
            "manifest_sha256": "a" * 64,
            "model": "fixture-model",
            "signature": signature,
            "profile": "action-list-v3",
        }
        checked.append(True)

    def execute(argv, **kw):
        assert checked == [True]
        start, end = interval(argv)
        assert ("--resume-certificate" in argv) == (start == 77)
        seen.append((start, end))
        seal_success(tmp_path, start, end)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher, "verify_certificate", verify)
    assert launcher.collect(options, tmp_path, IDS, execute=execute) == 0
    assert seen == [(77, 100), (100, 125), (125, 135)]


@pytest.mark.parametrize("wanted", [IDS[78:100], IDS[77:103], IDS[77:100][::-1]])
def test_initial_certificate_cannot_skip_reorder_or_exceed_first_batch(
    tmp_path, monkeypatch, wanted
):
    options, _, _ = initial_fixture(tmp_path, monkeypatch, wanted)
    monkeypatch.setattr(launcher, "verify_certificate", lambda *a, **kw: pytest.fail("bad range"))
    with pytest.raises(ValueError, match="complete contiguous"):
        launcher.collect(
            options, tmp_path, IDS, execute=lambda *a, **kw: pytest.fail("must not launch")
        )


def test_initial_certificate_full_audit_failure_prevents_even_dry_run(tmp_path, monkeypatch):
    options, _, _ = initial_fixture(tmp_path, monkeypatch)
    options.allow_network = False

    def reject(*a, **kw):
        raise ValueError("changed ancestor evidence")

    monkeypatch.setattr(launcher, "verify_certificate", reject)
    with pytest.raises(ValueError, match="changed ancestor"):
        launcher.collect(
            options, tmp_path, IDS, execute=lambda *a, **kw: pytest.fail("must not launch")
        )


def test_explicit_certificate_does_not_override_existing_target_claim(tmp_path, monkeypatch):
    options, _, _ = initial_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "verify_certificate", lambda *a, **kw: None)
    dump(tmp_path / "runs" / f"{launcher.run_name(77, 100)}.claim.json", {})
    with pytest.raises(ValueError, match="already exists"):
        launcher.collect(
            options, tmp_path, IDS, execute=lambda *a, **kw: pytest.fail("must not replay")
        )


def recorded_model_fixture(project, start=0, end=5, component="reader"):
    certificate = failed_fixture(project, start, end, error="ValueError")
    directory = project / "runs" / launcher.run_name(start, end)
    reports = json.loads((directory / "predictions.json").read_text())
    failed = reports[-1]["arms"]["fresh"]
    failed.update(question_id=IDS[start], arm="fresh")
    call = failed["calls"][0]
    call.update(status="completed", api_requests=1, prompt_version="frozen-fixture")
    call["trace_id"] = f"{directory.name}/{IDS[start]}/fresh/{component}"
    payload = {"original_question": "Synthetic fixture, no real dataset"}
    content = '{"invalid":"frozen model output"}'
    dump(
        Path(call["audit_path"]),
        {
            "status": "completed",
            "http_status": 200,
            "network_attempted": True,
            "retry_count": 0,
            "transport_source": "live_api",
            "api_requests": 1,
            "response_redacted": False,
            "finish_reason": "stop",
            "trace_id": call["trace_id"],
            "prompt_version": "frozen-fixture",
            "response": {"choices": [{"message": {"content": content}}]},
            "request": {
                "messages": [
                    {"role": "system", "content": "fixture"},
                    {"role": "user", "content": json.dumps(payload)},
                ]
            },
        },
    )
    dump(directory / "predictions.json", reports)
    dump(directory / "final_budget.json", {"block_reason": None, "calls": [call]})
    kind = "reader_record" if component == "reader" else "planner_record"
    context = {"question_id": IDS[start], "arm": "fresh"}
    events = [
        {
            **context,
            "kind": kind,
            "stage": "raw_wire",
            "raw_content": content,
            "prompt_version": "frozen-fixture",
            "payload": payload,
        }
    ]
    if kind == "planner_record":
        events.append({**context, "kind": kind, "stage": "normalized_parser_input"})
    events.extend(
        [
            {**context, "kind": "failure", "error_type": "ValueError"},
            {**context, "kind": "exit", "status": "failed"},
        ]
    )
    (directory / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )
    return certificate, directory


@pytest.mark.parametrize("component", ["reader", "plan/2"])
def test_recorded_model_rejection_requires_explicit_switch_and_full_suffix_audit(
    tmp_path, component
):
    cert, directory = recorded_model_fixture(tmp_path, component=component)
    assert launcher.recorded_model_progress(directory) == 0
    with pytest.raises(ValueError, match="not explicitly enabled"):
        launcher.continuation_progress(directory, args("--continue-untouched-transport"))
    options = args("--continue-untouched-recorded-model-errors")
    assert launcher.audited_suffix(
        options,
        tmp_path / "runs",
        directory.name,
        IDS[:5],
        certifier=lambda *a, **kw: cert,
        classify=lambda path: launcher.continuation_progress(path, options),
    )[1:] == (1, 0)


@pytest.mark.parametrize(
    "mutation", ["http", "response", "trace", "missing_raw", "budget", "later_work"]
)
def test_recorded_model_rejection_refuses_unproven_or_unrelated_failure(tmp_path, mutation):
    _, directory = recorded_model_fixture(tmp_path)
    audit_path = directory / "api_audit/failed.json"
    if mutation in {"http", "response", "trace"}:
        audit = json.loads(audit_path.read_text())
        if mutation == "http":
            audit["http_status"] = 429
        elif mutation == "trace":
            audit["trace_id"] += "/different"
        else:
            audit["response"]["choices"][0]["message"]["content"] = "different output"
        dump(audit_path, audit)
    elif mutation == "budget":
        budget = json.loads((directory / "final_budget.json").read_text())
        budget["block_reason"] = "estimated_budget_limit"
        dump(directory / "final_budget.json", budget)
    else:
        path = directory / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "missing_raw":
            events.pop(0)
        else:
            events.insert(1, {"kind": "operator_search", "question_id": IDS[0], "arm": "fresh"})
        path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.recorded_model_progress(directory)


def test_three_mixed_transport_and_recorded_model_failures_share_stop_counter(tmp_path):
    seen, certs = [], {}

    def execute(argv, **kw):
        start, end = interval(argv)
        seen.append((start, end))
        if start == 1:
            cert = failed_fixture(tmp_path, start, end)
        else:
            cert, _ = recorded_model_fixture(tmp_path, start, end)
        certs[launcher.run_name(start, end)] = cert
        return SimpleNamespace(returncode=1)

    assert (
        launcher.collect(
            args(
                "--allow-network",
                "--continue-untouched-transport",
                "--continue-untouched-recorded-model-errors",
            ),
            tmp_path,
            IDS,
            execute=execute,
            certifier=lambda r, n, **kw: certs[n],
        )
        == 1
    )
    assert seen == [(0, 5), (1, 5), (2, 5)]
    assert len(list((tmp_path / "runs").glob("*_untouched_certificate.json"))) == 2


def test_source_last_question_failure_advances_only_after_full_interval_audit(
    tmp_path, monkeypatch
):
    seen, audits = [], []

    def execute(argv, **kw):
        start, end = interval(argv)
        seen.append((start, end))
        if start == 0:
            failed_fixture(tmp_path, start, end, used=end - start)
            return SimpleNamespace(returncode=1)
        seal_success(tmp_path, start, end)
        return SimpleNamespace(returncode=0)

    def terminal(options, project, name, ids):
        audits.append(ids)
        return {"status": "terminal_failed", "completed_questions": len(ids) - 1}

    monkeypatch.setattr(launcher, "terminal_interval", terminal)
    assert (
        launcher.collect(
            args("--allow-network", "--continue-untouched-transport", "--batch-size", "2"),
            tmp_path,
            IDS,
            execute=execute,
            certifier=lambda *a, **kw: pytest.fail("no fake empty suffix proof"),
        )
        == 0
    )
    assert seen == [(0, 2), (2, 4), (4, 5)] and audits == [IDS[:2]]


def test_stop_file_is_not_read_and_prevents_next_child(tmp_path):
    flag = tmp_path / "stop.flag"
    flag.write_bytes(b"\xffnot text or instructions")
    assert (
        launcher.collect(
            args("--allow-network", "--stop-file", str(flag)),
            tmp_path,
            IDS,
            execute=lambda *a, **kw: pytest.fail("must stop before child"),
        )
        == 2
    )
    with pytest.raises(ValueError, match="stop file must stay inside"):
        launcher.collect(
            args("--stop-file", str(tmp_path.parent / "outside.flag")),
            tmp_path,
            IDS,
            execute=lambda *a, **kw: pytest.fail("outside path"),
        )
