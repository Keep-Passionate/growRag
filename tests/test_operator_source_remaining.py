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
