"""Small synthetic accounting fixtures; no historic user data or network."""

import json

import pytest

from growrag.experiments.prior_budget import ROOT_LEDGERS, reconcile_history


def write_fixture(tmp_path):
    for i, relative in enumerate(ROOT_LEDGERS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        audit = path.parent / "api_audit" / f"{i}.json"
        audit.parent.mkdir()
        row = {
            "trace_id": str(i),
            "audit_path": str(audit),
            "reserved_cny": 0.01,
            "estimated_actual_cny": 0.000001,
            "status": "completed",
            "input_tokens": 1,
            "output_tokens": 1,
            "api_requests": 1,
        }
        audit.write_text(json.dumps({**row, "transport_source": "live_api"}))
        path.write_text(
            json.dumps(
                {
                    "calls": [row],
                    "api_requests": 1,
                    "reserved_cny": 0.01,
                    "limits": {"input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
                }
            )
        )


def test_disjoint_roots_not_aggregate_copies(tmp_path):
    write_fixture(tmp_path)
    (tmp_path / "cumulative_budget.json").write_text('{"reserved_cny":999}')
    result = reconcile_history(tmp_path)
    assert result["prior_api_requests"] == 4
    assert result["prior_reserved_cny"] == pytest.approx(0.04)
    assert result["prior_known_estimated_cny"] == pytest.approx(0.000004)


def test_unknown_failure_retains_reservation_not_fake_zero(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / ROOT_LEDGERS[-1]
    value = json.loads(path.read_text())
    row = value["calls"][0]
    row.update(status="failed", input_tokens=None, output_tokens=None, estimated_actual_cny=None)
    path.write_text(json.dumps(value))
    audit = path.parent / "api_audit" / "3.json"
    audit.write_text(json.dumps({**row, "transport_source": "live_api"}))
    result = reconcile_history(tmp_path)
    assert result["prior_unknown_cost_requests"] == 1
    assert result["prior_total_actual_cny"] is None
    assert result["prior_reserved_cny"] == pytest.approx(0.04)


def test_overlapping_root_requests_rejected(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / ROOT_LEDGERS[1]
    value = json.loads(path.read_text())
    value["calls"][0]["trace_id"] = "0"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlapping"):
        reconcile_history(tmp_path)


def test_new_live_run_is_not_silently_ignored(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / "new-run" / "api_audit" / "new.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"transport_source": "live_api", "trace_id": "unbilled"}))
    with pytest.raises(ValueError, match="unaccounted"):
        reconcile_history(tmp_path)


def test_wrong_unit_cost_or_reservation_fails_closed(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / ROOT_LEDGERS[0]
    value = json.loads(path.read_text())
    value["reserved_cny"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reservation totals"):
        reconcile_history(tmp_path)
