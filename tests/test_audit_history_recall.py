"""Ranking diagnostics on synthetic text; no user questions or gold."""

import pytest

from growrag.experiments import audit_history_recall as audit
from growrag.experiments.protocol import RuntimeQuestion
from growrag.history_library import FrozenHistoryLibrary, HistoryCard, HistoryRecord
from growrag.macro_operators import GapField, OperatorSpec, QueryStep


def library():
    spec = OperatorSpec(
        "birth", "1", ("lookup",), (GapField("person"),), (QueryStep("s", "{person} born"),)
    )
    card = HistoryCard(
        "OP_birth", "1", "Template identity", "Generic metadata", "Opaque", operator_spec=spec
    )
    record = HistoryRecord(card, "learned", "synthetic", "1" * 64, ("source",), "published")
    return FrozenHistoryLibrary("synthetic", ("source",), (record,))


def test_comparison_does_not_create_an_applicability_or_benefit_label():
    bank = library()
    before = bank.fingerprint
    result = audit.compare_rankings(RuntimeQuestion("q", "When was Cedar born?"), bank)
    assert result["question_id"] == "q"
    assert len(result["rankings"]) == 3
    assert all(len(r) == 1 for r in result["rankings"].values())
    assert all(v["source_template_count"] == 1 for v in result["top3"].values())
    assert not {"answer", "supported", "benefit", "applicable", "gold"} & result.keys()
    assert bank.fingerprint == before


def test_existing_output_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "audit", lambda _: ({}, [], []))
    target = tmp_path / audit.OUTPUT
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="never overwrite"):
        audit.run(tmp_path, write=True)


def test_default_preflight_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "audit", lambda _: ({"gold_loaded": False}, [], []))
    assert audit.run(tmp_path) == {"gold_loaded": False}
    assert not (tmp_path / audit.OUTPUT).exists()
