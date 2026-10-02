"""Candidate-only hooks are checked with synthetic cards and API responses."""

import json
from types import SimpleNamespace

import pytest

from growrag.experiments.history_candidate_runtime import CandidateHistoryPlanner, planner_factory
from growrag.experiments.history_rank_diagnostics import rank_cards
from growrag.experiments.history_runtime import (
    HistoryContractClient,
    HistoryPlanner,
    shortlist_cards,
)
from growrag.experiments.protocol import RuntimeQuestion
from growrag.history_library import FrozenHistoryLibrary, HistoryCard, HistoryRecord
from growrag.operator_loop import Observation


def bank():
    cards = [
        HistoryCard(f"C{i}", "1", f"Rule {i}", "Subject born year", "Clarify subject.")
        for i in range(10)
    ]
    return FrozenHistoryLibrary(
        "synthetic",
        (),
        tuple(HistoryRecord(c, "reference", "synthetic", "1" * 64, (), "published") for c in cards),
    )


class Client:
    config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)
    block_reason = None

    def __init__(self):
        self.calls = []

    def complete(self, messages, *, trace_id, prompt_version):
        self.calls.append((messages, trace_id, prompt_version))
        return SimpleNamespace(
            content=json.dumps({"selected_card_id": None, "reason": "no_suitable_card"})
        )


def state():
    return Observation(RuntimeQuestion("q", "When was Cedar born?"), (), (), 2, 1)


def test_default_planner_hook_keeps_exact_legacy_ranking():
    library = bank()
    assert HistoryPlanner(None, library, trace_prefix="s").candidate_ranking(
        state()
    ) == shortlist_cards(state().question, library)


def test_legacy_intervention_keeps_request_and_proposal_identical_to_v2():
    from growrag.experiments.history_runtime_v2 import HistoryPlannerV2

    old, new = Client(), Client()
    kwargs = dict(trace_prefix="same")
    a = HistoryPlannerV2(HistoryContractClient(old), bank(), **kwargs)
    b = CandidateHistoryPlanner(
        HistoryContractClient(new), bank(), method="history_legacy3", **kwargs
    )
    assert a(state()) == b(state())
    assert old.calls == new.calls


@pytest.mark.parametrize("method,limit", [("history_body3", 3), ("history_body8", 8)])
def test_only_offered_candidates_change_and_bank_stays_frozen(method, limit):
    library, client = bank(), Client()
    before = library.fingerprint
    planner = CandidateHistoryPlanner(
        HistoryContractClient(client), library, trace_prefix="same", method=method
    )
    proposal = planner(state())
    payload = json.loads(client.calls[0][0][1]["content"])
    expected = rank_cards(state().question.text, library, "action_text_bm25", limit)
    assert [c["card_id"] for c in payload["candidate_cards"]] == [c["card_id"] for c in expected]
    assert proposal.spec is None and len(client.calls) == 1
    assert library.fingerprint == before
    assert not {"gold", "answer", "effectiveness", "source_qids"} & payload.keys()


def test_unknown_methods_fail_before_any_request():
    with pytest.raises(ValueError, match="unregistered"):
        planner_factory("secret-oracle")
