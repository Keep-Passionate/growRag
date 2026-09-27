import json

import pytest

from growrag.experiments.memory_source_plan import COUNTS, select_roles, source_questions


def test_roles_are_disjoint_deterministic_and_exclude_seen_questions():
    pool = [f"{i:024x}" for i in range(210)]
    blocked = set(pool[:20])
    roles = select_roles(pool, blocked)
    assert roles == select_roles(list(reversed(pool)), blocked)
    assert {key: len(value) for key, value in roles.items()} == COUNTS
    selected = [qid for ids in roles.values() for qid in ids]
    assert len(set(selected)) == 160
    assert not set(selected) & blocked


def test_roles_reject_insufficient_or_duplicate_pool():
    with pytest.raises(ValueError, match="not enough"):
        select_roles(["a" * 24], set())
    with pytest.raises(ValueError, match="duplicate"):
        select_roles(["a" * 24, "a" * 24], set())
    with pytest.raises(ValueError, match="invalid"):
        select_roles(["../invalid"], set())


def test_source_projection_never_decodes_labels_or_probe_text(tmp_path):
    path = tmp_path / "raw.json"
    # These unwanted values are deliberately objects, not usable QA strings.
    path.write_text(
        json.dumps(
            [
                {"_id": "a", "question": "source question", "answer": {"forbidden": True}},
                {"_id": "p", "question": {"heldout": True}, "answer": "hidden"},
            ]
        ),
        encoding="utf-8",
    )
    rows = source_questions(path, {"source": ["a"], "probe": ["p"]}, set())
    assert len(rows) == 1
    assert set(rows[0]) == {"question_id", "text", "dataset"}
    assert rows[0]["text"] == "source question"


def test_source_projection_preserves_frozen_order(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(
        json.dumps([{"_id": "a", "question": "alpha"}, {"_id": "b", "question": "beta"}]),
        encoding="utf-8",
    )
    rows = source_questions(path, {"source": ["b", "a"]}, set())
    assert [row["question_id"] for row in rows] == ["b", "a"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"_id": "a", "question": "same"}, {"_id": "b", "question": "same"}],
        [{"_id": "a", "question": "same"}, {"_id": "a", "question": "different"}],
        [{"_id": "a", "question": "same"}],
    ],
)
def test_source_projection_stops_on_overlap_or_missing_without_resampling(tmp_path, rows):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        source_questions(path, {"source": ["a", "b"]}, set())
