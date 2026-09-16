"""Planning contracts; no real calls, labels or quality claims."""

import pytest

from growrag.experiments.plan_prompt_comparison import build_plan


def data():
    return {
        "official_split": "train",
        "role": "selector_train",
        "official_dev_test_used": False,
        "reserved_representation_targets_used": False,
        "question_ids": ["a", "b"],
        "selection_seed": 42,
        "source_sha256": "example-hash",
        "gold": "MUST_NOT_ENTER_PLAN",
    }


def test_plan_deterministic_and_contains_no_gold_or_actual_cost():
    plan = build_plan(data())
    assert plan == build_plan(data())
    assert "MUST_NOT_ENTER_PLAN" not in str(plan)
    assert plan["actual_api_calls"] == 0
    assert plan["planned_max_api_calls"] == 10
    assert plan["run_authorized_by_this_file"] is False
    assert plan["model_training_performed"] is False
    assert len(plan["plan_sha256"]) == 64


@pytest.mark.parametrize(
    "key,value",
    [
        ("official_split", "test"),
        ("role", "final_test"),
        ("official_dev_test_used", True),
        ("reserved_representation_targets_used", True),
        ("question_ids", []),
        ("question_ids", ["a", "a"]),
    ],
)
def test_plan_rejects_wrong_roles_and_duplicates(key, value):
    records = data()
    records[key] = value
    with pytest.raises(ValueError):
        build_plan(records)


def test_custom_matrix_cannot_mislabel_its_primary_contrast():
    with pytest.raises(ValueError, match="PROMPT_ARMS"):
        build_plan(data(), variants=("BASE", "QUERY2DOC"))
