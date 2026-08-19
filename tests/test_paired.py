from growrag.evaluation.paired import PairedOutcome, evaluate_pair


def test_labels_harm_when_direct_is_good_and_reuse_is_bad() -> None:
    result = evaluate_pair(1.0, 0.0, good_threshold=1.0)
    assert result.outcome is PairedOutcome.HARM
    assert result.difference == -1.0


def test_labels_benefit_when_reuse_repairs_a_bad_direct_result() -> None:
    result = evaluate_pair(0.0, 1.0, good_threshold=1.0)
    assert result.outcome is PairedOutcome.BENEFIT
    assert result.difference == 1.0


def test_threshold_is_explicit() -> None:
    result = evaluate_pair(0.8, 0.7, good_threshold=0.5)
    assert result.outcome is PairedOutcome.BOTH_GOOD

