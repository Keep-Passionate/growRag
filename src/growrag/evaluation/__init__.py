"""Paired and oracle evaluation for benefit, harm, and abstention."""

from growrag.evaluation.oracle import (
    ACTION_COST_ORDER,
    OracleAnalysis,
    OracleInputRow,
    TargetOracleResult,
    analyze_oracle,
    read_oracle_csv,
)
from growrag.evaluation.paired import PairedEvaluation, PairedOutcome, evaluate_pair
from growrag.evaluation.selective import (
    OperatingPoint,
    RiskCoveragePoint,
    SelectiveCandidate,
    evaluate_operating_point,
    sweep_operating_points,
)

__all__ = [
    "ACTION_COST_ORDER",
    "OracleAnalysis",
    "OracleInputRow",
    "PairedEvaluation",
    "PairedOutcome",
    "OperatingPoint",
    "RiskCoveragePoint",
    "SelectiveCandidate",
    "TargetOracleResult",
    "analyze_oracle",
    "evaluate_pair",
    "evaluate_operating_point",
    "read_oracle_csv",
    "sweep_operating_points",
]
