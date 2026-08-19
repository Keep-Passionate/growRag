from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from growrag.config import ConfigError, load_pilot_config, load_runtime_config
from growrag.experience import LifecyclePolicy
from growrag.models import EnvironmentFingerprint
from growrag.selection import GatePolicy, QueryBudget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "configs" / "pilot.example.toml"


def _example_text() -> str:
    return EXAMPLE.read_text(encoding="utf-8")


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pilot.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_example_loads_into_typed_runtime_components() -> None:
    config = load_pilot_config(EXAMPLE)

    assert config.experiment.name == "gate1_oracle_feasibility"
    assert config.experiment.seed == 42
    assert config.actions.direct is True
    assert isinstance(config.environment, EnvironmentFingerprint)
    assert config.environment.corpus_id == "UNSET"
    assert config.environment.rewriter_version == "UNSET"
    assert isinstance(config.lifecycle_policy, LifecyclePolicy)
    assert config.lifecycle_policy.promotion_min_observations == 3
    assert config.lifecycle_policy.promotion_min_direct_good_observations == 2
    assert config.lifecycle_policy.quarantine_at_harm_count == 1
    assert isinstance(config.gate_policy, GatePolicy)
    assert config.gate_policy.minimum_observations == 3
    assert config.gate_policy.minimum_applicability == pytest.approx(0.5)
    assert isinstance(config.query_budget, QueryBudget)
    assert config.query_budget.max_retrieval_queries == 1
    assert config.query_budget.max_top_k == 5
    assert config.query_budget.max_context_tokens == 4096
    assert config.query_budget.max_query_characters == 2048
    assert config.candidate_recall.scorer == "source-query-jaccard-v1"
    assert config.candidate_recall.maximum_candidates == 5
    assert config.applicability.scorer == "signature-coverage-v1"
    assert config.application.applier_id == "UNSET"
    assert config.write_gate.minimum_source_gain == pytest.approx(0.05)


def test_template_is_allowed_for_inspection_but_fails_closed_for_runtime() -> None:
    template = load_pilot_config(EXAMPLE)

    with pytest.raises(ConfigError, match=r"\[environment\]\.corpus_id"):
        template.require_runtime_ready()
    with pytest.raises(ConfigError, match=r"\[application\]\.applier_id"):
        load_runtime_config(EXAMPLE)


def test_fully_resolved_configuration_is_runtime_ready(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _example_text().replace('"UNSET"', '"configured-v1"'))

    config = load_pilot_config(path, for_runtime=True)

    assert config.require_runtime_ready() is config
    assert config.environment.index_version == "configured-v1"
    assert config.application.application_version == "configured-v1"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda text: text + "\n[unexpected]\nvalue = 1\n", "unknown top-level"),
        (
            lambda text: text.replace("seed = 42", "seed = 42\nsead = 42"),
            r"unknown fields in \[experiment\].*sead",
        ),
        (
            lambda text: text.replace("seed = 42", 'seed = "42"'),
            r"\[experiment\]\.seed must be an integer",
        ),
        (
            lambda text: text.replace("top_k = 5", "top_k = true"),
            r"\[budget\]\.top_k must be an integer",
        ),
        (
            lambda text: text.replace('rewriter_version = "UNSET"\n', ""),
            r"missing fields in \[environment\].*rewriter_version",
        ),
        (
            lambda text: text.replace("good_threshold = 1.0", "good_threshold = nan"),
            r"\[evaluation\]\.good_threshold must be finite",
        ),
    ],
)
def test_unknown_missing_and_wrong_typed_fields_fail_clearly(
    tmp_path: Path, mutator: object, message: str
) -> None:
    mutate = mutator
    path = _write_config(tmp_path, mutate(_example_text()))  # type: ignore[operator]

    with pytest.raises(ConfigError, match=message):
        load_pilot_config(path)


def test_candidate_set_size_must_have_one_source_of_truth(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _example_text().replace("maximum_candidates = 5", "maximum_candidates = 4"),
    )

    with pytest.raises(ConfigError, match="candidate-set condition is unambiguous"):
        load_pilot_config(path)


def test_invalid_cross_field_policy_is_wrapped_with_section(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _example_text().replace("minimum_benefits = 1", "minimum_benefits = 4"),
    )

    with pytest.raises(ConfigError, match=r"invalid \[reliability\] policy"):
        load_pilot_config(path)


def test_runtime_validation_also_rejects_programmatic_placeholder() -> None:
    config = load_pilot_config(EXAMPLE)
    resolved_environment = replace(
        config.environment,
        corpus_id="toy",
        corpus_version="1",
        retriever_version="1",
        index_id="toy-index",
        index_version="1",
        analyzer_id="words",
        analyzer_version="1",
        rewriter_id="writer",
        rewriter_version="1",
    )
    partially_resolved = replace(config, environment=resolved_environment)

    with pytest.raises(ConfigError, match=r"\[application\]\.application_version"):
        partially_resolved.require_runtime_ready()
