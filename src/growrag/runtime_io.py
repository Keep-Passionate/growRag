"""Strict file schemas for reproducible GrowRAG runs.

Runtime requests use JSON Lines (one exact request object per line). A plan
bundle uses one versioned JSON object whose plan entries are keyed internally by
``(target_query_id, experience_id)``. Parsing always fails closed.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from growrag.rewrite import PrecomputedPlanApplier

PLAN_SCHEMA_VERSION = 1


class RuntimeIOError(ValueError):
    """Raised when a runtime input file violates its frozen schema."""


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    """One target query and its precomputed applicability signatures."""

    target_query_id: str
    target_query: str
    signatures: tuple[str, ...]
    signature_extractor_id: str
    signature_extractor_version: str
    signature_input_scope: str

    def __post_init__(self) -> None:
        _require_non_empty(self.target_query_id, "target_query_id")
        _require_non_empty(self.target_query, "target_query")
        _require_non_empty(self.signature_extractor_id, "signature_extractor_id")
        _require_non_empty(
            self.signature_extractor_version,
            "signature_extractor_version",
        )
        if self.signature_input_scope.strip().casefold() != "query_only":
            raise RuntimeIOError("signature_input_scope must be 'query_only'")
        object.__setattr__(self, "signature_input_scope", "query_only")
        if not isinstance(self.signatures, tuple):
            raise RuntimeIOError("signatures must be a tuple of strings")
        for index, signature in enumerate(self.signatures):
            if not isinstance(signature, str):
                raise RuntimeIOError(f"signatures[{index}] must be a string")
            _require_non_empty(signature, f"signatures[{index}]")


@dataclass(frozen=True, slots=True)
class PlanBundle:
    """Versioned target-specific plans, independent of retrieval budgets."""

    schema_version: int
    applier_id: str
    application_version: str
    plans: Mapping[tuple[str, str], str]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PLAN_SCHEMA_VERSION
        ):
            raise RuntimeIOError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {PLAN_SCHEMA_VERSION}"
            )
        _require_non_empty(self.applier_id, "applier_id")
        _require_non_empty(self.application_version, "application_version")
        if not isinstance(self.plans, Mapping):
            raise RuntimeIOError("plans must be a mapping")

        frozen: dict[tuple[str, str], str] = {}
        for key, transformed_query in self.plans.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise RuntimeIOError("each plan key must be (target_query_id, experience_id)")
            target_query_id, experience_id = key
            if not isinstance(target_query_id, str) or not isinstance(experience_id, str):
                raise RuntimeIOError("plan key components must be strings")
            _require_non_empty(target_query_id, "plan target_query_id")
            _require_non_empty(experience_id, "plan experience_id")
            if not isinstance(transformed_query, str):
                raise RuntimeIOError("plan transformed_query must be a string")
            _require_non_empty(transformed_query, "plan transformed_query")
            frozen[(target_query_id, experience_id)] = transformed_query
        object.__setattr__(self, "plans", MappingProxyType(frozen))

    def to_applier(
        self,
        *,
        retrieval_query_count: int,
        requested_top_k: int,
        context_token_budget: int,
        estimated_cost: float = 0.0,
    ) -> PrecomputedPlanApplier:
        """Create an applier with budgets supplied explicitly by the caller."""

        return PrecomputedPlanApplier(
            plans=self.plans,
            applier_id=self.applier_id,
            application_version=self.application_version,
            retrieval_query_count=retrieval_query_count,
            requested_top_k=requested_top_k,
            context_token_budget=context_token_budget,
            estimated_cost=estimated_cost,
        )


def load_requests(path: str | os.PathLike[str]) -> tuple[RuntimeRequest, ...]:
    """Load strict request JSONL, rejecting blank lines and duplicate IDs."""

    text = _read_utf8(path, "request JSONL")
    if not text:
        return ()
    requests: list[RuntimeRequest] = []
    seen_ids: set[str] = set()
    signature_protocol: tuple[str, str, str] | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise RuntimeIOError(f"request JSONL line {line_number} is blank")
        raw = _loads_json(line, f"request JSONL line {line_number}")
        request = _request_from_object(raw, f"request JSONL line {line_number}")
        if request.target_query_id in seen_ids:
            raise RuntimeIOError(f"duplicate target_query_id: {request.target_query_id}")
        current_protocol = (
            request.signature_extractor_id,
            request.signature_extractor_version,
            request.signature_input_scope,
        )
        if signature_protocol is None:
            signature_protocol = current_protocol
        elif current_protocol != signature_protocol:
            raise RuntimeIOError("all requests must share one frozen signature protocol")
        seen_ids.add(request.target_query_id)
        requests.append(request)
    return tuple(requests)


def load_plan_bundle(path: str | os.PathLike[str]) -> PlanBundle:
    """Load one strict, versioned JSON plan bundle."""

    raw = _loads_json(_read_utf8(path, "plan bundle"), "plan bundle")
    root = _expect_object(raw, "plan bundle")
    _require_exact_keys(
        root,
        {"schema_version", "applier_id", "application_version", "plans"},
        "plan bundle",
    )
    schema_version = _expect_int(root["schema_version"], "schema_version")
    applier_id = _expect_string(root["applier_id"], "applier_id")
    application_version = _expect_string(
        root["application_version"],
        "application_version",
    )
    raw_plans = _expect_list(root["plans"], "plans")
    plans: dict[tuple[str, str], str] = {}
    for index, value in enumerate(raw_plans):
        path_label = f"plans[{index}]"
        item = _expect_object(value, path_label)
        _require_exact_keys(
            item,
            {"target_query_id", "experience_id", "transformed_query"},
            path_label,
        )
        target_id = _expect_string(item["target_query_id"], f"{path_label}.target_query_id")
        experience_id = _expect_string(
            item["experience_id"],
            f"{path_label}.experience_id",
        )
        transformed_query = _expect_string(
            item["transformed_query"],
            f"{path_label}.transformed_query",
        )
        key = (target_id, experience_id)
        if key in plans:
            raise RuntimeIOError(
                "duplicate plan key: "
                f"target_query_id={target_id!r}, experience_id={experience_id!r}"
            )
        plans[key] = transformed_query
    return PlanBundle(
        schema_version=schema_version,
        applier_id=applier_id,
        application_version=application_version,
        plans=plans,
    )


def write_decision_jsonl(
    path: str | os.PathLike[str],
    decisions: Iterable[dict[str, object]],
) -> Path:
    """Atomically write deterministic decision dictionaries as JSON Lines."""

    lines: list[str] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise RuntimeIOError(f"decisions[{index}] must be a dict")
        if any(not isinstance(key, str) for key in decision):
            raise RuntimeIOError(f"decisions[{index}] keys must be strings")
        try:
            lines.append(
                json.dumps(
                    decision,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeIOError(f"decisions[{index}] is not JSON-safe: {exc}") from exc
    content = "".join(f"{line}\n" for line in lines)
    return _atomic_write_utf8(path, content)


def _request_from_object(value: object, path: str) -> RuntimeRequest:
    raw = _expect_object(value, path)
    _require_exact_keys(
        raw,
        {
            "target_query_id",
            "target_query",
            "signatures",
            "signature_extractor_id",
            "signature_extractor_version",
            "signature_input_scope",
        },
        path,
    )
    signatures_raw = _expect_list(raw["signatures"], f"{path}.signatures")
    signatures = tuple(
        _expect_string(item, f"{path}.signatures[{index}]")
        for index, item in enumerate(signatures_raw)
    )
    return RuntimeRequest(
        target_query_id=_expect_string(raw["target_query_id"], f"{path}.target_query_id"),
        target_query=_expect_string(raw["target_query"], f"{path}.target_query"),
        signatures=signatures,
        signature_extractor_id=_expect_string(
            raw["signature_extractor_id"],
            f"{path}.signature_extractor_id",
        ),
        signature_extractor_version=_expect_string(
            raw["signature_extractor_version"],
            f"{path}.signature_extractor_version",
        ),
        signature_input_scope=_expect_string(
            raw["signature_input_scope"],
            f"{path}.signature_input_scope",
        ),
    )


def _read_utf8(path: str | os.PathLike[str], label: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeIOError(f"cannot read {label}: {exc}") from exc


def _loads_json(text: str, label: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except RuntimeIOError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeIOError(f"invalid {label}: {exc}") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeIOError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise RuntimeIOError(f"non-finite JSON number is forbidden: {value}")


def _require_exact_keys(raw: dict[str, object], expected: set[str], path: str) -> None:
    actual = set(raw)
    if actual != expected:
        raise RuntimeIOError(
            f"{path} fields differ from schema; "
            f"unknown={sorted(actual - expected)}, missing={sorted(expected - actual)}"
        )


def _expect_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeIOError(f"{path} must be an object")
    return value


def _expect_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeIOError(f"{path} must be an array")
    return value


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise RuntimeIOError(f"{path} must be a string")
    return value


def _expect_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeIOError(f"{path} must be an integer")
    return value


def _require_non_empty(value: str, path: str) -> None:
    if not value.strip():
        raise RuntimeIOError(f"{path} must not be empty")


def _atomic_write_utf8(path: str | os.PathLike[str], content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return target
