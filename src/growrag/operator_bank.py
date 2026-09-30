"""Strict data-only operator codec and source-only, frozen memory snapshots.

本模块不执行模型、检索、文件读写或反馈更新。候选记录可以留下，但只有明确标为
validated 的记录进入 published_specs。这个标记和来源清单仍须上层协议核查；
结构隔离不能证明自由文本没有夹带答案，也不能证明一个算子已经有效。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .macro_operators import FieldEquals, GapField, OperatorSpec, QueryStep

_LABEL_FIELDS = frozenset(
    {
        "answer",
        "gold",
        "gold_answer",
        "gold_answers",
        "reference_answer",
        "final_answer",
        "expected_answer",
        "supporting_facts",
    }
)


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _strings(value: object, name: str) -> None:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty immutable tuple")
    for item in value:
        _text(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"duplicate {name}")


def _object(value: object, keys: str) -> dict:
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError(f"expected an object containing exactly: {keys}")
    return value


def _array(value: object) -> list:
    if type(value) is not list:
        raise ValueError("expected a JSON array")
    return value


def _json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _load(text: str) -> object:
    if type(text) is not str:
        raise TypeError("JSON input must be text")

    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _check_spec(spec: OperatorSpec) -> None:
    if not isinstance(spec, OperatorSpec):
        raise TypeError("spec must be an OperatorSpec")
    names = {field.name.casefold() for field in spec.gap_schema}
    names.update(name.casefold() for step in spec.steps for name in step.requires_bindings)
    if names & _LABEL_FIELDS:
        raise ValueError("answer/gold label fields are not runtime operator inputs")


def operator_to_dict(spec: OperatorSpec) -> dict:
    """JSON-compatible objects only, including arrays instead of Python tuples."""
    _check_spec(spec)
    return json.loads(_json(asdict(spec)))


def operator_from_dict(value: object) -> OperatorSpec:
    data = _object(value, "operator_id version supported_intents gap_schema steps")
    gaps = tuple(
        GapField(**_object(item, "name kind required")) for item in _array(data["gap_schema"])
    )
    steps = []
    for item in _array(data["steps"]):
        step = _object(item, "step_id template when requires_bindings")
        conditions = tuple(
            FieldEquals(**_object(item, "field value")) for item in _array(step["when"])
        )
        steps.append(
            QueryStep(
                step["step_id"],
                step["template"],
                conditions,
                tuple(_array(step["requires_bindings"])),
            )
        )
    spec = OperatorSpec(
        data["operator_id"],
        data["version"],
        tuple(_array(data["supported_intents"])),
        gaps,
        tuple(steps),
    )
    _check_spec(spec)
    return spec


def operator_to_json(spec: OperatorSpec) -> str:
    return _json(operator_to_dict(spec))


def operator_from_json(text: str) -> OperatorSpec:
    return operator_from_dict(_load(text))


@dataclass(frozen=True, slots=True)
class OperatorRecord:
    """No answer, gold, score, full trajectory or mutable confidence payload.

    source_trace_ref points to an audit record, not content supplied to a selector.
    validated is an explicit publication declaration, NOT a computed trust score.
    """

    spec: OperatorSpec
    source_qids: tuple[str, ...]
    protocol_id: str
    source_trace_ref: str
    status: str = "candidate"
    validation_ref: str | None = None

    def __post_init__(self) -> None:
        _check_spec(self.spec)
        _strings(self.source_qids, "source_qids")
        _text(self.protocol_id, "protocol_id")
        _text(self.source_trace_ref, "source_trace_ref")
        if self.status not in {"candidate", "validated"}:
            raise ValueError("status must be candidate or validated")
        if self.status == "validated":
            _text(self.validation_ref, "validation_ref")
        elif self.validation_ref is not None:
            raise ValueError("candidate cannot carry a validation declaration")
        object.__setattr__(self, "source_qids", tuple(sorted(self.source_qids)))


def record_to_dict(record: OperatorRecord) -> dict:
    if not isinstance(record, OperatorRecord):
        raise TypeError("record must be an OperatorRecord")
    return {
        "spec": operator_to_dict(record.spec),
        "source_qids": sorted(record.source_qids),
        "protocol_id": record.protocol_id,
        "source_trace_ref": record.source_trace_ref,
        "status": record.status,
        "validation_ref": record.validation_ref,
    }


def record_from_dict(value: object) -> OperatorRecord:
    data = _object(value, "spec source_qids protocol_id source_trace_ref status validation_ref")
    return OperatorRecord(
        operator_from_dict(data["spec"]),
        tuple(_array(data["source_qids"])),
        data["protocol_id"],
        data["source_trace_ref"],
        data["status"],
        data["validation_ref"],
    )


def _check_record(record: OperatorRecord, source_ids: tuple[str, ...], protocol: str) -> None:
    if not isinstance(record, OperatorRecord):
        raise TypeError("record must be an OperatorRecord")
    if record.protocol_id != protocol:
        raise ValueError("record protocol differs from source protocol")
    if not set(record.source_qids) <= set(source_ids):
        raise ValueError("record contains non-source question IDs")


def _key(record: OperatorRecord) -> tuple[str, str]:
    return record.spec.operator_id, record.spec.version


@dataclass(frozen=True, slots=True)
class FrozenOperatorBank:
    """Immutable evaluation snapshot. Validation refs are audit-only metadata.

    Pass published_specs, not records or traces, to runtime selection. A stored
    fingerprint detects accidental changes, not adversarial re-signing. Expected
    source IDs/protocol must additionally be checked against the experiment manifest.
    """

    protocol_id: str
    allowed_source_ids: tuple[str, ...]
    records: tuple[OperatorRecord, ...]

    def __post_init__(self) -> None:
        _text(self.protocol_id, "protocol_id")
        _strings(self.allowed_source_ids, "allowed_source_ids")
        if type(self.records) is not tuple:
            raise TypeError("records must be an immutable tuple")
        seen = set()
        for record in self.records:
            _check_record(record, self.allowed_source_ids, self.protocol_id)
            if _key(record) in seen:
                raise ValueError("duplicate operator id/version")
            seen.add(_key(record))
        object.__setattr__(self, "allowed_source_ids", tuple(sorted(self.allowed_source_ids)))
        object.__setattr__(self, "records", tuple(sorted(self.records, key=_key)))

    @property
    def published_specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(record.spec for record in self.records if record.status == "validated")

    def _payload(self) -> dict:
        return {
            "schema_version": 1,
            "protocol_id": self.protocol_id,
            "allowed_source_ids": sorted(self.allowed_source_ids),
            "records": [record_to_dict(item) for item in sorted(self.records, key=_key)],
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_json(self._payload()).encode("utf-8")).hexdigest()

    def verify_fingerprint(self, expected: str) -> None:
        if self.fingerprint != expected:
            raise ValueError("bank fingerprint mismatch")

    def to_dict(self) -> dict:
        return {**self._payload(), "fingerprint": self.fingerprint}

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> FrozenOperatorBank:
        data = _object(value, "schema_version protocol_id allowed_source_ids records fingerprint")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported bank schema version")
        bank = cls(
            data["protocol_id"],
            tuple(_array(data["allowed_source_ids"])),
            tuple(record_from_dict(item) for item in _array(data["records"])),
        )
        bank.verify_fingerprint(data["fingerprint"])
        return bank

    @classmethod
    def from_json(cls, text: str) -> FrozenOperatorBank:
        return cls.from_dict(_load(text))


class MemoryBuilder:
    """Source-only offline builder; a new source-size prefix uses a new builder.

    The caller supplies IDs from its already audited source manifest, never from
    calibration/test sets. After freeze this object also rejects further writes.
    """

    def __init__(self, allowed_source_ids: tuple[str, ...], protocol_id: str) -> None:
        _strings(allowed_source_ids, "allowed_source_ids")
        _text(protocol_id, "protocol_id")
        self._source_ids = allowed_source_ids
        self._protocol = protocol_id
        self._records: dict[tuple[str, str], OperatorRecord] = {}
        self._frozen = False

    def add(self, record: OperatorRecord) -> None:
        if self._frozen:
            raise RuntimeError("builder is frozen; evaluation cannot update memory")
        _check_record(record, self._source_ids, self._protocol)
        key = _key(record)
        if key in self._records:
            raise ValueError("operator id/version cannot be overwritten")
        self._records[key] = record

    def freeze(self) -> FrozenOperatorBank:
        snapshot = FrozenOperatorBank(
            self._protocol,
            tuple(sorted(self._source_ids)),
            tuple(sorted(self._records.values(), key=_key)),
        )
        self._frozen = True
        return snapshot
