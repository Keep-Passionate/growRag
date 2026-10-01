"""历史基底：规则卡与可执行模板共用外壳，但不共用虚构的能力。

ReFormeR式规则供后续改写模型解释；OperatorSpec仍是模板的唯一执行正文。
本模块不调用API、不读取gold、不发现新动作、不推断适用条件，也不更新旧库。
published只表示允许进入实验候选池，绝不表示跨题可信。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from .macro_operators import OperatorSpec
from .operator_bank import FrozenOperatorBank, operator_from_dict, operator_to_dict

REPRESENTATIONS = ("rule", "examples", "conditions")


def _text(value: object, label: str, limit: int = 2000) -> None:
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be nonempty bounded text")


def _identifier(value: object) -> None:
    if type(value) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", value):
        raise ValueError("identity must be a bounded plain identifier")


def _tuple(value: object, kind: type, label: str, limit: int) -> None:
    if type(value) is not tuple or len(value) > limit or any(type(x) is not kind for x in value):
        raise TypeError(f"{label} must be a bounded immutable tuple of {kind.__name__}")


def _sha(value: object) -> None:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA256")


def _json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _load(raw: str | bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonstandard JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _object(value: object, fields: str) -> dict:
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError(f"expected exactly: {fields}")
    return value


def _array(value: object) -> list:
    if type(value) is not list:
        raise ValueError("expected a JSON array")
    return value


@dataclass(frozen=True, slots=True)
class HistoryExample:
    """规则的输入/输出示例，不含历史答案；参考示例也不等于成功标签。"""

    query: str
    rewrite: str

    def __post_init__(self):
        _text(self.query, "example query")
        _text(self.rewrite, "example rewrite")


@dataclass(frozen=True, slots=True)
class HistoryCondition:
    """明确的适用条件描述；满足与否是当前题的观察，不是卡片永久属性。"""

    condition_id: str
    description: str

    def __post_init__(self):
        _identifier(self.condition_id)
        _text(self.description, "condition description", 500)


@dataclass(frozen=True, slots=True)
class HistoryCard:
    card_id: str
    version: str
    name: str
    description: str
    transformation_rule: str
    examples: tuple[HistoryExample, ...] = ()
    conditions: tuple[HistoryCondition, ...] = ()
    operator_spec: OperatorSpec | None = None

    def __post_init__(self):
        _identifier(self.card_id)
        _text(self.version, "version", 100)
        _text(self.name, "name", 200)
        _text(self.description, "description")
        _text(self.transformation_rule, "transformation rule", 8000)
        _tuple(self.examples, HistoryExample, "examples", 3)
        _tuple(self.conditions, HistoryCondition, "conditions", 8)
        if len({x.condition_id for x in self.conditions}) != len(self.conditions):
            raise ValueError("duplicate condition identity")
        if self.operator_spec is not None:
            operator_to_dict(self.operator_spec)  # 复用旧正文的严格校验，不改旧codec。

    @property
    def action_kind(self) -> str:
        return "template" if self.operator_spec is not None else "rewrite_rule"


def _card_dict(card: HistoryCard) -> dict:
    result = asdict(card)
    result["operator_spec"] = (
        None if card.operator_spec is None else operator_to_dict(card.operator_spec)
    )
    return json.loads(_json(result))


def _card_from_dict(value: object) -> HistoryCard:
    data = _object(
        value,
        "card_id version name description transformation_rule examples conditions operator_spec",
    )
    examples = tuple(
        HistoryExample(**_object(item, "query rewrite")) for item in _array(data["examples"])
    )
    conditions = tuple(
        HistoryCondition(**_object(item, "condition_id description"))
        for item in _array(data["conditions"])
    )
    return HistoryCard(
        **{
            key: data[key]
            for key in ("card_id", "version", "name", "description", "transformation_rule")
        },
        examples=examples,
        conditions=conditions,
        operator_spec=None
        if data["operator_spec"] is None
        else operator_from_dict(data["operator_spec"]),
    )


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    """来源只供审计；这些字段不能整体传给选择器。"""

    card: HistoryCard
    source_kind: str
    source_ref: str
    source_sha256: str
    source_qids: tuple[str, ...] = ()
    status: str = "candidate"

    def __post_init__(self):
        if type(self.card) is not HistoryCard:
            raise TypeError("record requires a HistoryCard")
        if self.source_kind not in {"reference", "learned"}:
            raise ValueError("source_kind must be reference or learned")
        _text(self.source_ref, "source ref")
        _sha(self.source_sha256)
        _tuple(self.source_qids, str, "source qids", 100000)
        for qid in self.source_qids:
            _text(qid, "source qid", 200)
        if len(set(self.source_qids)) != len(self.source_qids):
            raise ValueError("duplicate source qid")
        if (self.source_kind == "learned") != bool(self.source_qids):
            raise ValueError("learned records need source qids; references cannot carry them")
        if self.status not in {"candidate", "published"}:
            raise ValueError("status must be candidate or published")


@dataclass(frozen=True, slots=True)
class FrozenHistoryLibrary:
    """新实验的sidecar，绝不覆盖旧FrozenOperatorBank及其冻结证书。"""

    protocol_id: str
    allowed_source_ids: tuple[str, ...]
    records: tuple[HistoryRecord, ...]

    def __post_init__(self):
        _text(self.protocol_id, "protocol id")
        _tuple(self.allowed_source_ids, str, "allowed source ids", 100000)
        _tuple(self.records, HistoryRecord, "records", 10000)
        for qid in self.allowed_source_ids:
            _text(qid, "allowed source id", 200)
        if len(set(self.allowed_source_ids)) != len(self.allowed_source_ids):
            raise ValueError("duplicate allowed source id")
        identities = [(r.card.card_id, r.card.version) for r in self.records]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate card id/version")
        # 首版同库同ID不同时展示两个版本，避免选择ID时产生歧义。
        if len({r.card.card_id for r in self.records}) != len(self.records):
            raise ValueError("only one version of each card may be in this snapshot")
        for record in self.records:
            if not set(record.source_qids) <= set(self.allowed_source_ids):
                raise ValueError("history contains non-source question ids")
        object.__setattr__(self, "allowed_source_ids", tuple(sorted(self.allowed_source_ids)))
        object.__setattr__(
            self, "records", tuple(sorted(self.records, key=lambda r: r.card.card_id))
        )

    @property
    def published_cards(self) -> tuple[HistoryCard, ...]:
        return tuple(r.card for r in self.records if r.status == "published")

    def _payload(self) -> dict:
        records = []
        for record in self.records:
            item = asdict(record)
            item["card"] = _card_dict(record.card)
            records.append(item)
        return {
            "schema_version": 1,
            "protocol_id": self.protocol_id,
            "allowed_source_ids": list(self.allowed_source_ids),
            "records": records,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_json(self._payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return _json({**self._payload(), "fingerprint": self.fingerprint})

    @classmethod
    def from_json(cls, text: str) -> FrozenHistoryLibrary:
        data = _object(
            _load(text), "schema_version protocol_id allowed_source_ids records fingerprint"
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported history schema")
        records = []
        for item in _array(data["records"]):
            record = _object(item, "card source_kind source_ref source_sha256 source_qids status")
            records.append(
                HistoryRecord(
                    _card_from_dict(record["card"]),
                    record["source_kind"],
                    record["source_ref"],
                    record["source_sha256"],
                    tuple(_array(record["source_qids"])),
                    record["status"],
                )
            )
        result = cls(data["protocol_id"], tuple(_array(data["allowed_source_ids"])), tuple(records))
        if result.fingerprint != data["fingerprint"]:
            raise ValueError("history fingerprint mismatch")
        return result


def card_view(card: HistoryCard, representation: str = "conditions") -> dict:
    """同一动作的三种展示；规则/执行正文始终相同，只删选择辅助字段。"""
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown history representation")
    view = {
        "card_id": card.card_id,
        "version": card.version,
        "name": card.name,
        "description": card.description,
        "transformation_rule": card.transformation_rule,
        "action_kind": card.action_kind,
        "operator_spec": None
        if card.operator_spec is None
        else operator_to_dict(card.operator_spec),
    }
    if representation in {"examples", "conditions"}:
        view["examples"] = [asdict(x) for x in card.examples]
    if representation == "conditions":
        view["conditions"] = [asdict(x) for x in card.conditions]
        view["applicability_status"] = "unknown"  # 条件未核验，空条件不是普遍适用。
    return view


def condition_observations(card: HistoryCard, observed: dict[str, bool | None]) -> list[dict]:
    """只编码外部明确提供的当前题观察，不猜测、不把unknown变成true。"""
    if type(observed) is not dict or set(observed) - {x.condition_id for x in card.conditions}:
        raise ValueError("unknown condition observation")
    if any(value is not None and type(value) is not bool for value in observed.values()):
        raise TypeError("condition observations must be bool or None")
    return [
        {
            "condition_id": condition.condition_id,
            "status": "unknown"
            if observed.get(condition.condition_id) is None
            else "supported"
            if observed[condition.condition_id]
            else "contradicted",
        }
        for condition in card.conditions
    ]


def resolve_card(library: FrozenHistoryLibrary, offered_ids: tuple[str, ...], selected_id: str):
    _tuple(offered_ids, str, "offered ids", 20)
    if len(set(offered_ids)) != len(offered_ids):
        raise ValueError("duplicate offered identity")
    published = {card.card_id: card for card in library.published_cards}
    if set(offered_ids) - set(published) or selected_id not in offered_ids:
        raise ValueError("selection must name an offered published card")
    return published[selected_id]


def resolve_operator(library, offered_ids, selected_id) -> OperatorSpec:
    card = resolve_card(library, offered_ids, selected_id)
    if card.operator_spec is None:
        raise ValueError("rewrite_rule needs a rewriter; it is not a compiled template")
    return card.operator_spec  # 不从名称/描述重建，不接收模型修改过的正文。


def import_reformer_patterns(
    raw: bytes,
    expected_sha256: str,
    source_ref: str,
    protocol_id: str = "history-foundation-v1",
) -> FrozenHistoryLibrary:
    """SHA锁定的数据导入，不执行作者Python，不修改或补全作者模式。"""
    _sha(expected_sha256)
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("ReFormeR pattern SHA256 mismatch")
    records = []
    for value in _array(_load(raw)):
        data = _object(value, "pattern_name description transformation_rule examples")
        _text(data["pattern_name"], "pattern name", 200)
        examples = []
        for item in _array(data["examples"]):
            pair = _array(item)
            if len(pair) != 2:
                raise ValueError("reference example must be a query/rewrite pair")
            examples.append(HistoryExample(*pair))
        identity = "RF_" + hashlib.sha256(data["pattern_name"].encode("utf-8")).hexdigest()[:16]
        card = HistoryCard(
            identity,
            "1",
            data["pattern_name"],
            data["description"],
            data["transformation_rule"],
            tuple(examples),
        )
        records.append(
            HistoryRecord(card, "reference", source_ref, expected_sha256, (), "published")
        )
    if not records:
        raise ValueError("reference library cannot be empty")
    return FrozenHistoryLibrary(protocol_id, (), tuple(records))


def import_operator_bank(
    bank: FrozenOperatorBank,
    source_ref: str,
    source_sha256: str,
    protocol_id: str = "history-foundation-v1",
) -> FrozenHistoryLibrary:
    """旧来源库的无损sidecar；不臆测用途、条件、真实示例或单动作因果收益。

    调用方须把bank的source名单与已审计train manifest核对。此函数只继承范围，
    不能凭一个任意对象证明它来自官方train。
    """
    if type(bank) is not FrozenOperatorBank:
        raise TypeError("expected a FrozenOperatorBank")
    records = []
    for item in bank.records:
        spec = item.spec
        identity = (
            "OP_"
            + hashlib.sha256(_json([spec.operator_id, spec.version]).encode("utf-8")).hexdigest()[
                :16
            ]
        )
        card = HistoryCard(
            identity,
            "1",
            f"Template {spec.operator_id}",
            f"Source-executed template with {len(spec.steps)} step(s); applicability unknown.",
            "Execute the unchanged declared steps and current-question parameters. "
            "A source episode does not establish cross-question transfer or per-step gain.",
            operator_spec=spec,
        )
        records.append(
            HistoryRecord(
                card,
                "learned",
                source_ref,
                source_sha256,
                item.source_qids,
                "published" if item.status == "validated" else "candidate",
            )
        )
    return FrozenHistoryLibrary(protocol_id, bank.allowed_source_ids, tuple(records))


def combine_libraries(*libraries: FrozenHistoryLibrary, protocol_id="history-foundation-v1"):
    """保留各来源，不合并语义相似卡，不覆盖卡片版本。"""
    if not libraries or any(type(x) is not FrozenHistoryLibrary for x in libraries):
        raise TypeError("at least one frozen history library is required")
    return FrozenHistoryLibrary(
        protocol_id,
        tuple(sorted({qid for library in libraries for qid in library.allowed_source_ids})),
        tuple(record for library in libraries for record in library.records),
    )
