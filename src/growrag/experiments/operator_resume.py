"""Offline proof for untouched questions in a closed, failed operator batch.

不释放失败题/完成题，也不删除原claim。证书每次使用都重审原始审计文件。
只读预测、请求和日志，不读数据集答案；不创建API客户端。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .operator_execution_signature import validate_execution_signature
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .shared_continuation import _event_mentions, _read, _request_evidence, _safe_file

SCHEMA = "growrag-operator-unstarted-certificate-v1"
PROTOCOL = "growrag-operator-study-v1"
PREFIX = "2026-09-30_operator_v1_"
_ID = re.compile(r"[0-9a-f]{24}")
_ARMS = {"base", "fresh", "static", "memory50", "memory100", "memory250", "memory500"}


def _ids(value):
    if (
        type(value) is not list
        or not value
        or any(not isinstance(qid, str) or not _ID.fullmatch(qid) for qid in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("invalid immutable question ID list")
    return value


def _audit(runs, parent_id, *, visited=()):
    if (
        not isinstance(parent_id, str)
        or not re.fullmatch(re.escape(PREFIX) + r"[A-Za-z0-9_]+", parent_id)
        or parent_id in visited
        or len(visited) >= 16
    ):
        raise ValueError("unsafe parent identity or continuation ancestry")
    root = Path(runs).resolve(strict=True)
    directory = (root / parent_id).resolve(strict=True)
    if directory.parent != root or not directory.is_dir():
        raise ValueError("parent path escapes runs root")
    files = {}

    def read(name):
        value, digest = _read(_safe_file(directory, name))
        files[name] = digest
        return value

    launch, reports, seal, budget = (
        read(name)
        for name in (
            "launch_plan.json",
            "predictions.json",
            "predictions_frozen.json",
            "final_budget.json",
        )
    )
    claim, claim_sha = _read(_safe_file(root, f"{parent_id}.claim.json"))
    files["../" + f"{parent_id}.claim.json"] = claim_sha
    if (
        launch.get("run_id") != parent_id
        or launch.get("protocol") != PROTOCOL
        or launch.get("phase") not in {"source", "calibration"}
        or launch.get("gold_loaded") is not False
        or launch.get("memory_updates") is not False
        or type(launch.get("arms")) is not list
        or not launch["arms"]
        or len(set(launch["arms"])) != len(launch["arms"])
        or not set(launch["arms"]) <= _ARMS
        or claim != {**launch, "plan_sha256": fingerprint(launch)}
    ):
        raise ValueError("parent launch/claim identity mismatch")
    planned = _ids(launch.get("question_ids"))
    if (
        type(reports) is not list
        or not reports
        or seal.get("status") != "failed"
        or seal.get("cleanup_errors") != []
        or seal.get("sha256") != fingerprint(reports)
        or seal.get("phase") != launch["phase"]
        or seal.get("gold_loaded") is not False
        or type(budget.get("calls")) is not list
    ):
        raise ValueError("parent must have a clean failed prediction seal and final ledger")
    started = _ids([report.get("question_id") for report in reports])
    if started != planned[: len(started)] or seal.get("question_ids") != started:
        raise ValueError("started questions are not the recorded sequential prefix")
    untouched = planned[len(started) :]
    if not untouched:
        raise ValueError("no entirely untouched questions remain")
    wanted, owned, expected_records = set(untouched), [], set()
    for number, report in enumerate(reports):
        qid = report["question_id"]
        outcomes = report.get("arms")
        if (
            set(report) != {"question_id", "arms"}
            or type(outcomes) is not dict
            or not outcomes
            or list(outcomes) != launch["arms"][: len(outcomes)]
            or read(f"checkpoint_{number:04d}.json") != report
        ):
            raise ValueError("question checkpoint or arm order mismatch")
        expected_records.add(f"checkpoint_{number:04d}.json")
        for arm, result in outcomes.items():
            name = f"{qid}_{arm}.json"
            expected_records.add(name)
            if (
                result.get("status") not in {"completed", "failed"}
                or result.get("feedback") is not None
                or result.get("memory_updated") is not False
                or result.get("question_id") != qid
                or result.get("arm") != arm
                or read(name) != result
                or type(result.get("calls")) is not list
            ):
                raise ValueError("per-arm report is missing, scored, mutated or invalid")
            owned.extend(result["calls"])
    if owned != budget["calls"] or len({c.get("trace_id") for c in owned}) != len(owned):
        raise ValueError("all calls must have exactly one question/arm owner")
    for call in owned:
        parts = call.get("trace_id", "").split("/")
        if (
            len(parts) < 4
            or parts[0] != parent_id
            or parts[1] not in started
            or parts[2] not in launch["arms"]
        ):
            raise ValueError("call trace outside the completed parent prefix")
    event_path = _safe_file(directory, "events.jsonl")
    raw = event_path.read_bytes()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    files["events.jsonl"] = hashlib.sha256(raw).hexdigest()
    if (
        not events
        or events[-1].get("kind") != "exit"
        or events[-1].get("status") != "failed"
        or any(event.get("kind") == "exit" for event in events[:-1])
        or any(_event_mentions(event, wanted) for event in events)
        or events[-1].get("requests") != budget.get("api_requests")
    ):
        raise ValueError("parent exit/event evidence is incomplete or touches a target")
    requests = _request_evidence(directory, budget, wanted, reports, events[-4:])
    # Refuse stray per-question/checkpoint files not represented in the final seal.
    actual = {
        p.name
        for p in directory.iterdir()
        if re.fullmatch(r"(?:[0-9a-f]{24}_.+|checkpoint_.+)\.json", p.name)
    }
    if actual != expected_records:
        raise ValueError("stray question/checkpoint artifact needs audit")
    for name in ("live.log", "process.json", "source_snapshot.json", "cumulative_budget.json"):
        path = _safe_file(directory, name)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = launch.get("execution_signature")
    source_signature = None
    if launch["phase"] == "source":
        source_signature = validate_execution_signature(signature)
    ancestors = [parent_id]
    if launch.get("resume_parent_run_id") is not None:
        certificate = read("resume_certificate.json")
        if fingerprint(certificate) != launch.get("resume_certificate_sha256"):
            raise ValueError("parent continuation certificate changed")
        previous = _validate(root, certificate, visited=(*visited, parent_id))
        if (
            previous["parent_run_id"] != launch["resume_parent_run_id"]
            or previous["phase"] != launch["phase"]
            or previous["manifest_sha256"] != launch["manifest_sha256"]
            or previous["model"] != launch["model"]
            or previous["arms"] != launch["arms"]
            or not set(planned) <= set(previous["question_ids"])
            or (
                source_signature is not None
                and previous["source_execution_sha256"] != source_signature
            )
        ):
            raise ValueError("parent continuation ancestry is inconsistent")
        ancestors.extend(previous["ancestor_run_ids"])
    return {
        "parent_run_id": parent_id,
        "phase": launch["phase"],
        "model": launch["model"],
        "manifest_sha256": launch["manifest_sha256"],
        "arms": launch["arms"],
        "question_ids": untouched,
        "started_question_ids": started,
        "ancestor_run_ids": ancestors,
        "source_execution_sha256": source_signature,
        "evidence_files": files,
        "request_evidence": requests,
    }


def build_certificate(runs, parent_run_id):
    body = {"schema": SCHEMA, "protocol": PROTOCOL, "proof": _audit(runs, parent_run_id)}
    return {**body, "sha256": fingerprint(body)}


def _validate(runs, certificate, *, visited=()):
    if type(certificate) is not dict or set(certificate) != {
        "schema",
        "protocol",
        "proof",
        "sha256",
    }:
        raise ValueError("invalid explicit resume certificate")
    body = {k: certificate[k] for k in ("schema", "protocol", "proof")}
    if (
        certificate["schema"] != SCHEMA
        or certificate["protocol"] != PROTOCOL
        or fingerprint(body) != certificate["sha256"]
    ):
        raise ValueError("resume certificate checksum/schema mismatch")
    proof = _audit(runs, certificate["proof"]["parent_run_id"], visited=visited)
    if proof != certificate["proof"]:
        raise ValueError("parent evidence changed since certificate creation")
    return proof


def verify_certificate(runs, path, *, phase, question_ids, arms, manifest_sha256, model, signature):
    certificate, _ = _read(Path(path).resolve(strict=True))
    proof = _validate(runs, certificate)
    if (
        proof["phase"] != phase
        or proof["manifest_sha256"] != manifest_sha256
        or proof["model"] != model
        or proof["arms"] != list(arms)
        or not set(_ids(list(question_ids))) <= set(proof["question_ids"])
        or (
            phase == "source"
            and proof["source_execution_sha256"] != validate_execution_signature(signature)
        )
    ):
        raise ValueError("resume only permits untouched questions under the approved method/role")
    return certificate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    certificate = build_certificate(args.runs_root, args.parent_run_id)
    write_json(args.output, certificate)
    count = len(certificate["proof"]["question_ids"])
    print(f"Offline certificate: {count} untouched questions; no API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
