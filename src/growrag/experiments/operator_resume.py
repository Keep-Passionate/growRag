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
from .operator_profiles import (
    LEGACY,
    profile_from_protocol,
    study_profile,
    validate_profile_execution_signature,
)
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .shared_continuation import _event_mentions, _read, _request_evidence, _safe_file

SCHEMA = "growrag-operator-unstarted-certificate-v1"
EVALUATION_SCHEMA = "growrag-operator-evaluation-unstarted-certificate-v1"
PROTOCOL = "growrag-operator-study-v1"
PREFIX = "2026-09-30_operator_v1_"
_ID = re.compile(r"[0-9a-f]{24}")
_ARMS = {"base", "fresh", "static", "memory50", "memory100", "memory250", "memory500"}
_EVALUATION_ARMS = ["base", "fresh", "static", "memory50", "memory100", "memory250", "memory500"]
_SHA = re.compile(r"[0-9a-f]{64}")
# Each evaluation batch has at most 25 questions; every failed ancestor consumes
# at least one whole question. Support all untouched suffixes without unbounded recursion.
_EVALUATION_DEPTH = 25


def _ids(value):
    if (
        type(value) is not list
        or not value
        or any(not isinstance(qid, str) or not _ID.fullmatch(qid) for qid in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("invalid immutable question ID list")
    return value


def _method_signature(value, profile):
    return (
        validate_execution_signature(value)
        if profile == LEGACY
        else validate_profile_execution_signature(value, profile)
    )


def _evaluation_binding(runs, launch, read, profile=LEGACY):
    """Revalidate the exact evaluation freeze, not just its copied hash string."""
    from .operator_evaluation_freeze import validate_evaluation_freeze

    project = runs.parent
    path_value, digest = (
        launch.get("evaluation_freeze_path"),
        launch.get("evaluation_freeze_sha256"),
    )
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(digest, str)
        or not _SHA.fullmatch(digest)
    ):
        raise ValueError("evaluation parent lacks its original freeze identity")
    path = Path(path_value).resolve(strict=True)
    if not path.is_relative_to(project):
        raise ValueError("evaluation freeze path escapes the project")
    body = validate_evaluation_freeze(
        project,
        path,
        expected_certificate_sha256=digest,
        **({} if profile == LEGACY else {"profile": profile.name}),
    )
    method = _method_signature(launch.get("execution_signature"), profile)
    ids, planned = body["evaluation_ids"], launch["question_ids"]
    start = ids.index(planned[0]) if planned[0] in ids else -1
    if (
        read("evaluation_freeze_verified.json") != body
        or body["protocol"] != profile.protocol
        or (project / body["paths"]["runs"]).resolve(strict=True) != runs
        or launch["arms"] != _EVALUATION_ARMS
        or launch["manifest_sha256"] != body["expected"]["manifest"]
        or launch["model"] != body["execution_signature"]["configuration"]["model"]
        or launch["execution_signature"] != body["execution_signature"]
        or method != body["expected"]["execution"]
        or launch.get("evaluation_order_sha256") != body["evaluation_order_sha256"]
        or body["evaluation_order_sha256"] != fingerprint(ids)
        or not isinstance(body.get("runner_sha256"), str)
        or not _SHA.fullmatch(body["runner_sha256"])
        or not 1 <= len(planned) <= 25
        or start < 0
        or ids[start : start + len(planned)] != planned
        or launch.get("bank_sha256")
        != {
            f"memory{size}": body["banks"][size]["fingerprint"]
            for size in ("50", "100", "250", "500")
        }
        or launch.get("bank_file_sha256")
        != {
            f"memory{size}": body["banks"][size]["file_sha256"]
            for size in ("50", "100", "250", "500")
        }
    ):
        raise ValueError("evaluation parent differs from its frozen method/banks/order")
    return {
        "freeze_path": str(path),
        "freeze_sha256": digest,
        "freeze_body_sha256": fingerprint(body),
        "execution_sha256": method,
        "runner_sha256": body["runner_sha256"],
        "evaluation_order_sha256": body["evaluation_order_sha256"],
    }


def _audit(runs, parent_id, *, visited=(), profile=LEGACY.name):
    selected = study_profile(profile)
    if (
        not isinstance(parent_id, str)
        or not re.fullmatch(re.escape(selected.prefix) + r"[A-Za-z0-9_]+", parent_id)
        or parent_id in visited
        or len(visited) >= _EVALUATION_DEPTH
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
        or launch.get("protocol") != selected.protocol
        or selected != LEGACY
        and launch.get("profile") != selected.name
        or launch.get("phase") not in {"source", "calibration", "evaluation"}
        or launch.get("gold_loaded") is not False
        or launch.get("memory_updates") is not False
        or type(launch.get("arms")) is not list
        or not launch["arms"]
        or len(set(launch["arms"])) != len(launch["arms"])
        or not set(launch["arms"]) <= _ARMS
        or claim != {**launch, "plan_sha256": fingerprint(launch)}
    ):
        raise ValueError("parent launch/claim identity mismatch")
    if launch["phase"] != "evaluation" and len(visited) >= 16:
        raise ValueError("unsafe parent identity or continuation ancestry")
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
    if launch["phase"] == "source" or selected != LEGACY:
        source_signature = _method_signature(signature, selected)
    evaluation_binding = (
        _evaluation_binding(root, launch, read, selected)
        if launch["phase"] == "evaluation"
        else None
    )
    ancestors = [parent_id]
    if launch.get("resume_parent_run_id") is not None:
        certificate = read("resume_certificate.json")
        if fingerprint(certificate) != launch.get("resume_certificate_sha256"):
            raise ValueError("parent continuation certificate changed")
        previous = _validate(root, certificate, visited=(*visited, parent_id), profile=selected)
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
            or (
                evaluation_binding is not None
                and (
                    previous.get("evaluation_binding") != evaluation_binding
                    or planned != previous["question_ids"]
                )
            )
        ):
            raise ValueError("parent continuation ancestry is inconsistent")
        ancestors.extend(previous["ancestor_run_ids"])
    proof = {
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
    # Do not add nullable keys to legacy proofs: historical v1 fingerprints stay valid.
    if evaluation_binding is not None:
        proof["evaluation_binding"] = evaluation_binding
    return proof


def build_certificate(runs, parent_run_id, *, profile=LEGACY.name):
    selected = study_profile(profile)
    proof = _audit(runs, parent_run_id, profile=selected)
    schema = EVALUATION_SCHEMA if proof["phase"] == "evaluation" else SCHEMA
    body = {"schema": schema, "protocol": selected.protocol, "proof": proof}
    return {**body, "sha256": fingerprint(body)}


def _validate(runs, certificate, *, visited=(), profile=None):
    if type(certificate) is not dict or set(certificate) != {
        "schema",
        "protocol",
        "proof",
        "sha256",
    }:
        raise ValueError("invalid explicit resume certificate")
    body = {k: certificate[k] for k in ("schema", "protocol", "proof")}
    selected = study_profile(profile_from_protocol(certificate["protocol"]))
    if profile is not None and study_profile(profile) != selected:
        raise ValueError("resume certificate profile differs from requested profile")
    if (
        certificate["schema"]
        != (EVALUATION_SCHEMA if certificate["proof"].get("phase") == "evaluation" else SCHEMA)
        or fingerprint(body) != certificate["sha256"]
    ):
        raise ValueError("resume certificate checksum/schema mismatch")
    proof = _audit(runs, certificate["proof"]["parent_run_id"], visited=visited, profile=selected)
    if proof != certificate["proof"]:
        raise ValueError("parent evidence changed since certificate creation")
    return proof


def verify_certificate(
    runs,
    path,
    *,
    phase,
    question_ids,
    arms,
    manifest_sha256,
    model,
    signature,
    evaluation_freeze=None,
    expected_freeze_sha256=None,
    profile=None,
):
    certificate, _ = _read(Path(path).resolve(strict=True))
    proof = _validate(runs, certificate, profile=profile)
    selected = study_profile(profile_from_protocol(certificate["protocol"]))
    if (
        proof["phase"] != phase
        or proof["manifest_sha256"] != manifest_sha256
        or proof["model"] != model
        or proof["arms"] != list(arms)
        or not set(_ids(list(question_ids))) <= set(proof["question_ids"])
        or (
            (phase == "source" or selected != LEGACY)
            and proof["source_execution_sha256"] != _method_signature(signature, selected)
        )
    ):
        raise ValueError("resume only permits untouched questions under the approved method/role")
    if phase == "evaluation":
        binding = proof["evaluation_binding"]
        if (
            evaluation_freeze is None
            or binding["freeze_path"] != str(Path(evaluation_freeze).resolve(strict=True))
            or binding["freeze_sha256"] != expected_freeze_sha256
            or binding["execution_sha256"] != _method_signature(signature, selected)
            or list(question_ids) != proof["question_ids"]
        ):
            raise ValueError(
                "evaluation resume requires the same freeze and complete untouched suffix"
            )
    return certificate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["legacy-v1", "action-list-v3"], default=LEGACY.name)
    args = parser.parse_args(argv)
    certificate = build_certificate(args.runs_root, args.parent_run_id, profile=args.profile)
    write_json(args.output, certificate)
    count = len(certificate["proof"]["question_ids"])
    print(f"Offline certificate: {count} untouched questions; no API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
