"""Audited serial operator batches: predictions only, no scoring or memory writes.

显式 allow-network 才调用普通按量 API。评测须先通过独立来源冻结证书。
每条请求、每题输出、失败和预算独立保存；旧 claim 不自动释放或重试。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from growrag.operator_bank import FrozenOperatorBank
from growrag.operator_loop import run_operator_episode

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .operator_data_plan import SCHEMA, SOURCE_SIZES
from .operator_execution_signature import execution_configuration, execution_signature
from .operator_model import ModelOperatorPlanner, answer_episode, seed_specs, shortlist_specs
from .operator_profiles import (
    ACTION_LIST,
    LEGACY,
    PROFILES,
    STRUCTURED,
    action_list_execution_configuration,
    action_list_execution_signature,
    get_profile,
    structured_execution_configuration,
    structured_execution_signature,
    validate_profile_options,
)
from .operator_resume import verify_certificate
from .pre_pilot import write_json
from .protocol import Evidence, RuntimeQuestion
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient
from .run_shared_s2g import reviewed_history
from .shared_s2g_corpus import SharedBM25Index

PROTOCOL = "growrag-operator-study-v1"
PREFIX = "2026-09-30_operator_v1_"
KEY_VARIABLE = "GROWRAG_OPERATOR_API_KEY"
ARMS = ("base", "fresh", "static", "memory50", "memory100", "memory250", "memory500")
COUNTS = {"source": 500, "calibration": 100, "evaluation": 500}
_ID = re.compile(r"[0-9a-f]{24}")
_SHA = re.compile(r"[0-9a-f]{64}")


def _jsonable(value):
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


class OperatorLog:
    def __init__(self, output):
        self.output = output
        self.question_id, self.arm = "", "setup"

    def __call__(self, event):
        record = {
            "utc": datetime.now(UTC).isoformat(),
            "question_id": self.question_id,
            "arm": self.arm,
            **_jsonable(event),
        }
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        brief = {
            k: record[k]
            for k in (
                "utc",
                "question_id",
                "arm",
                "kind",
                "status",
                "stop_reason",
                "requests",
                "error_type",
                "query",
            )
            if k in record
        }
        if "search" in record:
            brief["query"] = record["search"]["query"]
        line = json.dumps(brief, ensure_ascii=False)
        with (self.output / "live.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print(line, flush=True)


def artifact(root, manifest, name):
    root = Path(root).resolve(strict=True)
    if Path(name).name != name or not re.fullmatch(r"[a-z_]+\.jsonl", name):
        raise ValueError("artifact name must be local")
    info = manifest["artifacts"][name]
    if (
        info.get("contains_gold") is not False
        or info.get("runtime_safe") is not True
        or not isinstance(info.get("sha256"), str)
        or not _SHA.fullmatch(info["sha256"])
        or type(info.get("rows")) is not int
        or info["rows"] <= 0
    ):
        raise ValueError("unreviewed runtime artifact metadata")
    path = (root / name).resolve(strict=True)
    if path.parent != root or not path.is_file() or _sha(path) != info["sha256"]:
        raise ValueError("artifact path/hash mismatch")
    return path


def _evaluation_gate(options, project, *, chosen_ids=None):
    """Validate provenance before reading evaluation text or launching paid work.

    Local import is essential: the offline freeze builder itself imports this
    runner's source-only loader. This gate never scores or opens dev labels.
    """
    if (
        options is None
        or options.phase != "evaluation"
        or not options.evaluation_freeze
        or not isinstance(options.expected_freeze_sha256, str)
        or not _SHA.fullmatch(options.expected_freeze_sha256)
    ):
        raise ValueError("evaluation locked until source/calibration freeze certificates exist")
    if (
        tuple(options.arms) != ARMS
        or options.banks is None
        or not isinstance(options.expected_execution_sha256, str)
        or not _SHA.fullmatch(options.expected_execution_sha256)
        or not isinstance(options.expected_manifest_sha256, str)
        or not _SHA.fullmatch(options.expected_manifest_sha256)
    ):
        raise ValueError(
            "evaluation requires all seven ordered arms, frozen banks and fingerprints"
        )
    from .operator_evaluation_freeze import validate_evaluation_freeze

    project = Path(project).resolve(strict=True)
    body = validate_evaluation_freeze(
        project,
        options.evaluation_freeze,
        expected_certificate_sha256=options.expected_freeze_sha256,
    )
    if (
        body["protocol"] != PROTOCOL
        or (project / body["paths"]["manifest"]).resolve(strict=True)
        != options.manifest.resolve(strict=True)
        or (project / body["paths"]["banks"]).resolve(strict=True)
        != options.banks.resolve(strict=True)
        or (project / body["paths"]["runs"]).resolve(strict=True) != project / "runs"
        or body["expected"]["manifest"] != options.expected_manifest_sha256
        or body["expected"]["execution"] != options.expected_execution_sha256
        or execution_signature(project) != body["execution_signature"]
        or set(body["controls"]) != set(ARMS)
    ):
        raise ValueError("evaluation request differs from the frozen manifest/banks/method")
    ids = body["evaluation_ids"]
    if (
        body["evaluation_order_sha256"] != fingerprint(ids)
        or chosen_ids is not None
        and chosen_ids != ids[options.start : options.start + options.count]
    ):
        raise ValueError("evaluation order differs from frozen question IDs")
    if options.resume_certificate is not None:
        # Reject source/calibration certificates before decoding any evaluation text.
        verify_certificate(
            project / "runs",
            options.resume_certificate,
            phase="evaluation",
            question_ids=ids[options.start : options.start + options.count],
            arms=options.arms,
            manifest_sha256=options.expected_manifest_sha256,
            model=body["execution_signature"]["configuration"]["model"],
            signature=body["execution_signature"],
            evaluation_freeze=options.evaluation_freeze,
            expected_freeze_sha256=options.expected_freeze_sha256,
        )
    return body


def load_inputs(path, phase, *, evaluation_options=None):
    # Lock BEFORE opening even the manifest/evaluation text for direct callers.
    evaluation = None
    if phase == "evaluation":
        evaluation = _evaluation_gate(evaluation_options, Path.cwd())
        if Path(path).resolve(strict=True) != evaluation_options.manifest.resolve(strict=True):
            raise ValueError("evaluation loader manifest differs from the authorized request")
    elif phase not in {"source", "calibration"}:
        raise ValueError("unknown experiment phase")
    elif evaluation_options is not None:
        raise ValueError("evaluation options cannot authorize another phase")
    path = Path(path).resolve(strict=True)
    if path.name != "manifest.json":
        raise ValueError("expected frozen manifest.json")
    manifest = json.loads(path.read_bytes())
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("official_splits")
        != {"source": "train", "calibration": "train", "evaluation": "dev"}
        or manifest.get("counts") != COUNTS
        or manifest.get("source_sizes") != list(SOURCE_SIZES)
        or manifest.get("official_test_used") is not False
        or manifest.get("evaluation_memory_updates_allowed") is not False
        or manifest.get("calibration_memory_updates_allowed") is not False
        or manifest.get("gold_projected") is not False
    ):
        raise ValueError("frozen train500/train100/dev500 manifest required")
    if path.with_suffix(".sha256").read_text().strip() != f"{_sha(path)}  manifest.json":
        raise ValueError("manifest hash sidecar mismatch")
    roles = manifest.get("roles")
    if type(roles) is not dict or set(roles) != set(COUNTS):
        raise ValueError("unexpected role registry")
    for role, count in COUNTS.items():
        ids = roles[role]
        if (
            type(ids) is not list
            or len(ids) != count
            or any(not isinstance(qid, str) or not _ID.fullmatch(qid) for qid in ids)
        ):
            raise ValueError("invalid role IDs/counts")
    all_ids = [qid for group in roles.values() for qid in group]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("role overlap")
    if manifest.get("nested_source_ids") != {
        str(size): roles["source"][:size] for size in SOURCE_SIZES
    }:
        raise ValueError("source scales must be exact nested prefixes")
    if evaluation is not None and roles[phase] != evaluation["evaluation_ids"]:
        raise ValueError("evaluation order differs from frozen question IDs")
    filename = f"{phase}_runtime_questions.jsonl"
    runtime_path = artifact(path.parent, manifest, filename)
    rows = [json.loads(line) for line in runtime_path.read_text(encoding="utf-8").splitlines()]
    if any(type(row) is not dict or set(row) != {"question_id", "text", "dataset"} for row in rows):
        raise ValueError("runtime contains unallowlisted fields or labels")
    if (
        [row["question_id"] for row in rows] != roles[phase]
        or manifest["artifacts"][filename]["rows"] != len(rows)
        or any(not isinstance(row["dataset"], str) for row in rows)
    ):
        raise ValueError("runtime role/order/count mismatch")
    return manifest, tuple(RuntimeQuestion(**row) for row in rows)


def load_banks(directory, arms, manifest):
    banks = {}
    for arm in arms:
        if not arm.startswith("memory"):
            continue
        if directory is None:
            raise ValueError("memory arms require explicitly frozen banks")
        directory = Path(directory).resolve(strict=True)
        size = arm.removeprefix("memory")
        path = (directory / f"bank_{size}.json").resolve(strict=True)
        if path.parent != directory:
            raise ValueError("bank path escapes explicit directory")
        bank = FrozenOperatorBank.from_json(path.read_text(encoding="utf-8"))
        if set(bank.allowed_source_ids) != set(manifest["nested_source_ids"][size]):
            raise ValueError("bank source scope does not match the expected prefix")
        if bank.protocol_id != PROTOCOL:
            raise ValueError("bank must have reviewed protocol")
        banks[arm] = (bank, path, _sha(path))
    return banks


def verify_banks(banks):
    for bank, path, digest in banks.values():
        if _sha(path) != digest:
            raise ValueError("frozen bank file changed")
        FrozenOperatorBank.from_json(path.read_text(encoding="utf-8")).verify_fingerprint(
            bank.fingerprint
        )


def check_unstarted(runs, phase, ids, arms, *, resume_certificate=None, profile=LEGACY.name):
    active_profile = get_profile(profile)
    released_parents = (
        set(resume_certificate["proof"]["ancestor_run_ids"]) if resume_certificate else set()
    )
    paths = [
        (registered, path)
        for registered in PROFILES
        for path in (
            *runs.glob(f"{registered.prefix}*/launch_plan.json"),
            *runs.glob(f"{registered.prefix}*.claim.json"),
        )
    ]
    for registered, path in paths:
        if not path.resolve().is_relative_to(runs.resolve()):
            raise ValueError("old claim escapes runs root")
        old = json.loads(path.read_bytes())
        if (
            old.get("protocol") != registered.protocol
            or old.get("phase") not in COUNTS
            or type(old.get("question_ids")) is not list
            or not old["question_ids"]
            or any(not isinstance(q, str) or not _ID.fullmatch(q) for q in old["question_ids"])
            or len(set(old["question_ids"])) != len(old["question_ids"])
            or type(old.get("arms")) is not list
            or not old["arms"]
            or not set(old["arms"]) <= set(ARMS)
            or (
                registered != LEGACY
                and (
                    old["phase"] != "calibration"
                    or not set(old["arms"]) <= {"base", "fresh", "static"}
                )
            )
        ):
            raise ValueError("old pending claim/launch requires offline audit")
        if set(ids) & set(old["question_ids"]):
            if old["phase"] != phase:
                raise ValueError("question previously claimed in a different phase")
            if set(arms) & set(old["arms"]):
                identity = (
                    path.parent.name
                    if path.name == "launch_plan.json"
                    else path.name.removesuffix(".claim.json")
                )
                if registered == active_profile and identity in released_parents:
                    continue
                raise ValueError("question/arm already claimed; no automatic replay")


@contextmanager
def serial_lock(runs, *, profile=LEGACY.name):
    """Only this runner's processes cooperate; all project paid runners must be serial."""
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / ".operator-study.lock"
    # Crash leaves the exclusive lock for inspection, never silently steals it.
    write_json(path, {"pid": os.getpid(), "protocol": get_profile(profile).protocol})
    digest = _sha(path)
    try:
        yield
    finally:
        if _sha(path) != digest:
            raise ValueError("serial lock changed; preserve it for audit")
        path.unlink()


def execute_one(question, arm, index, client, *, bank, output, trace, log, profile=LEGACY.name):
    active_profile = get_profile(profile)
    configuration = {
        LEGACY: execution_configuration,
        STRUCTURED: structured_execution_configuration,
        ACTION_LIST: action_list_execution_configuration,
    }[active_profile]()
    planner_type, reader = ModelOperatorPlanner, answer_episode
    if active_profile != LEGACY:
        if bank is not None or arm not in {"base", "fresh", "static"}:
            raise ValueError(f"{active_profile.name} calibration cannot use memory banks")
        from .operator_model_v2 import answer_episode_v2

        reader = answer_episode_v2
        if active_profile == STRUCTURED:
            from .operator_model_v2 import ModelOperatorPlannerV2

            planner_type = ModelOperatorPlannerV2
        else:
            from .operator_model_v3 import ModelOperatorPlannerV3

            planner_type = ModelOperatorPlannerV3
    specs = bank.published_specs if bank else seed_specs() if arm == "static" else ()
    mode = "memory" if arm.startswith("memory") else "static" if arm == "static" else "fresh"
    start = len(client.calls)
    report = {
        "status": "started",
        "question_id": question.question_id,
        "arm": arm,
        "episode": None,
        "reader": None,
        "feedback": None,
        "memory_updated": False,
    }
    try:
        if arm.startswith("memory"):
            if bank is None:
                raise ValueError("memory arm requires a frozen bank; missing is not empty")
            # 空库不是额外的提示处理：让模型输入逐字等同 FRESH，避免把模式标签
            # 或过滤掉的卡片数量造成的差异误认为经验收益。审计信息只留在日志。
            published_count = len(specs)
            candidate_count = len(shortlist_specs(question.text, specs))
            fallback = candidate_count == 0
            if fallback:
                mode, specs = "fresh", ()
            memory_context = {
                "policy": "empty-visible-library-is-fresh-v1",
                "published_count": published_count,
                "visible_candidate_count": candidate_count,
                "effective_planner_mode": mode,
                "fallback_reason": (
                    "empty_published_library"
                    if fallback and published_count == 0
                    else "all_specs_exceed_prompt_limit"
                    if fallback
                    else None
                ),
            }
            report["memory_context"] = memory_context
            log({"kind": "memory_context", **memory_context})
        planner = planner_type(
            client,
            mode=mode,
            specs=specs,
            trace_prefix=trace,
            on_record=lambda item: log({"kind": "planner_record", **item}),
        )

        def retrieve(query, top_k):
            return tuple(Evidence(d.doc_id, d.title, 0, d.text) for d in index(query, top_k))

        result = run_operator_episode(
            question,
            retrieve,
            planner,
            retrieval_budget=configuration["base_retrieval_budget"]
            if arm == "base"
            else configuration["retrieval_budget"],
            max_decisions=configuration["max_decisions"],
            top_k=configuration["top_k"],
            on_event=log,
        )
        report["episode"] = _jsonable(result)
        reader_options = (
            {"on_record": lambda item: log({"kind": "reader_record", **item})}
            if active_profile != LEGACY
            else {}
        )
        report["reader"] = reader(
            client, question, result.evidence, trace_id=f"{trace}/reader", **reader_options
        )
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["calls"] = list(client.calls[start:])
        write_json(output, report)
    return report


def _live(args, project, runs, manifest, chosen, banks, run_id, manifest_sha):
    active_profile = get_profile(getattr(args, "profile", LEGACY.name))
    validate_profile_options(active_profile, args)
    # One lock covers history reconciliation, claim, all calls and ledger finalization.
    with serial_lock(runs, profile=active_profile.name):
        signature = {
            LEGACY: execution_signature,
            STRUCTURED: structured_execution_signature,
            ACTION_LIST: action_list_execution_signature,
        }[active_profile](project)
        if (
            args.expected_execution_sha256 is not None
            and signature["sha256"] != args.expected_execution_sha256
        ):
            raise ValueError("expected execution signature mismatch; no request sent")
        evaluation = (
            _evaluation_gate(args, project, chosen_ids=[q.question_id for q in chosen])
            if args.phase == "evaluation"
            else None
        )
        configuration = signature["configuration"]
        resume = (
            verify_certificate(
                runs,
                args.resume_certificate,
                phase=args.phase,
                question_ids=[q.question_id for q in chosen],
                arms=args.arms,
                manifest_sha256=manifest_sha,
                model=configuration["model"],
                signature=signature,
                evaluation_freeze=args.evaluation_freeze,
                expected_freeze_sha256=args.expected_freeze_sha256,
            )
            if args.resume_certificate
            else None
        )
        check_unstarted(
            runs,
            args.phase,
            [q.question_id for q in chosen],
            args.arms,
            resume_certificate=resume,
            profile=active_profile.name,
        )
        history = reviewed_history(runs)
        prior = history["prior_reserved_cny"]
        if type(prior) not in (int, float) or not math.isfinite(prior) or prior < 0:
            raise ValueError("invalid prior budget accounting")
        subcap = min(args.budget_cny, args.project_cap_cny - prior)
        if subcap <= 0:
            raise ValueError(f"project cumulative {args.project_cap_cny:g} CNY budget exhausted")
        git, snapshot = _git_state(), source_snapshot(project)
        if not args.api_config or git["worktree_dirty"] or not git.get("commit"):
            raise ValueError("live execution requires specified config and clean committed code")
        settings = read_local_bailian_settings(args.api_config)
        host = urlsplit(settings.base_url).hostname or ""
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("reviewed ordinary Beijing endpoint required")
        verify_banks(banks)
        corpus = artifact(args.manifest.parent, manifest, "corpus.jsonl")
        if (
            _sha(args.manifest) != manifest_sha
            or source_snapshot(project)["sha256"] != snapshot["sha256"]
        ):
            raise ValueError("manifest/source changed before launch")
        info = manifest["artifacts"]["corpus.jsonl"]
        max_calls = args.count * sum(
            1 if arm == "base" else configuration["max_decisions"] + 1 for arm in args.arms
        )
        config = ChatConfig(
            settings.base_url,
            configuration["model"],
            KEY_VARIABLE,
            max_calls=max_calls,
            max_output_tokens=configuration["max_output_tokens"],
            output_limit_parameter=configuration["output_limit_parameter"],
            enable_thinking=configuration["enable_thinking"],
            temperature=configuration["temperature"],
            top_p=configuration["top_p"],
            json_object_mode=configuration["json_object_mode"],
            json_schema_mode=configuration.get("json_schema_mode", False),
        )
        limits = PriceLimits(
            budget_cny=subcap,
            max_prompt_bytes=configuration["max_prompt_bytes"],
            max_elapsed_seconds=1800,
        )
        plan = {
            **configuration,
            "protocol": active_profile.protocol,
            "run_id": run_id,
            "phase": args.phase,
            "question_ids": [q.question_id for q in chosen],
            "arms": args.arms,
            "manifest_sha256": manifest_sha,
            "max_calls": max_calls,
            "execution_signature": signature,
            "evaluation_freeze_sha256": args.expected_freeze_sha256 if evaluation else None,
            "evaluation_order_sha256": (
                evaluation["evaluation_order_sha256"] if evaluation else None
            ),
            "evaluation_freeze_path": (
                str(args.evaluation_freeze.resolve(strict=True)) if evaluation else None
            ),
            "resume_parent_run_id": resume["proof"]["parent_run_id"] if resume else None,
            "resume_certificate_sha256": fingerprint(resume) if resume else None,
            "git": git,
            "source_sha256": snapshot["sha256"],
            "bank_sha256": {a: b[0].fingerprint for a, b in banks.items()},
            "bank_file_sha256": {a: b[2] for a, b in banks.items()},
            "project_cap_cny": args.project_cap_cny,
            "budget_authorization_note": (
                "Effective project ceiling for this launch; includes all reconciled historical "
                "reservations, not an additional allocation. Historical authorization is unchanged."
            ),
            "historical_authorized_total_cny": history.get("authorized_total_cny"),
            "historical_authorization_date": history.get("authorization_date"),
            "subcap_cny": subcap,
            "prior_reserved_cny": prior,
            "timeout_seconds": 1800,
            "price_checked_date": "2026-09-30",
            "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
            "gold_loaded": False,
            "memory_updates": False,
        }
        if active_profile != LEGACY:
            plan["profile"] = active_profile.name
            plan["profile_scope"] = "calibration_only_not_source_or_evaluation"
        output = runs / run_id
        if output.exists():
            raise FileExistsError("old run output requires audit; never overwrite")
        write_json(runs / f"{run_id}.claim.json", {**plan, "plan_sha256": fingerprint(plan)})
        output.mkdir(exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        if evaluation is not None:
            write_json(output / "evaluation_freeze_verified.json", evaluation)
        if resume:
            write_json(output / "resume_certificate.json", resume)
        write_json(output / "source_snapshot.json", snapshot)
        write_json(output / "process.json", {"pid": os.getpid(), "argv": sys.argv})
        previous, client, index = os.environ.get(KEY_VARIABLE), None, None
        log, reports, failed = OperatorLog(output), [], False
        cleanup_errors = []
        try:
            index = SharedBM25Index(
                corpus.parent / "index.sqlite3",
                corpus,
                info["sha256"],
                info["rows"],
                k1=configuration["bm25_k1"],
                b=configuration["bm25_b"],
            )
            os.environ[KEY_VARIABLE] = settings.api_key
            client = DurableBudgetClient(
                LiveChatClient(config, output / "api_audit", allow_network=True),
                limits,
                output / "request_journal",
            )
            log({"kind": "launch"})
            for number, question in enumerate(chosen):
                log.question_id = question.question_id
                item = {"question_id": question.question_id, "arms": {}}
                reports.append(item)
                for arm in args.arms:
                    log.arm = arm
                    bank = banks[arm][0] if arm in banks else None
                    start = len(client.calls)
                    target = output / f"{question.question_id}_{arm}.json"
                    try:
                        verify_banks(banks)
                        item["arms"][arm] = execute_one(
                            question,
                            arm,
                            index,
                            client,
                            bank=bank,
                            output=target,
                            trace=f"{run_id}/{question.question_id}/{arm}",
                            log=log,
                            **(
                                {"profile": active_profile.name} if active_profile != LEGACY else {}
                            ),
                        )
                        verify_banks(banks)
                    except BaseException as error:
                        failed = True
                        partial = (
                            json.loads(target.read_bytes())
                            if target.is_file()
                            else {
                                "calls": list(client.calls[start:]),
                                "episode": None,
                                "reader": None,
                            }
                        )
                        partial.update(
                            status="failed", error_type=type(error).__name__, feedback=None
                        )
                        item["arms"][arm] = partial
                        if not target.exists():
                            write_json(target, partial)
                        log({"kind": "failure", "error_type": type(error).__name__})
                        break
                    log(
                        {"kind": "arm_complete", "status": "completed", "requests": client.attempts}
                    )
                # Exclusive files: checkpoints do not overwrite previous predictions.
                write_json(output / f"checkpoint_{number:04d}.json", item)
                if failed:
                    break
        except BaseException as error:
            failed = True
            log({"kind": "failure", "error_type": type(error).__name__})
        finally:
            # Never return inside try: late bank/ledger errors must affect exit status.
            try:
                try:
                    verify_banks(banks)
                    if _sha(args.manifest) != manifest_sha or _sha(corpus) != info["sha256"]:
                        raise ValueError("frozen input changed")
                    if (
                        evaluation is not None
                        and _evaluation_gate(
                            args, project, chosen_ids=[q.question_id for q in chosen]
                        )
                        != evaluation
                    ):
                        raise ValueError("evaluation freeze changed during execution")
                except Exception as error:
                    cleanup_errors.append(type(error).__name__)
                if index is not None:
                    try:
                        index.close()
                    except Exception as error:
                        cleanup_errors.append(type(error).__name__)
                failed = failed or bool(cleanup_errors) or bool(client and client.block_reason)
                write_json(output / "predictions.json", reports)
                write_json(
                    output / "predictions_frozen.json",
                    {
                        "sha256": fingerprint(reports),
                        "phase": args.phase,
                        "question_ids": [r["question_id"] for r in reports],
                        "gold_loaded": False,
                        "status": "failed" if failed else "completed",
                        "cleanup_errors": cleanup_errors,
                    },
                )
                if client is not None:
                    write_json(output / "final_budget.json", client.report())
                    write_json(
                        output / "cumulative_budget.json",
                        {
                            "prior_reserved_cny": prior,
                            "new_reserved_cny": client.reserved_cny,
                            "cumulative_reserved_cny": prior + client.reserved_cny,
                            "project_cap_cny": args.project_cap_cny,
                        },
                    )
                log(
                    {
                        "kind": "exit",
                        "status": "failed" if failed else "completed",
                        "requests": client.attempts if client else 0,
                    }
                )
            finally:
                if previous is None:
                    os.environ.pop(KEY_VARIABLE, None)
                else:
                    os.environ[KEY_VARIABLE] = previous
        return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=[p.name for p in PROFILES], default=LEGACY.name)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-execution-sha256")
    parser.add_argument("--resume-certificate", type=Path)
    parser.add_argument("--evaluation-freeze", type=Path)
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--phase", choices=tuple(COUNTS), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["base", "fresh", "static"])
    parser.add_argument("--banks", type=Path)
    parser.add_argument("--budget-cny", type=float, default=1.0)
    parser.add_argument(
        "--project-cap-cny",
        type=float,
        default=50.0,
        help="Explicitly authorized cumulative project ceiling, including all historical costs.",
    )
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    active_profile = get_profile(args.profile)
    validate_profile_options(active_profile, args)
    if args.phase == "evaluation":
        if args.resume_certificate is not None and not args.evaluation_freeze:
            raise ValueError("evaluation resume requires its original evaluation freeze")
        if not args.evaluation_freeze or not args.expected_freeze_sha256:
            raise ValueError("evaluation locked until source/calibration freeze certificates exist")
    elif args.evaluation_freeze is not None or args.expected_freeze_sha256 is not None:
        raise ValueError("evaluation freeze options require the evaluation phase")
    if not 1 <= args.count <= 25 or args.start < 0 or len(set(args.arms)) != len(args.arms):
        raise ValueError("unique arms and a fixed batch of 1-25 questions required")
    if not math.isfinite(args.budget_cny) or not 0 < args.budget_cny <= 10:
        raise ValueError("batch cap must be finite, positive and at most 10 CNY")
    if not math.isfinite(args.project_cap_cny) or not 0 < args.project_cap_cny <= 300:
        raise ValueError("project cap must be finite, positive and at most 300 CNY")
    if args.phase == "source" and any(arm.startswith("memory") for arm in args.arms):
        raise ValueError("source collection cannot reuse its own future source banks")
    if args.allow_network and (
        not isinstance(args.expected_manifest_sha256, str)
        or not _SHA.fullmatch(args.expected_manifest_sha256)
    ):
        raise ValueError("live launch requires explicit expected manifest SHA256")
    if args.expected_execution_sha256 is not None and not _SHA.fullmatch(
        args.expected_execution_sha256
    ):
        raise ValueError("invalid expected execution SHA256")
    if args.allow_network and args.phase == "source" and args.expected_execution_sha256 is None:
        raise ValueError("source launch requires explicit expected execution SHA256")
    project = Path.cwd().resolve()
    runs = (project / "runs").resolve()
    args.manifest = args.manifest.resolve(strict=True)
    if not args.manifest.is_relative_to(project) or not runs.is_relative_to(project):
        raise ValueError("manifest and run output must remain inside the project")
    manifest_sha = _sha(args.manifest)
    if args.expected_manifest_sha256 is not None and manifest_sha != args.expected_manifest_sha256:
        raise ValueError("expected manifest SHA256 mismatch")
    manifest, questions = (
        load_inputs(args.manifest, args.phase, evaluation_options=args)
        if args.phase == "evaluation"
        else load_inputs(args.manifest, args.phase)
    )
    chosen = questions[args.start : args.start + args.count]
    if len(chosen) != args.count:
        raise ValueError("batch exceeds frozen split")
    banks = load_banks(args.banks, args.arms, manifest)
    run_id = (
        f"{active_profile.prefix}{args.phase}_{'_'.join(args.arms)}"
        f"_{args.start:04d}_{args.start + args.count:04d}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid run identity")
    resume = (
        verify_certificate(
            runs,
            args.resume_certificate,
            phase=args.phase,
            question_ids=[q.question_id for q in chosen],
            arms=args.arms,
            manifest_sha256=manifest_sha,
            model=PILOT_MODEL,
            signature=execution_signature(project),
            evaluation_freeze=args.evaluation_freeze,
            expected_freeze_sha256=args.expected_freeze_sha256,
        )
        if args.resume_certificate
        else None
    )
    check_unstarted(
        runs,
        args.phase,
        [q.question_id for q in chosen],
        args.arms,
        resume_certificate=resume,
        profile=active_profile.name,
    )
    if not args.allow_network:
        print(
            json.dumps(
                {
                    "protocol": active_profile.protocol,
                    "profile": active_profile.name,
                    "run_id": run_id,
                    "phase": args.phase,
                    "question_ids": [q.question_id for q in chosen],
                    "arms": args.arms,
                    "manifest_sha256": manifest_sha,
                    "requested_subcap_cny": args.budget_cny,
                    "project_cap_cny": args.project_cap_cny,
                    "network_enabled": False,
                    "notice": "Live launch rechecks cumulative budget under "
                    "an exclusive serial lock.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return _live(args, project, runs, manifest, chosen, banks, run_id, manifest_sha)


if __name__ == "__main__":
    raise SystemExit(main())
