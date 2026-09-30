"""Explicit operator study profiles; structured v2/v3 are calibration-only.

Legacy method files and signature validation stay untouched. A new prefix never
releases an old claim, and selecting a profile is not permission to spend money.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .fresh_dev_manifest import _sha
from .operator_execution_signature import METHOD_FILES, execution_configuration
from .representation_runner import fingerprint


@dataclass(frozen=True)
class OperatorProfile:
    name: str
    protocol: str
    prefix: str


LEGACY = OperatorProfile("legacy-v1", "growrag-operator-study-v1", "2026-09-30_operator_v1_")
STRUCTURED = OperatorProfile(
    "structured-v2", "growrag-operator-study-structured-v2", "2026-09-30_operator_v2_"
)
ACTION_LIST = OperatorProfile(
    "action-list-v3", "growrag-operator-study-action-list-v3", "2026-09-30_operator_v3_"
)
PROFILES = (LEGACY, STRUCTURED, ACTION_LIST)
STRUCTURED_SCHEMA = "growrag-operator-execution-signature-v2"
STRUCTURED_METHOD_FILES = (
    *METHOD_FILES,
    "src/growrag/experiments/operator_model_v2.py",
    "src/growrag/experiments/operator_schemas.py",
    "src/growrag/experiments/output_schemas.py",
    "src/growrag/experiments/operator_profiles.py",
)
ACTION_LIST_SCHEMA = "growrag-operator-execution-signature-v3"
ACTION_LIST_METHOD_FILES = (
    *STRUCTURED_METHOD_FILES,
    "src/growrag/experiments/operator_model_v3.py",
    "src/growrag/experiments/operator_schemas_v3.py",
)


def get_profile(name: str = LEGACY.name) -> OperatorProfile:
    for profile in PROFILES:
        if type(name) is str and name == profile.name:
            return profile
    raise ValueError("unknown operator profile")


def structured_execution_configuration() -> dict:
    return {**execution_configuration(), "json_object_mode": False, "json_schema_mode": True}


def structured_execution_signature(project_root: Path) -> dict:
    root = Path(project_root).resolve(strict=True)
    files = {}
    for name in STRUCTURED_METHOD_FILES:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("structured method source escapes project")
        files[name] = _sha(path)
    body = {
        "schema": STRUCTURED_SCHEMA,
        "configuration": structured_execution_configuration(),
        "files": files,
    }
    return {**body, "sha256": fingerprint(body)}


def validate_structured_execution_signature(value: dict) -> str:
    if type(value) is not dict or set(value) != {"schema", "configuration", "files", "sha256"}:
        raise ValueError("missing structured execution signature")
    body = {key: value[key] for key in ("schema", "configuration", "files")}
    if (
        value["schema"] != STRUCTURED_SCHEMA
        or value["configuration"] != structured_execution_configuration()
        or type(value["files"]) is not dict
        or set(value["files"]) != set(STRUCTURED_METHOD_FILES)
        or any(
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in value["files"].values()
        )
        or value["sha256"] != fingerprint(body)
    ):
        raise ValueError("invalid structured execution signature")
    return value["sha256"]


def action_list_execution_configuration() -> dict:
    return structured_execution_configuration()


def action_list_execution_signature(project_root: Path) -> dict:
    root = Path(project_root).resolve(strict=True)
    files = {}
    for name in ACTION_LIST_METHOD_FILES:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("action-list method source escapes project")
        files[name] = _sha(path)
    body = {
        "schema": ACTION_LIST_SCHEMA,
        "configuration": action_list_execution_configuration(),
        "files": files,
    }
    return {**body, "sha256": fingerprint(body)}


def validate_action_list_execution_signature(value: dict) -> str:
    if type(value) is not dict or set(value) != {"schema", "configuration", "files", "sha256"}:
        raise ValueError("missing action-list execution signature")
    body = {key: value[key] for key in ("schema", "configuration", "files")}
    if (
        value["schema"] != ACTION_LIST_SCHEMA
        or value["configuration"] != action_list_execution_configuration()
        or type(value["files"]) is not dict
        or set(value["files"]) != set(ACTION_LIST_METHOD_FILES)
        or any(
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in value["files"].values()
        )
        or value["sha256"] != fingerprint(body)
    ):
        raise ValueError("invalid action-list execution signature")
    return value["sha256"]


def validate_profile_options(profile: OperatorProfile, options) -> None:
    """Run before reading data, certificates, banks, secrets or history."""
    if profile == LEGACY:
        return
    if profile not in {STRUCTURED, ACTION_LIST}:
        raise ValueError("unknown operator profile")
    if (
        options.phase != "calibration"
        or not options.arms
        or not set(options.arms) <= {"base", "fresh", "static"}
        or options.banks is not None
        or options.resume_certificate is not None
        or options.evaluation_freeze is not None
        or options.expected_freeze_sha256 is not None
    ):
        raise ValueError(
            f"{profile.name} is calibration-only: source/evaluation, banks and resume remain locked"
        )
    if options.allow_network and options.expected_execution_sha256 is None:
        raise ValueError("structured calibration requires explicit expected execution SHA256")
