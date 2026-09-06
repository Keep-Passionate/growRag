"""One opt-in ordinary Bailian API check, never a benchmark or Token Plan client.

Read only the user-selected local file. Never print credentials, load Codex login
state, change the configured region, retry, or substitute another model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .api_client import APIRequestError, ChatConfig, LiveChatClient


@dataclass(frozen=True)
class LocalBailianSettings:
    base_url: str
    api_key: str = field(repr=False)


def read_local_bailian_settings(path: Path) -> LocalBailianSettings:
    """Extract data only, not instructions, from one explicitly selected file."""
    if path.stat().st_size > 65536:
        raise ValueError("configuration file exceeds 64 KiB")
    content = path.read_text(encoding="utf-8-sig")
    keys = set(re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}", content))
    urls = set(re.findall(r"https://[^\s\"'`<>]+", content))
    if len(keys) != 1 or len(urls) != 1:
        raise ValueError("configuration must contain exactly one API key and one HTTPS base URL")
    key, base_url = keys.pop(), urls.pop().rstrip("/")
    if key.startswith("sk-sp-"):
        raise ValueError("subscription-plan credentials cannot be used by this experiment client")
    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    legacy_hosts = {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
    }
    workspace_host = re.fullmatch(
        r"ws-[a-z0-9-]+\.(cn-beijing|ap-southeast-1|ap-northeast-1)\.maas\.aliyuncs\.com",
        host,
    )
    if (
        (host not in legacy_hosts and workspace_host is None)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or parsed.path != "/compatible-mode/v1"
    ):
        raise ValueError("expected an ordinary official Bailian endpoint; no credential was sent")
    return LocalBailianSettings(base_url, key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        parser.error("network disabled; pass --allow-network only for an authorized paid check")
    if not re.fullmatch(r"qwen[a-z0-9.-]*-\d{4}-\d{2}-\d{2}", args.model):
        parser.error("use an explicitly dated Qwen snapshot, not an updating alias")
    try:
        settings = read_local_bailian_settings(args.config)
    except (OSError, ValueError):
        print("Configuration invalid or unavailable; no request sent. Check local file privately.")
        return 2
    args.output.mkdir(parents=True, exist_ok=False)
    variable = "GROWRAG_PREFLIGHT_API_KEY"
    previous = os.environ.get(variable)
    os.environ[variable] = settings.api_key
    client = LiveChatClient(
        ChatConfig(
            base_url=settings.base_url,
            model=args.model,
            key_environment_variable=variable,
            max_calls=1,
            max_output_tokens=64,
            timeout_seconds=45,
            output_limit_parameter="max_tokens",
            enable_thinking=False,
        ),
        args.output / "api_audit",
        allow_network=True,
    )
    summary = {
        "purpose": "connectivity_only_not_HotpotQA_or_reuse_evaluation",
        "requested_model": args.model,
        "max_calls": 1,
        "max_output_tokens": 64,
        "enable_thinking": False,
        "research_effectiveness": "not_evaluated",
    }
    try:
        response = client.complete(
            [{"role": "user", "content": 'Reply with only this JSON: {"ok": true}'}],
            trace_id="connectivity-v1",
            prompt_version="growrag-connectivity-v1",
        )
        try:
            valid = json.loads(response.content) == {"ok": True}
        except ValueError:
            valid = False
        summary.update(
            status="completed" if valid else "invalid_probe_response",
            returned_model=response.returned_model,
            returned_model_matches=response.returned_model == args.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            api_requests=client.attempts,
            audit_path=str(response.audit_path.resolve()),
        )
    except APIRequestError as error:
        summary.update(
            status="failed",
            error=str(error),
            api_requests=error.api_requests,
            input_tokens=error.input_tokens,
            output_tokens=error.output_tokens,
            audit_path=str(error.audit_path.resolve()) if error.audit_path else None,
        )
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
