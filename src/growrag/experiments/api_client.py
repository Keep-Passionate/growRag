"""Opt-in HTTPS chat transport; never substitutes fixtures for a failed API call.

This is transport infrastructure, not a validated RAG backend or a paid-run CLI.
Compatibility must be checked for the selected provider/model before a pilot.
Only the caller's explicitly supplied environment key is used; Codex login
credentials and the surrounding assistant conversation are never accessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit


class APIRequestError(RuntimeError):
    """A failed request, with no provider body or credential in the exception."""

    def __init__(
        self,
        message: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        api_requests: int = 0,
        request_id: str | None = None,
        audit_path: Path | None = None,
        transport_source: str = "live_api",
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.api_requests = api_requests
        self.request_id = request_id
        self.audit_path = audit_path
        self.transport_source = transport_source


@dataclass(frozen=True, slots=True)
class ChatConfig:
    base_url: str
    model: str
    key_environment_variable: str
    max_calls: int
    max_output_tokens: int = 512
    timeout_seconds: float = 45.0
    output_limit_parameter: str = "max_completion_tokens"
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be HTTPS without credentials, query or fragment")
        for value in (self.model, self.key_environment_variable):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("model and key environment variable must be explicit")
        for value in (self.max_calls, self.max_output_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("call and output limits must be positive integers")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be finite and within (0, 60]")
        if self.output_limit_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("unsupported output limit parameter")
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be bool or None")


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    requested_model: str
    returned_model: str | None
    response_id: str | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float
    audit_path: Path
    transport_source: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise APIRequestError("HTTP redirects are disabled; verify the configured endpoint")


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _redact(value: object, secret: str) -> object:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED_API_KEY]")
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    return value


class LiveChatClient:
    """Single-threaded, bounded client with an append-only file per API attempt.

    ``allow_network`` must be explicitly enabled by the caller after a provider,
    data scope and budget are agreed. max_calls/output limits constrain requests,
    not currency spend. A provider-side spending limit is still recommended.
    """

    transport_source = "live_api"

    def __init__(
        self, config: ChatConfig, audit_directory: Path, *, allow_network: bool = False
    ) -> None:
        self.config = config
        self.audit_directory = Path(audit_directory)
        self.allow_network = allow_network
        self.attempts = 0

    def complete(
        self, messages: list[dict[str, str]], *, trace_id: str, prompt_version: str
    ) -> ChatResponse:
        if not self.allow_network:
            raise APIRequestError("live API calls are disabled; no request was sent")
        if self.attempts >= self.config.max_calls:
            raise APIRequestError("API call limit reached; no request was sent")
        if not trace_id.strip() or not prompt_version.strip():
            raise ValueError("trace_id and prompt_version are required")
        if not messages or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            for message in messages
        ):
            raise ValueError("messages must contain only role and text content")
        api_key = os.environ.get(self.config.key_environment_variable, "")
        if not api_key.strip():
            raise APIRequestError("configured API key is absent; no request was sent")

        payload = {
            "model": self.config.model,
            "messages": messages,
            self.config.output_limit_parameter: self.config.max_output_tokens,
            "stream": False,
        }
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if api_key in body.decode("utf-8"):
            raise APIRequestError("credential detected in request content; no request was sent")
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        self.audit_directory.mkdir(parents=True, exist_ok=True)
        # A digest avoids using user-supplied trace IDs as filesystem paths.
        file_id = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()
        audit_path = self.audit_directory / f"{file_id}.json"
        start = time.perf_counter()
        event: dict[str, object] = {
            "schema_version": 1,
            "execution_kind": "api_transport",
            "trace_id": trace_id,
            "prompt_version": prompt_version,
            "started_at": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "request": payload,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "status": "started",
            "network_attempted": False,
            "retry_count": 0,
            "transport_source": self.transport_source,
            "api_requests": 0,
            "input_tokens": None,
            "output_tokens": None,
        }
        # Reserve the audit ID before sending: never charge twice for a reused ID.
        with audit_path.open("x", encoding="utf-8") as handle:
            json.dump(_redact(event, api_key), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        self.attempts += 1
        event["network_attempted"] = True
        event["api_requests"] = 1
        try:
            opener = urllib.request.build_opener(_NoRedirect())
            with opener.open(request, timeout=self.config.timeout_seconds) as response:
                event["http_status"] = response.status
                event["request_id"] = _redact(response.headers.get("x-request-id"), api_key)
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise APIRequestError("provider response exceeded the configured size limit")
            event["response_sha256"] = hashlib.sha256(raw).hexdigest()
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise APIRequestError("provider response is not a JSON object")
            event["response"] = data
            event["response_redacted"] = api_key in json.dumps(data, ensure_ascii=False)
            usage = data.get("usage")
            if isinstance(usage, dict):
                event["input_tokens"] = _token_count(usage.get("prompt_tokens"))
                event["output_tokens"] = _token_count(usage.get("completion_tokens"))
            if event["response_redacted"]:
                raise APIRequestError("provider response contained credentials and was rejected")
            choices = data.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise APIRequestError("provider did not return exactly one completion")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                raise APIRequestError(
                    "completion was truncated, refused or did not finish normally"
                )
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("refusal") or message.get("tool_calls"):
                raise APIRequestError("provider did not return a plain text completion")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise APIRequestError("provider returned no usable text")
            event["status"] = "completed"
            return ChatResponse(
                content=content,
                requested_model=self.config.model,
                returned_model=data.get("model") if isinstance(data.get("model"), str) else None,
                response_id=data.get("id") if isinstance(data.get("id"), str) else None,
                request_id=event.get("request_id"),
                input_tokens=event["input_tokens"],
                output_tokens=event["output_tokens"],
                elapsed_seconds=time.perf_counter() - start,
                audit_path=audit_path,
                transport_source=self.transport_source,
            )
        except urllib.error.HTTPError as error:
            event["status"] = "failed"
            event["error_type"] = "HTTPError"
            event["http_status"] = error.code
            # Do not store exception/provider error strings: they can echo secrets.
            raise self._failure(
                f"API HTTP {error.code}; no retry or fallback", event, audit_path
            ) from None
        except (OSError, ValueError, KeyError, TypeError, APIRequestError) as error:
            event["status"] = "failed"
            event["error_type"] = type(error).__name__
            raise self._failure(
                "API request/response failed; no retry or fallback", event, audit_path
            ) from None
        finally:
            event["elapsed_seconds"] = time.perf_counter() - start
            # This is the file reserved by this attempt, not a previous run's file.
            with audit_path.open("w", encoding="utf-8") as handle:
                json.dump(_redact(event, api_key), handle, ensure_ascii=False, indent=2)

    def _failure(self, message: str, event: dict, audit_path: Path) -> APIRequestError:
        return APIRequestError(
            message,
            input_tokens=event["input_tokens"],
            output_tokens=event["output_tokens"],
            api_requests=event["api_requests"],
            request_id=event.get("request_id"),
            audit_path=audit_path,
            transport_source=self.transport_source,
        )
