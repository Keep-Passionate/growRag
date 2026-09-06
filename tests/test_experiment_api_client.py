"""Offline transport tests only: no provider calls or benchmark measurements."""

import io
import json
import urllib.error

import pytest

from growrag.experiments.api_client import APIRequestError, ChatConfig, LiveChatClient


def config(**changes):
    values = dict(
        base_url="https://api.example.invalid/v1",
        model="explicit-model-snapshot",
        key_environment_variable="GROWRAG_TEST_ONLY_KEY",
        max_calls=1,
    )
    return ChatConfig(**(values | changes))


def messages():
    return [{"role": "user", "content": "Return one query, not its answer."}]


class Response(io.BytesIO):
    status = 200
    headers = {"x-request-id": "TEST-REQUEST-ID"}


def fake_provider(monkeypatch, payload):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return Response(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    monkeypatch.setenv("GROWRAG_TEST_ONLY_KEY", "TEST-ONLY-NOT-A-REAL-KEY")
    return calls


def completion(**changes):
    return {
        "id": "TEST-RESPONSE-ID",
        "model": "test-returned-model",
        "choices": [{"finish_reason": "stop", "message": {"content": "new query"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2},
    } | changes


def test_network_disabled_even_when_key_exists(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, completion())
    client = LiveChatClient(config(), tmp_path)
    with pytest.raises(APIRequestError, match="disabled"):
        client.complete(messages(), trace_id="x", prompt_version="v1")
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_missing_key_does_not_send(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, completion())
    monkeypatch.delenv("GROWRAG_TEST_ONLY_KEY")
    with pytest.raises(APIRequestError, match="absent"):
        LiveChatClient(config(), tmp_path, allow_network=True).complete(
            messages(), trace_id="x", prompt_version="v1"
        )
    assert calls == []


def test_audit_uses_provider_counts_without_authorization_secret(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, completion())
    client = LiveChatClient(config(), tmp_path, allow_network=True)
    result = client.complete(messages(), trace_id="q/fresh", prompt_version="v1")
    assert len(calls) == 1
    assert result.input_tokens == 11
    assert result.response_id == "TEST-RESPONSE-ID"
    event_text = result.audit_path.read_text(encoding="utf-8")
    assert "TEST-ONLY-NOT-A-REAL-KEY" not in event_text
    event = json.loads(event_text)
    assert event["status"] == "completed"
    assert event["request"]["model"] == "explicit-model-snapshot"
    assert event["response"]["model"] == "test-returned-model"
    with pytest.raises(APIRequestError, match="limit"):
        client.complete(messages(), trace_id="second", prompt_version="v1")
    assert len(calls) == 1


def test_missing_usage_remains_unknown(tmp_path, monkeypatch):
    fake_provider(monkeypatch, completion(usage=None))
    result = LiveChatClient(config(), tmp_path, allow_network=True).complete(
        messages(), trace_id="x", prompt_version="v1"
    )
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_qwen_thinking_flag_is_explicit_and_audited(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, completion())
    client = LiveChatClient(
        config(enable_thinking=False, output_limit_parameter="max_tokens"),
        tmp_path,
        allow_network=True,
    )
    client.complete(messages(), trace_id="thinking-off", prompt_version="v1")
    payload = json.loads(calls[0][0].data)
    assert payload["enable_thinking"] is False
    assert "max_completion_tokens" not in payload
    assert payload["max_tokens"] == 512


def test_non_boolean_thinking_flag_rejected():
    with pytest.raises(ValueError, match="enable_thinking"):
        config(enable_thinking="false")


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls", None])
def test_incomplete_response_is_failure_without_fake_result(tmp_path, monkeypatch, reason):
    calls = fake_provider(
        monkeypatch, completion(choices=[{"finish_reason": reason, "message": {"content": "x"}}])
    )
    client = LiveChatClient(config(), tmp_path, allow_network=True)
    with pytest.raises(APIRequestError) as error:
        client.complete(messages(), trace_id="x", prompt_version="v1")
    assert len(calls) == 1
    event = json.loads(next(tmp_path.iterdir()).read_text(encoding="utf-8"))
    assert event["status"] == "failed"
    assert event["input_tokens"] == 11  # A failed/truncated result can still cost tokens.
    assert error.value.input_tokens == 11
    assert error.value.api_requests == 1
    assert error.value.audit_path is not None


def test_http_failure_is_logged_without_echoing_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("GROWRAG_TEST_ONLY_KEY", "secret-test-value")

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 401, "secret-test-value", {}, None)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    client = LiveChatClient(config(), tmp_path, allow_network=True)
    with pytest.raises(APIRequestError, match="HTTP 401") as error:
        client.complete(messages(), trace_id="x", prompt_version="v1")
    assert "secret-test-value" not in str(error.value)
    text = next(tmp_path.iterdir()).read_text(encoding="utf-8")
    assert "secret-test-value" not in text
    assert json.loads(text)["status"] == "failed"


def test_reused_trace_id_cannot_make_another_request(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, completion())
    for attempt in range(2):
        client = LiveChatClient(config(), tmp_path, allow_network=True)
        if attempt == 0:
            client.complete(messages(), trace_id="same", prompt_version="v1")
        else:
            with pytest.raises(FileExistsError):
                client.complete(messages(), trace_id="same", prompt_version="v1")
    assert len(calls) == 1


@pytest.mark.parametrize("url", ["http://example.com/v1", "https://u:p@x/v1", "https://x/?key=x"])
def test_unsafe_endpoint_rejected(url):
    with pytest.raises(ValueError):
        config(base_url=url)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, 61])
def test_unbounded_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        config(timeout_seconds=timeout)


def test_http_200_error_body_cannot_leak_key_into_audit(tmp_path, monkeypatch):
    fake_provider(monkeypatch, {"error": {"message": "TEST-ONLY-NOT-A-REAL-KEY"}})
    with pytest.raises(APIRequestError):
        LiveChatClient(config(), tmp_path, allow_network=True).complete(
            messages(), trace_id="x", prompt_version="v1"
        )
    text = next(tmp_path.iterdir()).read_text(encoding="utf-8")
    assert "TEST-ONLY-NOT-A-REAL-KEY" not in text
    assert "[REDACTED_API_KEY]" in text
    assert json.loads(text)["response_sha256"]


def test_accidental_key_in_prompt_is_blocked_before_network(tmp_path, monkeypatch):
    calls = fake_provider(monkeypatch, completion())
    with pytest.raises(APIRequestError, match="credential detected"):
        LiveChatClient(config(), tmp_path, allow_network=True).complete(
            [{"role": "user", "content": "TEST-ONLY-NOT-A-REAL-KEY"}],
            trace_id="x",
            prompt_version="v1",
        )
    assert calls == []


def test_key_in_response_header_cannot_escape_in_returned_metadata(tmp_path, monkeypatch):
    fake_provider(monkeypatch, completion())
    monkeypatch.setattr(Response, "headers", {"x-request-id": "TEST-ONLY-NOT-A-REAL-KEY"})
    result = LiveChatClient(config(), tmp_path, allow_network=True).complete(
        messages(), trace_id="header-test", prompt_version="v1"
    )
    assert result.request_id == "[REDACTED_API_KEY]"
    assert "TEST-ONLY-NOT-A-REAL-KEY" not in result.audit_path.read_text(encoding="utf-8")
