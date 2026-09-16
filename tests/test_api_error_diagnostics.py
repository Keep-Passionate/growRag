"""HTTP diagnostics use mocked failures only; no credentials or network calls."""

import io
import json
import urllib.error

import pytest

from growrag.experiments.api_client import APIRequestError, ChatConfig, LiveChatClient

SAFE_ID = "12345678-1234-1234-1234-123456789abc"
SECRET = "diagnostics-test-only-not-a-real-key"


class ObservedBody(io.BytesIO):
    def __init__(self, raw):
        super().__init__(raw)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


def failed_request(tmp_path, monkeypatch, payload, *, headers=None, secret=SECRET):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    body = ObservedBody(raw)
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "provider exception message with " + secret,
                headers or {},
                body,
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    monkeypatch.setenv("GROWRAG_DIAGNOSTIC_TEST_KEY", secret)
    config = ChatConfig(
        base_url="https://api.example.invalid/v1",
        model="mock-model",
        key_environment_variable="GROWRAG_DIAGNOSTIC_TEST_KEY",
        max_calls=3,
    )
    client = LiveChatClient(config, tmp_path, allow_network=True)
    with pytest.raises(APIRequestError, match="API HTTP 400; no retry or fallback") as failure:
        client.complete(
            [{"role": "user", "content": "mock query"}],
            trace_id="mock-error",
            prompt_version="mock-v1",
        )
    event_text = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert len(calls) == 1
    assert client.attempts == 1
    assert body.read_sizes == [16_385]
    assert body.closed
    assert secret not in event_text
    assert secret not in str(failure.value)
    return json.loads(event_text), failure.value, event_text


def test_allowlisted_code_and_validated_header_id_are_the_only_error_metadata(
    tmp_path, monkeypatch
):
    payload = {
        "error": {
            "code": "InvalidParameter",
            "message": SECRET + " C:\\private\\error-data.txt\nECHOED_USER_CONTENT",
            "param": SECRET,
            "type": "UNTRUSTED_TYPE",
        },
        "usage": {"prompt_tokens": 500, "completion_tokens": 20},
        "request_id": "req-abcdef1234567890",
    }
    event, failure, text = failed_request(
        tmp_path, monkeypatch, payload, headers={"x-request-id": SAFE_ID, "secret": SECRET}
    )
    assert event["schema_version"] == 1
    assert event["status"] == "failed"
    assert event["http_status"] == failure.http_status == 400
    assert event["provider_error_code"] == failure.provider_error_code == "InvalidParameter"
    assert event["request_id"] == failure.request_id == SAFE_ID
    assert event["retry_count"] == 0
    assert event["api_requests"] == failure.api_requests == 1
    assert event["input_tokens"] is failure.input_tokens is None
    assert event["output_tokens"] is failure.output_tokens is None
    assert "response" not in event
    assert "response_sha256" not in event
    for forbidden in ("private", "ECHOED_USER_CONTENT", "UNTRUSTED_TYPE", "param", "headers"):
        assert forbidden not in text


@pytest.mark.parametrize(
    "code",
    [
        SECRET,
        "InvalidParameter " + SECRET,
        "C:\\private\\token.txt",
        "InvalidParameter\nECHOED_USER_CONTENT",
        "UNKNOWN_PROVIDER_CODE",
        {"value": "InvalidParameter"},
        ["InvalidParameter"],
        400,
        None,
    ],
)
def test_unknown_or_adversarial_error_codes_are_not_persisted(tmp_path, monkeypatch, code):
    event, failure, text = failed_request(tmp_path, monkeypatch, {"error": {"code": code}})
    assert event["provider_error_code"] is failure.provider_error_code is None
    assert "UNKNOWN_PROVIDER_CODE" not in text
    assert "ECHOED_USER_CONTENT" not in text
    assert "private" not in text


@pytest.mark.parametrize(
    "request_id",
    [
        SECRET,
        "C:\\private\\token.txt",
        SAFE_ID + "\n",
        "\t" + SAFE_ID,
        SAFE_ID + "../",
        "req-" + "a" * 65,
        "arbitrary-user-text",
        "req-" + "１" * 32,
        {"id": SAFE_ID},
        None,
    ],
)
def test_untrusted_header_and_body_request_ids_are_dropped(tmp_path, monkeypatch, request_id):
    event, failure, text = failed_request(
        tmp_path,
        monkeypatch,
        {"code": "DataInspectionFailed", "request_id": request_id},
        headers={"x-request-id": request_id},
    )
    assert event["request_id"] is failure.request_id is None
    assert "arbitrary-user-text" not in text
    assert "private" not in text


@pytest.mark.parametrize("request_id", [SAFE_ID, "req-abcdef1234567890"])
def test_top_level_provider_code_and_safe_body_id_are_supported(tmp_path, monkeypatch, request_id):
    event, failure, _ = failed_request(
        tmp_path, monkeypatch, {"code": "DataInspectionFailed", "request_id": request_id}
    )
    assert event["provider_error_code"] == failure.provider_error_code == "DataInspectionFailed"
    assert event["request_id"] == failure.request_id == request_id


@pytest.mark.parametrize(
    "raw",
    [
        b"<html>server failure with private user data</html>",
        b"not json",
        b"[]",
        b"null",
        b'\xff{"code":"InvalidParameter"}',
        b'{"code":"InvalidParameter","code":"DataInspectionFailed"}',
        b"{" + b'"nested":{' * 1_100 + b'"code":"InvalidParameter"' + b"}" * 1_101,
        b'{"code":"InvalidParameter","message":"' + b"x" * 16_385 + b'"}',
    ],
)
def test_non_json_ambiguous_deep_or_oversized_body_remains_unknown(tmp_path, monkeypatch, raw):
    event, failure, text = failed_request(
        tmp_path, monkeypatch, raw, headers={"x-request-id": SAFE_ID}
    )
    assert event["provider_error_code"] is failure.provider_error_code is None
    assert event["request_id"] == SAFE_ID
    assert "private user data" not in text
    assert len(text) < 3_000


@pytest.mark.parametrize("secret", [SAFE_ID, "InvalidParameter"])
def test_even_safe_shaped_metadata_cannot_equal_configured_credential(
    tmp_path, monkeypatch, secret
):
    event, failure, _ = failed_request(
        tmp_path,
        monkeypatch,
        {"code": "InvalidParameter", "request_id": SAFE_ID},
        headers={"x-request-id": SAFE_ID},
        secret=secret,
    )
    if secret == SAFE_ID:
        assert event["request_id"] is failure.request_id is None
    else:
        assert event["provider_error_code"] is failure.provider_error_code is None


def test_diagnostic_read_failure_does_not_mask_http_error(tmp_path, monkeypatch):
    def broken_read(self, size=-1):
        self.read_sizes.append(size)
        raise OSError("read failed with " + SECRET)

    monkeypatch.setattr(ObservedBody, "read", broken_read)
    event, failure, _ = failed_request(
        tmp_path, monkeypatch, b"ignored", headers={"x-request-id": SAFE_ID}
    )
    assert event["provider_error_code"] is failure.provider_error_code is None
    assert event["request_id"] == SAFE_ID
    assert failure.input_tokens is failure.output_tokens is None
