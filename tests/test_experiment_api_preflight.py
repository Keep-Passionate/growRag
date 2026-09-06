"""Pure local tests; never open the user's credential file or call a provider."""

import pytest

from growrag.experiments.api_preflight import read_local_bailian_settings

TEST_KEY = "sk-ONLY_TEST_NOT_REAL_123456"


def config_file(tmp_path, key=TEST_KEY, url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
    path = tmp_path / "test_config.md"
    path.write_text(f"api_key={key}\nbase_url={url}", encoding="utf-8")
    return path


def test_settings_preserve_endpoint_without_printable_key(tmp_path):
    url = "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    settings = read_local_bailian_settings(config_file(tmp_path, url=url))
    assert settings.base_url == url
    assert settings.api_key == TEST_KEY
    assert TEST_KEY not in repr(settings)


@pytest.mark.parametrize(
    "url",
    [
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "https://coding.dashscope.aliyuncs.com/v1",
        "https://dashscope.aliyuncs.com.attacker.invalid/compatible-mode/v1",
        "https://user:password@dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1?token=hidden",
    ],
)
def test_nonstandard_endpoint_rejected(tmp_path, url):
    with pytest.raises(ValueError):
        read_local_bailian_settings(config_file(tmp_path, url=url))


def test_plan_credential_rejected(tmp_path):
    with pytest.raises(ValueError, match="subscription"):
        read_local_bailian_settings(config_file(tmp_path, key="sk-sp-TEST_NOT_REAL_123456"))


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        read_local_bailian_settings(path)
