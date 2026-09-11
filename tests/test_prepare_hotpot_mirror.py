"""Mirror helper contracts with fake byte streams; no pyarrow or network needed."""

import builtins
import copy
import hashlib
import importlib.util
import io
import json
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture
def mirror(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            pytest.fail("importing the mirror helper must not load pyarrow")
        return original_import(name, *args, **kwargs)

    def no_network(*args, **kwargs):
        pytest.fail("tests must not perform a real download")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_hotpot_mirror.py"
    spec = importlib.util.spec_from_file_location("growrag_test_mirror_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def row():
    return {
        "id": "synthetic-row-1",
        "question": "Which synthetic records refer to the same location?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "context": {
            "title": ["Second appearing title", "First alphabetic title"],
            "sentences": [["sentence 0", "", "sentence 2", " "], ["", "nonempty"]],
        },
        "supporting_facts": {
            "title": ["First alphabetic title", "Second appearing title"],
            "sent_id": [1, 2],
        },
    }


def test_import_is_side_effect_free_without_optional_dependencies(mirror):
    assert callable(mirror.official_shape)
    assert callable(mirror.download)
    assert mirror.EXPECTED_ROWS == 90447


def test_official_shape_maps_fields_without_losing_order_or_empty_sentences(mirror, row):
    original = copy.deepcopy(row)
    converted = mirror.official_shape(row)
    assert converted == {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": "comparison",
        "level": "hard",
        "context": [
            ["Second appearing title", ["sentence 0", "", "sentence 2", " "]],
            ["First alphabetic title", ["", "nonempty"]],
        ],
        "supporting_facts": [["First alphabetic title", 1], ["Second appearing title", 2]],
    }
    assert "id" not in converted
    assert row == original
    assert converted["context"][0][1][2] == "sentence 2"
    assert converted["context"][1][1][1] == "nonempty"


@pytest.mark.parametrize(
    "field,component",
    [
        ("context", "title"),
        ("context", "sentences"),
        ("supporting_facts", "title"),
        ("supporting_facts", "sent_id"),
    ],
)
def test_official_shape_rejects_parallel_list_length_mismatch(mirror, row, field, component):
    row[field][component].pop()
    with pytest.raises(ValueError):
        mirror.official_shape(row)


class ByteResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"ETag": '"synthetic-etag"'}

    def geturl(self):
        return "https://public-cdn.example/data/shard.parquet?signature=DO_NOT_SAVE#private"


def test_fake_download_preserves_bytes_hash_and_removes_redirect_query(
    mirror, tmp_path, monkeypatch
):
    payload = b"PAR1" + bytes(range(256)) * 5000 + "synthetic Unicode \u4e2d\u6587".encode()
    response = ByteResponse(payload)
    called = []

    def fake_open(url, *, timeout):
        called.append((url, timeout))
        return response

    monkeypatch.setattr(mirror.urllib.request, "urlopen", fake_open)
    public_url = "https://public.example/train/0.parquet"
    destination = tmp_path / "train-0.parquet"
    result = mirror.download(public_url, destination)
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".part").exists()
    assert response.closed
    assert called == [(public_url, 60)]
    assert result == {
        "url": public_url,
        "resolved_url_without_query": "https://public-cdn.example/data/shard.parquet",
        "etag": '"synthetic-etag"',
        "file": destination.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert "DO_NOT_SAVE" not in json.dumps(result)
    assert "private" not in json.dumps(result)
    assert mirror.digest_file(destination) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("existing", ["destination", "partial"])
def test_download_refuses_existing_artifacts_before_opening_url(mirror, tmp_path, existing):
    destination = tmp_path / "train-0.parquet"
    protected = destination if existing == "destination" else destination.with_suffix(".part")
    protected.write_bytes(b"user-owned previously saved bytes")
    with pytest.raises(FileExistsError):
        mirror.download("https://public.example/train/0.parquet", destination)
    assert protected.read_bytes() == b"user-owned previously saved bytes"
    if existing == "destination":
        assert not destination.with_suffix(".part").exists()
    else:
        assert not destination.exists()


def test_interrupted_fake_download_retains_partial_and_cannot_implicitly_retry(
    mirror, tmp_path, monkeypatch
):
    class InterruptedResponse(ByteResponse):
        def __init__(self):
            super().__init__(b"partial bytes")
            self.read_calls = 0

        def read(self, size=-1):
            self.read_calls += 1
            if self.read_calls > 1:
                raise OSError("synthetic disconnected byte stream")
            return super().read(size)

    response = InterruptedResponse()
    opens = []

    def fake_open(url, *, timeout):
        opens.append(url)
        return response

    monkeypatch.setattr(mirror.urllib.request, "urlopen", fake_open)
    destination = tmp_path / "train-0.parquet"
    with pytest.raises(OSError, match="synthetic"):
        mirror.download("https://public.example/train/0.parquet", destination)
    assert not destination.exists()
    assert destination.with_suffix(".part").read_bytes() == b"partial bytes"
    assert response.closed
    with pytest.raises(FileExistsError):
        mirror.download("https://public.example/train/0.parquet", destination)
    assert len(opens) == 1
