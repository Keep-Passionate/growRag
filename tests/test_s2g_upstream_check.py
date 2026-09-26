from pathlib import Path

import pytest

from growrag.experiments import run_fresh_continuation as old_runner
from growrag.experiments import s2g_upstream_check as check


@pytest.mark.parametrize("split", ["dev", "validation", "test", "unknown", "train[:10]", ""])
def test_only_verified_train_split_can_feed_memory(split):
    with pytest.raises(ValueError):
        check.check_memory_source(
            official_split=split, candidate_ids=["q1"], allowed_train_ids=["q1"], protected_ids=[]
        )


@pytest.mark.parametrize(
    "ids,allowed,protected",
    [
        ([], [], []),
        (["q1", "q1"], ["q1"], []),
        (["q2"], ["q1"], []),
        (["q1"], ["q1"], ["q1"]),
        ([None], [None], []),
        ([" "], [" "], []),
    ],
)
def test_memory_allowlist_and_protected_overlap_fail_closed(ids, allowed, protected):
    with pytest.raises(ValueError):
        check.check_memory_source(
            official_split="train",
            candidate_ids=ids,
            allowed_train_ids=allowed,
            protected_ids=protected,
        )


def test_disjoint_audited_train_ids_accepted():
    check.check_memory_source(
        official_split="train",
        candidate_ids=["q1"],
        allowed_train_ids=["q1", "q2"],
        protected_ids=["q3"],
    )


def test_changed_author_source_is_not_executed(tmp_path):
    path = tmp_path / "inference/inference_bm25.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('must not execute unknown source')")
    with pytest.raises(ValueError, match="fingerprint"):
        check.load_pinned_utilities(tmp_path)


def test_old_live_plan_disabled_before_any_data_or_api(monkeypatch, tmp_path):
    monkeypatch.setattr(old_runner, "load_fresh_dev", lambda _: pytest.fail("no data access"))
    with pytest.raises(ValueError, match="superseded"):
        old_runner.main(
            [
                "--manifest",
                "unused",
                "--runs-root",
                str(tmp_path),
                "--output",
                str(tmp_path / old_runner.RUN_ID),
                "--allow-network",
            ]
        )


def test_pinned_author_utilities_if_snapshot_is_available():
    upstream = Path("external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6")
    if not upstream.exists():
        pytest.skip("author snapshot is intentionally not redistributed in our repository")
    result = check.audit(upstream)
    assert all(result["pure_function_checks"].values())
    assert result["python_files_syntax_checked"] > 15
    assert result["model_api_requests"] == 0
    assert not result["full_s2g_inference_completed"]


def test_cli_creates_new_parent_and_never_overwrites(monkeypatch, tmp_path):
    result = {"python_files_syntax_checked": 20, "pure_function_checks": {"synthetic": True}}
    monkeypatch.setattr(check, "audit", lambda _: result)
    output = tmp_path / "new" / "audit.json"
    args = ["--upstream", str(tmp_path), "--output", str(output)]
    assert check.main(args) == 0 and output.exists()
    with pytest.raises(FileExistsError):
        check.main(args)
