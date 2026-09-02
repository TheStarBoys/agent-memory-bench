"""公开题库的接入机制。

⛔ 这一层的纪律只有一条：**用它们的判分代码，不自己重写。**
测试盯着这条纪律，而不是盯着某个具体题库的分数。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amb.suites.public import (
    DatasetMissing,
    Pin,
    PublicSuite,
    REGISTRY,
    pin_for,
)


def test_missing_data_is_not_a_zero(tmp_path: Path) -> None:
    """⛔ 数据没取下来 = 未接入，不是 0 分。"""
    suite = PublicSuite(name="x", pin=Pin("repo", "abc"), data_dir=tmp_path / "nope")
    assert not suite.available()
    with pytest.raises(DatasetMissing, match="不是 0 分"):
        suite.require_available()


def test_available_once_data_is_there(tmp_path: Path) -> None:
    (tmp_path / "d.json").write_text("{}", encoding="utf-8")
    assert PublicSuite(name="x", pin=Pin("r", "c"), data_dir=tmp_path).available()


def test_registry_lookup_is_exact() -> None:
    """⛔ 精确查找，与适配器注册表同一条纪律。"""
    with pytest.raises(KeyError):
        pin_for("loco")           # locomo 的前缀


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_upstream_records_its_known_caveats(name: str) -> None:
    """⚠️ 上游判分有缺陷时**照旧调用**，但缺陷要进报告。

    ⛔ 不静默修好——悄悄修了，我们的数就和所有引用上游成绩的论文对不上，
    而读者不知道差在哪。
    """
    pin = pin_for(name)
    assert pin.caveats, f"{name} 一条 caveat 都没有？那多半是漏写了"
    assert pin.provenance()["caveats"] == list(pin.caveats)


def test_memorydata_records_the_license_and_conflict_of_interest() -> None:
    """⚠️ 两条最要紧的：无 LICENSE，以及 memorybench 的利益关系。"""
    joined = " ".join(pin_for("memorydata").caveats)
    assert "无 LICENSE" in joined
    assert "利益关系" in joined and "supermemory" in joined


def test_beam_records_that_it_does_not_report_cost() -> None:
    """⚠️ 上一轮核实推翻的那条说法，⛔ 别让它再漂回来。"""
    joined = " ".join(pin_for("beam").caveats)
    assert "不报成本" in joined
    assert "评委" in joined, "它的判分带评委漂移，不可与确定性分数并列"


def test_commits_are_pinned_or_explicitly_empty() -> None:
    """⛔ 钉死 commit。⚠️ 现在都是空的——接入时必须填。"""
    unpinned = sorted(n for n, p in REGISTRY.items() if not p.commit)
    assert unpinned == sorted(REGISTRY), (
        "有的填了有的没填？⛔ 要么全都钉死，要么这条测试跟着改"
    )
