"""运行结果存档的完整性。

⛔ 一份没有版本 / 抽样 / 成本的结果，读者没法判断它能不能信，
也没法复现——那样的存档比没有更糟，它看起来像证据。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RUNS = Path(__file__).resolve().parents[1] / "docs" / "runs"
ARCHIVES = sorted(p for p in RUNS.glob("*.md") if p.name != "README.md")


def test_there_is_at_least_one_archive() -> None:
    assert ARCHIVES, "⛔ 跑过就该留档"


@pytest.mark.parametrize("doc", ARCHIVES, ids=lambda p: p.stem)
def test_archive_records_what_makes_it_reproducible(doc: Path) -> None:
    """⛔ 缺任何一项，这份结果就不可复现。"""
    body = doc.read_text(encoding="utf-8")
    for needed, why in [
        ("seed", "⛔ 没种子，抽样不可复现"),
        ("抽样", "⛔ 抽样方式变了分数就不可比"),
        ("区间", "⛔ 抽样分必须带置信区间"),
        ("成本", "⛔ 「又快又好」才是好"),
        ("地板", "⛔ 绝对分不单独读"),
    ]:
        assert needed in body, f"{doc.name} 缺「{needed}」：{why}"


@pytest.mark.parametrize("doc", ARCHIVES, ids=lambda p: p.stem)
def test_archive_pins_the_versions_it_ran(doc: Path) -> None:
    """⚠️ 换一个被测系统的版本等于换了被测对象。"""
    body = doc.read_text(encoding="utf-8")
    # 至少要有一个 40 位 commit sha 或一个精确版本号
    assert (re.search(r"\b[0-9a-f]{40}\b", body)
            or re.search(r"\b\d+\.\d+\.\d+\b", body)), \
        f"{doc.name} 没钉任何版本"


@pytest.mark.parametrize("doc", ARCHIVES, ids=lambda p: p.stem)
def test_archive_says_what_cannot_be_concluded(doc: Path) -> None:
    """⭐ 最要紧的一节：**这份结果得不出什么**。

    ⛔ 一份只说结论不说边界的存档会被当成评测结果引用。
    """
    body = doc.read_text(encoding="utf-8")
    assert "不能从这份结果得出的" in body or "得不出" in body, (
        f"{doc.name} 没写清边界——⛔ 它会被当成评测结果引用")


@pytest.mark.parametrize("doc", ARCHIVES, ids=lambda p: p.stem)
def test_archive_is_listed_in_the_index(doc: Path) -> None:
    index = (RUNS / "README.md").read_text(encoding="utf-8")
    assert doc.name in index, f"{doc.name} 没进索引"


def test_index_demands_the_right_fields() -> None:
    """⚠️ 索引本身要写清「一份存档必须带什么」。"""
    index = (RUNS / "README.md").read_text(encoding="utf-8")
    for field in ("版本", "抽样", "种子", "区间", "成本", "地板"):
        assert field in index, f"索引没要求「{field}」"


def test_index_says_archives_are_not_publications() -> None:
    """⛔ 存档不是发布——题量小、系统少、backbone 单一。"""
    index = (RUNS / "README.md").read_text(encoding="utf-8")
    assert "不是评测结果" in index or "存档不是发布" in index
