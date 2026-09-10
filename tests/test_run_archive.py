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


# ── ⭐ 报告是给人读的 ───────────────────────────────────────────
def _sample_report():
    """一份带齐三种结构的报告：⚠️ 分档曲线 / 分层 / 普通指标。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from amb.core import Observation, SuiteRun
    from amb.report import ArmResult, Report
    from amb.scoring import score

    rep = Report(run_id="x", at="now",
                 world={"name": "toy", "seed": 42, "documents": 8,
                        "digest": "sha256:a", "corpus": "c"},
                 backbone={"model": "fake", "thinking": False})
    for arm in ("null", "bm25", "mem0_raw"):
        a = ArmResult(arm=arm, is_control=arm != "mem0_raw", declared=["search"])
        run = SuiteRun("n6_structure", "scored")
        for i in range(112):
            run.observations.append(Observation(f"i{i}", {
                "fan": 2 ** (i % 7), "reached": i % 4, "cues": 3,
                "precise": i % 3 == 0}))
        a.scores["n6_structure"] = score(run)
        a.cost_profile = {"canary": {}, "items_ingested": 8, "items_probed": 112}
        a.participation = {"declared": 1, "total_caps": 11, "items": 112}
        rep.lanes.setdefault("library", []).append(a)
    return rep


def test_no_line_in_the_report_is_a_wall_of_text() -> None:
    """⛔ 明细行早先把 18 个指标堆成**一行**——⚠️ 实测最长 595 个字符。

    ⭐ 而 `n6` 那 18 个值本质是一条曲线上的 7 个点：⛔ 横着排读不出趋势。
    ⚠️ 这条检查此前**不存在**，所以行长从来没人管。
    """
    from amb.report import render

    long = [ln for ln in render(_sample_report()).splitlines() if len(ln) > 220]
    assert not long, ("⛔ 这些行太长，读者要横向滚动：\n  "
                      + "\n  ".join(f"{len(ln)} 字符: {ln[:70]}…" for ln in long[:3]))


def test_a_fan_curve_is_rendered_as_a_table() -> None:
    """⭐ 扇形退化是**趋势**：⛔ 竖着排才看得出来。"""
    from amb.report import render

    text = render(_sample_report())
    assert "随扇形度的变化" in text, "⛔ 分档没有单独成表"
    # ⚠️ 表头是扇形度，⭐ 每档一行
    assert "| 扇形度 |" in text
    for lv in (1, 2, 4, 8, 16, 32, 64):
        assert f"| {lv} |" in text, f"⛔ 缺 fan{lv} 那一行"
