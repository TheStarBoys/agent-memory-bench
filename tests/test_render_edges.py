"""报告渲染的退化分支——⛔ 错了不炸，⚠️ 只是**印错**。

## ⭐ 为什么这一块要守

⚠️ 报告是所有努力的出口。⛔ 地板选错、外部依赖漏印、缓存命中不标——
⭐ 都不会报错，只会让读者据此下一个错的结论。

⚠️ 这些分支每一条都对着一次**真实的误报**（见各条 docstring）。
"""

from __future__ import annotations

import pytest

from amb.core import Observation, SuiteRun
from amb.report import ArmResult, Report, render
from amb.scoring import score


def _report(**kw) -> Report:
    base = dict(
        run_id="x", at="now",
        world={"name": "toy", "seed": 42, "documents": 8,
               "digest": "sha256:abcdef0123456789", "corpus": "c1"},
        backbone={"model": "m", "thinking": False},
    )
    base.update(kw)
    return Report(**base)


def _arm(name: str, *, control: bool, top1: float, n: int = 40,
         suite: str = "retrieval") -> ArmResult:
    a = ArmResult(arm=name, is_control=control, declared=["search"])
    run = SuiteRun(suite, "scored")
    hit = round(top1 * n)
    for i in range(n):
        run.observations.append(Observation(f"q{i}", {
            "gold": ["d"], "top1": "d" if i < hit else "x",
            "retrieved": ["d"]}))
    a.scores[suite] = score(run)
    a.cost = {"setup": 1, "ingest": 100, "probe": 1000}
    a.cost_profile = {"canary": {}, "items_ingested": 8, "items_probed": n}
    a.participation = {"declared": 1, "total_caps": 11, "items": n}
    return a


# ── ⭐ 报告头：⛔ 漏印的每一项都对着一次误读 ────────────────────
def test_an_external_that_failed_to_install_is_still_printed() -> None:
    """⛔ **装失败的也要印**：⚠️ 早先 `if row.get("ok")` 静默丢掉它们，
    ⭐ 于是读者看到的是一份**干净的钉版清单**，而实际有依赖没装上。
    """
    rep = _report(externals={
        "mem0": {"actual": "2.0.19", "ok": True},
        "a_mem": {"actual": "", "ok": False},
    })
    rep.lanes["library"] = [_arm("bm25", control=True, top1=0.5)]
    text = render(rep)
    assert "mem0@2.0.19" in text
    assert "a_mem" in text and "未装上" in text, "⛔ 装失败的被吞了"


def test_a_cache_hit_run_is_flagged_loudly() -> None:
    """⛔ 命中率高的跑测出来的「延迟」**不是真延迟**——⚠️ 不标出来，
    「它变快了」会被读成系统变快了。
    """
    rep = _report(cache={"hits": 90, "misses": 10})
    rep.lanes["library"] = [_arm("bm25", control=True, top1=0.5)]
    text = render(rep)
    assert "缓存命中 90/100" in text and "不是独立测量" in text


def test_a_cache_that_never_engaged_says_why() -> None:
    """⚠️ 命中率为 0 有很多种原因——⛔ 不说是哪一种就查不出来。"""
    rep = _report(cache={"hits": 0, "skipped": {"采样温度": 3},
                         "diagnosis": "被测系统在用采样温度"})
    rep.lanes["library"] = [_arm("bm25", control=True, top1=0.5)]
    assert "采样温度" in render(rep)


def test_thinking_on_is_marked_because_it_changes_cost() -> None:
    """⚠️ 思考开关直接改变成本与输出长度——⛔ 不能只躺在 JSON 里。"""
    rep = _report(backbone={"model": "m", "thinking": True})
    rep.lanes["library"] = [_arm("bm25", control=True, top1=0.5)]
    assert "思考开" in render(rep)


def test_the_sampling_seed_is_in_the_header() -> None:
    """⛔ 抽样方式变了分数就不可比——⚠️ 种子也要在，⭐ 否则不可复现。"""
    rep = _report(sampling={"strategy": "stratified", "sampled": 50,
                            "total": 1973, "seed": 7})
    rep.lanes["library"] = [_arm("bm25", control=True, top1=0.5)]
    text = render(rep)
    assert "n=50/1973" in text and "seed=7" in text


def test_the_agent_host_version_is_pinned_in_the_header() -> None:
    """⚠️ 换宿主版本等于换尺子——⛔ 报告里必须写着。"""
    rep = _report(host={"version": "0.1.2a3"})
    rep.lanes["library"] = [_arm("bm25", control=True, top1=0.5)]
    assert "0.1.2a3" in render(rep)


# ── ⛔ 成本 × 质量：地板选错是那次「四条真臂全没救」的成因 ──────
def test_no_cost_table_without_a_qualified_control() -> None:
    """⛔ **没有够格的对照组就不出这张表**。

    ⚠️ 早先兜底 `max(quality, …)` 在剔除退化臂**之前**取值、
    也不要求是对照组——⭐ 于是没有对照时把**被测系统**当地板，
    ⛔ 正是那次「四条真臂全部没有存在理由」的成因。
    """
    rep = _report()
    rep.lanes["library"] = [_arm("mem0", control=False, top1=0.9)]
    text = render(rep)
    assert "不出这张表" in text and "没有够格的对照组" in text


def test_a_degenerate_arm_is_never_the_floor() -> None:
    """⛔ `full_context` 在检索档里**不检索**——⚠️ recall 恒为 1.000
    且总耗时约 1ms。⭐ 拿它当分母，真实臂的耗时比会变成天文数字
    （实测 938102x）并被判「没有存在理由」——⛔ 两个结论都是错的。
    """
    rep = _report()
    rep.lanes["library"] = [
        _arm("full_context", control=True, top1=1.0),
        _arm("mem0", control=False, top1=0.8),
    ]
    text = render(rep)
    assert "不出这张表" in text, (
        "⛔ 退化臂当了地板——⚠️ 那张表里的倍数会是天文数字")


def test_the_cost_table_appears_with_a_real_control() -> None:
    """⭐ 反向：⛔ 永远不出表的闸门等于没有表。"""
    rep = _report()
    rep.lanes["library"] = [
        _arm("bm25", control=True, top1=0.3),
        _arm("mem0", control=False, top1=0.8),
    ]
    text = render(rep)
    assert "成本 × 质量" in text and "不出这张表" not in text


# ── ⛔ 缺席与未跑：⚠️ 静默消失会被读成「没参赛」 ────────────────
def test_an_arm_that_never_ran_is_shown_as_such() -> None:
    """⛔ 一条臂没跑要**印出来**——⚠️ 行整个消失的话，
    读者会以为它压根没报名。
    """
    rep = _report()
    good = _arm("bm25", control=True, top1=0.5)
    absent = ArmResult(arm="mem0", is_control=False, declared=["search"])
    absent.cost_profile = {"canary": {}}
    rep.lanes["library"] = [good, absent]
    text = render(rep)
    assert "mem0" in text, "⛔ 没跑的臂整行消失了"


def test_a_crashed_arm_is_not_confused_with_a_zero_score() -> None:
    """⛔ 「跑挂了」与「得 0 分」是两回事——⚠️ 混起来的话，
    一次网络抖动会被读成这个系统很差。
    """
    rep = _report()
    dead = ArmResult(arm="mem0", is_control=False, declared=[],
                     crashed="TimeoutError: timed out")
    dead.cost_profile = {"canary": {}}
    rep.lanes["library"] = [_arm("bm25", control=True, top1=0.5), dead]
    text = render(rep)
    assert "TimeoutError" in text or "crashed" in text.lower()
