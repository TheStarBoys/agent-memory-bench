"""端到端：五阶段 · 世界守卫 · 四态纪律 · 地板线。

⛔ 四态：有分 / 不支持 / 不适用 / 跑挂了，⭐ 外加**框架自己的错**——
它与「跑挂了」是两件事（core/fault.py）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from amb.core import Document
from amb.report import ArmResult, Report, best_floor, render
from amb.runner import Plan, WorldTampered, build, run_one
from amb.world import digest

import worlds.toy as toy

OFFLINE = ("null", "host_default", "bm25", "full_context")


def plan() -> Plan:
    return Plan(manifest=toy.MANIFEST, documents=toy.all_documents(),
                changes=toy.CHANGES, suites_for=toy.suites)


@pytest.mark.parametrize("arm", OFFLINE)
def test_five_phases_complete(arm: str, tmp_path: Path) -> None:
    result, world_digest = run_one(arm, build(arm), plan(), tmp_path / arm,
                                   is_control=True)
    assert world_digest.startswith("sha256:")
    # ⚠️ 套件加多了不该让这条测试失效——它验的是「跑完了」，不是「跑了哪几个」
    assert {"retrieval", "n2_provenance", "n1_prompted", "n1_spontaneous", "qa"} \
        <= set(result.scores)
    assert result.cost, "⚠️ 墙钟必须记账（原则⑥）"


@pytest.mark.parametrize("arm", OFFLINE)
def test_unsupported_is_not_zero(arm: str, tmp_path: Path) -> None:
    """⛔ 没声明的能力记不支持：不计分母，不记 0，不产生任何指标。"""
    from amb.core import Capability

    adapter = build(arm)
    declares_reality = Capability.REALITY in adapter.capabilities()
    result, _ = run_one(arm, adapter, plan(), tmp_path / arm, is_control=True)
    n1 = result.scores["n1_prompted"]
    if declares_reality:
        assert n1.status == "scored"
        return
    assert n1.status == "unsupported"
    assert n1.metrics == {}, "不支持不该产生任何指标——⛔ 0 也是指标"
    assert n1.denominator == 0, "⛔ 不支持不进分母"


def test_no_span_means_unsupported_not_wrong(tmp_path: Path) -> None:
    """⛔ 给不出区间是诚实的能力缺失，与「给错」必须分开。"""
    silent, _ = run_one("null", build("null"), plan(), tmp_path / "n", is_control=True)
    speaks, _ = run_one("bm25", build("bm25"), plan(), tmp_path / "b", is_control=True)
    assert silent.scores["n2_provenance"].status == "unsupported"
    assert speaks.scores["n2_provenance"].status == "scored"
    # 沉默的那个不该出现在任何分数列里
    assert "精确匹配率" not in silent.scores["n2_provenance"].metrics


def test_world_guard_catches_tampering(tmp_path: Path) -> None:
    """⛔ 被测系统改了世界 → 本次跑作废，不是扣分。"""

    class Vandal:
        """一个在摄入时偷偷改世界的适配器。"""

        def capabilities(self):
            from amb.core import BASELINE
            return set(BASELINE)

        def setup(self, world) -> None:
            self._root = Path(world.root)

        def reset(self) -> None: ...
        def close(self) -> None: ...

        def ingest(self, doc) -> None:
            target = self._root / "notes" / "cat.md"
            if target.exists():
                os.chmod(target, 0o644)
                target.write_text("被篡改", encoding="utf-8")

        def finalize(self) -> None: ...
        def search(self, query, k, *, principal=None): return []
        def count(self) -> int: return 0

    with pytest.raises(WorldTampered, match="ingest"):
        run_one("vandal", Vandal(), plan(), tmp_path / "v", is_control=False)


def test_world_is_reproducible(tmp_path: Path) -> None:
    """⛔ 同一份清单 + 同一个种子 → 同一个哈希，含 mtime 钉死。"""
    from amb.world import materialize

    a = materialize(toy.MANIFEST, tmp_path / "a")
    b = materialize(toy.MANIFEST, tmp_path / "b")
    facts = dict(toy.MANIFEST.facts)
    assert digest(a, toy.CLOCK_START, facts) == digest(b, toy.CLOCK_START, facts)
    stamps = {p.stat().st_mtime for p in a.rglob("*") if p.is_file()}
    assert len(stamps) == 1, "⛔ mtime 必须全部钉死在时钟起点"


def test_floor_picks_the_strongest_control(tmp_path: Path) -> None:
    """⛔ 地板取对照组里最强的，不是最弱的——挑弱的是在抬高自己。"""
    arms = []
    for name in OFFLINE:
        r, _ = run_one(name, build(name), plan(), tmp_path / name, is_control=True)
        arms.append(r)
    floor = best_floor(arms, "retrieval", "top1")
    assert floor is not None
    best = max(a.scores["retrieval"].metrics["top1"] for a in arms)
    assert floor.value == best


def test_report_shows_unsupported_as_dash_not_zero(tmp_path: Path) -> None:
    r, d = run_one("null", build("null"), plan(), tmp_path / "n", is_control=True)
    report = Report(run_id="t", at="t", world={"name": "toy", "seed": 42, "digest": d},
                    backbone={"model": "—"}, lanes={"library": [r]})
    text = render(report)
    n1_row = next(li for li in text.splitlines()
                  if li.startswith("| null") and "unsupported" in li)
    assert "0.000" not in n1_row, "⛔ 不支持在表里是 —，不是 0"


def test_memory_is_required_to_detect_modification(tmp_path: Path) -> None:
    """⭐ 没有记忆，就检测不了记忆的腐化。

    重读世界告诉你「现在是什么」，不告诉你「你记的东西过期了没有」。
    host_default 判得出「消失」，判不出「改值」——它诚实地报 unknown，
    ⛔ 而不是猜成 holds。这条固化实测发现，防止有人「优化」掉那份诚实。
    """
    with_memory, _ = run_one("bm25", build("bm25"), plan(), tmp_path / "b",
                             is_control=True)
    without, _ = run_one("host_default", build("host_default"), plan(),
                         tmp_path / "h", is_control=True)

    m = with_memory.scores["n1_prompted"].metrics
    n = without.scores["n1_prompted"].metrics

    assert m["检出率"] > n["检出率"], "有快照的应当检得更全"
    assert n["broken→unknown"] > 0, "无记忆的应当在改值上弃权"
    assert n["broken→holds"] == 0, "⛔ 判不了就报 unknown，不许猜成 holds"
    # 两边都不许误报——把什么都标 broken 就能刷检出率
    assert m["误报率"] == 0.0 and n["误报率"] == 0.0


def test_two_modes_are_reported_separately(tmp_path: Path) -> None:
    """⛔ 有提示与无提示分开报，永不合并成一个 N1 分数。"""
    r, _ = run_one("bm25", build("bm25"), plan(), tmp_path / "b", is_control=True)
    assert "n1_prompted" in r.scores and "n1_spontaneous" in r.scores
    assert "n1_reality" not in r.scores, "⛔ 不许有一个合并后的 N1 分数"


def test_spontaneous_mode_distinguishes_a_prompt_only_system(tmp_path: Path) -> None:
    """⭐ 两种模式存在的唯一理由：区分「被问了才查」与「自己就发现」。

    造一个只在 audit() 里认真、search() 从不表态的适配器。
    它应当在有提示上拿满分，在无提示上垫底——⛔ 如果两种模式给出同一个数，
    这一档就白设了。
    """
    from amb.adapters.impl.bm25 import BM25Adapter
    from amb.core import Entry

    class PromptOnly(BM25Adapter):
        """被问了才查；平时检索绝口不提自己可能过期了。"""

        def search(self, query: str, k: int, *, principal: str | None = None):
            hits = super().search(query, k, principal=principal)
            for h in hits:
                h.state = None          # ⛔ 不表态
            return hits

    r, _ = run_one("prompt_only", PromptOnly(), plan(), tmp_path / "p",
                   is_control=False)
    prompted = r.scores["n1_prompted"]
    spontaneous = r.scores["n1_spontaneous"]

    assert prompted.metrics["检出率"] == 1.0, "被问了它是查得出来的"
    assert spontaneous.metrics["检出率"] == 0.0, "没人问它就不吭声"
    assert spontaneous.metrics["弃权率"] == 1.0
    # ⭐ 这个差就是「主动」这一维的量度
    assert prompted.metrics["检出率"] > spontaneous.metrics["检出率"]


def test_cannot_reconcile_is_unsupported_not_zero(tmp_path: Path) -> None:
    """⛔ 对不上账 → 不支持，不是 0 分。

    host_default 声明了 REALITY，但 search 不返回条目，
    评测器无从把条目对回被破坏的事实——那不是它答错了。
    """
    r, _ = run_one("host_default", build("host_default"), plan(),
                   tmp_path / "h", is_control=True)
    sp = r.scores["n1_spontaneous"]
    assert sp.status == "unsupported"
    assert "无从对账" in (sp.reason or "")
    assert sp.metrics == {}, "⛔ 不支持不产生任何指标，0 也是指标"


# ── answer 档：用假 backbone，⛔ 测试不依赖网络 ──────────────────
class _FakeLLM:
    """按题面给定答案。⚠️ 只用于验判分逻辑，不验模型质量。"""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        from amb.adapters.llm import Meter

        self.meter = Meter()

    def complete(self, system: str, user: str) -> str:
        self.meter.add({"prompt_tokens": 10, "completion_tokens": 3})
        return self._reply


def _arm_with_fake_llm(name: str, reply: str):
    arm = build(name)
    arm._llm = _FakeLLM(reply)   # noqa: SLF001 —— 测试替身
    return arm


def test_qa_is_unsupported_without_a_backbone(tmp_path: Path) -> None:
    """⛔ 没配 backbone = 压根没这能力，不是这次没做成。"""
    r, _ = run_one("bm25", build("bm25"), plan(), tmp_path / "b", is_control=True)
    qa = r.scores["qa"]
    assert qa.status == "unsupported"
    assert qa.metrics == {}


def test_abstention_is_not_counted_as_wrong(tmp_path: Path) -> None:
    """⭐ 拒答不是加分项，也不该被记成答错——单列。"""
    arm = _arm_with_fake_llm("bm25", "资料未提及")
    r, _ = run_one("always_abstain", arm, plan(), tmp_path / "a", is_control=False)
    m = r.scores["qa"].metrics
    assert m["正确弃权率"] == 1.0, "该弃权的题它弃权了"
    assert m["编造率"] == 0.0
    assert m["该答却弃权"] == 1.0, "该答的题也弃权了——单列，⛔ 不混进准确率"
    assert m["准确率"] == 0.0


def test_fabrication_is_reported_alongside_accuracy(tmp_path: Path) -> None:
    """⛔ 只报准确率的话，见题就编的系统会比诚实弃权的好看。"""
    liar = _arm_with_fake_llm("bm25", "新皮层")   # 什么都答「新皮层」
    r, _ = run_one("liar", liar, plan(), tmp_path / "l", is_control=False)
    m = r.scores["qa"].metrics
    assert m["准确率"] > 0.0, "蒙对了一部分"
    assert m["编造率"] == 1.0, "⭐ 该弃权的题它编了——必须与准确率同屏"


def test_answer_reports_token_usage(tmp_path: Path) -> None:
    """原则⑥：token 只有适配器报得出来。"""
    from amb.core import Usage

    arm = _arm_with_fake_llm("bm25", "新皮层")
    run_one("u", arm, plan(), tmp_path / "u", is_control=False)
    usage = arm.usage()
    assert isinstance(usage, list) and isinstance(usage[0], Usage)
    assert usage[0].tokens_in > 0 and usage[0].llm_calls > 0
