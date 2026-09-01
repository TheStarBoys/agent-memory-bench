"""端到端：五阶段 · 世界守卫 · 三态纪律 · 地板线。"""

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
    return Plan(manifest=toy.MANIFEST, documents=toy.DOCUMENTS,
                changes=toy.CHANGES, suites=toy.suites())


@pytest.mark.parametrize("arm", OFFLINE)
def test_five_phases_complete(arm: str, tmp_path: Path) -> None:
    result, world_digest = run_one(arm, build(arm), plan(), tmp_path / arm,
                                   is_control=True)
    assert world_digest.startswith("sha256:")
    assert set(result.scores) == {"retrieval", "n2_provenance", "n1_reality"}
    assert result.cost, "⚠️ 墙钟必须记账（原则⑥）"


@pytest.mark.parametrize("arm", OFFLINE)
def test_unsupported_is_not_zero(arm: str, tmp_path: Path) -> None:
    """⛔ 没声明的能力记不支持：不计分母，不记 0，不产生任何指标。"""
    from amb.core import Capability

    adapter = build(arm)
    declares_reality = Capability.REALITY in adapter.capabilities()
    result, _ = run_one(arm, adapter, plan(), tmp_path / arm, is_control=True)
    n1 = result.scores["n1_reality"]
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
                    backbone={"model": "—"}, arms=[r])
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

    m = with_memory.scores["n1_reality"].metrics
    n = without.scores["n1_reality"].metrics

    assert m["检出率"] > n["检出率"], "有快照的应当检得更全"
    assert n["broken→unknown"] > 0, "无记忆的应当在改值上弃权"
    assert n["broken→holds"] == 0, "⛔ 判不了就报 unknown，不许猜成 holds"
    # 两边都不许误报——把什么都标 broken 就能刷检出率
    assert m["误报率"] == 0.0 and n["误报率"] == 0.0
