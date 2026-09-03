"""N5 巩固与遗忘：该留的留了，⭐ **该丢的丢了吗**。

⛔ 判据是需求概率，不是「像人一样遗忘」——
人之所以那样遗忘，是因为那样在真实环境里最优（Anderson & Schooler 1991）。

两种：
    外部观察  只发 search，看什么还捞得到 —— ⭐ 任何系统都能跑
    系统自报  调 recall(claims)，逐条问「还留着吗、多强」
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import (
    Adapter,
    Capability,
    Claim,
    Failed,
    Observation,
    SuiteRun,
    Unsupported,
)
from amb.world import WorldState
from amb.world.stream.events import EventStream, Fact
from amb.world.stream.need import NeedCurve


@dataclass(frozen=True, slots=True)
class RetentionProbe:
    """一条事实，以及它此刻「还会被需要」的概率。"""

    fact: Fact
    query: str
    need: float             # ⭐ ground truth
    should_keep: bool       # need 是否超过保留阈值


def probes_from(stream: EventStream, curve: NeedCurve, *, now_s: float,
                keep_threshold: float = 0.5) -> list[RetentionProbe]:
    """按需求概率给每条事实定真值。

    ⚠️ 用最后一次出现算 elapsed——⭐ 这正是「间隔」起作用的地方：
    分散组的最后一次更靠近现在，需求概率更高。
    """
    last_seen: dict[str, float] = {}
    for occ in stream.occurrences:
        last_seen[occ.fact_id] = max(last_seen.get(occ.fact_id, 0.0), occ.at)

    out: list[RetentionProbe] = []
    for fact in stream.facts:
        elapsed = max(1.0, now_s - last_seen.get(fact.fact_id, fact.first_at))
        need = curve.at(elapsed)
        # ⭐ 频率与显著性抬高需求：一件反复发生、或后果重大的事，更可能再被问起
        need = min(1.0, need * (1.0 + 0.1 * (fact.frequency - 1))
                   * (1.5 if fact.salient else 1.0))
        out.append(RetentionProbe(fact, fact.text, need, need >= keep_threshold))
    return out


#: ⛔ 曲线没拟合时，这一档的数**不得发布**的理由。⚠️ 机制照跑，分照算——
#: ⭐ 但 ground truth 本身立不住：自己拍参数等于自己定义什么叫「该记住」。
UNFITTED_WHY = ("需求概率曲线未从真实语料拟合（source=None）——"
                "⛔ 自己拍参数等于自己定义什么叫「该记住」，那是自证。"
                "见 docs/adapters/world.md#need-probability")


def _why_unpublishable(curve: NeedCurve | None) -> str:
    """⚠️ 没给曲线就不表态：⛔ 不拿「没说」冒充「可发布」。"""
    if curve is None:
        return "未申报需求概率曲线的来源——⛔ 说不清 ground truth 从哪来"
    return "" if curve.fitted else UNFITTED_WHY


class ObservedRetentionSuite:
    """外部观察：⭐ 不需要声明任何能力，任何系统都能跑。"""

    name: ClassVar[str] = "n5_observed"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.SEARCH})

    def __init__(self, probes: list[RetentionProbe],
                 curve: NeedCurve | None = None) -> None:
        self._probes = probes
        # ⛔ 曲线的出身必须跟着分走到报告：⚠️ 断在这里，一个「不得发布」的数
        # 就会照常进对比表——实测踩到，它还当上了成本×质量表的质量列。
        self._why = _why_unpublishable(curve)

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored", not_publishable=self._why)
        for p in self._probes:
            hits = adapter.search(p.query, 5)
            retained = any(p.fact.fact_id in h.doc_ids for h in hits)
            run.observations.append(Observation(p.fact.fact_id, {
                "should_keep": p.should_keep,
                "retained": retained,
                "need": p.need,
                # ⭐ 三个因子，判分要分别算它们的贡献
                "frequency": p.fact.frequency,
                "spacing": str(p.fact.spacing),
                "salient": p.fact.salient,
            }))
        return run


class SelfReportedRetentionSuite:
    """系统自报：逐条问「这条你还留着吗、多强」。"""

    name: ClassVar[str] = "n5_self_reported"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.RETENTION})

    def __init__(self, probes: list[RetentionProbe],
                 curve: NeedCurve | None = None) -> None:
        self._probes = probes
        self._why = _why_unpublishable(curve)      # ⛔ 同上，曲线出身跟着分走

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        claims = [Claim(p.fact.fact_id, p.query, [p.fact.fact_id])
                  for p in self._probes]
        got = adapter.recall(claims)
        if isinstance(got, Unsupported):
            return SuiteRun(self.name, "unsupported", reason=got.reason)
        if isinstance(got, Failed):
            return SuiteRun(self.name, "scored", reason=got.reason,
                            failed=len(claims), not_publishable=self._why)

        run = SuiteRun(self.name, "scored", not_publishable=self._why)
        by_id = {v.claim_id: v for v in got}
        for p in self._probes:
            v = by_id.get(p.fact.fact_id)
            run.observations.append(Observation(p.fact.fact_id, {
                "should_keep": p.should_keep,
                "retained": bool(v and v.state == "retained"),
                "strength": (v.strength if v else None),
                "need": p.need,
                "frequency": p.fact.frequency,
                "spacing": str(p.fact.spacing),
                "salient": p.fact.salient,
            }))
        return run
