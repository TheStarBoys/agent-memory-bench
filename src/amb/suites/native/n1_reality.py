"""N1 对现实求值：世界变了，你发现没有。

两种，分开报：
    有提示  调 audit(claims)，把命题交过去
    无提示  什么都不给，只看 search() 顺带返回的 Entry.state

⛔ 有提示由评测器出命题，不是让系统列举「我的哪些记忆坏了」——
后者预设系统持有可枚举的离散条目集合，是形状偏心。
"""

from __future__ import annotations

from typing import ClassVar

from amb.core import Adapter, Capability, Claim, Failed, Unsupported
from amb.core import Observation, SuiteRun
from amb.world import WorldState


class PromptedRealitySuite:
    """有提示：把命题交过去，看你查得准不准。"""

    name: ClassVar[str] = "n1_prompted"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.REALITY})

    def __init__(self, claims: list[Claim], truth: dict[str, str]) -> None:
        """truth: claim_id → holds | broken（变更之后的真值）"""
        self._claims = claims
        self._truth = truth

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        got = adapter.audit(self._claims)
        if isinstance(got, Unsupported):
            # ⛔ 进独立的「不支持」列，不计入分母，不记 0
            return SuiteRun(self.name, "unsupported", reason=got.reason)
        if isinstance(got, Failed):
            return SuiteRun(self.name, "scored", reason=got.reason,
                            failed=len(self._claims))

        run = SuiteRun(self.name, "scored")
        by_id = {v.claim_id: v for v in got}
        for claim in self._claims:
            v = by_id.get(claim.claim_id)
            run.observations.append(
                Observation(
                    claim.claim_id,
                    {
                        "truth": self._truth[claim.claim_id],
                        "reported": v.state if v else "unknown",
                        # ⛔ 空 grounds 判 Failed，不是 unknown
                        "grounds": list(v.grounds) if v else [],
                    },
                )
            )
        return run


class SpontaneousRealitySuite:
    """无提示：什么都不给，只发 search，看 `Entry.state`。

    ⛔ 与有提示分开报，永不合并——一个是「给了机会你查得准不准」，
    一个是「没人提醒你自己发现了吗」，合并之后两者都读不出来。

    绝大多数系统在这一档上会全军覆没。**那是这一档的意义，不是它的缺陷。**
    """

    name: ClassVar[str] = "n1_spontaneous"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.REALITY})

    def __init__(self, claims: list[Claim], truth: dict[str, str]) -> None:
        self._claims = claims
        self._truth = truth

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        reconciled = 0
        for claim in self._claims:
            # ⚠️ 只发普通检索——⛔ 不调 audit，不给任何提示
            hits = adapter.search(claim.text, 5)
            # 靠 doc_ids 把条目对回这条命题；⛔ 对不上就不是「答错」
            matched = [
                h for h in hits
                if h.doc_ids and set(h.doc_ids) & set(claim.doc_ids)
            ]
            reconciled += bool(matched)
            states = {h.state for h in matched if h.state is not None}
            if not states:
                reported = "unknown"      # 没表态 = 没发现
            elif "broken" in states:
                reported = "broken"       # 任一条目说坏了，就是发现了
            else:
                reported = "holds"
            run.observations.append(
                Observation(claim.claim_id, {
                    "truth": self._truth[claim.claim_id],
                    "reported": reported,
                    "grounds": sorted({d for h in matched for d in h.doc_ids}),
                })
            )

        if reconciled == 0:
            # ⛔ 一条都对不上账 → 不支持，不是 0 分。
            # 不是它答错了，是评测器无从把条目对回被破坏的事实。
            return SuiteRun(self.name, "unsupported",
                            reason="search 未返回带 doc_ids 的条目，无从对账")
        return run
