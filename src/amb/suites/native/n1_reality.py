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


class RealitySuite:
    name: ClassVar[str] = "n1_reality"
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
