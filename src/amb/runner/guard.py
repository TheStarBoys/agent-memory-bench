"""世界哈希守卫。

⛔ 每个阶段边界都校验，不只是 probe 前后——
ingest 与 finalize 期间系统同样在运行，同样够得着世界。

⛔ 不一致直接判本次跑作废，不是扣分。
"""

from __future__ import annotations

from pathlib import Path

from amb.core import Phase
from amb.world import WorldState, digest


class WorldTampered(RuntimeError):
    """被测系统改了世界——⛔ 本次跑作废。

    能写世界的系统可以靠改变现实让自己的记忆重新为真，
    那不是通过测试，是把考题改了。
    """


class WorldGuard:
    def __init__(self, state: WorldState) -> None:
        self._state = state
        self._expected = self._now()

    def _now(self) -> str:
        return digest(self._state.root, self._state.now, self._state.facts)

    def rebaseline(self) -> str:
        """评测器自己改完世界之后重设基线——⚠️ 只有 mutate 阶段可以调。"""
        self._expected = self._now()
        return self._expected

    def check(self, phase: Phase) -> None:
        actual = self._now()
        if actual != self._expected:
            raise WorldTampered(
                f"{phase} 阶段结束时世界哈希不一致——本次跑作废\n"
                f"  期望 {self._expected}\n  实际 {actual}"
            )

    @property
    def expected(self) -> str:
        return self._expected
