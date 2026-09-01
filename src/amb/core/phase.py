"""五个阶段。评测器固定按这个顺序驱动，⛔ 适配器不得跨阶段留后门。

规格：docs/adapters/protocol.md#阶段
"""

from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    SETUP = "setup"     # 建世界 → reset() → setup(world)
    INGEST = "ingest"   # 逐条 ingest(doc) → finalize()
    MUTATE = "mutate"   # ⚠️ 只有评测器动世界。适配器不参与，也不被通知
    PROBE = "probe"     # search / answer / audit / recall / audit_log
    SCORE = "score"     # 确定性判分


#: 每个阶段边界都要校验一次世界哈希——不只是 probe 前后。
#: ingest 与 finalize 期间系统同样在运行，同样够得着世界。
BOUNDARIES: tuple[Phase, ...] = (Phase.SETUP, Phase.INGEST, Phase.MUTATE, Phase.PROBE)
