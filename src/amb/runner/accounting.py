"""成本与延迟记账（原则⑥）。

⚠️ 墙钟两份都要：评测器从外部独立计时一份，适配器自报一份。
两者差得多本身就是信息（排队、重试、后台补偿）。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class Ledger:
    wall_ms_harness: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def measure(self, phase: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = int((time.perf_counter() - t0) * 1000)
            self.wall_ms_harness[phase] = self.wall_ms_harness.get(phase, 0) + ms
