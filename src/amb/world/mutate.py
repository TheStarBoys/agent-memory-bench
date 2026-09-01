"""五类变更。每一类对应一种真实的腐化方式。

⛔ 「无关变更」这一类必须有，它是这套题的反方向：
只考「该 broken 的有没有被标出来」，一个把所有东西都标 broken 的系统会拿满分。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ChangeKind(StrEnum):
    VANISH = "vanish"        # 文件被删 → 依赖它存在的命题 broken
    REVALUE = "revalue"      # 内容/取值变了 → 断定旧值的 broken
    APPEAR = "appear"        # 新增 → 多数命题不受影响，仍 holds
    ADVANCE = "advance"      # 时钟前推 → 带有效期的 broken
    IRRELEVANT = "irrelevant"  # ⛔ 与任何命题无关 → 全部仍 holds


@dataclass(frozen=True, slots=True)
class Change:
    kind: ChangeKind
    target: str            # 文件路径 / 事实表键 / 新的 RFC3339
    value: str | None = None


@dataclass(slots=True)
class WorldState:
    root: Path
    now: str
    facts: dict[str, str]

    def apply(self, change: Change) -> None:
        """⚠️ 只有评测器调用这个。适配器不参与，也不被通知。"""
        match change.kind:
            case ChangeKind.VANISH:
                p = self.root / change.target
                os.chmod(p, 0o644)
                p.unlink()
            case ChangeKind.REVALUE:
                if change.target in self.facts:
                    self.facts[change.target] = change.value or ""
                else:
                    p = self.root / change.target
                    os.chmod(p, 0o644)
                    p.write_text(change.value or "", encoding="utf-8")
                    os.chmod(p, 0o444)
            case ChangeKind.APPEAR | ChangeKind.IRRELEVANT:
                p = self.root / change.target
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(change.value or "", encoding="utf-8")
                os.chmod(p, 0o444)
            case ChangeKind.ADVANCE:
                self.now = change.target
