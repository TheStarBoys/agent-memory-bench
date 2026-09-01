"""把清单物化成一个真实目录。

⛔ mtime 全部钉死在时钟起点，不是物化时的真实时间——
不钉死的话两次物化内容相同、stat 不同，而「时间流逝」正是一类变更，
一个去读 mtime 判新鲜度的系统会因此在两次跑之间给出不同答案。
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from amb.world.manifest import WorldManifest


def _epoch(rfc3339: str) -> float:
    return dt.datetime.fromisoformat(rfc3339.replace("Z", "+00:00")).timestamp()


def materialize(manifest: WorldManifest, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = _epoch(manifest.clock_start)
    for spec in sorted(manifest.files, key=lambda f: f.path):  # ⚠️ 字典序，可复现
        target = root / spec.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(spec.text, encoding="utf-8")
        os.chmod(target, spec.mode)
        os.utime(target, (stamp, stamp))
    return root


def pin_mtimes(root: Path, clock_start: str) -> None:
    """变更世界之后重新钉死 mtime——⛔ 否则修改会泄漏真实时间。"""
    stamp = _epoch(clock_start)
    for p in sorted(root.rglob("*")):
        os.utime(p, (stamp, stamp))
