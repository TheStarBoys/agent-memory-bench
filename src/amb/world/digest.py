"""世界哈希：规范化序列化，⛔ 不是逐字节 diff。

不能只哈希文件内容——时钟和事实表不是文件。
⚠️ mtime 不进哈希，但必须钉死（见 materialize），否则复现性靠不住。
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def digest(root: Path, now: str, facts: dict[str, str]) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        body = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{rel}\0{p.stat().st_mode & 0o777:o}\0{body}\n".encode())
    h.update(f"CLOCK\0{now}\n".encode())
    for k in sorted(facts):
        h.update(f"{k}\0{facts[k]}\n".encode())
    return "sha256:" + h.hexdigest()
