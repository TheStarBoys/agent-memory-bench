"""对着当前世界核一条命题。对照组共用。

⛔ 只读：读文件、GET 时钟与事实表，绝不写。
⚠️ 这不是「记忆系统该怎么做 N1」的示范——协议只说要回答什么，
不说怎么做到（原则②）。这里是**地板线**的做法：笨办法，重读一遍。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from amb.core import WorldHandle


@dataclass(frozen=True, slots=True)
class Reading:
    """一次世界读取的结果。⛔ grounds 记下读了什么，判分要它非空。"""

    exists: bool
    text: str | None
    ground: str          # 读的是哪个路径 / 哪个键


class WorldReader:
    def __init__(self, world: WorldHandle) -> None:
        self._world = world
        self._root = Path(world.root)

    def file(self, rel: str) -> Reading:
        p = self._root / rel
        if not p.is_file():
            return Reading(False, None, f"file:{rel}")
        return Reading(True, p.read_text(encoding="utf-8"), f"file:{rel}")

    def fact(self, key: str) -> Reading:
        try:
            with urllib.request.urlopen(f"{self._world.facts_url}/{key}", timeout=10) as r:
                return Reading(True, json.loads(r.read())["value"], f"fact:{key}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return Reading(False, None, f"fact:{key}")
            raise

    def now(self) -> str:
        with urllib.request.urlopen(self._world.clock_url, timeout=10) as r:
            return json.loads(r.read())["now"]
