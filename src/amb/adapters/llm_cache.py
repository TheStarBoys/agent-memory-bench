"""LLM 响应缓存：内容寻址,⛔ 不改变被测系统的行为。

⭐ 为什么安全：缓存键是 (模型, 全部参数, 完整请求体) 的哈希。
同样的输入本来就该得到同样的输出——⚠️ temperature=0 时这是确定的，
⛔ temperature>0 时**不该缓存**，那会把随机性冻住。

⭐ 收益：第一次跑照常花钱，之后重跑同一批语料**几乎免费**。
⚠️ 迭代评测代码时这是最大的一笔节省——摄入占了 86% 的时间。

⛔ 但缓存命中的跑**不是一次独立的延迟测量**：
报告里必须标出来，否则「它变快了」会被读成系统变快了。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_PATH = Path(".external/llm-cache.sqlite")


#: 跳过缓存的原因。⛔ 每一种都要计数——
#: ⚠️ 「命中率 0」有很多种原因，不分开记就查不出是哪一种（踩过：
#: mem0 默认 temperature=0.1，缓存静默失效，**连异常都没抛**）。
class Skip(StrEnum):
    DISABLED = "未启用"                    # AMB_LLM_CACHE 没开
    SAMPLING = "temperature>0"             # ⛔ 采样时不缓存
    READ_ERROR = "读取出错"
    WRITE_ERROR = "写入出错"


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    #: ⭐ 省下的墙钟（毫秒）——按未命中时的实测均值估
    saved_ms: int = 0
    #: ⭐ 跳过的原因分布。⛔ 这一栏是「为什么没生效」的答案。
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, why: "Skip") -> None:
        self.skipped[str(why)] = self.skipped.get(str(why), 0) + 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())

    def diagnosis(self) -> str:
        """⭐ 一句话说清「缓存为什么没生效」。

        ⛔ 这是这一层存在的理由——命中率为 0 时，
        它必须答得出「因为什么」，⚠️ 而不是让人去猜。
        """
        looked = self.hits + self.misses
        if looked and self.hit_rate > 0:
            return (f"✓ 命中 {self.hits}/{looked}（{self.hit_rate:.0%}）"
                    f"，省下 {self.saved_ms / 1000:.0f}s")
        if not looked and not self.skipped:
            return "⚠️ 一次 LLM 调用都没发生——缓存无从谈起"
        if self.skipped:
            top = max(self.skipped, key=lambda k: self.skipped[k])
            detail = " · ".join(f"{k}×{v}" for k, v in sorted(self.skipped.items()))
            hint = _HINTS.get(top, "")
            return (f"⛔ {self.total_skipped} 次调用跳过了缓存（{detail}）"
                    + (f"——{hint}" if hint else ""))
        return (f"⚠️ 查了 {looked} 次一次没中——"
                f"⛔ 请求内容每次都不同（提示里带了时间戳？随机 id？）")

    def as_dict(self) -> dict[str, object]:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hit_rate, "saved_ms": self.saved_ms,
                "skipped": dict(self.skipped),
                "diagnosis": self.diagnosis()}


#: 每种跳过原因该怎么办。⚠️ 光说「跳过了」不够，要说**下一步做什么**。
_HINTS: dict[str, str] = {
    str(Skip.DISABLED): "设 AMB_LLM_CACHE=1 打开",
    str(Skip.SAMPLING): (
        "⭐ 被测系统在用采样温度。判分要可复现，"
        "把它的 temperature 钉成 0（mem0 默认是 0.1）"),
    str(Skip.READ_ERROR): "缓存库可能损坏，删掉 .external/llm-cache.sqlite 重来",
    str(Skip.WRITE_ERROR): "检查 .external/ 的写权限与磁盘空间",
}


class LLMCache:
    """SQLite 内容寻址缓存。⚠️ 进程内加锁，跨进程靠 SQLite 自己。"""

    def __init__(self, path: Path = DEFAULT_PATH, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._path = path
        self._conn: sqlite3.Connection | None = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses "
                "(key TEXT PRIMARY KEY, body TEXT, wall_ms INTEGER)")
            self._conn.commit()
        return self._conn

    @staticmethod
    def key(payload: dict) -> str:
        """⛔ 键必须覆盖**所有**影响输出的东西——漏一个就会串味。"""
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, payload: dict) -> dict | None:
        # ⛔ 每条跳过路径都记原因——⚠️ 静默返回 None 会让「没生效」查不出来
        if not self.enabled:
            self.stats.skip(Skip.DISABLED)
            return None
        if not _cacheable(payload):
            self.stats.skip(Skip.SAMPLING)
            return None
        with self._lock:
            row = self._db().execute(
                "SELECT body, wall_ms FROM responses WHERE key = ?",
                (self.key(payload),)).fetchone()
        if row is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        self.stats.saved_ms += int(row[1] or 0)
        return json.loads(row[0])

    def put(self, payload: dict, body: dict, wall_ms: int) -> None:
        # ⚠️ get 那边已经记过原因了，这里不重复计数
        if not self.enabled or not _cacheable(payload):
            return
        with self._lock:
            self._db().execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                (self.key(payload), json.dumps(body, ensure_ascii=False), wall_ms))
            self._db().commit()


def _cacheable(payload: dict) -> bool:
    """⛔ temperature>0 时不缓存——那会把随机性冻成一个固定答案。"""
    temp = payload.get("temperature")
    return temp is None or float(temp) == 0.0


#: 全局缓存。⚠️ 由 AMB_LLM_CACHE 控制，⛔ 默认**关闭**——
#: 打开它的跑不是独立的延迟测量，得有人明确要求。
_GLOBAL: LLMCache | None = None


def global_cache() -> LLMCache:
    global _GLOBAL  # noqa: PLW0603
    if _GLOBAL is None:
        flag = os.environ.get("AMB_LLM_CACHE", "").lower()
        _GLOBAL = LLMCache(enabled=flag in ("1", "true", "yes", "on"))
    return _GLOBAL
