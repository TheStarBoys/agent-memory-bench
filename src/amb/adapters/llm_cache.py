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


_WARNED: set[str] = set()


def warn_once(message: str) -> None:
    """⛔ 缓存故障必须可见——⚠️ 静默降级会让它变成查不出来的 bug。"""
    import sys

    if message not in _WARNED:
        _WARNED.add(message)
        print(f"⚠️ {message}", file=sys.stderr)


#: 限流重试。⛔ 供应商 429 不是被测系统的错，也不是我们的分数——
#: ⚠️ 实测：mem0 摄入到第 ~130 条时撞 TPM 上限，整条臂当场判「跑挂了」，
#: 而它已经跑了 16 分钟。⭐ 退避重试放在这一层，被测系统完全看不见。
RETRY_MAX = int(os.environ.get("AMB_LLM_RETRY", "6"))
#: 退避基数（秒）。⚠️ TPM 是**按分钟**的窗口，所以要退到分钟级才有用。
RETRY_BASE_S = float(os.environ.get("AMB_LLM_RETRY_BASE_S", "8"))


def _is_transient(exc: Exception) -> bool:
    """限流与超时可以重试；⛔ 别的错误照抛——⚠️ 把真 bug 重试掉比慢更糟。"""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429 or str(status) == "429":
        return True
    name = type(exc).__name__
    # ⚠️ 超时跟限流一样是暂时的。⭐ 但我们的重试**会说话**，
    # SDK 自己那 2 次静默重试不会——那正是 19 分钟空档查不出来的原因。
    if name in ("APITimeoutError", "APIConnectionError", "Timeout"):
        return True
    text = str(exc)
    return "429" in text and ("rate limit" in text.lower()
                              or "TPM" in text or "RPM" in text)


class Meter:
    """⭐ **我们自己**在包装层测的 token 用量。

    ⛔ 原则⑥ 说「token 只有适配器报得出来」——那是指被测系统自报。
    ⚠️ 但我们拦着每一次 openai 调用，usage 就在响应里，
    ⭐ 这是**我们测的**，比自报更可信，且不要求它声明 ACCOUNTING。

    ⚠️ 缓存命中的调用不计——⛔ 那次没真花钱，算进去会虚高。
    """

    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0
        self.cached_calls = 0

    def add(self, usage) -> None:
        self.tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        self.calls += 1

    def as_dict(self) -> dict:
        return {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "llm_calls": self.calls, "cached_calls": self.cached_calls,
                "retries": RETRIES.retries,
                # ⚠️ 重试等待单独报——⛔ 那是供应商配额的成本，不是系统的
                "retry_waited_s": round(RETRIES.waited_s, 1),
                # ⭐ embedding 也要报——⛔ 早先这一层完全不可见
                **EMBED.as_dict()}


METER = Meter()


class EmbedMeter:
    """⭐ 被测系统发出的 **embedding** 调用，我们在包装层实测。

    ⛔ 补的是一个明确的缺口：`wrap_openai_client` 只包了 `chat.completions`，
    ⚠️ embedding 调用既没有超时钉子、也没有重试、更没有计量——
    于是它用的是 openai SDK 的默认值（600s 超时 + **静默**重试 2 次），
    一次卡住的调用能吃掉半小时而日志上一片空白。

    ⚠️ 实测踩到：一次摄入 60478ms，正常 30395ms，**慢一倍且全程无告警**。
    ⭐ 慢不是罪，慢而没人知道才是。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0
        self.wall_ms = 0
        self.slowest_ms = 0

    def add(self, *, texts: int, ms: float) -> None:
        self.calls += 1
        self.texts += texts
        self.wall_ms += int(ms)
        self.slowest_ms = max(self.slowest_ms, int(ms))

    def as_dict(self) -> dict:
        return {"embed_calls": self.calls, "embed_texts": self.texts,
                "embed_wall_ms": self.wall_ms,
                "embed_slowest_ms": self.slowest_ms}


EMBED = EmbedMeter()


class RetryStats:
    """⚠️ 重试等了多久要能报出来——⛔ 否则它会被算进「这个系统很慢」。"""

    def __init__(self) -> None:
        self.retries = 0
        self.waited_s = 0.0


RETRIES = RetryStats()


def backbone_overrides() -> dict:
    """被测系统自己发的调用里，**我们有权钉死**的那几项。

    ⛔ 这些是 backbone 的受控变量，不是被测系统的设置——
    所有臂必须一致，否则分数不可比。⚠️ 每一项都要进报告。

    | 钉什么 | 为什么 |
    |---|---|
    | `temperature=0.0` | 判分要可复现。⚠️ mem0 默认 0.1、A-mem 默认 1.0，两个都踩过 |
    | `enable_thinking=False` | ⭐ 思考型 backbone 输出 token 大 6～8 倍。实测 A-mem 摄入 3 条：418.4s → 27.4s |

    ⚠️ 不支持 `enable_thinking` 的服务端会忽略它，⛔ 不是错误。
    要开思考就 `AMB_LLM_THINKING=1`，⭐ 报告里会写着开了。
    """
    thinking = os.environ.get("AMB_LLM_THINKING", "").lower() in (
        "1", "true", "yes", "on")
    # ⛔ 单次调用超时。⚠️ openai SDK 默认 read=600s **且自带 2 次重试**，
    # 一次卡住的调用能吃掉 30 分钟——实测 A-mem 摄入中间空了 19 分钟，
    # 没有任何日志（SDK 的重试是静默的）。
    # ⭐ 关思考后输出都在 500 token 以内，120s 已经很宽。超了就让**我们的**
    # 退避重试接手——那个会说话。
    out: dict = {"temperature": 0.0,
                 "timeout": float(os.environ.get("AMB_LLM_TIMEOUT_S", "120"))}
    if not thinking:
        out["extra_body"] = {"enable_thinking": False}
    return out


def wrap_openai_client(client: object, *,
                       force_temperature: float | None = None,
                       overrides: dict | None = None) -> bool:
    """钉死 backbone 的受控变量，并（在启用时）套缓存。返回是否打上了。

    ⛔ **缓存没开也会打补丁**——受控变量比缓存重要：⚠️ 早先只在缓存启用时
    才打，那样不开缓存的跑用的是被测系统自己的 temperature，
    判分不可复现，而且没人会发现。

    ⛔ 打在类上不行：openai 的 `@required_args` 装饰器在**导入时**就绑定了
    原函数——⚠️ 替换类属性对已存在的调用路径不生效（踩过，表现是命中数恒为 0）。
    ⭐ 打在实例的 `chat.completions` 对象上才拦得到。

    `overrides` 不传就用 `backbone_overrides()`；⛔ 传空字典 `{}` 才是
    「什么都不钉」。`force_temperature` 覆盖其中的温度那一项。
    """
    import time

    cache = global_cache()
    target = getattr(getattr(client, "chat", None), "completions", None)
    if target is None or getattr(target, "_amb_cached", False):
        return False

    original = target.create
    pins = dict(backbone_overrides() if overrides is None else overrides)
    if force_temperature is not None:
        pins["temperature"] = force_temperature

    def cached(**kwargs):
        for name, value in pins.items():
            if name == "extra_body":
                kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **value}
            else:
                kwargs[name] = value
        # ⛔ 缓存没开也要走到这儿——受控变量比缓存重要
        if not cache.enabled:
            got = _with_retry(original, kwargs)
            METER.add(getattr(got, "usage", None))
            return got
        payload = _jsonable({k: v for k, v in kwargs.items()
                             if k != "extra_headers"})
        try:
            hit = cache.get(payload)
        except Exception as exc:  # noqa: BLE001 —— 退回真调用，⛔ 但要说话
            warn_once(f"缓存读取失败：{type(exc).__name__}: {exc}")
            hit = None
        if hit is not None:
            from openai.types.chat import ChatCompletion

            # ⚠️ 命中不计 token——⛔ 那次没真花钱
            METER.cached_calls += 1
            return ChatCompletion.model_validate(hit)
        t0 = time.perf_counter()
        got = _with_retry(original, kwargs)
        METER.add(getattr(got, "usage", None))
        try:
            cache.put(payload, got.model_dump(),
                      int((time.perf_counter() - t0) * 1000))
        except Exception as exc:  # noqa: BLE001
            warn_once(f"缓存写入失败：{type(exc).__name__}: {exc}")
        return got

    target.create = cached
    target._amb_cached = True
    return True


def _with_retry(call, kwargs: dict):
    """撞限流就退避重试。⛔ 只对限流生效。

    ⚠️ 等待时间累进 `RETRIES`，报告要把它跟「系统本身多慢」分开——
    ⛔ 否则供应商的配额会被读成被测系统的成本。
    """
    import random
    import time as _time

    for attempt in range(RETRY_MAX + 1):
        try:
            return call(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt >= RETRY_MAX or not _is_transient(exc):
                raise
            # ⚠️ 指数退避 + 抖动：TPM 是按分钟的窗口，退到分钟级才有意义
            wait = RETRY_BASE_S * (2 ** attempt) * (0.5 + random.random())
            RETRIES.retries += 1
            RETRIES.waited_s += wait
            warn_once(f"{type(exc).__name__} → 退避重试"
                      f"（第 {attempt + 1}/{RETRY_MAX} 次，等 {wait:.0f}s）"
                      f"——⚠️ 这段等待不算被测系统的成本")
            _time.sleep(wait)
    raise RuntimeError("不可达")


def _jsonable(payload: dict) -> dict:
    import json

    return json.loads(json.dumps(payload, default=str, sort_keys=True))


def wrap_openai_embeddings(client: object) -> bool:
    """给被测系统的 **embedding** 调用钉超时、套重试、上计量。返回是否打上了。

    ⛔ 为什么必须单独打一层：`wrap_openai_client` 拦的是
    `chat.completions.create`，⚠️ `embeddings.create` 走的是**另一条路径**，
    早先完全裸奔——openai SDK 默认 600s 超时 + 静默重试 2 次。

    ⚠️ 这里**不做缓存**：向量是大块二进制，缓存收益小；
    ⛔ 更要紧的是缓存会把端点的抖动冻住——实测同一句话两次调用
    余弦 0.99989（不是 1.0），⭐ 冻住它等于把一个真实的不确定性藏起来。
    """
    import time

    target = getattr(client, "embeddings", None)
    if target is None or getattr(target, "_amb_wrapped", False):
        return False

    original = target.create
    timeout = float(os.environ.get("AMB_EMBED_TIMEOUT_S", "120"))

    def wrapped(**kwargs):
        kwargs.setdefault("timeout", timeout)
        n = len(kwargs.get("input") or [])
        t0 = time.perf_counter()
        got = _with_retry(original, kwargs)
        EMBED.add(texts=n, ms=(time.perf_counter() - t0) * 1000)
        return got

    target.create = wrapped
    target._amb_wrapped = True
    return True
