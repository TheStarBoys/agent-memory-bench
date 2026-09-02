"""Embedding 客户端。OpenAI 兼容接口，stdlib 实现，不引第三方依赖。

⛔ API key 只从环境变量读，绝不写进仓库或配置文件——
配置里存的是**变量名**，不是值。
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from amb.core import load_dotenv, require


class EmbeddingError(RuntimeError):
    pass


@dataclass(slots=True)
class EmbedMeter:
    """embedding 调用的用量与**可见性**。

    ⛔ 补的是一个明确的缺口：早先重试与计量**只包了 chat 调用**，
    embedding 两样都没有——⚠️ 于是一次跑的摄入慢了一倍（60478ms vs
    30395ms）而全程无人知晓，查了几个小时。

    ⭐ 慢不是罪，**慢而没人知道**才是。
    """

    calls: int = 0
    texts: int = 0
    wall_ms: int = 0
    slowest_ms: int = 0
    retries: int = 0
    waited_s: float = 0.0

    def add(self, *, texts: int, ms: float) -> None:
        self.calls += 1
        self.texts += texts
        self.wall_ms += int(ms)
        self.slowest_ms = max(self.slowest_ms, int(ms))

    def as_dict(self) -> dict[str, object]:
        return {"embed_calls": self.calls, "embed_texts": self.texts,
                "embed_wall_ms": self.wall_ms,
                "embed_slowest_ms": self.slowest_ms,
                "embed_retries": self.retries,
                # ⚠️ 重试等待单独报——⛔ 那是供应商的成本，不是被测系统的
                "embed_retry_waited_s": round(self.waited_s, 1)}


#: ⚠️ 进程内累计。⛔ 与评测器从外部测的墙钟分开报，两者的差本身就是信息。
METER = EmbedMeter()

#: 单次调用超过这个倍数的均值就出声。⚠️ 阈值宽一点，⛔ 但不能没有。
_SLOW_FACTOR = float(os.environ.get("AMB_EMBED_SLOW_FACTOR", "4"))
_SLOW_FLOOR_MS = float(os.environ.get("AMB_EMBED_SLOW_FLOOR_MS", "8000"))


def _say(message: str) -> None:
    print(f"⚠️ {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model: str
    base_url: str
    api_key_env: str          # ⛔ 变量名，不是 key 本身
    timeout_s: float = 120.0
    #: ⚠️ 端点在大批量上会截断连接（实测 IncompleteRead）——重试并缩批
    max_retries: int = 3
    #: 单次最多几条。⛔ 太大就撞上截断，⚠️ 太小则调用次数暴涨
    max_batch: int = 16

    def api_key(self) -> str:
        load_dotenv()  # 幂等；已存在的环境变量不覆盖
        try:
            return require(self.api_key_env)
        except KeyError as exc:
            raise EmbeddingError(str(exc)) from None


class EmbeddingClient:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg

    def embed(self, texts: list[str]) -> list[list[float]]:
        """⚠️ 自动分批 + 重试。

        ⛔ 实测端点在大批量上会 IncompleteRead（读到一半断流）——
        那不是端点坏了，是客户端没扛住。**评测框架不该因为传输抖动就丢一条臂。**
        """
        if not texts:
            return []
        out: list[list[float]] = []
        step = max(1, self.cfg.max_batch)
        for i in range(0, len(texts), step):
            out.extend(self._embed_batch(texts[i : i + step]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """⚠️ 退避重试 + 计量 + **出声**。

        ⛔ 三件事早先都没有：静默重试、不计量、慢了不说话。
        ⚠️ 后果实测过——一次摄入慢一倍，查了几小时才发现是这一层。
        """
        import http.client
        import random
        import time

        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            t0 = time.perf_counter()
            try:
                got = self._post(texts)
            except (urllib.error.URLError, http.client.IncompleteRead,
                    ConnectionError, TimeoutError) as exc:
                last = exc
                # ⛔ 只有限流、超时、传输抖动才重试——⚠️ 把 400 重试掉
                # 比慢更糟：它会把「请求本身就是错的」拖成一次超时
                if not _retryable(exc) or attempt + 1 >= self.cfg.max_retries:
                    break
                # ⚠️ 指数退避 + 抖动。⛔ 429 是按分钟的窗口，1.5s 不够用
                wait = 2.0 * (2 ** attempt) * (0.5 + random.random())
                METER.retries += 1
                METER.waited_s += wait
                _say(f"embedding {type(exc).__name__} → 退避重试"
                     f"（第 {attempt + 1}/{self.cfg.max_retries} 次，"
                     f"等 {wait:.0f}s）——⛔ 这段等待不算被测系统的成本")
                time.sleep(wait)
                continue
            ms = (time.perf_counter() - t0) * 1000
            mean = METER.wall_ms / METER.calls if METER.calls else ms
            METER.add(texts=len(texts), ms=ms)
            # ⭐ 慢不是罪，慢而没人知道才是
            if ms > _SLOW_FLOOR_MS and ms > _SLOW_FACTOR * mean:
                _say(f"embedding 单次 {ms / 1000:.1f}s（{len(texts)} 条），"
                     f"均值 {mean / 1000:.1f}s——⚠️ 端点在抖，"
                     f"这一跑的耗时不代表被测系统")
            return got
        raise EmbeddingError(
            f"embedding 调用失败（重试 {self.cfg.max_retries} 次）：{last}")

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.cfg.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.cfg.base_url.rstrip('/')}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.cfg.api_key()}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            body = json.loads(resp.read())
        return [row["embedding"] for row in body["data"]]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else num / (na * nb)


def _retryable(exc: Exception) -> bool:
    """限流 / 服务端错 / 传输抖动可以重试。⛔ 其余照抛。

    ⚠️ `HTTPError` 是 `URLError` 的子类——⛔ 早先一并重试了，
    于是一个 400（请求本身就是错的）会被重试三次才报出来。
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return True
