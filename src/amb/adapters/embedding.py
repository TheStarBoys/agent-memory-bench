"""Embedding 客户端。OpenAI 兼容接口，stdlib 实现，不引第三方依赖。

⛔ API key 只从环境变量读，绝不写进仓库或配置文件——
配置里存的是**变量名**，不是值。
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass

from amb.core import load_dotenv, require


class EmbeddingError(RuntimeError):
    pass


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
        import http.client
        import time

        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                return self._post(texts)
            except (urllib.error.URLError, http.client.IncompleteRead,
                    ConnectionError, TimeoutError) as exc:
                last = exc
                if attempt + 1 < self.cfg.max_retries:
                    # ⚠️ 退避后重试；⛔ 不静默返回空向量——那会让分数变成假的
                    time.sleep(1.5 * (attempt + 1))
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
