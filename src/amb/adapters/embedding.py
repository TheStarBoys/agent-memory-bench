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
        if not texts:
            return []
        payload = json.dumps({"model": self.cfg.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.cfg.base_url.rstrip('/')}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.cfg.api_key()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:  # 网络/鉴权
            raise EmbeddingError(f"embedding 调用失败：{exc}") from exc
        return [row["embedding"] for row in body["data"]]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else num / (na * nb)
