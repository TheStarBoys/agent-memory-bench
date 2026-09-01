"""LLM 客户端。OpenAI 兼容，stdlib 实现，不引第三方依赖。

⛔ API key 只从环境变量读。
⭐ 顺带记 token——原则⑥ 要求成本是一等指标，而 token 只有适配器报得出来。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LLMConfig:
    model: str
    base_url: str
    api_key_env: str          # ⛔ 变量名，不是 key 本身
    temperature: float = 0.0  # ⛔ 判分要可复现，不采样
    timeout_s: float = 600.0
    max_tokens: int = 512


@dataclass(slots=True)
class Meter:
    """累计用量。⚠️ 与评测器从外部测的墙钟分开报，两者差本身就是信息。"""

    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0

    def add(self, usage: dict[str, int]) -> None:
        self.tokens_in += int(usage.get("prompt_tokens", 0))
        self.tokens_out += int(usage.get("completion_tokens", 0))
        self.calls += 1


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.meter = Meter()

    def _key(self) -> str:
        from amb.core import load_dotenv, require

        load_dotenv()
        try:
            return require(self.cfg.api_key_env)
        except KeyError as exc:
            raise LLMError(str(exc)) from None

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self._key()}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc
        self.meter.add(body.get("usage", {}))
        return body["choices"][0]["message"]["content"].strip()


def from_env() -> LLMConfig:
    """⛔ 全局唯一的 backbone——跑 answer 档时所有系统必须用同一个。"""
    from amb.core import load_dotenv, require

    load_dotenv()
    return LLMConfig(
        model=require("AMB_LLM_MODEL"),
        base_url=require("AMB_LLM_BASE_URL"),
        api_key_env=os.environ.get("AMB_LLM_API_KEY_ENV", "SILICONFLOW_API_KEY"),
    )
