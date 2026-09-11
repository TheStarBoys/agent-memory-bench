"""LLM 客户端。OpenAI 兼容，stdlib 实现，不引第三方依赖。

⛔ API key 只从环境变量读。
⭐ 顺带记 token——原则⑥ 要求成本是一等指标，而 token 只有适配器报得出来。
"""

from __future__ import annotations

import json
import time
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class LLMError(RuntimeError):
    pass


class BadResponse(LLMError):
    """端点回了个不成形的东西。⭐ **可重试**——⚠️ 它跟连接抖动是同一类抽风。

    ⛔ 早先这两种情形**裸抛底层异常**：
      ⚠️ 返回一坨 HTML → `json.decoder.JSONDecodeError`
      ⚠️ 没有 choices  → `IndexError: list index out of range`

    ⭐ 两个都**不可重试**（`_retryable` 只认 `URLError`），
    ⛔ 而错误话毫无信息——真跑里看到 `IndexError`，
    ⚠️ 得查半天才知道是端点抽风而不是我们的 bug。

    ⚠️ embedding 那条路早有同样的防护（`IncompleteEmbedding`），⛔ 这条漏了。
    ⭐ 是 lint 的 `B017`（测试里 `raises(Exception)` 太宽）把它翻出来的：
    那两条测试**之前是假绿**——通过了，而抛的根本不是 `LLMError`。
    """


@dataclass(frozen=True, slots=True)
class LLMConfig:
    model: str
    base_url: str
    api_key_env: str          # ⛔ 变量名，不是 key 本身
    temperature: float = 0.0  # ⛔ 判分要可复现，不采样
    timeout_s: float = 600.0
    max_tokens: int = 512
    #: ⭐ 思考型 backbone 会把输出 token 撑大 6～8 倍。实测 A-mem 摄入 3 条：
    #: 思考开 418.4s，关 27.4s——**15 倍**，而抽取质量没变。
    #: ⚠️ 这是 backbone 的受控变量，不是被测系统的设置；⛔ 必须进报告。
    thinking: bool = False


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
            # ⚠️ 不支持这个字段的服务端会忽略它，⛔ 不是错误
            **({} if self.cfg.thinking else {"enable_thinking": False}),
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
        # ⛔ **必须重试**：⚠️ 被测系统那条路（`llm_cache._with_retry`）有退避，
        # 这里早先一次瞬时 5xx/超时就 `LLMError` 冒到 suite → 整条臂记 crashed。
        # ⭐ 同一次网络抖动，对照组被记「跑挂了」、被测系统被重试救回来——
        # 那是**框架的缺陷被记成臂的失败**（见 core/fault.py 那一类）。
        body = self._post(req)
        self.meter.add(body.get("usage", {}))
        return _content_of(body)

    def _post(self, req, attempts: int = 3) -> dict:
        """发一次请求，⭐ 瞬时故障退避重试。⛔ 三次都不成才算真失败。"""
        last: Exception = LLMError("没发出去")
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    raw = resp.read()
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError as exc:
                    # ⚠️ 网关的 HTML 错误页最常见
                    raise BadResponse(
                        f"响应不是 JSON（前 80 字节：{raw[:80]!r}）") from exc
                # ⛔ **校验要在重试循环内**：⚠️ 放在 `complete` 里的话，
                # 「BadResponse 可重试」就只是句空话——⭐ 实测过：
                # 空 choices 抛出来之后一次都没重试过。
                _content_of(body)
                return body
            except (urllib.error.URLError, BadResponse) as exc:
                last = exc
                if not _retryable(exc) or attempt == attempts - 1:
                    break
                time.sleep(2 ** attempt)
        raise LLMError(f"LLM 调用失败（试了 {attempts} 次）：{last}") from last


def _content_of(body: dict) -> str:
    """从响应里取回答。⛔ **不许裸下标**——⚠️ 空 choices 抛的
    `IndexError: list index out of range` 不告诉任何人是端点返回了空。"""
    choices = body.get("choices") or []
    if not choices:
        raise BadResponse(f"响应里没有 choices（keys={sorted(body)}）")
    return choices[0]["message"]["content"].strip()


def _retryable(exc: Exception) -> bool:
    """值不值得重试。⛔ 4xx（除 429）是请求本身错了，重试没有意义。

    ⭐ `BadResponse` 值得重试：⚠️ 端点回一坨 HTML 或空 choices 是典型的
    **瞬时**抽风，⛔ 与「请求本身错了」不是一回事。
    """
    if isinstance(exc, BadResponse):
        return True
    code = getattr(exc, "code", None)
    return code is None or code == 429 or code >= 500


def from_env() -> LLMConfig:
    """⛔ 全局唯一的 backbone——跑 answer 档时所有系统必须用同一个。"""
    from amb.core import load_dotenv, require

    load_dotenv()
    return LLMConfig(
        model=require("AMB_LLM_MODEL"),
        base_url=require("AMB_LLM_BASE_URL"),
        api_key_env=os.environ.get("AMB_LLM_API_KEY_ENV", "SILICONFLOW_API_KEY"),
        # ⛔ 默认关思考。要开就显式 AMB_LLM_THINKING=1，⚠️ 并且报告里会写着。
        thinking=os.environ.get("AMB_LLM_THINKING", "").lower()
        in ("1", "true", "yes", "on"),
        # ⭐ 读超时可配：⚠️ 默认 600s 是给真端点留的余量，
        # ⛔ 但那让「超时之后会怎样」在测试里**触发不了**——
        # 实测代价：一条断言超时行为的测试等了 3 秒就通过了，
        # ⚠️ 它从没超时过，于是那个真 bug（超时被判配置问题）从它底下溜走。
        timeout_s=float(os.environ.get("AMB_LLM_TIMEOUT_S") or 600.0),
    )
