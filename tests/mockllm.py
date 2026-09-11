"""可编程的 LLM / embedding 端点——⭐ **真 HTTP**，⛔ 除了它全是真的。

## ⛔ 与 `offline.py` 的区别

⚠️ `offline.py` 把 `urllib.request.urlopen` 整个换掉——⭐ 快，
⛔ 但它同时蒙掉了**我们自己的**网络层：分批、重试、退避、超时、
计量、缓存、`IncompleteRead` 处理。⚠️ 而那几样正是 bug 爱藏的地方
（实测：一次读超时被判「配置问题」，整条地板臂被跳过）。

⭐ 这里起一个**真的 HTTP 服务**：⚠️ 请求真的经过 socket、真的有
状态码和响应头，⛔ 于是那一层是被测的，不是被绕过的。

## ⭐ 它能被编程成什么样

⚠️ 边界情况靠它构造，⛔ 不靠等真端点碰巧抽风：

    always(text=...)          固定回答
    fail(n, status=500)       前 n 次返回错误码
    hang(n, seconds=...)      前 n 次挂住不回（⭐ 逼出超时与重试）
    truncated(n)              前 n 次响应**读到一半断流**（IncompleteRead）
    malformed(n)              前 n 次返回不是 JSON 的东西
    empty_choices(n)          前 n 次返回合法 JSON 但没有 choices
    rate_limit(n)             前 n 次返回 429
    script([...])             按顺序返回一串回答

⛔ 每一条都对着一个真实发生过的失效：⚠️ 端点在大批量上 IncompleteRead、
限流、读超时、返回空 choices……⭐ 它们此前只能靠真跑碰上。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: ⚠️ 假向量的维度。⛔ 取小——⭐ 这一层测的是接线不是语义。
DIM = 16


def _vector(text: str) -> list[float]:
    """确定性假向量：⭐ 按字符算，⛔ 刻意不随机。

    ⚠️ 随机向量下排名是任意的，⛔ 那样「原文能查到自己」这类断言
    就成了掷硬币。
    """
    v = [0.0] * DIM
    for ch in text:
        v[ord(ch) % DIM] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@dataclass
class Behaviour:
    """下一次请求该怎么答。⛔ 计数器一次性——⚠️ 用完回到正常。"""

    #: 固定回答；None = 从提示里挑一句最像的
    text: str | None = None
    #: 按顺序发这些回答，⚠️ 发完回到 `text`
    script: list[str] = field(default_factory=list)
    fail_times: int = 0
    fail_status: int = 500
    hang_times: int = 0
    hang_seconds: float = 5.0
    truncate_times: int = 0
    malformed_times: int = 0
    empty_times: int = 0
    rate_limit_times: int = 0


@dataclass
class Wire:
    """线上真实发生了什么。⭐ 断言看它，⛔ 不看我们以为发生了什么。"""

    chat_calls: int = 0
    embed_calls: list[int] = field(default_factory=list)
    #: ⚠️ 每次请求的完整 body——⛔ 「提示里到底放了什么」只有它说得准
    chat_bodies: list[dict] = field(default_factory=list)
    #: ⭐ 被我们**故意**弄坏的那些次
    injected: list[str] = field(default_factory=list)

    @property
    def embed_texts(self) -> int:
        return sum(self.embed_calls)

    @property
    def biggest_embed_batch(self) -> int:
        """⭐ 摄入是批量的、答题是一条一条的——⛔ 这个数分得开。"""
        return max(self.embed_calls, default=0)


def _pick_answer(body: dict) -> str:
    """从提示里挑一句最像答案的。⚠️ 它不聪明也不该聪明——
    ⛔ 这一层测接线，不测回答质量。
    """
    import re

    user = next((m["content"] for m in reversed(body.get("messages", []))
                 if m.get("role") == "user"), "")
    lines = [ln.strip(" -·") for ln in user.splitlines() if len(ln.strip()) > 4]
    if not lines:
        return "资料未提及"
    question = lines[-1]
    q = set(question)
    best, score = "", 0
    for ln in lines[:-1] or lines:
        overlap = len(q & set(ln))
        if overlap > score:
            best, score = ln, overlap
    if not best:
        return "资料未提及"
    # ⚠️ 截短：⛔ 判分按包含匹配，整段返回会让所有题「答对」——
    # ⭐ 那是假的满分，比 0 分更糟。
    m = re.search(r"[：:]\s*(\S{1,20})", best)
    return m.group(1) if m else best[:20]


class MockLLM:
    """一个可编程的 OpenAI 兼容端点。⭐ 用作上下文管理器。"""

    def __init__(self) -> None:
        self.behaviour = Behaviour()
        self.wire = Wire()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── ⭐ 编程接口 ────────────────────────────────────────────
    def always(self, text: str) -> "MockLLM":
        self.behaviour.text = text
        return self

    def script(self, answers: list[str]) -> "MockLLM":
        self.behaviour.script = list(answers)
        return self

    def fail(self, times: int, status: int = 500) -> "MockLLM":
        self.behaviour.fail_times = times
        self.behaviour.fail_status = status
        return self

    def rate_limit(self, times: int) -> "MockLLM":
        self.behaviour.rate_limit_times = times
        return self

    def hang(self, times: int, seconds: float = 5.0) -> "MockLLM":
        self.behaviour.hang_times = times
        self.behaviour.hang_seconds = seconds
        return self

    def truncated(self, times: int) -> "MockLLM":
        """⚠️ 响应读到一半断流——⛔ 实测端点在大批量上就是这么坏的。"""
        self.behaviour.truncate_times = times
        return self

    def malformed(self, times: int) -> "MockLLM":
        self.behaviour.malformed_times = times
        return self

    def empty_choices(self, times: int) -> "MockLLM":
        self.behaviour.empty_times = times
        return self

    def reset(self) -> None:
        self.behaviour = Behaviour()
        self.wire = Wire()

    # ── 生命周期 ──────────────────────────────────────────────
    @property
    def base_url(self) -> str:
        assert self._server is not None, "⛔ 还没 start"
        host, port = self._server.server_address[:2]
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> "MockLLM":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):      # ⛔ 别刷屏
                pass

            def do_POST(self):
                outer._handle(self)

        # ⚠️ 端口给 0：⛔ 固定端口会在并行跑测试时撞车
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ── 请求处理 ──────────────────────────────────────────────
    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        length = int(h.headers.get("Content-Length") or 0)
        raw = h.rfile.read(length)
        try:
            body = json.loads(raw.decode())
        except Exception:
            body = {}
        b = self.behaviour
        is_embed = "embeddings" in h.path

        # ⭐ 注入的失效，⚠️ 一次性：用掉一次就少一次
        if b.hang_times:
            b.hang_times -= 1
            self.wire.injected.append("hang")
            time.sleep(b.hang_seconds)       # ⛔ 逼出客户端的读超时
            return
        if b.rate_limit_times:
            b.rate_limit_times -= 1
            self.wire.injected.append("429")
            return self._send(h, 429, b'{"error":{"message":"rate limited"}}')
        if b.fail_times:
            b.fail_times -= 1
            self.wire.injected.append(str(b.fail_status))
            return self._send(h, b.fail_status, b'{"error":{"message":"boom"}}')
        if b.malformed_times:
            b.malformed_times -= 1
            self.wire.injected.append("malformed")
            return self._send(h, 200, b"<html>not json</html>")
        if b.truncate_times:
            b.truncate_times -= 1
            self.wire.injected.append("truncated")
            # ⛔ 声明一个长度然后**只发一半**再断连——⚠️ 这就是 IncompleteRead
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", "4096")
            h.end_headers()
            h.wfile.write(b'{"data":[{"embedding":[0.1,0.2')
            h.wfile.flush()
            h.close_connection = True
            return

        if is_embed:
            texts = body.get("input")
            texts = [texts] if isinstance(texts, str) else (texts or [])
            self.wire.embed_calls.append(len(texts))
            if b.empty_times:
                b.empty_times -= 1
                self.wire.injected.append("empty")
                return self._send(h, 200, b'{"data":[]}')
            payload = {"data": [{"embedding": _vector(t)} for t in texts]}
        else:
            self.wire.chat_calls += 1
            self.wire.chat_bodies.append(body)
            if b.empty_times:
                b.empty_times -= 1
                self.wire.injected.append("empty")
                return self._send(h, 200, b'{"choices":[]}')
            if b.script:
                text = b.script.pop(0)
            elif b.text is not None:
                text = b.text
            else:
                text = _pick_answer(body)
            usage = {"prompt_tokens": max(1, len(str(body)) // 4),
                     "completion_tokens": max(1, len(text) // 4)}
            # ⭐ **DSH 走流式**：⚠️ 它发 `stream: True`，
            # ⛔ 拿非流式的 `choices[].message` 回它，它判 `EMPTY_RESPONSE`
            # 并重试 5 次然后 `finish_reason='error'`——⚠️ 实测踩到。
            if body.get("stream"):
                return self._send_sse(h, text, usage)
            payload = {
                "choices": [{"message": {"content": text}}],
                # ⭐ 带用量：⚠️ 计量层也要被测到，⛔ 空的会让成本档静默为 0
                "usage": usage,
            }
        self._send(h, 200, json.dumps(payload, ensure_ascii=False).encode())

    @staticmethod
    def _send_sse(h: BaseHTTPRequestHandler, text: str, usage: dict) -> None:
        """OpenAI 兼容的流式响应。⚠️ 逐块发，⛔ 最后一块带 finish_reason。"""
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.end_headers()

        def emit(obj) -> None:
            h.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
                          .encode())
            h.wfile.flush()

        # ⭐ 切成几块：⚠️ 一次性发完的话「分块拼接」那条路径测不到
        step = max(1, len(text) // 3 or 1)
        for i in range(0, len(text), step):
            emit({"choices": [{"index": 0,
                               "delta": {"content": text[i:i + step]},
                               "finish_reason": None}]})
        emit({"choices": [{"index": 0, "delta": {},
                           "finish_reason": "stop"}], "usage": usage})
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()
        h.close_connection = True

    @staticmethod
    def _send(h: BaseHTTPRequestHandler, status: int, data: bytes) -> None:
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)
        h.wfile.flush()


def env_for(mock: MockLLM) -> dict[str, str]:
    """指向这个 mock 的环境变量。⭐ 除了端点，别的全走真代码。"""
    return {
        "AMB_EMBED_MODEL": "mock-embed",
        "AMB_EMBED_BASE_URL": mock.base_url,
        "AMB_EMBED_API_KEY_ENV": "AMB_MOCK_KEY",
        "AMB_EMBED_DIMS": str(DIM),
        "AMB_LLM_MODEL": "mock-llm",
        "AMB_LLM_BASE_URL": mock.base_url,
        "AMB_LLM_API_KEY_ENV": "AMB_MOCK_KEY",
        "AMB_MOCK_KEY": "not-a-real-key",
    }
