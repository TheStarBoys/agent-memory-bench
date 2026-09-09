"""离线跑真流水线：⛔ **假在网络出口，不假在我们自己的代码上**。

## 为什么要有这一层

⚠️ 本次会话 7 个 bug **全部**只有真跑才发现，而全套 713 个测试是绿的。
⛔ 查下来成因是同一个：**测试不走生产那扇门**。

    真正的入口   build(臂名) → run_one(五阶段) → score
    测试走的     create(手填参数) 或 手搓一个假臂

⭐ `build()` 正是 env 变量、storage_dir、embedding 配置、臂名分派的所在地——
⚠️ 也正是那 7 个 bug 全部的所在地。而它此前只被 `null` / `bm25` /
`host_default` 走过：⛔ 恰好是三条**不需要外部配置**的臂。

## ⭐ 接缝选在 `urlopen`

⛔ 不假 `EmbeddingClient` / `LLMClient` 本身：⚠️ 那会把**分批、重试、
计量、缓存**一起蒙掉——而那些正是 bug 爱藏的地方。
⭐ 假在它们的网络出口，于是那些逻辑全都还是被测的真代码。

⚠️ 按 Google 的尺度这是 **medium**：可以碰文件系统，⛔ 不许碰网络。
⭐ 秒级，所以它能进 pre-commit——而真跑要 2.5 小时。
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field


#: ⚠️ 维度取小：⛔ 2560 维会让这一层慢下来，⭐ 而它测的是接线不是语义。
DIM = 16

#: ⭐ 假端点的主机名。⛔ 只有发往它的请求才被拦——⚠️ 其余（世界服务器）透传。
_HOST = "offline.invalid"


@dataclass
class Wire:
    """记下**每一次**出网请求。⭐ 断言「没重跑摄入」靠它。"""

    embed_calls: list[int] = field(default_factory=list)
    chat_calls: int = 0

    @property
    def embed_texts(self) -> int:
        return sum(self.embed_calls)

    @property
    def biggest_embed_batch(self) -> int:
        """⭐ 摄入是**批量**的，答题是一条一条的——⛔ 这个数分得开两者。"""
        return max(self.embed_calls, default=0)


def _vector(text: str) -> list[float]:
    """确定性假向量：⭐ 按字符算，⛔ 刻意不随机。

    ⚠️ 随机向量下排名是任意的，那样断言「原文能查到自己」就成了掷硬币。
    """
    v = [0.0] * DIM
    for ch in text:
        v[ord(ch) % DIM] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


#: ⭐ 答题用：从提示里把资料抠出来，⛔ 挑与问题字符重合最多的一句。
#: ⚠️ 它不聪明，也不该聪明——这一层测的是**接线**，⛔ 不是回答质量。
def _answer(user: str) -> str:
    lines = [ln.strip(" -·") for ln in user.splitlines() if len(ln.strip()) > 4]
    question = lines[-1] if lines else ""
    q = set(question)
    best, score = "", 0
    for ln in lines[:-1] if len(lines) > 1 else lines:
        overlap = len(q & set(ln))
        if overlap > score:
            best, score = ln, overlap
    if not best:
        return "资料未提及"
    # ⚠️ 截短：⛔ 判分按包含匹配，整段返回会让所有题都「答对」——
    # ⭐ 那是假的满分，比 0 分更糟。
    m = re.search(r"[：:]\s*(\S{1,20})", best)
    return m.group(1) if m else best[:20]


def install(monkeypatch, wire: Wire | None = None) -> Wire:
    """把网络出口换成确定性的假实现。⭐ 返回的 `Wire` 记着所有调用。"""
    wire = wire or Wire()

    import urllib.request

    real = urllib.request.urlopen

    def _fake_urlopen(req, timeout=None):
        # ⭐ **只截我们那两个假端点**：⚠️ 世界服务器（钟 / 事实）也走 urlopen，
        # ⛔ 而它是**本机的真服务**，是被测系统的一部分——截了就等于把
        # N1 那一整档的「世界现在什么样」换成假的。
        # ⚠️ 按 Google 的尺度，medium 测试允许 localhost。
        url = getattr(req, "full_url", req if isinstance(req, str) else "")
        if _HOST not in url:
            return real(req, timeout=timeout)
        body = json.loads(req.data.decode())
        if "embeddings" in url:
            texts = body["input"]
            texts = [texts] if isinstance(texts, str) else texts
            wire.embed_calls.append(len(texts))
            payload = {"data": [{"embedding": _vector(t)} for t in texts]}
        else:
            wire.chat_calls += 1
            user = next((m["content"] for m in reversed(body["messages"])
                         if m["role"] == "user"), "")
            payload = {
                "choices": [{"message": {"content": _answer(user)}}],
                # ⭐ 带上用量：⚠️ 计量层也要被测到，⛔ 空的会让成本档静默为 0
                "usage": {"prompt_tokens": len(user) // 4 or 1,
                          "completion_tokens": 8},
            }
        return _Resp(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    return wire


class _Resp(io.BytesIO):
    """⚠️ `urlopen` 的返回值要能当上下文管理器用。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


#: ⭐ `build()` 要的 env。⚠️ 值是假的，⛔ 但**这条路径是真的**——
#: 那 7 个 bug 里有 3 个就住在这几行的下游。
ENV = {
    "AMB_EMBED_MODEL": "fake-embed",
    "AMB_EMBED_BASE_URL": "http://offline.invalid/v1",
    "AMB_EMBED_API_KEY_ENV": "AMB_FAKE_KEY",
    "AMB_EMBED_DIMS": str(DIM),
    "AMB_LLM_MODEL": "fake-llm",
    "AMB_LLM_BASE_URL": "http://offline.invalid/v1",
    "AMB_LLM_API_KEY_ENV": "AMB_FAKE_KEY",
    "AMB_FAKE_KEY": "not-a-real-key",
}


def use_env(monkeypatch, **extra: str) -> None:
    for k, v in {**ENV, **extra}.items():
        monkeypatch.setenv(k, v)
