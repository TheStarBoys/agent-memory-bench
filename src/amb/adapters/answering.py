"""共用的答题代码。

⛔ 所有臂共用这一份：同一个 backbone、同一套提示、同一段答题逻辑。
**唯一允许不同的是记忆层本身**——否则「差别只来自记忆层」就不成立，
Δ 也就不可信了（docs/baselines.md）。
"""

from __future__ import annotations

from amb.adapters.llm import LLMClient
from amb.core import Answer, Entry, Usage

#: ⚠️ 提示是判分口径的一部分。改它等于换尺子，两次跑不可比。
SYSTEM = (
    "你是一个问答助手。只依据给出的资料回答，用最简短的词或短语作答，"
    "不要解释、不要复述问题。"
    "如果资料里没有答案，只回答四个字：资料未提及。"
)

NO_CONTEXT = "（没有可用资料）"


def build_prompt(question: str, entries: list[Entry]) -> str:
    if not entries:
        body = NO_CONTEXT
    else:
        body = "\n".join(f"[{i + 1}] {e.digest}" for i, e in enumerate(entries))
    return f"资料：\n{body}\n\n问题：{question}\n答案："


def answer_with(client: LLMClient, question: str, entries: list[Entry]) -> Answer:
    text = client.complete(SYSTEM, build_prompt(question, entries))
    return Answer(text=text, used=[e.id for e in entries])


def usage_of(client: LLMClient) -> list[Usage]:
    m = client.meter
    return [Usage(phase="probe", tokens_in=m.tokens_in,
                  tokens_out=m.tokens_out, llm_calls=m.calls)]
