"""共用的答题代码。

⛔ 所有臂共用这一份：同一个 backbone、同一套提示、同一段答题逻辑。
**唯一允许不同的是记忆层本身**——否则「差别只来自记忆层」就不成立，
Δ 也就不可信了（docs/baselines.md）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from amb.adapters.llm import LLMClient
from amb.core import Answer, AnswerStyle, Entry, Usage

#: ⚠️ 提示是判分口径的一部分。改它等于换尺子，两次跑不可比。
#: ⛔ 但**一次跑里所有臂必须用同一个**——那才是「差别只来自记忆层」的前提。


@dataclass(frozen=True, slots=True)
class Prompt:
    """一套答题口径：提示 + 弃权词 + 无资料时的占位。

    ⛔ 语言必须跟题库走。⚠️ 实测踩过：中文提示 + 英文题库（LoCoMo），
    模型一律用中文答——`by dancing` 答成「跳舞」、`19 January, 2023`
    答成「昨天」，逐字比对全判错。⭐ 那不是记忆层不行，是**尺子在量语言**。
    ⛔ 加一句「用问题的语言作答」没用，实测反而更差（0.077 → 0.077，
    中文答案变多了）——⭐ 得换成英文提示才行（0.077 → 0.154）。
    """

    system: str
    #: ⭐ 允许外推那一套的 system。⛔ 没有默认值是**故意**的：
    #: 少写一个变体，声明它的套件就会静默退回默认口径——
    #: ⚠️ 那正是 N8 四条臂全 0.000 的成因，不能让它再发生一次。
    inductive: str
    #: ⛔ 判分要逐字认的那个弃权词
    abstain: str
    no_context: str
    #: 资料 / 问题 / 答案三个标签——⚠️ 也得跟着语言走
    labels: tuple[str, str, str]

    def styled(self, style: AnswerStyle) -> "Prompt":
        """换成这个套件要的口径。⛔ 只换 system——语言、标签、弃权词不动。

        ⚠️ 换的是**要求**，不是语言：语言仍然跟题库走。
        """
        if style is AnswerStyle.INDUCTIVE:
            return replace(self, system=self.inductive)
        return self


ZH = Prompt(
    system=(
        "你是一个问答助手。只依据给出的资料回答，用最简短的词或短语作答，"
        "不要解释、不要复述问题。"
        "如果资料里没有答案，只回答四个字：资料未提及。"
    ),
    inductive=(
        "你是一个问答助手。资料里是一批同类个体的观察记录。"
        "先从这些记录里归纳出这一类通常成立的规律，再回答问题。"
        "用最简短的词或短语作答，不要解释、不要复述问题。"
        "如果资料里没有直接提到问题问的那个个体，就按归纳出的规律推断作答，"
        "不要回答「资料未提及」。"
        "但如果资料里明确写了那个个体的情况，以资料写的为准。"
    ),
    abstain="资料未提及",
    no_context="（没有可用资料）",
    labels=("资料", "问题", "答案"),
)

EN = Prompt(
    system=(
        "You are a question-answering assistant. Answer ONLY from the material "
        "below. Reply with the shortest possible word or phrase — no "
        "explanation, no restating the question. If the material does not "
        "contain the answer, reply exactly: NOT IN THE MATERIAL."
    ),
    inductive=(
        "You are a question-answering assistant. The material below is a set "
        "of observations about individuals of the same kind. First infer the "
        "general pattern that usually holds for them, then answer. Reply with "
        "the shortest possible word or phrase — no explanation. If the "
        "material does not mention the individual the question asks about, "
        "answer by extrapolating from the pattern — do NOT reply that the "
        "material does not say. But when the material states something "
        "explicitly about that individual, the explicit statement wins."
    ),
    abstain="NOT IN THE MATERIAL",
    no_context="(no material available)",
    labels=("Material", "Question", "Answer"),
)

#: 题库 → 口径。⛔ 一次跑里所有臂共用同一个，⚠️ 且必须进报告。
BY_BENCH = {"locomo": EN, "toy": ZH}


def for_bench(bench: str) -> Prompt:
    """⚠️ 认不出的题库退回中文那套——⛔ 不猜语言。"""
    return BY_BENCH.get(bench, ZH)


#: ⚠️ 兼容旧调用点：不传 prompt 时用中文那套
SYSTEM = ZH.system
NO_CONTEXT = ZH.no_context


def build_prompt(question: str, entries: list[Entry],
                 prompt: Prompt = ZH) -> str:
    if not entries:
        body = prompt.no_context
    else:
        body = "\n".join(f"[{i + 1}] {e.digest}" for i, e in enumerate(entries))
    material, question_label, answer_label = prompt.labels
    return (f"{material}:\n{body}\n\n"
            f"{question_label}: {question}\n{answer_label}:")


def answer_with(client: LLMClient, question: str, entries: list[Entry],
                prompt: Prompt = ZH) -> Answer:
    text = client.complete(prompt.system, build_prompt(question, entries, prompt))
    return Answer(text=text, used=[e.id for e in entries])


def usage_of(client: LLMClient) -> list[Usage]:
    m = client.meter
    return [Usage(phase="probe", tokens_in=m.tokens_in,
                  tokens_out=m.tokens_out, llm_calls=m.calls)]
