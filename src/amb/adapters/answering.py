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


def fit_to_window(entries: list[Entry], budget_chars: int | None
                  ) -> tuple[list[Entry], int]:
    """把交给模型的材料裁到窗口里。⭐ 返回 (留下的, 被裁掉的条数)。

    ## ⛔ 为什么这必须对**所有臂**一视同仁

    ⚠️ 窗口是**受控变量**：⛔ 只约束 `recency_window` 而放任别的臂
    随便塞，那扫描就不成立——⭐ 那时测的是「谁被限制了」，
    不是「谁选得准」。

    ⭐ 而这正是记忆层的价值所在：⚠️ 窗口一小，能塞的材料就少，
    ⛔ **选得准的臂掉得慢**。窗口不限时人人都能把 top-10 全交出去，
    那时候「会选」这件事量不到。

    ⚠️ 按**顺序**保留：⛔ 检索臂的第 1 条是它最有把握的那条，
    ⭐ 裁掉尾巴才是「窗口不够时它会失去什么」。

    ## ⛔ 但它在 library 档**现在不起作用**，这一条要说清楚

    ⚠️ 实测（2026-09-10）：`answer_k=5`，每条 digest 约 25 码点，
    合计 ~123 码点——⛔ 窗口卡到 120 也只砍掉 1 条，分数纹丝不动。

    ⭐ 成因是结构性的：**library 档从不把全量交给模型**，只交 top-k。
    ⚠️ 所以那一档没有「装不下」这回事——⛔ 瓶颈是检索质量，不是窗口。

    ⚠️ 那这个函数留着干什么：⭐ 它挡住将来出现的长 digest 臂
    （摘要型、整篇返回型）在窗口收紧时白占便宜。
    ⛔ **不要拿它当「我们控制了上下文窗口」的证据**——
    ⚠️ 真正模拟窗口失效的是 `recency_window` 那条臂。
    """
    if budget_chars is None or budget_chars <= 0:
        return entries, 0
    kept: list[Entry] = []
    used = 0
    for e in entries:
        size = len(e.digest or "")
        if kept and used + size > budget_chars:
            break
        kept.append(e)
        used += size
    return kept, len(entries) - len(kept)


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
                prompt: Prompt = ZH,
                budget_chars: int | None = None) -> Answer:
    """⚠️ `budget_chars`：⛔ 上下文窗口，**所有臂共用同一个值**。

    ⭐ `None` = 不限（默认）：⚠️ 那时窗口不是约束，
    ⛔ 而「记忆有没有用」这件事量不到——见 `fit_to_window`。
    """
    kept, dropped = fit_to_window(entries, budget_chars)
    text = client.complete(prompt.system, build_prompt(question, kept, prompt))
    # ⛔ 被裁掉的要记下来：⚠️ 静默裁剪会让一条臂的分被读成
    # 「材料都给它了还只有这么高」。
    return Answer(text=text, used=[e.id for e in kept], dropped=dropped)


def usage_of(client: LLMClient) -> list[Usage]:
    m = client.meter
    return [Usage(phase="probe", tokens_in=m.tokens_in,
                  tokens_out=m.tokens_out, llm_calls=m.calls)]
