"""LoCoMo 接入。

⭐ 它是唯一能**纯粹量检索质量**的公开题库：每题带 `evidence`
（ground-truth 轮次 id），所以不生成答案也能判分。

⚠️ 五类题占比悬殊：4 单跳 42% · **5 弃权 22%** · 2 时间 16% ·
1 多跳 14% · 3 开放域 5%。
⛔ **22% 是弃权题，比多跳还多**——只会返回 top-k 的系统在这一类上必然全错。

⛔ 许可 NOASSERTION：只读数据，代码不复制进本仓库。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from amb.core import Adapter, Capability, Document, Observation, SuiteRun
from amb.suites.public.sampling import SampleResult, SampleSpec, sample
from amb.suites.public.spec import DatasetMissing
from amb.world import WorldState

#: 类别 → 人话。⚠️ 与 docs/benchmarks.md 的表一致。
CATEGORIES = {
    1: "多跳", 2: "时间推理", 3: "开放域推断", 4: "单跳事实", 5: "弃权",
}

DATA = Path(".external/locomo/data/locomo10.json")


@dataclass(frozen=True, slots=True)
class LocomoQA:
    qa_id: str
    conversation_id: str
    question: str
    #: ⛔ 弃权题没有 answer，只有 adversarial_answer
    answer: str | None
    category: int
    #: ground-truth 轮次 id，⭐ 判检索靠它
    evidence: tuple[str, ...]

    @property
    def unanswerable(self) -> bool:
        return self.category == 5

    @property
    def stratum(self) -> str:
        return f"{self.category}-{CATEGORIES.get(self.category, '?')}"


@dataclass
class LocomoData:
    questions: list[LocomoQA]
    #: conversation_id → 该对话的全部轮次（dia_id → 文本）
    turns: dict[str, dict[str, str]]
    #: 每个 dia_id 属于哪一次会话，⚠️ 摄入时要按顺序
    order: dict[str, list[str]]


def load(path: Path = DATA) -> LocomoData:
    """⛔ 数据不在就抛异常，不是返回空。"""
    if not path.is_file():
        raise DatasetMissing(
            f"LoCoMo 数据不在 {path}。⛔ 该题库记「未接入」，不是 0 分。\n"
            f"    python -m amb.cli setup locomo"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions: list[LocomoQA] = []
    turns: dict[str, dict[str, str]] = {}
    order: dict[str, list[str]] = {}

    for conv in raw:
        cid = str(conv.get("sample_id", len(turns)))
        body = conv.get("conversation") or {}
        turns[cid] = {}
        order[cid] = []
        for key in sorted(k for k in body
                          if k.startswith("session_") and isinstance(body[k], list)):
            for turn in body[key]:
                dia = turn.get("dia_id")
                if not dia:
                    continue
                turns[cid][dia] = f"{turn.get('speaker', '')}: {turn.get('text', '')}"
                order[cid].append(dia)

        for i, qa in enumerate(conv.get("qa") or []):
            questions.append(LocomoQA(
                qa_id=f"{cid}#{i}",
                conversation_id=cid,
                question=str(qa.get("question", "")),
                answer=(str(qa["answer"]) if qa.get("answer") is not None else None),
                category=int(qa.get("category", 0)),
                evidence=tuple(str(e) for e in (qa.get("evidence") or [])),
            ))
    return LocomoData(questions=questions, turns=turns, order=order)


def documents_for(data: LocomoData, conversation_ids: set[str]) -> list[Document]:
    """把对话轮变成摄入单元。⚠️ doc_id 用 `<对话>/<轮次>`，判检索靠它。"""
    return [
        Document(doc_id=f"{cid}/{dia}", text=data.turns[cid][dia], kind="turn")
        for cid in sorted(conversation_ids)
        for dia in data.order[cid]
    ]


class LocomoRetrievalSuite:
    """⭐ 只判检索：不生成答案，靠 evidence 对账。

    ⛔ 这一档不需要 backbone——所以它量的是**记忆层本身**，
    没有把别人的生成器算进成绩。
    """

    name: ClassVar[str] = "locomo_retrieval"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.SEARCH})

    def __init__(self, questions: list[LocomoQA], k: int = 10) -> None:
        self._questions = questions
        self._k = k

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for q in self._questions:
            hits = adapter.search(q.question, self._k)
            got = {d.split("/", 1)[-1] for h in hits for d in h.doc_ids}
            gold = set(q.evidence)
            run.observations.append(Observation(q.qa_id, {
                "category": q.category,
                "stratum": q.stratum,
                "unanswerable": q.unanswerable,
                "gold": sorted(gold),
                "hit": sorted(gold & got),
                "retrieved": len(got),
            }))
        return run


def pick(data: LocomoData, spec: SampleSpec,
         max_conversations: int | None = None) -> SampleResult:
    """抽题。⚠️ 结果的 provenance 要进报告。

    ⛔ `max_conversations` 控的是**语料量**，与题数是两件事：
    抽 20 道题可能碰到全部 10 个对话 → 摄入 5882 轮。
    对 mem0 这种每条都调 LLM 的系统，**语料量才是那个约束**
    （实测：不限的话要跑几小时）。

    ⚠️ 先限对话再抽题——⛔ 反过来会抽出没有语料的题。
    """
    import random as _r

    questions = data.questions
    if max_conversations is not None:
        convs = sorted(data.turns)
        keep = set(_r.Random(spec.seed).sample(
            convs, k=min(max_conversations, len(convs))))
        questions = [q for q in questions if q.conversation_id in keep]

    got = sample(questions, spec, key=lambda q: q.qa_id,
                 stratum=lambda q: q.stratum)
    if max_conversations is not None:
        got.spec_note = f"max_conversations={max_conversations}"
    return got
