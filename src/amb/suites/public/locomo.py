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
from dataclasses import dataclass, field
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
    #: conversation_id → dia_id → 那次会话的日期时间原文。
    #: ⚠️ 缺日期的会话不在这张表里——⛔ 不编一个。
    dated: dict[str, dict[str, str]] = field(default_factory=dict)

    def undated(self) -> int:
        """没有会话日期的轮次有多少条。⛔ 进报告，⚠️ 不静默。"""
        return sum(len(v) - len(self.dated.get(k, {}))
                   for k, v in self.turns.items())


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

    dated: dict[str, dict[str, str]] = {}
    for conv in raw:
        cid = str(conv.get("sample_id", len(turns)))
        body = conv.get("conversation") or {}
        turns[cid] = {}
        order[cid] = []
        dated[cid] = {}
        for key in sorted(k for k in body
                          if k.startswith("session_") and isinstance(body[k], list)):
            # ⭐ 会话日期：`session_3_date_time` = "1:56 pm on 8 May, 2023"。
            # ⛔ 早先它被丢掉了，于是**时间推理那一类构造性不可答**——
            # 问 `When did Jon go to a fair?`（gold `24 April, 2023`），
            # ⚠️ 所有臂拿到的资料里根本没有日期。
            # ⭐ 检索档那一类还有 0.571~0.643 分（捞对了轮次，轮次里没日期），
            # ⛔ 那一格把「检索到证据 ≠ 答得对」演示到了极致。
            when = str(body.get(f"{key}_date_time", "") or "")
            for turn in body[key]:
                dia = turn.get("dia_id")
                if not dia:
                    continue
                said = f"{turn.get('speaker', '')}: {turn.get('text', '')}"
                # ⚠️ 日期放进**摄入单元本身**，不是放进一条单独的文档：
                # ⭐ 上游数据里会话头就管着这一段的每一轮，那是原文的意思；
                # ⛔ 单独放一条会凭空造出一道「把轮次连到日期」的连接题，
                # 而那道题不在这个题库里。
                turns[cid][dia] = f"[{when}] {said}" if when else said
                if when:
                    dated[cid][dia] = when
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
    return LocomoData(questions=questions, turns=turns, order=order,
                      dated=dated)


def documents_for(data: LocomoData, conversation_ids: set[str],
                  max_turns: int | None = None) -> list[Document]:
    """把对话轮变成摄入单元。⚠️ doc_id 用 `<对话>/<轮次>`，判检索靠它。

    ⚠️ max_turns 与 pick() 的必须一致——⛔ 不一致会让题指向不存在的语料。
    """
    return [
        Document(doc_id=f"{cid}/{dia}", text=data.turns[cid][dia], kind="turn")
        for cid in sorted(conversation_ids)
        for dia in (data.order[cid][:max_turns] if max_turns else data.order[cid])
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


class LocomoAnswerSuite:
    """⭐ 回答档：检索到证据 ≠ 答得对。

    ⛔ 与 [`LocomoRetrievalSuite`](#) 量的**不是同一件事**：
    那一档只问「证据捞到没有」，这一档问「捞到之后答对没有」。
    ⚠️ 两档的数不可互比，报分时必须写成「<系统> + <backbone>」——
    这一档含答案生成器，只报系统名等于把别人的生成器算进自己的成绩。

    ⛔ 判分沿用 [`qa` 那把尺](../../scoring/metrics.py)：逐字比对，不用评委。
    ⚠️ 它会**系统性漏判**——LoCoMo 的 gold 有 35% 长过 25 字符
    （实测：gold `September, 2023`，答 `September`，严格比对判错）。
    ⭐ 所以额外报一个**宽松准确率**当判分上界，两者的差就是这把尺的不确定度。
    ⛔ 但排名一律看严格那个：宁可漏判成错，不靠判分的宽松度刷分。
    """

    name: ClassVar[str] = "locomo_answer"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.ANSWER})

    def __init__(self, questions: list[LocomoQA]) -> None:
        self._questions = questions

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        from amb.core import Answer, Failed, Unsupported

        run = SuiteRun(self.name, "scored")
        for q in self._questions:
            got = adapter.answer(q.question)
            if isinstance(got, Unsupported):
                return SuiteRun(self.name, "unsupported", reason=got.reason)
            if isinstance(got, Failed):
                run.failed += 1        # ⛔ 计入分母，记为未答对
                continue
            assert isinstance(got, Answer)
            run.observations.append(Observation(q.qa_id, {
                "text": got.text,
                # ⛔ 弃权题没有 gold，只有 adversarial_answer——这里给空
                "gold": [q.answer] if q.answer else [],
                "unanswerable": q.unanswerable,
                "category": q.category,
                "stratum": q.stratum,
                "used": len(got.used),
            }))
        return run


def pick(data: LocomoData, spec: SampleSpec,
         max_conversations: int | None = None,
         max_turns: int | None = None,
         conversations: tuple[str, ...] = ()) -> SampleResult:
    """抽题。⚠️ 结果的 provenance 要进报告。

    ⛔ `max_conversations` 控的是**语料量**，与题数是两件事：
    抽 20 道题可能碰到全部 10 个对话 → 摄入 5882 轮。
    对 mem0 这种每条都调 LLM 的系统，**语料量才是那个约束**
    （实测：不限的话要跑几小时）。

    ⚠️ 先限语料再抽题——⛔ 反过来会抽出没有语料的题。

    `max_turns` 比 `max_conversations` 更细：⚠️ mem0 每条 add() 要多轮 LLM
    调用（抽取+比对+裁决），实测 369 轮要跑几小时——
    ⭐ 一个对话仍然太大，所以要能按轮数截。

    ⛔ 截了语料就必须**丢掉 evidence 落在被截部分的题**，
    否则那些题必然全错，分数是假的。丢了几道进 provenance。
    """
    import random as _r

    questions = data.questions
    notes: list[str] = []

    if conversations:
        # ⭐ 点名。⚠️ 各对话的**题目产出差 2.5 倍**（conv-30 给 105 题，
        # conv-42 给 258 题），随机抽会白付摄入成本。
        # ⛔ 这是抽样决定，不是挑结果：选的依据是「每份摄入能判多少题」，
        # ⚠️ 在看到任何分数之前就定了，且必须进 provenance。
        unknown = sorted(set(conversations) - set(data.turns))
        if unknown:
            raise KeyError(f"没有这些对话：{unknown}。"
                           f"已知：{sorted(data.turns)}")
        keep = set(conversations)
        questions = [q for q in questions if q.conversation_id in keep]
        notes.append(f"conversations={','.join(sorted(keep))}")
    elif max_conversations is not None:
        convs = sorted(data.turns)
        keep = set(_r.Random(spec.seed).sample(
            convs, k=min(max_conversations, len(convs))))
        questions = [q for q in questions if q.conversation_id in keep]
        notes.append(f"max_conversations={max_conversations}（随机抽）")

    # ⛔ evidence 不在语料里的题一律丢掉——⚠️ 留着它们**必然全错**，
    # 那不是系统答砸了，是这道题在这份语料上不可能命中（假的失分）。
    # ⭐ 三种成因分开记：合并成一个数就读不出该修哪一边。
    kept_turns = {cid: (set(order[:max_turns]) if max_turns is not None
                        else set(order))
                  for cid, order in data.order.items()}
    if max_turns is not None:
        notes.append(f"max_turns={max_turns}")

    causes = {"no_evidence": 0, "dangling": 0, "truncated": 0}
    keep_q: list[LocomoQA] = []
    for q in questions:
        known = set(data.turns.get(q.conversation_id, {}))
        kept = kept_turns.get(q.conversation_id, set())
        if not q.evidence:
            # ⚠️ 上游就没给证据（实测 4 条，全是开放域推断）。
            # ⛔ 留着的话「命中任一率」把它记成一次未命中——而它无从命中。
            causes["no_evidence"] += 1
        elif not set(q.evidence) <= known:
            # ⛔ 上游数据里那个 id 根本不存在（实测 9 条：`D`、`D:11:26`、
            # `D8:6; D9:17` 两个 id 挤在一个串里）。⚠️ 它会把 evidence_recall
            # 的分母顶大——⭐ 那是上游的数据问题，不是谁答砸了。
            causes["dangling"] += 1
        elif not set(q.evidence) <= kept:
            causes["truncated"] += 1        # ⚠️ 我们自己截语料截掉的
        else:
            keep_q.append(q)
    questions = keep_q
    notes += [f"dropped_{k}={v}" for k, v in causes.items() if v]

    got = sample(questions, spec, key=lambda q: q.qa_id,
                 stratum=lambda q: q.stratum)
    got.spec_note = " ".join(notes)
    return got
