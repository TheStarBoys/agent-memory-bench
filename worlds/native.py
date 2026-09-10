"""⭐ 自研题库：**够格下结论**的那一个。

## ⛔ 它跟 `toy` 的区别

⚠️ `toy` 自己的注释写着「它**不是**一个够格的题库——题量太小，
统计上说明不了任何事」。⛔ 而它此前是 N1–N8 **唯一**能跑的世界，
于是那八档的每一个数底下都是一个自称不合格的世界。

⭐ 这里按仓库自己的样本量公式定题量（`required_n` / 配对口径）：

| 档 | toy | 这里 | 够辨 |
|---|---:|---:|---|
| n1 | **3** | 128 | 0.05 |
| n4 | **4** | 90 | 0.10 |
| n5 | 30 | 80 | 0.10 |
| n6 | 112 | 112 | 0.05 |
| n8 | **12** | 64 | 0.10 |
| retrieval / qa / n2 | 38~42 | 135 | 0.05 |

⚠️ 语料 1009 篇（toy 623）。⛔ 摄入贵的是 `mem0`（约 163 分钟），
⭐ 但摄入快照让那一笔**只付一次**。

## ⚠️ 仍然是合成的，这一条要说清楚

⛔ N1–N8 量的是**结构**：扇形度、删除是否彻底、例外压不压得过规律……
⚠️ 自然语料里这些东西是碰巧长成的、不可控的——⭐ 要测「关联度从 1 涨到 64
会发生什么」，必须**构造**。所以合成不是妥协，是这一类测量的前提。

⛔ 但**合成不等于够格**：够格靠的是题量、四类变更成比例、
有反方向的对照面（`holds` 那一侧、答不对的题）——⭐ 这些这里都有，
而 `toy` 都没有。

⚠️ 它**不能**替代真语料上的检索/问答：⭐ 那是 `locomo` 的活（1973 题）。
"""

from __future__ import annotations

from amb.core import Claim, Document
from amb.suites.native.n1_reality import (
    PromptedRealitySuite,
    SpontaneousRealitySuite,
)
from amb.suites.native.n2_provenance import ProvenanceSuite, SpanProbe
from amb.suites.native.n3_reasoning import ReasoningSuite, questions_from
from amb.suites.native.n4_governance import DeletionProbe, GovernanceSuite
from amb.suites.native.n5_consolidation import (
    ObservedRetentionSuite,
    SelfReportedRetentionSuite,
    probes_from,
)
from amb.suites.native.n6_structure import StructureSuite
from amb.suites.native.n7_calibration import CalibrationItem, CalibrationSuite
from amb.suites.native.n8_induction import InductionSuite
from amb.suites.native.qa import QAItem, QASuite
from amb.suites.native.retrieval import Query, RetrievalSuite
from amb.world import FileSpec, WorldManifest
from amb.world.stream import corpus as _corpus
from amb.world.stream import events as _events
from amb.world.stream import factgraph as _factgraph
from amb.world.stream import governance as _governance
from amb.world.stream import need as _need
from amb.world.stream import reality as _reality
from amb.world.stream import regularity as _regularity
from amb.world.stream import topology as _topology

SEED = 42
CLOCK_START = "2026-01-01T00:00:00Z"
_SPAN_S = 86_400 * 30.0

# ── ⭐ 各档的生成器：⚠️ 题量由 required_n 定，⛔ 不拍脑袋 ────────
REALITY = _reality.build(seed=SEED, n_claims=128)
GOV_DOCS, _SECRETS = _governance.build(seed=SEED, n_probes=90)
#: ⭐ 生成器只出数据，⛔ 转成套件的题在这一层做——⚠️ `world` 不许依赖 `suites`。
DELETION_PROBES = [DeletionProbe(doc_id=s.doc_id, text=s.text,
                                 marker=s.marker, query=s.query)
                   for s in _SECRETS]
FACT_GRAPH = _factgraph.build(seed=SEED, chains=16, depth=3)
EVENT_STREAM = _events.build(seed=SEED, span_s=_SPAN_S, per_cell=8)
TOPOLOGY = _topology.build(seed=SEED, entities_per_fan=2)
#: ⭐ 12 档成立率 × 每档 4 条 = 48 题。⚠️ toy 只有 12 题（结构上下不了结论）。
#:
#: ⛔ **为什么不是 64 档各一条**：⚠️ 成立率是用 `round(seen × rate)` 实现的，
#: `seen_per_rate=12` 时只有 7 个可达的档位——⭐ 64 个不同的率里大部分
#: 落到同一个实现值上，那是**假的**分辨率。
#: ⭐ 同一档放 4 条独立规律更划算：⚠️ 题数一样多，而语料少一半
#: （每条规律要 `seen+1` 篇），⛔ 且「规律强度单调性」有了重复观测反而更稳。
_RATES = tuple(sorted(
    tuple(round(0.50 + i * (0.5 / 11), 3) for i in range(12)) * 4))
REGULARITIES = _regularity.build(seed=SEED, rates=_RATES, seen_per_rate=12)
CORPUS_GEN = _corpus.build(seed=SEED, entities=45, attrs_per_entity=3)

#: ⛔ 占位曲线：参数不是我们拟合的，⚠️ 拿它跑出来的 N5 分数**不得发布**。
NEED_CURVE = _need.PLACEHOLDER

CLAIMS = REALITY.claims
TRUTH = REALITY.truth
CHANGES = REALITY.changes

MANIFEST = WorldManifest(
    name="native", seed=SEED, clock_start=CLOCK_START,
    files=tuple(FileSpec(d.doc_id, d.text) for d in REALITY.documents),
    facts=dict(REALITY.facts),
)

#: ⚠️ 进世界的只有 N1 那批——⛔ 其余都是「被记住的东西」，不是外部现实。
DOCUMENTS = [
    Document(doc_id=d.doc_id, text=d.text, timestamp=CLOCK_START,
             principal=d.principal, kind="document")
    for d in REALITY.documents
]


def extra_documents() -> list[Document]:
    """只进记忆、不进世界的语料。⛔ 混进世界会污染 N1 的「文件还在不在」。"""
    docs: list[Document] = [
        Document(doc_id=d.doc_id, text=d.text, timestamp=CLOCK_START,
                 principal=d.principal, kind="document") for d in GOV_DOCS
    ]
    for t in FACT_GRAPH.facts:
        docs.append(Document(doc_id=str(t), text=t.sentence(),
                             timestamp=CLOCK_START, principal="alice"))
    for f in EVENT_STREAM.facts:
        docs.append(Document(doc_id=f.fact_id, text=f.text,
                             timestamp=CLOCK_START, principal="alice"))
    for f in TOPOLOGY.facts:
        docs.append(Document(doc_id=f.fact_id, text=f.text,
                             timestamp=CLOCK_START, principal="alice"))
    for reg in REGULARITIES:
        for i, inst in enumerate(reg.seen):
            docs.append(Document(doc_id=f"{reg.category}#{i}",
                                 text=inst.statement(reg.prop),
                                 timestamp=CLOCK_START, principal="alice"))
    # ⭐ 它轮流分主体，⚠️ N4 的隔离才有真实的对照面
    docs += CORPUS_GEN.documents(clock=CLOCK_START)
    return docs


def all_documents() -> list[Document]:
    return [*DOCUMENTS, *extra_documents()]


QUERIES = [Query(f"c{i}", f.question, frozenset({f.doc_id}))
           for i, f in enumerate(CORPUS_GEN.facts)]

QA_ITEMS = [*(QAItem(f"c{i}", f.question, (f.value,))
              for i, f in enumerate(CORPUS_GEN.facts)),
            # ⚠️ 该弃权的题：⛔ 问库里**从来没有**的实体。
            # ⭐ 不用「被删掉的那条」——那考的是记忆过时（N1 的活）。
            *(QAItem(f"cx{i}", f"E999_{i}的配额是多少？", (), unanswerable=True)
              for i in range(15))]

SPAN_PROBES = [SpanProbe(f"c{i}", f.question, f.doc_id,
                         f.value_start, f.value_end)
               for i, f in enumerate(CORPUS_GEN.facts)]

#: ⚠️ N2 判分要拿原文比对——⛔ 生成的那批也得进这张表
CORPUS = {**{d.doc_id: d.text for d in REALITY.documents},
          **{f.doc_id: f.text for f in CORPUS_GEN.facts}}

#: ⭐ N7 **两半各占一半**：⛔ 只放答得对的，ECE 就退化成
#: 「1 − 平均置信度」——⚠️ 一条把置信度钉在基线准确率上的臂拿满分。
_CAL_OK = tuple((f.question, (f.value,)) for f in CORPUS_GEN.facts[:20])
_CAL_BAD = tuple((f"Z999_{i} 一次最多能用多少？", ("__不可能答对__",))
                 for i in range(20))


def suites(rebuild=None, world_handle=None) -> list:
    return [
        RetrievalSuite(QUERIES),
        ProvenanceSuite(SPAN_PROBES, CORPUS),
        PromptedRealitySuite(CLAIMS, TRUTH),
        SpontaneousRealitySuite(CLAIMS, TRUTH),
        QASuite(QA_ITEMS),
        # ⚠️ N4 第 3 步要重开适配器——⛔ 没给 rebuild 就不跑这一档
        *([] if rebuild is None else [
            GovernanceSuite(DELETION_PROBES, rebuild, world_handle),
        ]),
        ReasoningSuite(FACT_GRAPH, questions_from(FACT_GRAPH)),
        # ⛔ 曲线一并传进去：⚠️ 它是占位的，这两档的数**不得发布**，
        # ⭐ 而那句话必须跟着分走到报告。
        ObservedRetentionSuite(probes_from(EVENT_STREAM, NEED_CURVE,
                                           now_s=_SPAN_S), NEED_CURVE),
        SelfReportedRetentionSuite(probes_from(EVENT_STREAM, NEED_CURVE,
                                               now_s=_SPAN_S), NEED_CURVE),
        StructureSuite(TOPOLOGY),
        CalibrationSuite([
            *(CalibrationItem(f"k{i}", q, g, salient=(i % 2 == 0))
              for i, (q, g) in enumerate(_CAL_OK)),
            *(CalibrationItem(f"kx{i}", q, g, salient=(i % 2 == 1))
              for i, (q, g) in enumerate(_CAL_BAD)),
        ]),
        InductionSuite(REGULARITIES),
    ]
