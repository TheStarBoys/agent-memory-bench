"""玩具世界：给流水线用的最小可跑清单。

⚠️ 它**不是**一个够格的题库——题量太小，统计上说明不了任何事。
它存在的唯一目的是让五阶段先通起来，好尽早发现协议哪里不好用。
"""

from __future__ import annotations

from amb.core import Claim, Document
from amb.suites.native.n1_reality import (
    PromptedRealitySuite,
    SpontaneousRealitySuite,
)
from amb.suites.native.n2_provenance import ProvenanceSuite, SpanProbe
from amb.suites.native.n4_governance import DeletionProbe, GovernanceSuite
from amb.suites.native.n3_reasoning import ReasoningSuite, questions_from
from amb.suites.native.n5_consolidation import (
    ObservedRetentionSuite,
    SelfReportedRetentionSuite,
    probes_from,
)
from amb.suites.native.n6_structure import StructureSuite
from amb.suites.native.n7_calibration import CalibrationItem, CalibrationSuite
from amb.suites.native.n8_induction import InductionSuite
from amb.suites.native.qa import QAItem, QASuite
from amb.world.stream import corpus as _corpus
from amb.world.stream import events as _events
from amb.world.stream import factgraph as _factgraph
from amb.world.stream import need as _need
from amb.world.stream import regularity as _regularity
from amb.world.stream import topology as _topology
from amb.suites.native.retrieval import Query, RetrievalSuite
from amb.world import Change, ChangeKind, FileSpec, WorldManifest

CLOCK_START = "2026-01-01T00:00:00Z"

_FILES = {
    "notes/hippocampus.md": "海马体负责情节记忆的快速编码，一次暴露就能记住具体事件。",
    "notes/neocortex.md": "新皮层学得慢，靠反复暴露抽取跨情节的统计规律。",
    "notes/cat.md": "橘猫喜欢在窗台上晒太阳，一睡就是一下午。",
    "config/retention.txt": "retention_days=30",
}

SEED = 42

MANIFEST = WorldManifest(
    name="toy",
    seed=SEED,
    clock_start=CLOCK_START,
    files=tuple(FileSpec(p, t) for p, t in _FILES.items()),
    facts={"retention_days": "30", "region": "cn-north"},
)

DOCUMENTS = [
    Document(doc_id=path, text=text, timestamp=CLOCK_START, principal="alice",
             kind="document")
    for path, text in _FILES.items()
] + [
    # ⚠️ N4 的探针语料：只进记忆，不进世界——它是「被记住的东西」，不是外部现实
    Document(doc_id="secret/formula.md", text="内部配方编号 K-7391，仅限研发组。",
             timestamp=CLOCK_START, principal="alice", kind="document"),
]

def all_documents() -> list[Document]:
    return [*DOCUMENTS, *extra_documents()]

_HAND_QUERIES = [
    Query("q1", "哪个结构学得慢？", frozenset({"notes/neocortex.md"})),
    Query("q2", "什么动物喜欢晒太阳？", frozenset({"notes/cat.md"})),
    Query("q3", "一次就能记住靠什么？", frozenset({"notes/hippocampus.md"})),
    Query("q4", "保留多少天？", frozenset({"config/retention.txt"})),
]

#: ⚠️ 手写的两条留着——⛔ 它们的语料同时被 N1 引用，删了 N1 就没了参照。
#: ⭐ 生成的那批在 `SPAN_PROBES` 里追加（见文件末尾）。
_HAND_SPAN_PROBES = [
    SpanProbe("s1", "新皮层", "notes/neocortex.md", 0, len(_FILES["notes/neocortex.md"])),
    SpanProbe("s2", "橘猫", "notes/cat.md", 0, len(_FILES["notes/cat.md"])),
]

#: mutate 阶段的变更。⛔ 必须含一条「无关变更」当反方向。
#:
#: ⚠️ 配置文件与事实表代表同一个值，所以必须一起改——
#: 只改一边会让世界自相矛盾，命题就没有确定的真值了。
#: （第一次跑就是这么错的：只改了事实表，而命题引的是文件。）
CHANGES = [
    Change(ChangeKind.VANISH, "notes/cat.md"),                        # 消失
    Change(ChangeKind.REVALUE, "retention_days", "7"),                # 改值：事实表
    Change(ChangeKind.REVALUE, "config/retention.txt",
           "retention_days=7"),                                       # 改值：文件，同上
    Change(ChangeKind.IRRELEVANT, "notes/unrelated.md", "无关内容"),   # ⛔ 反方向
]

CLAIMS = [
    Claim("c1", "橘猫的笔记存在", ["notes/cat.md"]),
    Claim("c2", "保留天数是 30", ["config/retention.txt"]),
    Claim("c3", "新皮层学得慢", ["notes/neocortex.md"]),
]

#: 变更之后每条命题的真值。⚠️ c3 仍成立——它是「无关变更」那一侧的对照。
TRUTH = {"c1": "broken", "c2": "broken", "c3": "holds"}


#: 端到端答题。⛔ 短事实题——判分要确定性，不能用评委。
#: ⚠️ 答案都取自变更**之后**的世界，与 N1 的真值一致。
_HAND_QA_ITEMS = [
    QAItem("a1", "哪个脑结构学得慢，靠反复暴露抽取规律？", ("新皮层",)),
    QAItem("a2", "哪个脑结构一次暴露就能记住？", ("海马",)),
    # ⭐ 该弃权的题：语料里**从来没提过**这件事。
    # ⚠️ 刻意不用「被删掉的那份笔记」——那考的是记忆过时，是 N1 的活。
    # 混进来会让「编造率」同时含两种成因，读不出是哪一种。
    QAItem("a3", "海马体是哪一年被命名的？", (), unanswerable=True),
]


#: N4：一条要被删掉的记忆，marker 是它独有的特征子串。
#: ⚠️ 独有很重要——带外搜索搜到别的会误判成「没删干净」。
DELETION_PROBES = [
    DeletionProbe(doc_id="secret/formula.md",
                  text="内部配方编号 K-7391，仅限研发组。",
                  marker="K-7391", query="配方编号"),
]


# ── N3 / N5 / N6 / N7 / N8：各自的合成结构 ─────────────────────
#: ⚠️ 都很小，只够验机制。⛔ 题量上统计上说明不了任何事。
FACT_GRAPH = _factgraph.build(seed=SEED, chains=4, depth=3)
EVENT_STREAM = _events.build(seed=SEED, span_s=86_400 * 30.0, per_cell=3)
TOPOLOGY = _topology.build(seed=SEED, entities_per_fan=2)
REGULARITIES = _regularity.build(seed=SEED)

#: ⭐ `retrieval` / `qa` / N2 的语料——**生成的，不是手写的**。
#: ⛔ 手写夹具只有 4/3/2 题且互不相似，[实测](../docs/runs/2026-09-03-native-suites-first.md)
#: 三条臂分数完全相同——⚠️ 什么方法都能在互不相干的三篇里找对。
#: ⭐ 生成的这份**每条都有 35 条干扰**（同实体不同属性、同属性不同实体），
#: 要同时认出实体和属性才找得对。⚠️ 难度由 entities × attrs 调。
CORPUS_GEN = _corpus.build(seed=SEED, entities=12, attrs_per_entity=3)

#: ⛔ 占位曲线：参数不是我们拟合的，拿它跑出来的 N5 分数**不得发布**。
NEED_CURVE = _need.PLACEHOLDER


def extra_documents() -> list[Document]:
    """N3/N5/N6/N8 的语料。⚠️ 只进记忆，不进世界。"""
    docs: list[Document] = []
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
    # ⭐ 生成的检索语料。⚠️ 它**轮流分主体**（alice/bob/carol）——
    # ⛔ 手写那几篇 principal 全是 alice，N4 的隔离没有真实的对照面。
    docs += CORPUS_GEN.documents(clock=CLOCK_START)
    return docs


#: ⭐ 手写 + 生成。⛔ 手写的留着是因为 N1 引用了它们的语料。
#: ⭐ 生成器只出数据，转成套件的题在这一层做——⛔ `world` 不许依赖 `suites`。
QUERIES = [*_HAND_QUERIES, *(
    Query(f"c{i}", f.question, frozenset({f.doc_id}))
    for i, f in enumerate(CORPUS_GEN.facts))]

QA_ITEMS = [*_HAND_QA_ITEMS, *(
    QAItem(f"c{i}", f.question, (f.value,))
    for i, f in enumerate(CORPUS_GEN.facts)), *(
    # ⚠️ 该弃权的题：问库里**从来没有**的实体。
    # ⛔ 不用「被删掉的那条」——那考的是记忆过时（N1 的活），
    # 混进来会让「编造率」同时含两种成因，读不出是哪一种。
    QAItem(f"cx{i}", f"E999_{i}的配额是多少？", (), unanswerable=True)
    for i in range(3))]

SPAN_PROBES = [*_HAND_SPAN_PROBES, *(
    SpanProbe(f"c{i}", f.question, f.doc_id, f.value_start, f.value_end)
    for i, f in enumerate(CORPUS_GEN.facts))]
#: ⚠️ N2 判分要拿原文比对——⛔ 生成的那批也得进这张表
CORPUS = {**_FILES, **{f.doc_id: f.text for f in CORPUS_GEN.facts}}


def suites(rebuild=None, world_handle=None) -> list:
    return [
        RetrievalSuite(QUERIES),
        ProvenanceSuite(SPAN_PROBES, CORPUS),
        PromptedRealitySuite(CLAIMS, TRUTH),
        SpontaneousRealitySuite(CLAIMS, TRUTH),
        QASuite(QA_ITEMS),
        *([] if rebuild is None else [
            GovernanceSuite(DELETION_PROBES, rebuild, world_handle),
        ]),
        ReasoningSuite(FACT_GRAPH, questions_from(FACT_GRAPH)),
        # ⛔ 曲线一并传进去：⚠️ 它是占位的，这两档的数**不得发布**，
        # 而那句话必须跟着分走到报告——⭐ 不然它会被当成正常的分用掉
        # （实测踩到：它当上了成本×质量表的质量列）。
        ObservedRetentionSuite(probes_from(EVENT_STREAM, NEED_CURVE,
                                           now_s=86_400 * 30.0), NEED_CURVE),
        SelfReportedRetentionSuite(probes_from(EVENT_STREAM, NEED_CURVE,
                                               now_s=86_400 * 30.0), NEED_CURVE),
        StructureSuite(TOPOLOGY),
        CalibrationSuite([
            CalibrationItem(f"c{i}", q, g, salient=(i % 2 == 0))
            for i, (q, g) in enumerate([
                ("哪个脑结构学得慢？", ("新皮层",)),
                ("一次暴露就能记住靠哪个结构？", ("海马",)),
            ])
        ]),
        InductionSuite(REGULARITIES),
    ]


# ── agent 档 ────────────────────────────────────────────────────
#: 无提示那一档用的普通问题 + 只有拿旧记忆才会说出来的那个词。
#: ⚠️ 与 CLAIMS 一一对应。
SPONTANEOUS_QUESTIONS = {
    "c1": "橘猫喜欢在哪里睡觉？如果现在查不到这条信息，就说「查不到」。",
    "c2": "保留天数是多少？",
    "c3": "哪个脑结构学得慢？",
}

#: 拿旧记忆作答才会冒出来的标记词。⛔ c3 没有——它本来就仍然成立。
STALE_MARKERS = {"c1": "窗台", "c2": "30", "c3": ""}

#: 跟上了世界变化才会说出来的词。
#: ⛔ 必须有：只看「没说旧值」的话，一个什么都不知道的系统会被判成全检出，
#: 而那是无知不是检出。
FRESH_MARKERS = {
    "c1": ("查不到", "没有找到", "不存在", "无法"),   # 笔记被删了
    "c2": ("7",),                                    # 30 → 7
    "c3": ("新皮层",),                                # 没变，答对即可
}


#: agent 档 N2：问一个问题并要求说出来源。
#: ⚠️ distractors 是「容易被答成的别的文件」——用来分辨「说错」与「没说」。
CITATION_PROBES = [
    ("s1", "哪个脑结构学得慢？", "neocortex", ("hippocampus", "cat")),
    ("s2", "一次暴露就能记住靠哪个结构？", "hippocampus", ("neocortex", "cat")),
]


def agent_suites(verdict_sink) -> list:
    """agent 档：判分口径与直接调库一致，⛔ 探针完全不同。

    ⚠️ 收 verdict_sink：有提示那一档靠**工具表态**，表态落盘评测器才读得到。
    """
    from amb.suites.agent_native import (
        AgentPromptedRealitySuite,
        AgentProvenanceSuite,
        AgentQASuite,
        AgentCalibrationSuite,
        AgentGovernanceSuite,
        AgentInductionSuite,
        AgentReasoningSuite,
        AgentRecallSuite,
        AgentSpontaneousRealitySuite,
        CitationProbe,
        ForgetProbe,
        retention_items,
        structure_items,
    )

    return [
        AgentPromptedRealitySuite(CLAIMS, TRUTH, verdict_sink),
        AgentSpontaneousRealitySuite(CLAIMS, TRUTH, SPONTANEOUS_QUESTIONS,
                                     STALE_MARKERS, FRESH_MARKERS),
        AgentQASuite(QA_ITEMS),
        AgentProvenanceSuite([
            CitationProbe(i, q, gold, dis) for i, q, gold, dis in CITATION_PROBES
        ]),
        # ⭐ N5/N6 共用通用召回探针——判分口径与直接调库那一档同源
        # ⚠️ agent 档每题一轮会话，很慢；这里只取一小撮验机制
        AgentRecallSuite("n5_agent", retention_items(
            probes_from(EVENT_STREAM, NEED_CURVE, now_s=86_400 * 30.0)[:4])),
        AgentRecallSuite("n6_agent", structure_items(TOPOLOGY)[:2],
                         cues_key="cues_list"),
        # ⚠️ agent 档每题一轮会话，很慢；这里都只取一小撮验机制
        AgentReasoningSuite(FACT_GRAPH, questions_from(FACT_GRAPH)[:2],
                            verdict_sink),
        AgentGovernanceSuite([
            ForgetProbe(
                probe=DELETION_PROBES[0],
                remember=f"请记住：{DELETION_PROBES[0].text}",
                forget=f"请忘掉关于{DELETION_PROBES[0].query}的那条记忆。",
            )
        ]),
        AgentCalibrationSuite([
            CalibrationItem("k1", "哪个脑结构学得慢？", ("新皮层",), salient=True),
            CalibrationItem("k2", "一次暴露就能记住靠哪个结构？", ("海马",)),
        ], verdict_sink),
        AgentInductionSuite(REGULARITIES[:1]),
    ]
