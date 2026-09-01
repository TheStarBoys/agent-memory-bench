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
from amb.suites.native.qa import QAItem, QASuite
from amb.suites.native.retrieval import Query, RetrievalSuite
from amb.world import Change, ChangeKind, FileSpec, WorldManifest

CLOCK_START = "2026-01-01T00:00:00Z"

_FILES = {
    "notes/hippocampus.md": "海马体负责情节记忆的快速编码，一次暴露就能记住具体事件。",
    "notes/neocortex.md": "新皮层学得慢，靠反复暴露抽取跨情节的统计规律。",
    "notes/cat.md": "橘猫喜欢在窗台上晒太阳，一睡就是一下午。",
    "config/retention.txt": "retention_days=30",
}

MANIFEST = WorldManifest(
    name="toy",
    seed=42,
    clock_start=CLOCK_START,
    files=tuple(FileSpec(p, t) for p, t in _FILES.items()),
    facts={"retention_days": "30", "region": "cn-north"},
)

DOCUMENTS = [
    Document(doc_id=path, text=text, timestamp=CLOCK_START, principal="alice",
             kind="document")
    for path, text in _FILES.items()
]

CORPUS = dict(_FILES)

QUERIES = [
    Query("q1", "哪个结构学得慢？", frozenset({"notes/neocortex.md"})),
    Query("q2", "什么动物喜欢晒太阳？", frozenset({"notes/cat.md"})),
    Query("q3", "一次就能记住靠什么？", frozenset({"notes/hippocampus.md"})),
    Query("q4", "保留多少天？", frozenset({"config/retention.txt"})),
]

SPAN_PROBES = [
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
QA_ITEMS = [
    QAItem("a1", "哪个脑结构学得慢，靠反复暴露抽取规律？", ("新皮层",)),
    QAItem("a2", "哪个脑结构一次暴露就能记住？", ("海马",)),
    # ⭐ 该弃权的题：语料里**从来没提过**这件事。
    # ⚠️ 刻意不用「被删掉的那份笔记」——那考的是记忆过时，是 N1 的活。
    # 混进来会让「编造率」同时含两种成因，读不出是哪一种。
    QAItem("a3", "海马体是哪一年被命名的？", (), unanswerable=True),
]


def suites() -> list:
    return [
        RetrievalSuite(QUERIES),
        ProvenanceSuite(SPAN_PROBES, CORPUS),
        PromptedRealitySuite(CLAIMS, TRUTH),
        SpontaneousRealitySuite(CLAIMS, TRUTH),
        QASuite(QA_ITEMS),
    ]
