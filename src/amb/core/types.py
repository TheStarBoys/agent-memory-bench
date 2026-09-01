"""协议里的数据类型。

⛔ 这一层零依赖，也不做任何 IO——它是所有层共同的词汇表。

规格：docs/adapters/protocol.md#数据类型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amb.core.rules import DefeasibleRule, Rule


@dataclass(frozen=True, slots=True)
class WorldHandle:
    """世界交给被测系统的方式。三样都是只读、进程外、⛔ 不推送的。

    ⛔ 时钟与事实表是端点不是快照值：发一个当前时间的字符串过去，
    系统就永远察觉不到时间流逝——而时间流逝是 N1 的五类变更之一。
    """

    root: str        # 只读文件树的绝对路径
    clock_url: str   # GET → {"now": "<RFC3339>"}
    facts_url: str   # GET {facts_url}/{key} → {"key":…, "value":…} | 404


@dataclass(frozen=True, slots=True)
class Document:
    """摄入单元。doc_id 与字符偏移由评测器分配并永久稳定——N2 靠它对账。"""

    doc_id: str
    text: str                                 # 已做 NFC 规范化
    timestamp: str | None = None              # RFC3339
    principal: str | None = None              # 谁写的（N4）
    kind: Literal["turn", "document"] = "turn"


@dataclass(frozen=True, slots=True)
class Span:
    """原文区间。⛔ 起止都要，只给起点不算。

    ⛔ 偏移单位是 Unicode 码点（Python str 索引），不是字节也不是 token——
    中文语料下字节口径会让所有区间偏三倍。
    """

    doc_id: str
    start: int  # 闭
    end: int    # 开


@dataclass(slots=True)
class Entry:
    """检索返回的一条。正文可选——多数系统不返回正文是对的。"""

    id: str
    digest: str
    score: float | None = None
    doc_ids: list[str] = field(default_factory=list)   # 给不出则 N1 无提示记不支持
    spans: list[Span] = field(default_factory=list)    # N2；不支持就空列表
    principal: str | None = None                       # N4
    state: Literal["holds", "broken", "unknown"] | None = None  # N1 无提示
    confidence: float | None = None                    # N7；⛔ 不是 score
    supersedes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Claim:
    """N1 / N5：评测器出的一条待检命题。

    ⛔ 不问系统内部存了什么，只问这句话对当前世界还成不成立——
    否则就预设了系统持有可枚举的离散条目集合，那是形状偏心。
    """

    claim_id: str
    text: str
    doc_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Verdict:
    """N1：一条命题对当前世界还成不成立。"""

    claim_id: str
    state: Literal["holds", "broken", "unknown"]
    grounds: list[str] = field(default_factory=list)  # 路径 / 事实表键 / Entry.id
    note: str | None = None                           # 散文，⛔ 判分不读


@dataclass(slots=True)
class RecallVerdict:
    """N5 系统自报：一条命题现在还留着吗、多强。"""

    claim_id: str
    state: Literal["retained", "dropped", "unknown"]
    strength: float | None = None  # [0,1]，给不出就 None


@dataclass(frozen=True, slots=True)
class Premise:
    kind: Literal["entry", "step"]
    ref: str  # Entry.id 或 Step.step_id


@dataclass(slots=True)
class Step:
    """N3 推导链的一步。"""

    step_id: str
    claim: str  # ⛔ 规范化三元组 "subject|relation|object"，不是散文
    premises: list[Premise] = field(default_factory=list)
    rule: Rule = Rule.ASSERT


@dataclass(slots=True)
class Answer:
    text: str
    derivation: list[Step] = field(default_factory=list)  # N3；不支持就空列表
    used: list[str] = field(default_factory=list)         # 用到的 Entry.id
    missing: list[str] = field(default_factory=list)      # 未决时缺哪些 claim_id
    confidence: float | None = None                       # N7


@dataclass(slots=True)
class Regularity:
    """N8 推导链一档：系统归纳到的一条规律。"""

    claim: str
    kind: DefeasibleRule
    strength: float | None = None
    exceptions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DeleteResult:
    deleted: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)  # entry_id → 原因


@dataclass(slots=True)
class AuditEvent:
    event_id: str
    action: Literal["ingest", "update", "delete", "read"]
    entry_ids: list[str] = field(default_factory=list)
    principal: str | None = None
    at: str = ""       # RFC3339
    detail: str | None = None


@dataclass(slots=True)
class Usage:
    """原则⑥。⚠️ 墙钟评测器自己能测，token 只有适配器报得出来。"""

    phase: Literal["ingest", "probe"]
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    wall_ms: int = 0
