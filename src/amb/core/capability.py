"""能力自述。

适配器预先声明支持哪些能力，评测器只跑声明过的套件；
未声明的记不支持，⛔ 永远不折进 0 分。

⚠️ 少声明会在报告的「声明与参与」一列上可见——
「不支持不计分母」本身会催生少声明的动机，那一列是用来堵它的。

规格：docs/adapters/protocol.md#能力
"""

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    INGEST = "ingest"          # 基线，必需
    SEARCH = "search"          # 基线，必需
    ANSWER = "answer"          # 端到端答题（含生成器）
    REALITY = "reality"        # N1
    PROVENANCE = "provenance"  # N2
    REASONING = "reasoning"    # N3
    GOVERNANCE = "governance"  # N4
    RETENTION = "retention"    # N5 系统自报（外部观察不需要声明）
    CONFIDENCE = "confidence"  # N7
    INDUCTION = "induction"    # N8 推导链一档（三个问题不需要声明）
    ACCOUNTING = "accounting"  # 成本计量（原则⑥）


#: 每个适配器都必须实现，⛔ 不得返回三态。
BASELINE: frozenset[Capability] = frozenset({Capability.INGEST, Capability.SEARCH})
