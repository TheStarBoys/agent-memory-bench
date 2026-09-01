"""推导规则词表。

⛔ 两张表分开，不合并：Rule 全是单调的，
混进非单调规则会让 N3 的判分变得可争议——而 N3 的价值来自判分干净。

规格：docs/suites/n3-reasoning.md · docs/suites/n8-induction.md
"""

from __future__ import annotations

from enum import StrEnum


class Rule(StrEnum):
    """N3 的封闭词表。⛔ 不在表内的一律判该步不成立。"""

    ASSERT = "assert"          # 前提直接是一条记忆，零步推导
    CONJOIN = "conjoin"        # 合取多个前提
    SUBSTITUTE = "substitute"  # 等值替换：a=b, P(a) ⊢ P(b)
    TRANSITIVE = "transitive"  # 传递：a→b, b→c ⊢ a→c
    COMPARE = "compare"        # 数值 / 时间比较
    EXCLUDE = "exclude"        # 候选集合减去被否定的项
    NEGATE = "negate"          # 封闭世界否定：找不到满足 X 的 ⊢ ¬X


class DefeasibleRule(StrEnum):
    """N8 的扩展词表：可废止推理。"""

    DEFAULT = "default"  # 默认成立的规律：通常 A → P
    EXCEPT = "except"    # 明确的例外：a 是 A，但 ¬P(a)
