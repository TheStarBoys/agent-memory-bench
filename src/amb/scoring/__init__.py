"""确定性判分。⛔ 不用 LLM 评委——约束①。"""

from amb.scoring.metrics import UNTRUSTED_THRESHOLD, Score, score

__all__ = ["Score", "UNTRUSTED_THRESHOLD", "score"]
