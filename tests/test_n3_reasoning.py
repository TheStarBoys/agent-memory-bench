"""N3 推理链正确性。

⛔ 全部价值在于**蒙对率**——公开题库只判最终答案，
于是结论蒙对而链条错误判满分。
"""

from __future__ import annotations

from amb.core import (
    AdapterBase,
    Answer,
    BASELINE,
    Capability,
    Premise,
    Rule,
    Step,
)
from amb.scoring import score
from amb.suites.native.n3_reasoning import ReasoningSuite, questions_from
from amb.world.stream.factgraph import build

GRAPH = build(seed=3, chains=4, depth=3)
QUESTIONS = questions_from(GRAPH)
BY_CONCLUSION = {str(d.conclusion): d for d in GRAPH.derivations}


class _Arm(AdapterBase):
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def capabilities(self):
        return set(BASELINE) | {Capability.ANSWER, Capability.REASONING}

    def answer(self, query, *, principal=None):
        # 找出这道题对应的推导
        q = next(q for q in QUESTIONS if q.question == query)
        d = BY_CONCLUSION[str(q.gold)]

        if self._mode == "no_chain":
            return Answer(text=q.gold.obj)
        if self._mode == "honest":
            steps = [Step(step_id="s0", claim=str(d.conclusion),
                          premises=[Premise("entry", str(p)) for p in d.premises],
                          rule=Rule.TRANSITIVE)]
            return Answer(text=q.gold.obj, derivation=steps)
        if self._mode == "guessed":
            # ⭐ 结论对，但链条引了一条不存在的记忆
            steps = [Step(step_id="s0", claim=str(d.conclusion),
                          premises=[Premise("entry", "不存在|上级|谁")],
                          rule=Rule.TRANSITIVE)]
            return Answer(text=q.gold.obj, derivation=steps)
        if self._mode == "undecided":
            return Answer(text="推不出来", missing=["缺一跳"])
        return Answer(text="不知道")


def run(mode: str):
    return score(ReasoningSuite(GRAPH, QUESTIONS).probe(_Arm(mode), None)).metrics


def test_honest_chain_scores_on_both() -> None:
    m = run("honest")
    assert m["结论准确率"] == 1.0 and m["链条完好率"] == 1.0
    assert m["蒙对率"] == 0.0


def test_guessing_is_caught_even_when_the_conclusion_is_right() -> None:
    """⭐ 这一条就是 N3 存在的理由。

    公开题库会给它满分——结论完全正确。
    ⛔ 但它的链条引了一条不存在的记忆，链条完好率 0。
    """
    m = run("guessed")
    assert m["结论准确率"] == 1.0, "公开题库会判它满分"
    assert m["链条完好率"] == 0.0, "⛔ 链条不成立"
    assert m["蒙对率"] == 1.0, "⭐ 这就是表面共现的贡献"


def test_no_chain_at_all_is_not_a_good_chain() -> None:
    """⛔ 不给链条 ≠ 链条完好。"""
    m = run("no_chain")
    assert m["结论准确率"] == 1.0
    assert m["给链条率"] == 0.0
    assert m["链条完好率"] == 0.0 and m["蒙对率"] == 1.0


def test_undecided_is_tracked_separately_from_wrong() -> None:
    """⛔ 「我缺前提 X」比直接答错更有用，判分要体现。"""
    m = run("undecided")
    assert m["未决率"] == 1.0
    assert m["结论准确率"] == 0.0


def test_a_step_citing_a_nonexistent_memory_fails() -> None:
    """⛔ 引了一条系统里没有的记忆，那一步不成立。"""
    assert run("guessed")["链条完好率"] == 0.0
