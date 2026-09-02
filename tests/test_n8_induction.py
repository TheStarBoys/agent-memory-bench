"""N8 归纳与可废止推理。

⛔ 全部价值在于**四种行为分列**——
「过度修正」与「过度泛化」是两种相反的毛病，
在只判对错的表里它们都是「答错一道题」。
"""

from __future__ import annotations

from amb.core import AdapterBase, Answer, BASELINE, Capability
from amb.scoring import score
from amb.suites.native.n8_induction import InductionSuite
from amb.world.stream.regularity import build

REGS = build(seed=2)


class _Arm(AdapterBase):
    """按一个策略回答「X 是不是 P 的」。"""

    def __init__(self, policy) -> None:
        self._policy = policy

    def capabilities(self):
        return set(BASELINE) | {Capability.ANSWER}

    def answer(self, query, *, principal=None):
        name = query.split()[0]
        return Answer(text="是" if self._policy(name) else "否")


def run(policy):
    return score(InductionSuite(REGS).probe(_Arm(policy), None)).metrics


def test_perfect_behaviour() -> None:
    """规律用在新正例上，例外照例外办，且例外不推翻规律。"""
    m = run(lambda n: not n.endswith("-X"))
    assert m["全对"] == 1.0
    assert m["过度修正"] == m["过度泛化"] == m["未归纳"] == 0.0


def test_over_generalisation_is_its_own_category() -> None:
    """⛔ 规律压过了明确的例外。"""
    m = run(lambda n: True)          # 一律说「是」
    assert m["过度泛化"] == 1.0 and m["全对"] == 0.0


def test_failure_to_induce_is_its_own_category() -> None:
    m = run(lambda n: False)         # 一律说「否」
    assert m["未归纳"] == 1.0 and m["全对"] == 0.0


def test_over_correction_is_distinguished_from_over_generalisation() -> None:
    """⭐ 这一条是 N8 存在的理由。

    「见了一个反例就把规律整个扔了」与「规律压过例外」
    在只判对错的表里都是「答错一道题」，⛔ 但改进方向相反。
    """
    seen: set[str] = set()

    def gives_up_after_the_exception(name: str) -> bool:
        # 第一个新正例答对；例外答对；之后就不敢再用规律了
        if name.endswith("-X"):
            seen.add(name.split("-")[0])
            return False
        return name.split("-")[0] not in seen

    m = run(gives_up_after_the_exception)
    assert m["过度修正"] == 1.0
    assert m["过度泛化"] == 0.0, "⛔ 与过度泛化必须分开"
    assert m["全对"] == 0.0


def test_counts_are_reported_alongside_rates() -> None:
    """⚠️ 只有比率的话，样本量小的时候读者会当真。"""
    m = run(lambda n: True)
    assert m["计数_过度泛化"] == float(len(REGS))


def test_unparseable_answers_are_tracked() -> None:
    class Mumbles(_Arm):
        def answer(self, query, *, principal=None):
            return Answer(text="嗯……")

    m = score(InductionSuite(REGS).probe(Mumbles(lambda n: True), None)).metrics
    assert m["未解析率"] == 1.0
