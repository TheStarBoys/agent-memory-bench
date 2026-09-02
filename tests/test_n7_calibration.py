"""N7 置信度校准。

⛔ 全部价值在于**ECE 不能单独报**——
低 ECE 可以靠把所有置信度压到基准准确率附近取得，
那样的系统校准很好、但完全没有区分能力。
"""

from __future__ import annotations

from amb.core import Answer, Observation, SuiteRun
from amb.scoring import score


def run(rows: list[tuple[float, bool, bool]]):
    """rows: (confidence, correct, salient)"""
    suite_run = SuiteRun("n7_calibration", "scored")
    for i, (conf, correct, salient) in enumerate(rows):
        suite_run.observations.append(Observation(f"i{i}", {
            "confidence": conf, "correct": correct, "salient": salient,
        }))
    return score(suite_run).metrics


def test_a_perfectly_calibrated_system_has_near_zero_ece() -> None:
    # 置信 0.9 的题九成对，0.1 的题一成对
    rows = [(0.9, i < 9, False) for i in range(10)] + \
           [(0.1, i < 1, False) for i in range(10)]
    m = run(rows)
    assert m["ECE"] < 0.05
    assert m["区分度"] > 0.7, "⭐ 而且它确实分得开对错"


def test_a_hedger_gets_low_ece_but_no_discrimination() -> None:
    """⭐ 这一条就是「ECE 不能单独报」的证明。

    一个把所有置信度都压到基准准确率的系统：
    ⛔ ECE 极低，看起来校准完美，但区分度是 0——它什么都没告诉你。
    """
    rows = [(0.5, i % 2 == 0, False) for i in range(20)]
    m = run(rows)
    assert m["ECE"] < 0.05, "校准看起来完美"
    assert m["区分度"] == 0.0, "⛔ 但它完全没有区分能力"


def test_overconfidence_shows_up_in_ece() -> None:
    rows = [(0.95, i < 5, False) for i in range(10)]   # 说 95% 实际 50%
    m = run(rows)
    assert m["ECE"] > 0.4
    assert m["Brier"] > 0.3


def test_reliability_diagram_accompanies_the_scalars() -> None:
    """⛔ 两个标量看不出退化，可靠性图能。"""
    m = run([(0.9, True, False), (0.1, False, False)])
    assert any(k.startswith("桶") and k.endswith("_置信") for k in m)
    assert any(k.startswith("桶") and k.endswith("_准确") for k in m)


def test_confident_but_not_more_accurate_is_penalised() -> None:
    """⭐ 闪光灯记忆那个已知的人类 bug：不是更准，只是自觉更准。

    ⛔ 这一次我们知道它是 bug，所以扣分——像人不是加分项。
    """
    # 显著题与普通题准确率相同（都 50%），但显著题置信度高得多
    rows = [(0.9, i < 5, True) for i in range(10)] + \
           [(0.5, i < 5, False) for i in range(10)]
    m = run(rows)
    assert m["显著_准确率"] == m["普通_准确率"]
    assert m["显著_置信度"] > m["普通_置信度"]
    assert m["自信但不更准"] > 0.3, "⛔ 必须被标出来"


def test_genuine_salience_benefit_is_not_penalised() -> None:
    """⚠️ 反方向：显著性真的带来了更好的巩固，那不该扣分。"""
    rows = [(0.9, i < 9, True) for i in range(10)] + \
           [(0.5, i < 5, False) for i in range(10)]
    m = run(rows)
    assert m["显著_准确率"] > m["普通_准确率"]
    assert m["自信但不更准"] == 0.0, "准确率跟上了，就不是那个 bug"


def test_declaring_confidence_without_giving_one_is_failed() -> None:
    """⛔ 声明了 CONFIDENCE 却返回 None = 这次没做成，不是不支持。"""
    from amb.core import AdapterBase, BASELINE, Capability
    from amb.suites.native.n7_calibration import CalibrationItem, CalibrationSuite

    class Silent(AdapterBase):
        def capabilities(self):
            return set(BASELINE) | {Capability.CONFIDENCE, Capability.ANSWER}

        def answer(self, query, *, principal=None):
            return Answer(text="也许吧", confidence=None)   # ⛔ 不给数

    run_ = CalibrationSuite([CalibrationItem("a", "问题", ("答",))]).probe(Silent(), None)
    assert run_.failed == 1 and not run_.observations
