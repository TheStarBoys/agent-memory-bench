"""只有上下文窗口、没有记忆层的那条臂。

⛔ 它才是记忆层真正要打败的对手：⚠️ `full_context` 装不下就记 N/A，
⭐ 于是**最强的对照臂在最该较量的地方弃权**。
"""

from __future__ import annotations

import pytest

from amb.adapters import create
from amb.core import Document


def _arm(budget: int):
    return create("recency_window", budget_chars=budget)


def _docs(n: int, size: int = 10):
    return [Document(doc_id=f"d{i}", text=f"{i:03d}" + "x" * (size - 3))
            for i in range(n)]


def test_it_drops_the_oldest_when_the_window_overflows() -> None:
    """⭐ 滑窗：⚠️ 新的进来，旧的被挤出去。⛔ 不做选择——
    「会选」正是记忆层要证明的价值。
    """
    arm = _arm(50)
    for d in _docs(10):
        arm.ingest(d)
    assert arm.count() == 5, "⛔ 窗口装不下 10 条"
    kept = [e.doc_ids[0] for e in arm.search("随便问", 10)]
    assert kept == ["d5", "d6", "d7", "d8", "d9"], f"⛔ 丢错了：{kept}"


def test_it_never_truncates_silently() -> None:
    """⛔ **这条是这条臂的前提**：⚠️ 静默截断给出的是一个**假的天花板**——
    ⭐ 读者会读成「全都读了还只有这个分」。
    """
    arm = _arm(50)
    for d in _docs(10):
        arm.ingest(d)
    st = arm.window_stats()
    assert st["window_dropped"] == 5
    assert st["window_dropped_chars"] == 50
    assert st["window_budget_chars"] == 50


def test_a_run_that_never_overflows_says_so() -> None:
    """⚠️ `window_dropped=0` 是**读数的前提**：⛔ 那时这条臂等价于
    `full_context`，⭐ 这一跑**测不到记忆的价值**。
    """
    arm = _arm(10_000)
    for d in _docs(10):
        arm.ingest(d)
    assert arm.window_stats()["window_dropped"] == 0
    assert arm.count() == 10


def test_shrinking_the_window_is_what_creates_the_test_condition() -> None:
    """⭐ **你不需要几千轮对话**：⚠️ 把窗口调小，几十条就进入同一个失效区间。

    ⛔ 这就是「记忆有没有用」与「语料多大」解耦的地方——
    ⚠️ 而语料大小正是摄入成本的主要驱动。
    """
    docs = _docs(40)
    curve = []
    for budget in (400, 200, 100, 50):
        arm = _arm(budget)
        for d in docs:
            arm.ingest(d)
        curve.append(arm.count())
    assert curve == sorted(curve, reverse=True), f"⛔ 窗口越小该留得越少：{curve}"
    assert curve[0] > curve[-1] * 4, "⛔ 曲线没拉开，⚠️ 扫描就没有信息量"


def test_it_honours_k() -> None:
    """⛔ 不遵守 `k` 的话，那些「指名要一条」的判据（N6 精确检索、
    N1 无提示）在这条臂上等于**白送分**。
    """
    arm = _arm(10_000)
    for d in _docs(10):
        arm.ingest(d)
    assert len(arm.search("随便问", 1)) == 1
    assert len(arm.search("随便问", 3)) == 3


def test_it_ignores_the_query() -> None:
    """⭐ 它**不检索**：⚠️ 换个问题拿回的是同一批内容。

    ⛔ 这是刻意的——它的含义就是「不做选择」。
    """
    arm = _arm(10_000)
    for d in _docs(10):
        arm.ingest(d)
    a = [e.doc_ids[0] for e in arm.search("问题甲", 5)]
    b = [e.doc_ids[0] for e in arm.search("完全不同的问题乙", 5)]
    assert a == b


def test_reset_clears_the_drop_counters() -> None:
    """⛔ 残留的计数会让下一跑虚报截断量。"""
    arm = _arm(50)
    for d in _docs(10):
        arm.ingest(d)
    arm.reset()
    assert arm.window_stats()["window_dropped"] == 0
    assert arm.count() == 0


def test_a_zero_budget_is_refused() -> None:
    """⛔ 0 或负的预算不是「很小的窗口」，⚠️ 是配置错了。"""
    with pytest.raises(ValueError):
        _arm(0)


def test_the_drop_reaches_the_report(monkeypatch, tmp_path) -> None:
    """⭐ 端到端：⛔ 截断量要跟着分走到报告里。"""
    import sys

    sys.path.insert(0, str(tmp_path.parent))
    import offline
    from amb.report import Report, render
    from amb.runner import Plan, backbone, build, run_one

    import worlds.toy as toy

    offline.use_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    offline.install(monkeypatch)
    plan = Plan(manifest=toy.MANIFEST, documents=toy.all_documents()[:40],
                changes=toy.CHANGES, suites_for=toy.suites)
    r, _ = run_one("recency_window", build("recency_window", llm=backbone(),
                                           context_budget=200),
                   plan, tmp_path / "w", is_control=True)
    assert r.cost_profile["window_dropped"] > 0, "⛔ 这个预算下该丢东西"

    rep = Report(run_id="x", at="now",
                 world={"name": "toy", "seed": 42, "documents": 40,
                        "digest": "sha256:a", "corpus": "c"},
                 backbone={"model": "fake", "thinking": False})
    rep.lanes["library"] = [r]
    text = render(rep)
    assert "上下文窗口" in text, "⛔ 报告没印截断"
    assert "被挤掉" in text


# ── ⭐ 窗口是**受控变量**：⛔ 所有臂一视同仁 ────────────────────
def test_the_window_caps_what_any_arm_hands_to_the_model() -> None:
    """⛔ 只约束 `recency_window` 而放任别的臂随便塞，扫描就不成立——
    ⚠️ 那时测的是「谁被限制了」，不是「谁选得准」。

    ⭐ 而这正是记忆层的价值：窗口一小能塞的材料就少，
    ⚠️ **选得准的臂掉得慢**。⛔ 窗口不限时人人都能把 top-10 全交出去，
    那时候「会选」这件事量不到。
    """
    from amb.adapters.answering import fit_to_window
    from amb.core import Entry

    hits = [Entry(id=f"e{i}", digest="x" * 100, score=1.0 - i * 0.1)
            for i in range(10)]
    kept, dropped = fit_to_window(hits, 250)
    assert len(kept) == 2 and dropped == 8
    # ⚠️ 按**顺序**保留：⛔ 检索臂的第 1 条是它最有把握的那条
    assert [e.id for e in kept] == ["e0", "e1"]


def test_no_window_means_no_constraint() -> None:
    """⚠️ 默认不限——⛔ 保持旧行为，⭐ 不静默改掉已有的分。"""
    from amb.adapters.answering import fit_to_window
    from amb.core import Entry

    hits = [Entry(id=f"e{i}", digest="x" * 100) for i in range(10)]
    assert fit_to_window(hits, None) == (hits, 0)
    assert fit_to_window(hits, 0) == (hits, 0)


def test_at_least_one_entry_always_gets_through() -> None:
    """⛔ 窗口比第一条还小时也要交一条——⚠️ 交空的话那道题变成
    「没有材料」，⭐ 而那与「材料被裁掉了」是两回事。
    """
    from amb.adapters.answering import fit_to_window
    from amb.core import Entry

    kept, dropped = fit_to_window(
        [Entry(id="e0", digest="x" * 500), Entry(id="e1", digest="y" * 500)], 10)
    assert len(kept) == 1 and dropped == 1


def test_every_answering_arm_accepts_the_window() -> None:
    """⛔ 漏挂一条臂，它就在窗口收紧时**白占便宜**。"""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import offline
    from amb.adapters import SYSTEMS
    from amb.adapters.registry import _REGISTRY
    from amb.runner import build

    old = dict(os.environ)
    try:
        os.environ.update(offline.ENV)
        for name in sorted(set(_REGISTRY) - set(SYSTEMS)):
            arm = build(name, answer_window=4096)
            if hasattr(arm, "attach_window"):
                assert arm._window == 4096, f"⛔ {name} 没收到窗口"
    finally:
        os.environ.clear()
        os.environ.update(old)
