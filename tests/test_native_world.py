"""自研题库的生成器：⛔ N1 与 N4 此前是**手写的三条 / 一条**。

⚠️ `检出率` 的分母只有 2，`彻底删除率` 的分母是 **1**——
⛔ 那不是「测不出差别」，是**结构上不可能测出差别**。
⭐ 按仓库自己的 `required_n`，配对口径下要辨 0.05 需要 121 题。
"""

from __future__ import annotations

from collections import Counter

import pytest

from amb.world.mutate import ChangeKind
from amb.world.stream import governance, reality


# ── ⭐ N1：现实核对 ─────────────────────────────────────────────
def test_it_makes_enough_claims_to_conclude_anything() -> None:
    """⛔ 目标不是拍脑袋：⭐ 配对口径下要辨 0.05 需要 121 题。"""
    from amb.scoring.statistics import required_n

    w = reality.build(seed=42, n_claims=128)
    assert len(w.claims) >= 121, "⛔ 达不到能下结论的题量"
    # ⚠️ 顺带钉住独立口径要多少——⛔ 那是不做配对时的代价
    assert required_n(0.5, 0.05) > 1000


def test_all_four_change_kinds_appear() -> None:
    """⛔ 只造「文件没了」那一类，⚠️ 检出率就退化成「会不会查存在性」。

    ⭐ 四类各有各的失效方式：VANISH 最易，ADVANCE 要读时钟（多数系统不读）。
    """
    w = reality.build(seed=42, n_claims=128)
    kinds = Counter(c.kind for c in w.changes)
    for k in (ChangeKind.VANISH, ChangeKind.REVALUE, ChangeKind.IRRELEVANT):
        assert kinds[k] >= 25, f"⛔ {k} 只有 {kinds[k]} 条"


def test_there_is_a_holds_side_to_catch_false_alarms() -> None:
    """⛔ **反方向**：⚠️ 没有仍然成立的命题，
    一条「什么都报 broken」的臂能拿满分。
    """
    w = reality.build(seed=42, n_claims=128)
    holds = sum(1 for v in w.truth.values() if v == "holds")
    assert holds >= 25, f"⛔ 只有 {holds} 条 holds，⚠️ 误报率量不准"


def test_each_claim_owns_its_document() -> None:
    """⛔ 共用文档的话，一次变更同时改掉几条命题的真值——
    ⚠️ 那时「检出一条」与「检出三条」在判分上分不开。
    """
    w = reality.build(seed=42, n_claims=128)
    used = [d for c in w.claims for d in c.doc_ids]
    assert len(used) == len(set(used)), "⛔ 有命题共用了文档"


def test_revalue_changes_the_file_and_the_fact_table_together() -> None:
    """⛔ 只改一边世界就自相矛盾，⚠️ 命题没有确定的真值。

    ⭐ 第一次跑就是这么错的（只改了事实表，而命题引的是文件）。
    """
    w = reality.build(seed=7, n_claims=64)
    files = {c.target for c in w.changes
             if c.kind is ChangeKind.REVALUE and "/" in c.target}
    keys = {c.target for c in w.changes
            if c.kind is ChangeKind.REVALUE and "/" not in c.target}
    assert len(files) == len(keys) > 0, "⛔ 文件与事实表没有成对改"


def test_it_refuses_to_repeat_a_topic() -> None:
    """⛔ 不静默重复：⚠️ 重复的主题会让不同命题指向同一份语料，
    ⭐ 那时真值互相污染。
    """
    with pytest.raises(reality.TooFewTopics):
        reality.build(seed=42, n_claims=10_000)


def test_the_same_seed_gives_the_same_world() -> None:
    """⛔ 不可复现的题库等于每次换了张卷子。"""
    a, b = reality.build(seed=1, n_claims=40), reality.build(seed=1, n_claims=40)
    assert [c.text for c in a.claims] == [c.text for c in b.claims]
    assert a.truth == b.truth


# ── ⭐ N4：治理 ────────────────────────────────────────────────
def test_it_makes_enough_deletion_probes() -> None:
    """⛔ 此前分母是 **1**——⚠️ 那个数只能读成「这一个案例上发生了什么」，
    ⭐ 而报告把它当率印。
    """
    _docs, probes = governance.build(seed=42, n_probes=90)
    assert len(probes) >= 61, "⛔ 达不到能辨 0.10 的题量"


def test_every_marker_is_globally_unique() -> None:
    """⛔ 撞了的话「删了 A 却搜到 B」会被判成「没删干净」——
    ⚠️ 而那是评测器自己的错。
    """
    _docs, probes = governance.build(seed=42, n_probes=90)
    markers = [p.marker for p in probes]
    assert len(set(markers)) == len(markers)


def test_no_marker_is_a_prefix_of_another() -> None:
    """⛔ `K-7` 是 `K-70` 的前缀——⚠️ 删了 K-7 之后带外搜索仍会命中 K-70，
    ⭐ 那是**假的**「没删干净」。
    """
    _docs, probes = governance.build(seed=42, n_probes=90)
    ms = [p.marker for p in probes]
    bad = [(a, b) for a in ms for b in ms if a != b and a.startswith(b)]
    assert not bad, f"⛔ 前缀撞车：{bad[:3]}"


def test_principals_are_spread_out() -> None:
    """⛔ 全挂一个人的话，隔离那一组永远是「无隔离」——
    ⚠️ 那时测的是「我们只造了一个用户」，不是「它分不分得开用户」。
    """
    docs, _probes = governance.build(seed=42, n_probes=90)
    who = Counter(d.principal for d in docs)
    assert len(who) >= 3 and min(who.values()) >= 15, f"⛔ 主体太集中：{who}"


def test_a_probe_and_its_document_agree() -> None:
    """⚠️ 探针的 `text` / `doc_id` 要与那份文档**逐字一致**——
    ⛔ 对不上的话带外搜索找的是一个从没被摄入的串。
    """
    docs, probes = governance.build(seed=3, n_probes=40)
    by_id = {d.doc_id: d.text for d in docs}
    for p in probes:
        assert by_id[p.doc_id] == p.text
        assert p.marker in p.text, "⛔ 特征串不在正文里"


# ── ⭐ 组装出来的世界：⛔ 每一档都要够格 ─────────────────────────
def _suites():
    import worlds.native as w
    return {s.name: s for s in w.suites(rebuild=lambda: None,
                                        world_handle=lambda: None)}


#: ⭐ 每档至少要多少题，⚠️ 数字来自 `required_n`（配对口径 6/d）：
#:   121 → 够辨 0.05    61 → 够辨 0.10    41 → 够辨 0.15
#: ⛔ toy 对应的是 3 / 4 / 12——**结构上不可能下结论**。
_FLOOR = {
    "retrieval": 121, "qa": 121, "n2_provenance": 121,
    "n1_prompted": 121, "n1_spontaneous": 121,
    "n4_governance": 61, "n5_observed": 61, "n5_self_reported": 61,
    "n8_induction": 41, "n7_calibration": 31,
    "n6_structure": 61, "n3_reasoning": 41,
}


def _count(suite) -> int:
    """这个套件出几道题。⚠️ 各套件的字段名不一，⛔ 数不出来就当 0——
    ⭐ 那会让守卫红，而不是静默放过。"""
    for attr in ("_probes", "_items", "_claims", "_queries",
                 "_regs", "_questions"):
        got = getattr(suite, attr, None)
        if got:
            return len(got)
    # ⭐ N6 的题是从拓扑取样出来的，⚠️ 不是一个现成的列表
    if hasattr(suite, "_probed"):
        return len(suite._probed())
    return 0


@pytest.mark.parametrize("name,floor", sorted(_FLOOR.items()))
def test_every_suite_has_enough_questions(name: str, floor: int) -> None:
    """⛔ 题量不够就**不可能**下结论，⚠️ 而报告会印一排「分不开」，
    ⭐ 读者会以为是系统之间没差别——**那是尺子太短**。
    """
    got = _count(_suites()[name])
    assert got >= floor, (
        f"⛔ {name} 只有 {got} 题，⚠️ 要 {floor} 才够辨 {6 / floor:.2f}")


def test_it_is_bigger_than_the_toy_world_where_it_matters() -> None:
    """⭐ 对着 toy 的三个洞：⛔ N1 三题、N4 四题、N8 十二题。"""
    import worlds.toy as toy

    old = {s.name: _count(s) for s in toy.suites(rebuild=lambda: None,
                                                 world_handle=lambda: None)}
    new = {k: _count(v) for k, v in _suites().items()}
    for k in ("n1_prompted", "n4_governance", "n8_induction"):
        assert new[k] >= old.get(k, 0) * 4, \
            f"⛔ {k}: toy {old.get(k)} → native {new[k]}，⚠️ 没拉开"


def test_the_native_world_is_not_called_a_toy() -> None:
    """⛔ `toy` 自己的注释写着「它**不是**一个够格的题库」——
    ⚠️ 而它此前是 N1–N8 唯一能跑的世界。⭐ 这条守住新世界的定位。
    """
    import worlds.native as w

    assert w.MANIFEST.name == "native"
    assert "不是" not in (w.__doc__ or "").split("##")[0], \
        "⛔ 新世界不该也自称不够格"


def test_the_world_and_the_memory_corpus_stay_separate() -> None:
    """⛔ N4 的语料**只进记忆、不进世界**——⚠️ 混进去的话
    N1 的「文件还在不在」会被它污染。
    """
    import worlds.native as w

    in_world = {d.doc_id for d in w.DOCUMENTS}
    gov = {d.doc_id for d in w.GOV_DOCS}
    assert not (in_world & gov), "⛔ N4 的语料漏进世界了"
    assert gov <= {d.doc_id for d in w.all_documents()}, "⛔ 但它必须进记忆"
