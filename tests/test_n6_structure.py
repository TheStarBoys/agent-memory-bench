"""N6 关联结构。

⛔ 全部价值在于**两条曲线一起报**——
只报可达性，连成完全图的系统满分；只报精确性，完全不建关联的系统满分。
"""

from __future__ import annotations

from amb.core import AdapterBase, BASELINE, Document, Entry, Observation, SuiteRun
from amb.scoring import score
from amb.suites.native.n6_structure import StructureSuite
from amb.world.stream.topology import build

TOPO = build(seed=5)
#: fact_id → 完整指名。⚠️ 判分用的就是它，⛔ 测试替身也只认它
DESIGNATION = {f.fact_id: f.designation for f in TOPO.facts}
#: ⚠️ 跟着生成器的默认走——⛔ 写死一份名单，加了档次就测不到新的那几档
FANS = tuple(sorted({f.fan for f in TOPO.facts}))


class _Arm(AdapterBase):
    def __init__(self, strategy: str) -> None:
        self._strategy = strategy
        self._facts: dict[str, str] = {}
        self._entity: dict[str, str] = {}

    def capabilities(self):
        return set(BASELINE)

    def ingest(self, doc: Document) -> None:
        self._facts[doc.doc_id] = doc.text
        self._entity[doc.doc_id] = doc.doc_id.split("#")[0]

    def search(self, query, k, *, principal=None):
        if self._strategy == "complete_graph":
            # ⭐ 只要实体沾边就全返回：可达性满分，精确性崩盘
            ents = {e for f, e in self._entity.items() if e in query}
            hits = [f for f, e in self._entity.items() if e in ents] or list(self._facts)
        elif self._strategy == "exact_only":
            # ⭐ 只认**完整指名**（整句或「实体+别名」）：
            # 精确性满分，⛔ 换个弱一点的说法就够不到
            hits = [f for f, t in self._facts.items()
                    if query in (t, DESIGNATION.get(f))]
        else:  # 逐词重合
            terms = set(query.replace("。", "").split())
            hits = sorted(
                (f for f in self._facts),
                key=lambda f: -len(terms & set(self._facts[f].replace("。", "").split())),
            )
            hits = [f for f in hits
                    if terms & set(self._facts[f].replace("。", "").split())]
        return [Entry(id=f, digest=self._facts[f], doc_ids=[f]) for f in hits[:k]]

    def count(self) -> int:
        return len(self._facts)


def run(strategy: str):
    arm = _Arm(strategy)
    for f in TOPO.facts:
        arm.ingest(Document(doc_id=f.fact_id, text=f.text))
    arm.finalize()
    return score(StructureSuite(TOPO).probe(arm, None)).metrics


def test_dumping_everything_reproduces_the_fan_effect() -> None:
    """⭐ 扇形效应的机制，实测撞见的。

    「实体沾边就全返回」的策略在低扇形度上完美（fan1 两条曲线都 1.00），
    到 fan16 双双塌到 0.21 / 0.06——**兄弟条目把目标挤出了 top-k**。
    ⚠️ 这不是实现 bug，这正是激活被分摊的那个机制。
    """
    m = run("complete_graph")
    assert m["精确检索_fan1"] == 1.0, "低扇形度上它是完美的"
    assert m["可达性_fan16"] < 0.3
    assert m["精确检索_fan16"] < 0.1
    # ⭐ 曲线一路走到 fan64 才见底——⛔ 前段全平的曲线量不出斜率
    assert m["精确检索_fan64"] < 0.05
    # 单调退化
    curve = [m[f"精确检索_fan{f}"] for f in FANS]
    assert all(a >= b for a, b in zip(curve[:-1], curve[1:], strict=True))


def test_an_under_specified_cue_cannot_reach_everything() -> None:
    """⚠️ `可达性_fan1` 不是 1.000，⛔ 那不是缺陷。

    第三个线索**只有宾语**、没有实体——⭐ 它本来就该够不到所有条目，
    fan1 那一档有 16 个实体在抢同一个兜底位置。
    ⛔ 把它调成 1.000 的唯一办法是让线索重新变成原文子串，
    那样这一档就又什么都不量了。
    """
    m = run("complete_graph")
    assert 0.5 < m["可达性_fan1"] < 1.0


def test_exact_only_maxes_precision_and_destroys_reach() -> None:
    """⭐ 「只报一条毫无意义」的证明：只认完整指名的系统精确性满分。

    ⛔ 只看精确检索，它是完美的；看可达性才发现**换个弱一点的说法**
    （实体+动词、或只有宾语）就够不到——这就是为什么两条曲线必须一起报。

    ⚠️ 可达性的地板是 1/3 而不是 0：⭐ 三个线索里有一个正是那句完整指名，
    ⛔ 这是刻意的——两条曲线是同一个挤压过程上的两道阈
    （先掉出第一名，再掉出 top-k）。
    """
    m = run("exact_only")
    for fan in FANS:
        assert m[f"精确检索_fan{fan}"] == 1.0
        assert abs(m[f"可达性_fan{fan}"] - 1 / 3) < 1e-9


def test_both_curves_are_reported_at_every_fan() -> None:
    m = run("overlap")
    for fan in FANS:
        assert f"可达性_fan{fan}" in m and f"精确检索_fan{fan}" in m


def test_no_single_composite_score() -> None:
    """⛔ 不设单一总分——权衡因用途而异，合成就是替使用者做了取舍。"""
    m = run("overlap")
    assert not any(k in m for k in ("总分", "结构效率", "score", "overall"))


def test_fan_effect_shows_as_a_negative_slope() -> None:
    """⭐ 关联越多，指名要某一条越难。"""
    assert run("complete_graph")["扇形退化斜率"] < 0
    # ⚠️ 一个不受扇形影响的策略斜率为 0——这一档能分辨出来
    assert run("exact_only")["扇形退化斜率"] == 0.0


# ── ⛔ 线索必须是「另一种说法」，不是原文子串 ──────────────────────
def test_a_cue_is_another_wording_not_a_substring() -> None:
    """⛔ 这一条是 `可达性` 这个指标成立的前提。

    ⚠️ 实测踩到：三个线索原先**全是原文的子串**，于是这一档量的是
    子串匹配——`bm25` 从 fan1 到 fan64 一路 1.000，
    ⭐ 而加大扇形度**改不了**这一点。
    ⛔ 「换个说法够不够得到」必须真的换了说法：别名与原词一个字都不共。
    """
    from amb.world.stream.topology import PAIRS

    for verb, obj, alias in PAIRS:
        assert not set(obj) & set(alias), f"{obj} / {alias} 共字"
        assert alias not in f"{verb} {obj}", f"{alias} 是原文子串"
    # ⚠️ 别名里也不许嵌着别的动词/宾语——⛔ 那会在同一个实体内指向两条事实
    verbs = {v for v, _, _ in PAIRS}
    objs = {o for _, o, _ in PAIRS}
    for _, _, alias in PAIRS:
        assert not any(v in alias for v in verbs)
        assert not any(o in alias for o in objs)


def test_every_fact_carries_a_paraphrase_cue() -> None:
    topo = build(seed=7)
    for f in topo.facts:
        assert len(f.cues) == 3
        # ⭐ 第二个线索是换过说法的那个：⛔ 不在原文里
        assert f.cues[1] not in f.text, f.cues[1]
        assert f.entity in f.cues[1], "⚠️ 但仍然指名了是哪个实体"


def test_one_entity_never_repeats_a_verb_or_object() -> None:
    """⛔ 否则扇形度涨的同时线索也变模糊了，两个成因读不开。"""
    topo = build(seed=11)
    per_entity: dict[str, list[str]] = {}
    for f in topo.facts:
        per_entity.setdefault(f.entity, []).append(f.text)
    for entity, texts in per_entity.items():
        verbs = [t.split()[1] for t in texts]
        objs = [t.split()[2] for t in texts]
        assert len(set(verbs)) == len(verbs), entity
        assert len(set(objs)) == len(objs), entity


def test_asking_for_a_wider_fan_than_the_pool_is_refused() -> None:
    """⛔ 不静默取满多少算多少——⚠️ 那会让 `fan128` 那一格其实只有 64 条，
    而**分数看上去很正常**。"""
    import pytest as _pytest

    from amb.world.stream.topology import FanTooWide, PAIRS

    with _pytest.raises(FanTooWide):
        build(seed=1, fans=(len(PAIRS) + 1,))


def test_the_curve_reaches_fan64() -> None:
    """⭐ fan1~fan8 全是 1.000——⛔ 一条前段全平的曲线量不出退化斜率。"""
    assert max(f.fan for f in build(seed=3).facts) == 64


# ── ⛔ 边界值最需要区间，⚠️ 而它们恰恰最容易丢 ──────────────────
def _run_with(rows) -> SuiteRun:
    run = SuiteRun("n6_structure", "scored")
    for i, payload in enumerate(rows):
        run.observations.append(Observation(f"i{i}", payload))
    return run


def _fan_metrics(got) -> list[str]:
    return [k for k in got.metrics if k.startswith(("可达性_fan", "精确检索_fan"))]


def test_a_degenerate_arm_still_gets_intervals_per_fan() -> None:
    """⛔ 全 0.000 的臂，每一档都要有区间。

    ⚠️ 2026-09-07 真跑实测：`null` 的 **14 个分档全部无区间**——
    ⭐ 因为它们名字里没有「率」，被判成非比例走了重抽样，
    而重抽样对常数每次都得同一个值 → 零宽 → 按规矩不给区间。
    ⛔ 于是「没有区间就不许声称差异」把这一档永远判成「分不开」。
    """
    got = score(_run_with([
        {"fan": 2 ** (i % 7), "reached": 0, "cues": 3, "precise": False}
        for i in range(112)]))
    missing = [k for k in _fan_metrics(got) if k not in got.intervals]
    assert not missing, f"⛔ 这些分档没有区间：{missing}"


def test_a_perfect_arm_still_gets_intervals_per_fan() -> None:
    """⛔ 1.000 也一样。

    ⚠️ 实测：`bm25` 的 `可达性_fan2=1.000` 不带区间印出来，
    ⭐ 读起来比同一列的 `0.938[0.87,1.00]` **更确定**——而两者 n 一样。
    """
    got = score(_run_with([
        {"fan": 2 ** (i % 7), "reached": 3, "cues": 3, "precise": True}
        for i in range(112)]))
    missing = [k for k in _fan_metrics(got) if k not in got.intervals]
    assert not missing, f"⛔ 这些分档没有区间：{missing}"


def test_each_fan_interval_uses_its_own_denominator() -> None:
    """⛔ 分档的分母是**这一档的条数**，⚠️ 不是全部观测数。

    ⭐ 用观测数会把区间压窄，⛔ 而区间重叠是唯一阻止「声称 A 比 B 好」的闸门。
    """
    got = score(_run_with([
        {"fan": 2 ** (i % 7), "reached": 0, "cues": 3, "precise": False}
        for i in range(112)]))
    for k in _fan_metrics(got):
        assert got.denominators.get(k) == 16, \
            f"⛔ {k} 的分母是 {got.denominators.get(k)}，⚠️ 该是这一档的 16"
        assert got.intervals[k].n == 16, f"⛔ {k} 区间的 n 也该是 16"
    # ⭐ 汇总仍然用全部观测数——⚠️ 它是跨档的
    assert got.intervals["可达性"].n == 112


def test_a_narrow_stratum_gets_a_wider_interval() -> None:
    """⭐ 分母小 → 区间宽。⛔ 这才是区间该有的行为。

    ⚠️ 反过来（分档与汇总一样宽）说明分母用错了。
    """
    got = score(_run_with([
        {"fan": 2 ** (i % 7), "reached": 0, "cues": 3, "precise": False}
        for i in range(112)]))
    per_fan = got.intervals["可达性_fan1"]
    overall = got.intervals["可达性"]
    assert per_fan.high > overall.high, \
        f"⛔ 一档 16 条的区间不该比 112 条的还窄：{per_fan.high} vs {overall.high}"
