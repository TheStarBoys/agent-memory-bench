"""⭐ 缺的那一层：**离线跑完整流水线**。

## ⛔ 为什么加它

⚠️ 一次会话里 7 个 bug **全部**只有真跑（2.5 小时）才发现，
而全套 713 个测试是绿的。⭐ 查下来成因是同一个：

    真正的入口   build(臂名) → run_one(五阶段) → score
    测试走的     create(手填参数) 或 手搓一个假臂

⛔ `build()` 正是 env 变量、`storage_dir`、embedding 配置、臂名分派的所在地，
⚠️ 也正是那 7 个 bug 全部的所在地——而它此前只被 `null` / `bm25` /
`host_default` 走过，⭐ 恰好是三条**不需要外部配置**的臂。

## ⚠️ 这一层**不测**什么

⛔ 不测分数对不对：假 embedding 与假 LLM 给的分没有意义。
⭐ 它测的是**接线**：造得出来吗、跑得完吗、该有的东西有没有、
不该发生的事发生没有。⚠️ 分数的正确性归各套件自己的单元测试。
"""

from __future__ import annotations

import pytest

import offline
import worlds.toy as toy
from amb.runner import Plan, backbone, build, run_one

#: ⛔ **每一条对照臂都要在**：⚠️ 早先只有不需要配置的三条进得来。
#: ⭐ `hybrid` 尤其——它此前**一次都没被 `build()` 造过**，
#: 于是给它多传一个参数就是 TypeError，而 683 个测试照样绿。
#: ⚠️ `test_architecture` 守着这份名单：⛔ 加了新臂却不加进来，那条守卫会红。
ARMS = ("null", "bm25", "naive_rag", "hybrid", "host_default", "full_context")

#: ⚠️ 40 篇够跑完全部套件，⛔ 623 篇只是让这一层变慢。
CORPUS = 40


@pytest.fixture
def offline_world(monkeypatch, tmp_path):
    """⭐ 假在**网络出口**，⛔ 不假在我们自己的代码上。"""
    offline.use_env(monkeypatch)
    monkeypatch.chdir(tmp_path)       # ⚠️ 快照根与默认 store 都是相对 cwd 的
    return offline.install(monkeypatch)


def _plan() -> Plan:
    return Plan(manifest=toy.MANIFEST, documents=toy.all_documents()[:CORPUS],
                changes=toy.CHANGES, suites_for=toy.suites)


#: ⚠️ 世界目录每次都要新的：⛔ `materialize` 会把文件钉成只读，
#: 同一个目录跑第二次会 PermissionError——⭐ 真跑里每条臂本来就各有一个。
_seq = iter(range(10_000))


def _run(name: str, tmp_path, **kw):
    world = tmp_path / f"w-{name}-{next(_seq)}"
    # ⛔ **必须给 `rebuild`**：⚠️ N4 第 3 步（删完重开，看还在不在）要能再造
    # 一个适配器，⭐ 没给就整档不跑——而 N4 是判定最集中的一档。
    # ⚠️ 踩过：这一层最早不传它，于是「判定必须是 0/1」那两条不变量
    # **一条数据都没覆盖到**，绿着却什么都没测。
    kw.setdefault("rebuild", lambda n=name: build(n, llm=backbone()))
    return run_one(name, build(name, llm=backbone()), _plan(),
                   world, is_control=True, **kw)[0]


# ── ⛔ ① 每条臂都造得出来、跑得完 ────────────────────────────────
@pytest.mark.parametrize("name", ARMS)
def test_every_arm_survives_the_whole_pipeline(offline_world, tmp_path,
                                               name: str) -> None:
    """⛔ 造得出来、五阶段跑得完、没有一个套件跑挂。

    ⚠️ 实测抓到过：`build("hybrid")` 直接 TypeError——
    ⭐ 给它传了 `storage_dir=` 而它的 `__init__` 不认识这个参数。
    ⛔ 全套测试当时是绿的，因为**没有一个测试走 `build()` 造这条臂**。
    """
    r = _run(name, tmp_path)
    assert r.crashed is None, f"⛔ {name} 跑挂了：{r.crashed}"
    assert r.scores, f"⛔ {name} 一个套件都没跑"
    broken = {s: v.reason for s, v in r.scores.items()
              if v.status in ("crashed", "harness_fault")}
    assert not broken, f"⛔ {name} 有套件跑挂：{broken}"


# ── ⛔ ② 声明了什么就得交出什么 ──────────────────────────────────
@pytest.mark.parametrize("name", ARMS)
def test_a_declared_capability_is_actually_delivered(offline_world, tmp_path,
                                                     name: str) -> None:
    """⛔ 声明 PROVENANCE 却给不出区间，是**静默降级**——

    ⚠️ 实测踩到：`mem0_raw` 的原文表在 store 外面，命中快照时 spans 空掉，
    ⭐ 而分数照常算得出来。这一条把「声明」与「行为」对上。
    """
    from amb.core import Capability

    r = _run(name, tmp_path)
    caps = set(r.declared)
    if str(Capability.PROVENANCE) not in {str(c) for c in caps}:
        pytest.skip(f"{name} 没声明 PROVENANCE")
    canary = r.cost_profile.get("canary") or {}
    if not canary.get("hits"):
        pytest.skip("这条臂在指纹检索上没命中，⚠️ 区间无从谈起")
    assert canary.get("with_spans"), \
        f"⛔ {name} 声明了 PROVENANCE，⚠️ 而行为指纹里一个区间都没有"


# ── ⛔ ③ 摄入前库必须是空的 ─────────────────────────────────────
@pytest.mark.parametrize("name", ARMS)
def test_nothing_is_left_over_from_a_previous_run(offline_world, tmp_path,
                                                  name: str) -> None:
    """⛔ 残留 = 这一跑的语料是**重的**，⚠️ 而分数看上去完全正常。

    ⭐ 实测后果：每条两份 → top-k 去重后只剩一半 → recall 0.789 → 0.474，
    ⛔ 全程无异常、无告警。
    """
    _run(name, tmp_path)
    r = _run(name, tmp_path)          # ⭐ 同一个 cwd 跑第二次
    assert not r.cost_profile.get("pre_ingest_count"), \
        f"⛔ {name} 第二跑摄入前库里还有 {r.cost_profile['pre_ingest_count']} 条"


# ── ⛔ ④ 两条臂不许共用一个持久层 ────────────────────────────────
def test_no_two_arms_share_a_store(offline_world) -> None:
    """⛔ 索引文件同名，⚠️ 共用等于后跑的覆盖先跑的——

    ⭐ 而覆盖之后 `count()` 仍然对，只是内容是**另一条臂的**。
    """
    from amb.core import Unsupported

    seen: dict[str, str] = {}
    for name in ARMS:
        places = build(name).storage_locations()
        if isinstance(places, Unsupported) or not places:
            continue
        for p in places:
            assert p not in seen, f"⛔ {name} 与 {seen[p]} 共用 {p}"
            seen[p] = name


# ── ⛔ ⑤ 摄入快照：存得下、命中了、且**真的没重摄** ──────────────
@pytest.mark.parametrize("name", ("naive_rag", "hybrid"))
def test_a_snapshot_actually_skips_the_ingest(offline_world, tmp_path,
                                              name: str) -> None:
    """⭐ 端到端：第二跑不许再发**批量** embedding。

    ⛔ 光看「命中」两个字不够：⚠️ 恢复之后适配器可能拿不到索引，
    于是它**静默地从零重摄**，而报告照样印「命中」。
    ⭐ 所以断言看的是**线上有没有批量请求**。
    """
    first = _run(name, tmp_path, backbone="bb")
    assert first.ingest_snapshot == "已存", first.ingest_snapshot

    offline_world.embed_calls.clear()
    second = _run(name, tmp_path / "second", backbone="bb")
    assert second.ingest_snapshot.startswith("命中"), second.ingest_snapshot
    assert offline_world.biggest_embed_batch <= 1, (
        f"⛔ {name} 命中快照却还在批量 embedding："
        f"最大批 {offline_world.biggest_embed_batch}——⚠️ 摄入被重跑了")


# ── ⛔ ⑥ 新检出的仓库，第一跑就该能存快照 ────────────────────────
@pytest.mark.parametrize("name", ("naive_rag", "hybrid"))
def test_a_fresh_checkout_can_snapshot_immediately(offline_world, tmp_path,
                                                   name: str) -> None:
    """⚠️ `_store_of()` 要求 store 的父目录存在（防「整个仓库当 store」），
    ⛔ 而新检出的仓库里 `.external/` 还不存在——⭐ 早先因此静默不存快照。
    """
    assert not (tmp_path / ".external").exists()
    r = _run(name, tmp_path, backbone="bb")
    assert r.ingest_snapshot == "已存", \
        f"⛔ {name} 第一跑没存快照：{r.ingest_snapshot}"


# ── ⛔ ⑦ 每个比例都要有区间 ─────────────────────────────────────
@pytest.mark.parametrize("name", ARMS)
def test_every_proportion_carries_an_interval(offline_world, tmp_path,
                                              name: str) -> None:
    """⛔ 不带区间的抽样分是在骗人。

    ⚠️ 实测抓到：值恰好是 0.000 或 1.000 时区间静默消失——
    ⭐ 而那两个值**恰恰最需要区间**：`0/16` 与 `0/1000` 都印成 `0.000`。
    ⛔ `null` 那条臂**每个指标都是 0.000**，所以它是这条检查的主力。
    """
    r = _run(name, tmp_path)
    missing: list[str] = []
    for suite, v in r.scores.items():
        if v.status != "scored":
            continue
        for k, val in v.metrics.items():
            if v.kinds.get(k) == "rate" and k not in v.intervals:
                missing.append(f"{suite}/{k}={val:.3f}")
    assert not missing, f"⛔ {name} 这些比例没有区间：{missing[:8]}"


# ── ⛔ ⑧ 这一层自己不许偷偷上网 ──────────────────────────────────
def test_this_layer_never_touches_the_real_network(offline_world, tmp_path):
    """⛔ 一个偷偷发真请求的「离线」测试，⚠️ 会在 CI 上时红时绿，
    ⭐ 而且花钱。这一条守住它自己的前提。
    """
    _run("naive_rag", tmp_path)
    assert offline_world.embed_texts > 0, "⛔ 一次都没走假端点，⚠️ 那它测了什么"
    assert offline_world.chat_calls > 0, "⛔ 答题档没被走到"


# ── ⛔ ⑨ 分类与实际值必须自洽 ───────────────────────────────────
def _all_metrics(tmp_path) -> list[tuple[str, str, str, float, str]]:
    """跑遍所有臂，收集**真实产生**的每一个指标值。

    ⭐ 不是我想出来的值，是流水线算出来的——⛔ 想出来的值只覆盖我想到的情形。
    """
    out = []
    for name in ARMS:
        r = _run(name, tmp_path)
        for suite, v in r.scores.items():
            if v.status == "scored":
                out += [(name, suite, k, val, v.kinds.get(k))
                        for k, val in v.metrics.items()]
    return out


def test_every_metric_without_an_interval_says_why(offline_world,
                                                   tmp_path) -> None:
    """⛔ 「没有区间」有两种，⚠️ 而报告里它们长得一模一样：

        ① 它是原始计数——本来就不配区间
        ② 估计量在这批观测上**没有信息**（退化臂每次重抽都是同一个值）

    ⭐ 读者分不出「估不出来」与「忘了算」，⛔ 而后者是个 bug。
    ⚠️ 这一条**不问分类器**——⛔ 拿被测代码去推导期望值，
    正确的和错误的实现同样容易通过。⭐ 它只问：沉默有没有署名。
    """
    naked = []
    for name in ARMS:
        r = _run(name, tmp_path)
        for suite, v in r.scores.items():
            if v.status != "scored":
                continue
            naked += [f"{name}/{suite}/{k}" for k in v.metrics
                      if k not in v.intervals and k not in v.no_interval]
    assert not naked, f"⛔ 这些指标没有区间也没说为什么：{naked[:8]}"


def test_the_same_metric_never_has_an_interval_on_one_arm_and_not_another(
        offline_world, tmp_path) -> None:
    """⛔ 同一个指标在 A 臂有区间、在 B 臂没有——⚠️ 那是估计量**在边界值上
    掉出去了**，⭐ 而边界值恰恰最需要区间。

    ⚠️ 实测：`null` 的 `保留追踪度` 印成 0.000 无区间，
    同一列 `bm25` 是 -0.452 有区间。⛔ 允许这样，但**必须署名**。
    ⭐ 这一条不问分类器，只问：差别有没有被解释。
    """
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for name in ARMS:
        r = _run(name, tmp_path)
        for suite, v in r.scores.items():
            if v.status != "scored":
                continue
            for k in v.metrics:
                seen.setdefault((suite, k), {})[name] = (
                    "有" if k in v.intervals else v.no_interval.get(k, "⛔ 没说"))
    silent = {f"{s}/{k}": per for (s, k), per in seen.items()
              if "有" in per.values() and "⛔ 没说" in per.values()}
    assert not silent, f"⛔ 区间时有时无且没解释：{list(silent)[:6]}"


def test_a_metric_classified_a_proportion_is_actually_one(offline_world,
                                                          tmp_path) -> None:
    """⛔ 比例不可能大于 1，⚠️ 也不可能小于 0。

    ⭐ 这条**不需要我维护任何名单**：分类错了，值自己会露馅。
    ⚠️ 实测抓到：`计数_全对`（一个原始条数）被判成比例——
    ⛔ 因为它名字里有「全对」二字，而那是比例关键词。
    """
    bad = [f"{a}/{s}/{k}={v}" for a, s, k, v, kind in _all_metrics(tmp_path)
           if kind == "rate" and not 0.0 <= v <= 1.0]
    assert not bad, f"⛔ 声明成比例却不在 [0,1]：{bad[:8]}"


def test_a_metric_classified_a_count_is_actually_one(offline_world,
                                                     tmp_path) -> None:
    """⛔ 计数是**条数**，⚠️ 不可能是小数。

    ⭐ 反向的那一半：把比例误判成计数，它就永远拿不到区间——
    ⛔ 而那是静默的，报告照常印出一个没有区间的数。
    """
    bad = [f"{a}/{s}/{k}={v}" for a, s, k, v, kind in _all_metrics(tmp_path)
           if kind == "count" and float(v) != int(v)]
    assert not bad, f"⛔ 声明成计数却是小数：{bad[:8]}"


def test_a_metric_classified_a_verdict_is_actually_one(offline_world,
                                                       tmp_path) -> None:
    """⛔ 声明成判定的，值必须**恰好**是 0 或 1。

    ⭐ 这是判定比计数紧的那一格不变量：⚠️ `留痕_有删除事件=3` 是个 bug，
    ⛔ 而「是整数」那条检查抓不住它。
    """
    bad = [f"{a}/{s}/{k}={v}" for a, s, k, v, kind in _all_metrics(tmp_path)
           if kind == "flag" and v not in (0.0, 1.0)]
    assert not bad, f"⛔ 声明成判定却不是 0/1：{bad[:8]}"


def test_a_verdict_never_carries_an_interval(offline_world, tmp_path) -> None:
    """⛔ 判定没有可估计的总体——⚠️ 给它区间等于假装做了 n 次试验。"""
    wrong = []
    for name in ARMS:
        r = _run(name, tmp_path)
        for suite, v in r.scores.items():
            if v.status != "scored":
                continue
            wrong += [f"{name}/{suite}/{k}" for k, kind in v.kinds.items()
                      if kind == "flag" and k in v.intervals]
    assert not wrong, f"⛔ 判定拿到了区间：{wrong[:8]}"


def test_every_produced_metric_declares_a_valid_kind(offline_world,
                                                     tmp_path) -> None:
    """⛔ 真跑出来的**每一个**指标都带着合法声明。

    ⚠️ `_finish()` 已经在卡口上拦了，⭐ 这一条是端到端的复核：
    6 条臂 × 11 套件真跑一遍，⛔ 确认没有指标从别的门溜出去。
    """
    from amb.scoring.metrics import KINDS

    bad = [f"{a}/{s}/{k}" for a, s, k, _v, kind in _all_metrics(tmp_path)
           if kind not in KINDS]
    assert not bad, f"⛔ 这些指标没有合法声明：{bad[:8]}"


def test_a_declared_rate_always_carries_a_denominator(offline_world,
                                                      tmp_path) -> None:
    """⛔ 声明成比例就必须有分母——⚠️ 没有分母的比例配不出 Wilson 区间。

    ⭐ 这一条守的是那个老 bug 的反面：分母缺失时区间退回用观测数，
    ⛔ 被压窄 → 与地板不重叠 → 报告直接印显著差异。
    """
    naked = []
    for name in ARMS:
        r = _run(name, tmp_path)
        for suite, v in r.scores.items():
            if v.status != "scored":
                continue
            naked += [f"{name}/{suite}/{k}" for k, kind in v.kinds.items()
                      if kind == "rate" and k not in v.denominators]
    assert not naked, f"⛔ 这些比例没有分母：{naked[:8]}"


def test_this_layer_actually_exercises_every_kind(offline_world,
                                                  tmp_path) -> None:
    """⛔ **防空转**：上面那些「声明成 X 的必须满足 Y」，
    ⚠️ 只有真产出过 X 才算测了东西。

    ⭐ 实测踩到两次：一次是这一层不传 `rebuild` → N4 整档不跑 →
    「判定必须是 0/1」一条数据都没覆盖到；⚠️ 另一次是比对判分改动的底片
    漏了 n4/n7/locomo，⛔ 于是「0 变化」那个结论当时只对一半成立。
    ⭐ 一条绿着却什么都没测的断言，比没有更糟——它让人以为查过了。
    """
    from amb.scoring.metrics import KINDS

    seen = {k for _a, _s, _k, _v, kind in _all_metrics(tmp_path) for k in [kind]}
    missing = sorted(set(KINDS) - seen)
    assert not missing, (
        f"⛔ 这一层没产出这些种类：{missing}——"
        f"⚠️ 相关的不变量因此是空转的，⭐ 补一条会产出它的臂或套件")
