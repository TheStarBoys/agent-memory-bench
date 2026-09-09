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
    from amb.scoring.statistics import looks_like_proportion

    r = _run(name, tmp_path)
    missing: list[str] = []
    for suite, v in r.scores.items():
        if v.status != "scored":
            continue
        for k, val in v.metrics.items():
            if looks_like_proportion(k) and k not in v.intervals:
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
def _all_metrics(tmp_path) -> list[tuple[str, str, str, float]]:
    """跑遍所有臂，收集**真实产生**的每一个指标值。

    ⭐ 不是我想出来的值，是流水线算出来的——⛔ 想出来的值只覆盖我想到的情形。
    """
    out = []
    for name in ARMS:
        r = _run(name, tmp_path)
        for suite, v in r.scores.items():
            if v.status == "scored":
                out += [(name, suite, k, val) for k, val in v.metrics.items()]
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
    from amb.scoring.statistics import kind_of

    bad = [f"{a}/{s}/{k}={v}" for a, s, k, v in _all_metrics(tmp_path)
           if kind_of(k) == "proportion" and not 0.0 <= v <= 1.0]
    assert not bad, f"⛔ 判成比例却不在 [0,1]：{bad[:8]}"


def test_a_metric_classified_a_count_is_actually_one(offline_world,
                                                     tmp_path) -> None:
    """⛔ 计数是**条数**，⚠️ 不可能是小数。

    ⭐ 反向的那一半：把比例误判成计数，它就永远拿不到区间——
    ⛔ 而那是静默的，报告照常印出一个没有区间的数。
    """
    from amb.scoring.statistics import kind_of

    bad = [f"{a}/{s}/{k}={v}" for a, s, k, v in _all_metrics(tmp_path)
           if kind_of(k) == "count" and float(v) != int(v)]
    assert not bad, f"⛔ 判成计数却是小数：{bad[:8]}"


def test_the_two_classifiers_never_disagree() -> None:
    """⛔ **只能有一个分类入口**。

    ⚠️ 早先 `looks_like_proportion` 与 `_COUNT_HINTS` 是两张互不相识的表，
    ⭐ `计数_全对` 被一张判成计数、被另一张判成比例——
    ⛔ 两者不打架只是因为调用方**恰好先用了计数那张**。那是运气。
    """
    from amb.scoring import metrics as m
    from amb.scoring.statistics import COUNT_HINTS

    assert m._COUNT_HINTS is COUNT_HINTS, "⛔ 又分成两张表了"
