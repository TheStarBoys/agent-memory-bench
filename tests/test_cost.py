"""成本 × 质量。

⭐ 一个什么都记得住但慢得要死的系统没有用——
用户要个东西等半天，那还不如不记。「又快又好」才是好。

⛔ 所以快慢必须**被判**，不只是被记录。
"""

from __future__ import annotations

from amb.scoring import CostProfile, Pricing, UNUSABLE_PROBE_MS, judge_cost


def profile(arm: str, ingest_ms: int, probe_ms: int, *, n_in: int = 60,
            n_probe: int = 20) -> CostProfile:
    return CostProfile(arm, {"ingest": ingest_ms, "probe": probe_ms},
                       items_ingested=n_in, items_probed=n_probe)


def test_the_best_quality_can_still_be_unusable() -> None:
    """⭐ 这一条就是这个维度存在的理由。

    质量最高的那条臂，如果回答一次要 45 秒，⛔ 记得住也救不回来。
    """
    costs = {"bm25": profile("bm25", 300, 200),
             "slow": profile("slow", 1000, 900_000)}
    verdicts = {v.arm: v for v in judge_cost({"bm25": 0.52, "slow": 0.95},
                                             costs, "bm25")}
    assert verdicts["slow"].quality > verdicts["bm25"].quality, "它确实更准"
    assert verdicts["slow"].label == "⛔ 慢到不可用"
    assert "记得住也救不回来" in verdicts["slow"].note


def test_worse_and_slower_is_dominated() -> None:
    """⛔ 既不如地板准、又比它慢 → 没有存在理由。"""
    costs = {"bm25": profile("bm25", 300, 200),
             "bad": profile("bad", 5000, 3000)}
    v = {x.arm: x for x in judge_cost({"bm25": 0.52, "bad": 0.40}, costs, "bm25")}
    assert v["bad"].label == "⛔ 被地板压制"
    assert "没有存在理由" in v["bad"].note


def test_better_and_cheaper_is_the_good_case() -> None:
    """⭐ 更准且更省——这才是记忆系统该有的样子。"""
    costs = {"bm25": profile("bm25", 300, 200),
             "good": profile("good", 200, 150)}
    v = {x.arm: x for x in judge_cost({"bm25": 0.52, "good": 0.70}, costs, "bm25")}
    assert v["good"].label == "⭐ 又快又好"


def test_better_but_pricier_shows_the_exchange_rate() -> None:
    """⚠️ 更准但更贵不是错——但要说清**多 1% 准确率花了多少倍时间**。"""
    costs = {"bm25": profile("bm25", 300, 200),
             "mem0": profile("mem0", 2_200_000, 40_000)}
    v = {x.arm: x for x in judge_cost({"bm25": 0.52, "mem0": 0.61}, costs, "bm25")}
    assert v["mem0"].label == "⚠️ 更准但更贵"
    assert "倍时间" in v["mem0"].note
    assert v["mem0"].cost_ratio > 1000


def test_per_item_timings_are_what_the_user_actually_feels() -> None:
    """⭐ 「每次回答多久」才是用户等的那个数，⚠️ 不是总耗时。"""
    p = profile("x", ingest_ms=60_000, probe_ms=40_000, n_in=60, n_probe=20)
    assert p.ingest_ms_per_item == 1000.0
    assert p.probe_ms_per_item == 2000.0


def test_missing_measurements_stay_none_not_zero() -> None:
    """⛔ 没测到就是 None，不拿 0 冒充「没花钱」。"""
    p = CostProfile("x")
    assert p.ingest_ms_per_item is None and p.probe_ms_per_item is None
    assert p.tokens_in is None and p.money_usd is None


def test_no_price_no_money_figure() -> None:
    """⛔ 没给价格就不报钱——不瞎估。"""
    assert Pricing("m").money(1000, 500) is None
    assert Pricing("m", 0.5, 1.5).money(1_000_000, 1_000_000) == 2.0


def test_the_unusable_threshold_is_documented_not_a_pass_mark() -> None:
    """⚠️ 它是个**要显眼标出来**的阈值，⛔ 不是及格线。"""
    from pathlib import Path

    from amb.scoring import cost as mod

    assert UNUSABLE_PROBE_MS == 10_000.0
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "不是及格线" in source, "⛔ 阈值旁边要写清它不是及格线"


def test_no_single_composite_score() -> None:
    """⛔ 不合成总分——快与准的权衡因用途而异。"""
    costs = {"a": profile("a", 100, 100), "b": profile("b", 200, 200)}
    for v in judge_cost({"a": 0.5, "b": 0.6}, costs, "a"):
        assert not hasattr(v, "score")
        assert not hasattr(v, "总分")


# ── ⭐ 不关 LLM 的提速手段 ──────────────────────────────────────
def test_cache_is_content_addressed_on_everything_that_matters(tmp_path) -> None:
    """⛔ 键漏掉任何一个影响输出的参数，就会串味。"""
    from amb.adapters.llm_cache import LLMCache

    c = LLMCache(tmp_path / "c.db")
    base = {"model": "m", "temperature": 0.0, "messages": [{"role": "u"}]}
    c.put(base, {"answer": "A"}, 1000)

    assert c.get(base)["answer"] == "A"
    # 换模型 / 换消息 / 换任一参数 → ⛔ 都不该命中
    assert c.get({**base, "model": "other"}) is None
    assert c.get({**base, "messages": [{"role": "v"}]}) is None
    assert c.get({**base, "max_tokens": 100}) is None


def test_sampling_temperature_is_never_cached(tmp_path) -> None:
    """⛔ temperature>0 时缓存会把随机性冻成一个固定答案。

    ⚠️ 系统本来会给出分布，缓存让它只给一个点——那改变了被测对象。
    """
    from amb.adapters.llm_cache import LLMCache

    c = LLMCache(tmp_path / "c.db")
    hot = {"model": "m", "temperature": 0.7, "messages": []}
    c.put(hot, {"answer": "A"}, 1000)
    assert c.get(hot) is None, "⛔ 不该缓存也不该命中"


def test_cache_stats_are_reportable(tmp_path) -> None:
    """⚠️ 命中率必须进报告——⛔ 命中 90% 的跑测的不是真延迟。"""
    from amb.adapters.llm_cache import LLMCache

    c = LLMCache(tmp_path / "c.db")
    p = {"model": "m", "temperature": 0.0, "messages": []}
    c.get(p)                       # miss
    c.put(p, {"a": 1}, 36700)
    c.get(p)                       # hit
    st = c.stats.as_dict()
    assert st["hits"] == 1 and st["misses"] == 1 and st["hit_rate"] == 0.5
    assert st["saved_ms"] == 36700


def test_snapshot_key_covers_everything_that_changes_ingest(tmp_path) -> None:
    """⛔ 拿错快照比慢更糟——它会静默给出别的系统的分。"""
    from amb.core import Document
    from amb.runner.snapshot import SnapshotKey, corpus_digest

    docs = [Document(doc_id="a", text="x"), Document(doc_id="b", text="y")]
    base = SnapshotKey("mem0", "2.0.19", "Qwen3-8B", corpus_digest(docs))
    for changed in (
        SnapshotKey("mem0_raw", "2.0.19", "Qwen3-8B", base.corpus_digest),
        SnapshotKey("mem0", "2.0.20", "Qwen3-8B", base.corpus_digest),
        SnapshotKey("mem0", "2.0.19", "other-llm", base.corpus_digest),
        SnapshotKey("mem0", "2.0.19", "Qwen3-8B", corpus_digest(docs[:1])),
    ):
        assert changed.digest != base.digest


def test_corpus_digest_is_order_sensitive() -> None:
    """⚠️ 归并型系统对摄入顺序敏感——⛔ 顺序必须进指纹。"""
    from amb.core import Document
    from amb.runner.snapshot import corpus_digest

    docs = [Document(doc_id="a", text="x"), Document(doc_id="b", text="y")]
    assert corpus_digest(docs) != corpus_digest(list(reversed(docs)))


def test_a_half_written_snapshot_is_not_restored(tmp_path) -> None:
    """⛔ 半截快照比没有更糟——只有 .complete 在才算数。"""
    from amb.core import Document
    from amb.runner.snapshot import SnapshotKey, corpus_digest, restore

    key = SnapshotKey("x", "1", "b", corpus_digest([Document(doc_id="a", text="t")]))
    root = tmp_path / "snap"
    (key.path(root) / "store").mkdir(parents=True)
    (key.path(root) / "store" / "f").write_text("half")
    # ⚠️ 故意不落 .complete
    assert restore(key, tmp_path / "out", root) is False


def test_concurrent_ingest_is_documented_as_unsafe() -> None:
    """⛔ 并发对归并型系统不安全——那不是加速，是换了个被测对象。

    ⚠️ 这条把理由钉在文档里，防止有人日后「顺手优化」。
    """
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "docs" / "cost-control.md"
           ).read_text(encoding="utf-8")
    assert "为什么并发摄入不能用" in doc
    assert "归并决策就变了" in doc
    assert "摄入结果依赖已摄入内容的系统" in doc


def test_mem0_pins_temperature_to_zero() -> None:
    """⛔ mem0 默认 temperature=0.1——判分要可复现，采样温度不该 >0。

    ⚠️ 这也是缓存能生效的前提：temperature>0 时我们不缓存
    （那会把随机性冻成一个固定答案）。
    ⭐ 实测：这一条没设时，缓存命中率恒为 0，而且**没有任何报错**——
    ⛔ 一个静默失效的优化，比没有优化更糟。
    """
    from amb.adapters.impl.mem0 import Mem0Adapter

    arm = Mem0Adapter(llm_model="m", llm_base_url="u", embed_model="e",
                      embed_base_url="u", embed_dims=8, storage_dir="/tmp/x")
    llm_cfg = arm._cfg["llm"]["config"]  # noqa: SLF001
    assert llm_cfg["temperature"] == 0.0
    assert llm_cfg["top_p"] == 1.0


def test_a_nonzero_temperature_payload_is_never_cached(tmp_path) -> None:
    """⛔ 这条纪律不因为「想让缓存生效」而放宽。"""
    from amb.adapters.llm_cache import LLMCache

    c = LLMCache(tmp_path / "c.db")
    for temp in (0.1, 0.7, 1.0):
        p = {"model": "m", "temperature": temp, "messages": []}
        c.put(p, {"a": 1}, 100)
        assert c.get(p) is None, f"temperature={temp} ⛔ 不该缓存"
    # 只有 0.0 才缓存
    zero = {"model": "m", "temperature": 0.0, "messages": []}
    c.put(zero, {"a": 1}, 100)
    assert c.get(zero) is not None


# ── ⭐ 「缓存为什么没生效」的可观测性 ─────────────────────────────
def test_every_skip_reason_is_counted(tmp_path) -> None:
    """⛔ 「命中率 0」有很多种原因，不分开记就查不出是哪一种。

    ⚠️ 实测踩过：mem0 默认 temperature=0.1，缓存静默失效，
    **连异常都没抛**——查了很久才定位。
    """
    from amb.adapters.llm_cache import LLMCache, Skip

    off = LLMCache(tmp_path / "a.db", enabled=False)
    off.get({"temperature": 0.0})
    assert off.stats.skipped[str(Skip.DISABLED)] == 1

    hot = LLMCache(tmp_path / "b.db")
    hot.get({"temperature": 0.1})
    assert hot.stats.skipped[str(Skip.SAMPLING)] == 1


def test_diagnosis_names_the_cause_and_the_next_step(tmp_path) -> None:
    """⭐ 光说「跳过了」不够——要说**下一步做什么**。"""
    from amb.adapters.llm_cache import LLMCache

    hot = LLMCache(tmp_path / "b.db")
    hot.get({"temperature": 0.1})
    d = hot.stats.diagnosis()
    assert "temperature>0" in d
    assert "钉成 0" in d, "⛔ 要给出下一步，不能只报现象"
    assert "mem0 默认是 0.1" in d, "⚠️ 把踩过的坑写进提示"


def test_diagnosis_distinguishes_never_called_from_never_hit(tmp_path) -> None:
    """⚠️ 「一次都没调 LLM」与「调了但没命中」是两件事。"""
    from amb.adapters.llm_cache import LLMCache

    idle = LLMCache(tmp_path / "c.db")
    assert "一次 LLM 调用都没发生" in idle.stats.diagnosis()

    never = LLMCache(tmp_path / "d.db")
    for i in range(3):                       # 每次内容都不同 → 永不命中
        never.get({"temperature": 0.0, "messages": [{"c": i}]})
    d = never.stats.diagnosis()
    assert "一次没中" in d and "每次都不同" in d


def test_a_working_cache_reports_what_it_saved(tmp_path) -> None:
    from amb.adapters.llm_cache import LLMCache

    c = LLMCache(tmp_path / "e.db")
    p = {"temperature": 0.0, "messages": []}
    c.get(p)
    c.put(p, {"x": 1}, 78_000)
    c.get(p)
    d = c.stats.diagnosis()
    assert "命中 1/2" in d and "78s" in d


def test_report_flags_a_cached_run_as_not_a_latency_measurement() -> None:
    """⛔ 命中率高的跑测出来的「延迟」不是真延迟——必须显眼。"""
    from amb.report import ArmResult, Report, render

    report = Report(run_id="t", at="t",
                    world={"name": "x", "seed": 1, "digest": "d"},
                    backbone={"model": "m"},
                    cache={"hits": 9, "misses": 1, "hit_rate": 0.9,
                           "saved_ms": 700_000, "skipped": {},
                           "diagnosis": "✓ 命中 9/10"},
                    lanes={"library": [ArmResult(arm="a", is_control=True)]})
    text = render(report)
    assert "缓存命中 9/10" in text
    assert "不是独立测量" in text


def test_money_is_measured_not_estimated() -> None:
    """⛔ 有价格表却一个 token 都没测——那时报告里钱那一列永远是空的。

    ⚠️ 原则⑥ 说「token 只有适配器报得出来」指的是**被测系统自报**。
    ⭐ 但我们拦着每一次 openai 调用，usage 就在响应里——
    那是**我们测的**，比自报可信，也不要求它声明 ACCOUNTING。
    """
    from amb.adapters.llm_cache import Meter

    m = Meter()

    class _U:
        prompt_tokens, completion_tokens = 8413, 150

    m.add(_U())
    m.add(_U())
    got = m.as_dict()
    assert got["tokens_in"] == 16826 and got["llm_calls"] == 2
    # ⚠️ 重试等待单独报——⛔ 不能算进「这个系统很慢」
    assert "retry_waited_s" in got


def test_cache_hits_are_not_counted_as_spend() -> None:
    """⚠️ 缓存命中那次没真花钱——⛔ 算进去会让成本虚高。"""
    from amb.adapters.llm_cache import Meter

    m = Meter()
    m.cached_calls += 3
    assert m.as_dict()["tokens_in"] == 0
    assert m.as_dict()["cached_calls"] == 3


def test_the_cost_table_shows_money() -> None:
    """⛔ 算出来不印出来等于没算。"""
    from amb.report.render import _render_cost
    from amb.report.schema import ArmResult, Score

    arms = [
        ArmResult(arm="mem0", is_control=False,
                  scores={"locomo_retrieval": Score(
                      suite="locomo_retrieval", status="scored",
                      metrics={"evidence_recall": 0.7})},
                  cost={"ingest": 1000, "probe": 100},
                  cost_profile={"items_ingested": 10, "items_probed": 5,
                                "tokens_in": 84130, "tokens_out": 1500}),
        ArmResult(arm="naive_rag", is_control=True,
                  scores={"locomo_retrieval": Score(
                      suite="locomo_retrieval", status="scored",
                      metrics={"evidence_recall": 0.6})},
                  cost={"ingest": 100, "probe": 100},
                  cost_profile={"items_ingested": 10, "items_probed": 5}),
    ]
    text = "\n".join(_render_cost(arms, ["locomo_retrieval"], "Qwen/Qwen3-8B"))
    assert "钱" in text and "$" in text
    # ⛔ 没测到的臂写 —，⚠️ 不拿 0 冒充「没花钱」
    assert "—" in text


def test_full_context_is_flagged_as_degenerate_in_retrieval() -> None:
    """⛔ `full_context` 在检索档里把**全部语料**交出去（query/k 刻意忽略），
    所以 evidence_recall 必然满分。

    ⚠️ 一行 `full_context = 1.000` 会被读成「天花板很高」，
    实际是**分母被绕过了**。⭐ 它有意义的地方在回答档。
    ⛔ 不标出来，这张表就在骗人。
    """
    from amb.report.render import _render_lane
    from amb.report.schema import ArmResult, Report, Score

    def _arm(name, control, value):
        return ArmResult(arm=name, is_control=control,
                         scores={"locomo_retrieval": Score(
                             suite="locomo_retrieval", status="scored",
                             metrics={"evidence_recall": value})})

    report = Report(run_id="r", at="t",
                    world={"name": "w", "seed": 1, "digest": ""},
                    backbone={}, externals={}, sampling={})
    text = _render_lane("library", [_arm("full_context", True, 1.0),
                                    _arm("naive_rag", True, 0.64),
                                    _arm("a_mem", False, 0.7)], report)

    assert "退化†" in text            # ⭐ 那一行被标了
    assert "分母被绕过" in text        # ⛔ 且脚注解释了为什么——不能是孤儿脚注


# ── 走子进程的臂：两个计量器都要报 ────────────────────────────
class _TwoMeterArm:
    """一条走子进程的臂：摄入的 token 在子进程，答题的在宿主。"""

    name = "mem0"

    def __init__(self, ingest: tuple[int, int], probe: tuple[int, int] | None):
        self._ingest = ingest
        self._probe = probe

    def usage(self):
        from amb.core import Usage

        out = [Usage(phase="ingest", tokens_in=self._ingest[0],
                     tokens_out=self._ingest[1], llm_calls=419)]
        if self._probe is not None:
            out.append(Usage(phase="probe", tokens_in=self._probe[0],
                             tokens_out=self._probe[1], llm_calls=126))
        return out


def test_snapshot_stores_ingest_tokens_only() -> None:
    """⛔ 快照存的是**摄入**成本——⚠️ 混进答题的 token 会虚报一次摄入的钱。

    ⭐ 实测背景：回答档上线后，同一条臂的 `usage()` 会同时带回
    子进程的摄入用量与宿主 backbone 的答题用量。
    """
    from amb.runner.phases import _ingest_tokens

    arm = _TwoMeterArm(ingest=(3_634_599, 40_142), probe=(31_898, 586))
    got = _ingest_tokens(arm.usage())
    assert got == {"tokens_in": 3_634_599, "tokens_out": 40_142,
                   "llm_calls": 419}, "⛔ 答题的 token 混进摄入成本了"


def test_snapshot_cost_is_empty_when_nothing_was_ingested() -> None:
    """⚠️ 命中快照那一跑没摄入——⛔ 别把答题的 token 当成摄入成本存下去。"""
    from amb.runner.phases import _ingest_tokens
    from amb.core import Unsupported, Usage

    only_probe = [Usage(phase="probe", tokens_in=31_898, tokens_out=586,
                        llm_calls=126)]
    assert _ingest_tokens(only_probe) == {}
    assert _ingest_tokens(Unsupported("没跑过")) == {}
    assert _ingest_tokens([]) == {}


def test_subprocess_arm_reports_both_meters() -> None:
    """⛔ 回答档里 `mem0` 答了 126 次，`tokens_in` 却报 0——修的就是这个。

    ⚠️ 两个来源分得开：摄入的 LLM 在子进程，答题的 backbone 在宿主。
    ⭐ 只报一个，钱那一列就是错的。
    """
    from amb.adapters.impl.mem0.adapter import Mem0Adapter
    from amb.adapters.llm import LLMConfig
    from amb.core import Unsupported

    arm = Mem0Adapter(llm_model="m", llm_base_url="u", embed_model="e",
                      embed_base_url="u", embed_dims=8, storage_dir="/tmp/x")
    # 没起桥、没挂 backbone → ⛔ 无从计量，不拿 0 冒充
    assert isinstance(arm.usage(), Unsupported)

    arm.attach_llm(LLMConfig(model="m", base_url="u", api_key_env="K"))
    arm._llm.meter.add({"prompt_tokens": 31_898, "completion_tokens": 586})
    got = arm.usage()
    assert [u.phase for u in got] == ["probe"], \
        "⛔ 挂了 backbone 之后答题用量必须报出来"
    assert got[0].tokens_in == 31_898


def test_sub_cent_costs_are_not_all_rendered_as_zero() -> None:
    """⛔ 三位小数把一整档的成本压成 `$0.000`——那一栏就等于没有。

    ⚠️ 实测：17 题回答档四条臂全显示 `$0.000`，而它们真实差着 3 倍。
    """
    from amb.report.render import _money

    assert _money(0.2205) == "$0.221"
    assert _money(0.00024) != _money(0.00071), "⛔ 差 3 倍的两笔钱显示成了同一个数"
    assert _money(0.0) == "$0"


# ── ⛔ 一个不区分各条臂的数，排不出名次 ──────────────────────────
def test_a_tie_on_both_axes_is_not_domination() -> None:
    """⛔ 「被压制」要求至少有一轴**严格**更差。

    ⚠️ 实测踩到：质量列在所有臂上都是 0.000、耗时比 1.0x，
    报告照样印出「bm25 既不如它准，又不比它快——没有存在理由」。
    ⭐ 那不是判定，那是把一个不区分的数当成了排名依据。
    """
    from amb.scoring import CostProfile, judge_cost

    profiles = {n: CostProfile(arm=n, wall_ms={"probe": 1000},
                               items_probed=10) for n in ("floor", "same")}
    verdicts = {v.arm: v for v in judge_cost({"floor": 0.0, "same": 0.0},
                                             profiles, "floor")}
    assert verdicts["same"].label == "与地板并列"
    assert "没有存在理由" not in verdicts["same"].note


def test_still_dominated_when_strictly_slower() -> None:
    """⚠️ 反过来要还认得出来：⛔ 分一样但更慢，那确实没有存在理由。"""
    from amb.scoring import CostProfile, judge_cost

    profiles = {
        "floor": CostProfile(arm="floor", wall_ms={"probe": 1000},
                             items_probed=10),
        "slow": CostProfile(arm="slow", wall_ms={"probe": 3000},
                            items_probed=10),
    }
    verdicts = {v.arm: v for v in judge_cost({"floor": 0.5, "slow": 0.5},
                                             profiles, "floor")}
    assert verdicts["slow"].label == "⛔ 被地板压制"


def test_an_unpublishable_suite_cannot_become_the_quality_column() -> None:
    """⛔ ground truth 立不住的档，不许当成本×质量表的质量列。

    ⚠️ 实测踩到：N5 的需求概率曲线还是占位的（`不得发布`），
    ⭐ 但它参与面最广，于是当上了质量列——而它在所有臂上都是 0.000，
    ⛔ 表里于是印出「bm25 没有存在理由」。
    """
    from amb.report.render import _render_cost
    from amb.report.schema import ArmResult
    from amb.scoring import Score

    def arm(name: str, retrieval: float, retention: float) -> ArmResult:
        a = ArmResult(arm=name, is_control=True)
        a.scores["retrieval"] = Score("retrieval", "scored",
                                      metrics={"top1": retrieval})
        a.scores["n5_observed"] = Score(
            "n5_observed", "scored", metrics={"保留追踪度": retention},
            not_publishable="需求概率曲线未拟合")
        a.cost_profile = {"items_probed": 10}
        a.cost = {"probe": 1000}
        return a

    text = "\n".join(_render_cost([arm("bm25", 0.9, 0.0), arm("null", 0.0, 0.0)],
                                  ["n5_observed", "retrieval"]))
    assert "retrieval" in text and "n5_observed" not in text


def test_a_flat_quality_column_prints_no_verdict() -> None:
    """⚠️ 挑不出别的档时，⛔ 只报成本，不报判定。"""
    from amb.report.render import _render_cost
    from amb.report.schema import ArmResult
    from amb.scoring import Score

    def arm(name: str) -> ArmResult:
        a = ArmResult(arm=name, is_control=True)
        a.scores["retrieval"] = Score("retrieval", "scored",
                                      metrics={"top1": 0.0})
        a.cost_profile = {"items_probed": 10}
        a.cost = {"probe": 1000}
        return a

    text = "\n".join(_render_cost([arm("bm25"), arm("null")], ["retrieval"]))
    assert "不给判定" in text
    assert "没有存在理由" not in text and "被地板压制" not in text
