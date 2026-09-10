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
                      denominator=126,
                      metrics={"evidence_recall": 0.7})},
                  cost={"ingest": 1000, "probe": 100},
                  cost_profile={"items_ingested": 10, "items_probed": 5,
                                "tokens_in": 84130, "tokens_out": 1500}),
        ArmResult(arm="naive_rag", is_control=True,
                  scores={"locomo_retrieval": Score(
                      suite="locomo_retrieval", status="scored",
                      denominator=126,
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
        a.scores["retrieval"] = Score("retrieval", "scored", denominator=40, metrics={"top1": retrieval})
        a.scores["n5_observed"] = Score(
            "n5_observed", "scored", denominator=40,
            metrics={"保留追踪度": retention},
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
        a.scores["retrieval"] = Score("retrieval", "scored", denominator=40, metrics={"top1": 0.0})
        a.cost_profile = {"items_probed": 10}
        a.cost = {"probe": 1000}
        return a

    text = "\n".join(_render_cost([arm("bm25"), arm("null")], ["retrieval"]))
    assert "不给判定" in text
    assert "没有存在理由" not in text and "被地板压制" not in text


# ── ⛔ 指标的方向：少标一个，这个指标上每一句结论都是反的 ──────────
def test_a_lower_is_better_metric_picks_the_lowest_floor() -> None:
    """⛔ 实测踩到：`ECE` 没标方向 → `best_floor` 取 `max` →
    **校准最差**的那条当地板 → 一个 ECE 从 0.40 降到 0.05 的系统
    被印成「⚠️帮倒忙」。
    """
    from amb.report import ArmResult
    from amb.report.floor import best_floor, delta
    from amb.scoring import Score

    def arm(name: str, ece: float, control: bool = True) -> ArmResult:
        a = ArmResult(arm=name, is_control=control)
        a.scores["n7"] = Score("n7", "scored", denominator=40, metrics={"ECE": ece})
        return a

    floor = best_floor([arm("null", 0.40), arm("bm25", 0.11),
                        arm("sys", 0.05, False)], "n7", "ECE")
    assert floor.arm == "bm25", "⛔ 地板要取校准最**好**的那条对照"
    # ⭐ Δ 一律是「越大越好」口径：更好的 ECE 出正号
    assert delta(0.05, floor, "ECE") > 0
    assert delta(0.30, floor, "ECE") < 0


def test_every_headline_metric_has_a_declared_direction() -> None:
    """⚠️ 新加主指标时必须先回答「越大好还是越小好」。

    ⛔ 这条测试不判断谁对，它只保证**这个问题被回答过**——
    默认是「越大越好」，所以真正要守的是：越小越好的必须在名单里。
    """
    from amb.report.floor import (
        LOWER_IS_BETTER, SHAPE_NOT_QUALITY, is_quality_axis,
    )
    from amb.report.render import HEADLINE, _quality_unfit

    # ⭐ 越小越好的在这一份名单里
    assert "ECE" in LOWER_IS_BETTER
    # ⛔ **形状**是另一回事：⚠️ `扇形退化斜率` 的理想值是 0（越平越好），
    # 而一条什么都检索不到的臂每档 0.000 → 斜率完美等于 0。
    # ⭐ 它既不是「越高越好」也不是「越低越好」——是**不能当质量轴**。
    assert "扇形退化斜率" in SHAPE_NOT_QUALITY
    assert "扇形退化斜率" not in LOWER_IS_BETTER, "⛔ 更负并不更好，标错了"
    assert not is_quality_axis("扇形退化斜率") and not is_quality_axis("ECE")
    assert is_quality_axis("top1")

    # ⭐ 越小越好的**可以**当逐套件主指标（`best_floor`/`delta` 已按方向处理），
    # ⛔ 但一律不许进成本×质量表——那张表默认「越高越好」。
    for suite, metric in HEADLINE.items():
        if not is_quality_axis(metric):
            assert _quality_unfit(metric), f"{suite} 的 {metric} 没被挡在成本表外"


def test_a_degenerate_arm_cannot_widen_the_spread() -> None:
    """⛔ 退化的臂**先剔除再判 flat**。

    ⚠️ 顺序反了：`full_context` 的 1.000 把 spread 撑开 → `flat=False` →
    剩下几条全 0.000 的臂照样被排名，⭐ 正是「不许拿不区分的数排名」
    想堵的那个场景。
    """
    from amb.report.render import _render_cost
    from amb.report.schema import ArmResult
    from amb.scoring import Score

    def arm(name: str) -> ArmResult:
        a = ArmResult(arm=name, is_control=True)
        val = 1.0 if name == "full_context" else 0.0
        a.scores["retrieval"] = Score("retrieval", "scored", denominator=40, metrics={"top1": val})
        a.cost_profile = {"items_probed": 10}
        a.cost = {"probe": 1000}
        return a

    text = "\n".join(_render_cost(
        [arm("full_context"), arm("bm25"), arm("null")], ["retrieval"]))
    assert "不给判定" in text, "⛔ 剔除退化臂之后质量列是平的"
    assert "没有存在理由" not in text


def test_no_interval_means_refuse_to_claim() -> None:
    """⛔ 没有区间时**拒绝声称差异**，⚠️ 而不是免检。

    ⭐ 区间缺席（n<2、重抽样算不出来、零宽被压掉）说明这一跑
    **答不了这个问题**——早先这里直接落到「声称」分支，方向正好反了。
    """
    from amb.report.floor import Floor
    from amb.report.render import _delta_text

    text = _delta_text(0.9, None, Floor("bm25", 0.2), None, "top1")
    assert "不作判断" in text and "无区间" in text
    assert "帮倒忙" not in text


def test_a_three_question_suite_cannot_be_the_quality_axis() -> None:
    """⛔ 挑质量轴早先只看「有几条臂打了分」，完全不看**题数**。

    ⚠️ 于是一个 **3 道题**的套件当上了整份报告最显眼那张判定表的质量轴——
    ⭐ 那张表量的是抽样噪声。
    """
    from amb.report.render import MIN_AXIS_N, _render_cost
    from amb.report.schema import ArmResult
    from amb.scoring import Score

    def arm(name: str, tiny: float, big: float) -> ArmResult:
        a = ArmResult(arm=name, is_control=True)
        # ⚠️ 3 道题的套件：分高但没有信息
        a.scores["n8_induction"] = Score("n8_induction", "scored",
                                         denominator=3, metrics={"全对": tiny})
        a.scores["retrieval"] = Score("retrieval", "scored",
                                      denominator=40, metrics={"top1": big})
        a.cost_profile = {"items_probed": 10}
        a.cost = {"probe": 1000}
        return a

    text = "\n".join(_render_cost([arm("bm25", 1.0, 0.9), arm("null", 0.0, 0.1)],
                                  ["n8_induction", "retrieval"]))
    assert "`retrieval`" in text, "⭐ 该挑题多的那个"
    assert "`n8_induction`" not in text
    assert MIN_AXIS_N >= 10


def test_the_per_suite_table_shows_the_denominator() -> None:
    """⛔ 条件分母的指标只在一两道题上表态就能拿 1.000。

    ⚠️ `精确匹配率` 的分母是「给出了区间的题」——⭐ 一条只在 1 道题上
    给了区间且给对了的臂拿满分，而读者看不出那是**几分之几**。
    """
    from amb.report.render import _render_lane
    from amb.report.schema import ArmResult, Report
    from amb.scoring import Score

    a = ArmResult(arm="x", is_control=True)
    sc = Score("n2_provenance", "scored", denominator=40,
               metrics={"精确匹配率": 1.0, "回链率": 0.025})
    sc.denominators = {"精确匹配率": 1, "回链率": 40}
    a.scores["n2_provenance"] = sc
    report = Report(run_id="r", at="t", world={"name": "w", "seed": 1,
                                               "digest": ""},
                    backbone={}, externals={}, sampling={})
    text = _render_lane("library", [a], report)
    assert "n=1" in text, "⛔ 分母没同屏——读者看不出那是 1/1"


# ── ⛔ close 前后的用量要拼对，不能双份 ─────────────────────────
def _u(phase, tin=100, calls=1):
    from amb.core import Usage
    return Usage(phase=phase, tokens_in=tin, tokens_out=10, llm_calls=calls)


def test_a_subprocess_arm_keeps_its_ingest_tokens_across_the_close() -> None:
    """⛔ 摄入完存快照要先 `close()`，⚠️ 而子进程一退计量器就归零。

    ⭐ 所以 close 之前取一次，跑完再取一次，**拼起来**——
    ⛔ 不拼的话摄入那笔 token 从报告里消失，⚠️ 而钱是一等维度。
    """
    from amb.runner.phases import _merge_usage

    pre = [_u("ingest", 5000)]
    post = [_u("probe", 300)]          # ⚠️ 子进程重启，只剩探针那部分
    got = _merge_usage(pre, post)
    assert sum(u.tokens_in for u in got) == 5300
    assert {u.phase for u in got} == {"ingest", "probe"}


def test_an_in_process_arm_does_not_get_counted_twice() -> None:
    """⛔ 进程内的臂计量器活过了 close——⚠️ 再拼 `pre` 就是**双份**。

    ⭐ 判据是 `Usage.phase`：`post` 里已经有 ingest 行就只认 `post`。
    ⚠️ 双份不会报错，它只是让那条臂的成本凭空翻倍。
    """
    from amb.runner.phases import _merge_usage

    pre = [_u("ingest", 5000)]
    post = [_u("ingest", 5000), _u("probe", 300)]   # ⭐ 计量器没丢
    got = _merge_usage(pre, post)
    assert sum(u.tokens_in for u in got) == 5300, "⛔ 摄入那笔被算了两遍"


def test_nothing_to_merge_is_not_an_error() -> None:
    """⚠️ 没存快照的臂 `pre` 是 None——⛔ 不能因此丢掉 `post`。"""
    from amb.core import Unsupported
    from amb.runner.phases import _merge_usage

    post = [_u("probe", 300)]
    assert _merge_usage(None, post) is post
    assert list(_merge_usage([], post)) == post
    # ⭐ 不支持计量的臂原样透传，⛔ 不假装它有数
    assert isinstance(_merge_usage(None, Unsupported("没有")), Unsupported)
