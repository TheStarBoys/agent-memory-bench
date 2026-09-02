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
