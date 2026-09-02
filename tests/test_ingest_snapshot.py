"""摄入快照的**接线**，以及 backbone 受控变量**有没有真的到线上**。

⚠️ 快照键本身已由 `test_cost.py` 覆盖（键的每一项、顺序敏感、半截快照），
⛔ 这里不重复，只测那两层它没管的：

    ① 什么情况下**不用**快照——⛔ 宁可慢，不可拿错
    ② `wrap_openai_client` 打出去的钉子是否真的进了请求参数
       ⚠️ 这一层之前一条测试都没有，而它错了不会崩：
       只会让判分不可复现、让摄入慢 6 倍，且**没有任何报错**。
"""

from __future__ import annotations

from pathlib import Path

from amb.core import Document


# ── ① 接线：什么情况下不用快照 ──────────────────────────────────
class _Arm:
    """只实现快照关心的那一个方法。"""

    name = "mem0"

    def __init__(self, places: list[str]) -> None:
        self._places = places

    def storage_locations(self) -> list[str]:
        return self._places


def test_arm_without_a_declared_store_is_skipped() -> None:
    """⛔ 没申报持久层就不用快照——⚠️ 没东西可拷，也没法验证拷全了。"""
    from amb.runner.phases import _store_of

    assert _store_of(_Arm([])) is None


def test_arm_with_several_stores_is_skipped(tmp_path: Path) -> None:
    """⚠️ 状态不止一处，只拷一个会拿到**不一致**的快照——⛔ 不如不拷。"""
    from amb.runner.phases import _store_of

    assert _store_of(_Arm([str(tmp_path / "a"), str(tmp_path / "b")])) is None
    assert _store_of(_Arm([str(tmp_path / "a")])) == tmp_path / "a"


def test_no_backbone_means_no_snapshot(tmp_path: Path) -> None:
    """⛔ 没挂 backbone 时不用快照——键里缺了它就会拿错。"""
    from amb.runner.phases import Plan, _snapshot_key

    plan = Plan(manifest=None, documents=[Document(doc_id="a", text="x")])
    assert _snapshot_key("mem0", _Arm([str(tmp_path)]), plan, "") is None


def test_mem0_and_mem0_raw_never_share_a_snapshot() -> None:
    """⭐ 最容易踩的一个：这两条臂**是同一个适配器类**
    （`mem0_raw` 只是 `infer=False`），所以两者的 `adapter.name` 都是 `mem0`。

    ⛔ 键要是按适配器名取，它们会互相拿到对方的库——⚠️ 而那两个库内容
    完全不同：一个存抽出来的事实，一个存原文。
    """
    from amb.runner.snapshot import SnapshotKey

    same = dict(arm_version="2.0.19", backbone="Qwen/Qwen3-8B",
                corpus_digest="同一份语料")
    assert (SnapshotKey(arm="mem0", **same).digest
            != SnapshotKey(arm="mem0_raw", **same).digest)


# ── ② 受控变量：钉子有没有真的到线上 ────────────────────────────
class _FakeCompletions:
    def __init__(self) -> None:
        self.seen: list[dict] = []

    def create(self, **kwargs):
        self.seen.append(kwargs)
        return _Reply()


class _Reply:
    def model_dump(self) -> dict:
        return {}


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("chat", (), {"completions": _FakeCompletions()})()


def test_pins_reach_the_wire_even_with_the_cache_off(monkeypatch) -> None:
    """⛔ 这是我修过的一个真 bug：早先只在**缓存启用**时才打补丁。

    ⚠️ 那样不开缓存的跑用的是被测系统自己的 temperature 和思考模式——
    判分不可复现、摄入慢 6 倍，而且**没有任何报错**。
    """
    monkeypatch.delenv("AMB_LLM_CACHE", raising=False)      # ⭐ 缓存关着
    monkeypatch.delenv("AMB_LLM_THINKING", raising=False)
    from amb.adapters.llm_cache import wrap_openai_client

    client = _FakeClient()
    assert wrap_openai_client(client) is True
    client.chat.completions.create(model="m", messages=[])

    sent = client.chat.completions.seen[-1]
    assert sent["temperature"] == 0.0
    assert sent["extra_body"]["enable_thinking"] is False


def test_thinking_can_be_turned_back_on_explicitly(monkeypatch) -> None:
    """⚠️ 关思考是默认，不是强制——⛔ 但开了要显式说，且报告里会写着。"""
    monkeypatch.setenv("AMB_LLM_THINKING", "1")
    from amb.adapters.llm_cache import backbone_overrides

    assert "extra_body" not in backbone_overrides()
    assert backbone_overrides()["temperature"] == 0.0


def test_the_systems_own_extra_body_is_merged_not_dropped(monkeypatch) -> None:
    """⚠️ 被测系统自己传的 extra_body 要保留——⛔ 覆盖掉等于改了它别的行为。"""
    monkeypatch.delenv("AMB_LLM_CACHE", raising=False)
    monkeypatch.delenv("AMB_LLM_THINKING", raising=False)
    from amb.adapters.llm_cache import wrap_openai_client

    client = _FakeClient()
    wrap_openai_client(client)
    client.chat.completions.create(model="m", messages=[],
                                   extra_body={"它自己的": 1})

    sent = client.chat.completions.seen[-1]
    assert sent["extra_body"] == {"它自己的": 1, "enable_thinking": False}


def test_wrapping_twice_is_a_no_op() -> None:
    """⛔ 重复包一层会让钉子叠加、缓存计数翻倍——⚠️ 必须幂等。"""
    from amb.adapters.llm_cache import wrap_openai_client

    client = _FakeClient()
    assert wrap_openai_client(client) is True
    assert wrap_openai_client(client) is False


# ── ③ 快照的安全网：⛔ 别把整个仓库拷进去 ──────────────────────
def test_an_empty_env_var_never_makes_the_repo_the_store(monkeypatch) -> None:
    """⛔ 实测踩点：`AMB_MEM0_DIR=`（空串）会让 storage_dir 变成 `""`，
    而 `Path("")` 是 `.`——⚠️ 那时摄入快照会把**整个仓库**拷进
    `.external/snapshots`。

    两道防线都要在：① build 把空串当没设 ② `_store_of` 拒绝 `.`。
    """
    from amb.runner.phases import _store_of

    assert _store_of(_Arm([""])) is None
    assert _store_of(_Arm(["."])) is None
    assert _store_of(_Arm([str(Path.cwd())])) is None
    assert _store_of(_Arm(["/"])) is None


def test_empty_env_var_falls_back_to_the_default_dir(monkeypatch) -> None:
    """⚠️ `os.environ.get(k, 默认)` 只在**键不存在**时给默认值。"""
    from amb.runner.build import _env_dir

    monkeypatch.setenv("AMB_TEST_DIR", "")
    assert _env_dir("AMB_TEST_DIR", "默认") == "默认"
    monkeypatch.setenv("AMB_TEST_DIR", "给了值")
    assert _env_dir("AMB_TEST_DIR", "默认") == "给了值"


# ── ④ 三态：不适用 ≠ 跑挂了 ≠ 0 分 ─────────────────────────────
def test_not_applicable_is_not_crashed() -> None:
    """⛔ 实测踩点：`full_context` 遇到塞不下窗口的语料时抛 `ContextOverflow`，
    被通用 except 接住记成了 `crashed`。

    ⚠️ 但 docs/baselines.md 明确规定那该记 **N/A**——
    ⭐ 「不适用 / 失败 / 0 分」是三件事，压成一列这个项目就白做了。
    """
    from amb.report.render import _render_lane
    from amb.report.schema import ArmResult, Report

    report = Report(run_id="r", at="t", world={"name": "w", "seed": 1,
                                               "digest": ""},
                    backbone={}, externals={}, sampling={})
    text = _render_lane("library", [
        ArmResult(arm="full_context", is_control=True,
                  not_applicable="语料 55385 码点 > 预算 24000"),
        ArmResult(arm="mem0", is_control=False, crashed="BridgeError: 挂了"),
    ], report)

    assert "不适用" in text and "N/A" in text
    assert "没跑完" in text                       # 崩溃那一段还在
    # ⛔ 两者不许混为一谈
    na_at, crash_at = text.index("不适用"), text.index("没跑完")
    assert na_at != crash_at


# ── ⑤ 快照的键绑的是**摄入**用的 LLM，不是回答用的 backbone ──────
def test_snapshot_survives_no_answer_runs(monkeypatch) -> None:
    """⛔ 实测踩点：`--no-answer` 时没有回答用的 backbone，
    键绑在它上面就恒为空 → **快照一律不存**（跑完 419 条才发现没存上）。

    ⚠️ 但被测系统**摄入时照样调 LLM**（mem0 抽事实、A-mem 演化链接），
    用的是它自己配的 `AMB_LLM_MODEL`——那才是影响摄入结果的东西。
    """
    from amb.runner import ingest_identity

    monkeypatch.setenv("AMB_LLM_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.delenv("AMB_LLM_THINKING", raising=False)
    assert ingest_identity()          # ⭐ 没有回答 backbone 也拿得到身份


def test_thinking_is_part_of_the_ingest_identity(monkeypatch) -> None:
    """⭐ 思考开关把输出 token 变 25 倍——⛔ 抽出来的东西**不一样**，
    ⚠️ 不能共用一份快照。"""
    from amb.runner import ingest_identity

    monkeypatch.setenv("AMB_LLM_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.delenv("AMB_LLM_THINKING", raising=False)
    off = ingest_identity()
    monkeypatch.setenv("AMB_LLM_THINKING", "1")
    assert ingest_identity() != off


def test_unknown_ingest_model_disables_the_snapshot(monkeypatch) -> None:
    """⛔ 说不清摄入用了什么，就不敢复用——⚠️ 宁可慢，不可拿错。"""
    from amb.runner import ingest_identity

    monkeypatch.delenv("AMB_LLM_MODEL", raising=False)
    assert ingest_identity() == ""


# ── ⑥ 供应商限流不该变成被测系统的分数 ────────────────────────
def test_rate_limit_is_retried_not_fatal(monkeypatch) -> None:
    """⛔ 实测踩点：mem0 摄入到第 ~130 条撞 TPM 上限，整条臂判「跑挂了」，
    而它已经跑了 16 分钟。

    ⚠️ 供应商配额既不是被测系统的错，也不该变成它的分数。
    """
    from amb.adapters import llm_cache

    monkeypatch.setattr(llm_cache, "RETRY_BASE_S", 0.0)   # ⚠️ 测试里不真等
    attempts = []

    class _RateLimited(Exception):
        status_code = 429

    def flaky(**kw):
        attempts.append(1)
        if len(attempts) < 3:
            raise _RateLimited("Request was rejected due to rate limiting: TPM")
        return "成功了"

    assert llm_cache._with_retry(flaky, {}) == "成功了"
    assert len(attempts) == 3


def test_non_rate_limit_errors_are_never_retried(monkeypatch) -> None:
    """⛔ 只重试限流——⚠️ 把真 bug 重试掉比慢更糟：
    它会把一个确定性的失败变成一个「跑了很久然后失败」的失败。
    """
    import pytest

    from amb.adapters import llm_cache

    monkeypatch.setattr(llm_cache, "RETRY_BASE_S", 0.0)
    attempts = []

    def broken(**kw):
        attempts.append(1)
        raise ValueError("提示写错了")

    with pytest.raises(ValueError):
        llm_cache._with_retry(broken, {})
    assert len(attempts) == 1        # ⛔ 只试一次


def test_retry_gives_up_and_reports_the_real_error(monkeypatch) -> None:
    """⚠️ 退避不是无限的——⛔ 撑不过去要把**原始错误**抛出来，
    不能变成一个说不清原因的超时。"""
    import pytest

    from amb.adapters import llm_cache

    monkeypatch.setattr(llm_cache, "RETRY_BASE_S", 0.0)
    monkeypatch.setattr(llm_cache, "RETRY_MAX", 2)

    class _RateLimited(Exception):
        status_code = 429

    def always(**kw):
        raise _RateLimited("429 rate limit")

    with pytest.raises(_RateLimited):
        llm_cache._with_retry(always, {})


def test_retry_wait_is_accounted_separately() -> None:
    """⛔ 重试等待不能算进「这个系统很慢」——⚠️ 那是供应商配额的成本。"""
    from amb.adapters.llm_cache import RETRIES

    assert hasattr(RETRIES, "waited_s") and hasattr(RETRIES, "retries")


def test_a_hung_request_cannot_eat_thirty_minutes(monkeypatch) -> None:
    """⛔ 实测踩点：A-mem 摄入中间空了 **19 分钟**，没有任何日志。

    原因是 openai SDK 的默认值：read 超时 600s，**且自带 2 次静默重试**
    → 一次卡住的调用最坏吃掉 30 分钟，而且查不出来。

    ⭐ 关思考后输出都在 500 token 以内，120s 已经很宽。
    """
    monkeypatch.delenv("AMB_LLM_THINKING", raising=False)
    monkeypatch.delenv("AMB_LLM_TIMEOUT_S", raising=False)
    from amb.adapters.llm_cache import backbone_overrides

    assert backbone_overrides()["timeout"] == 120.0


def test_timeouts_are_retried_but_loudly(monkeypatch) -> None:
    """⚠️ 超时跟限流一样是暂时的，值得重试——
    ⭐ 但要走**我们的**退避（它会说话），⛔ 不是 SDK 那 2 次静默重试。
    """
    from amb.adapters import llm_cache

    monkeypatch.setattr(llm_cache, "RETRY_BASE_S", 0.0)
    tries = []

    class APITimeoutError(Exception):
        pass

    def slow(**kw):
        tries.append(1)
        if len(tries) < 2:
            raise APITimeoutError("请求超时")
        return "好了"

    assert llm_cache._with_retry(slow, {}) == "好了"
    assert llm_cache.RETRIES.retries >= 1     # ⭐ 记了账，不是静默的


# ── ⑦ 快照省了时间，⛔ 不该连成本测量一起丢掉 ──────────────────
def test_snapshot_carries_the_measured_ingest_cost(tmp_path: Path) -> None:
    """⛔ 命中快照时摄入被整段跳过，那一格本次约等于 0——
    ⚠️ 而成本恰恰是对比里最要紧的一列（a_mem 31.26s/条 vs mem0_raw 1.01s/条）。

    ⭐ 存快照那次的数字**可以**带回来：快照键锁死了臂 + 版本 + 摄入身份 +
    语料指纹，四项全同才命中，所以它量的就是这份语料上的这个系统。
    """
    from amb.runner.snapshot import SnapshotKey, save, saved_cost

    key = SnapshotKey("a_mem", "0.2.6", "Qwen/Qwen3-8B|thinking=0", "abc123")
    store = tmp_path / "store"
    store.mkdir()
    (store / "db").write_text("x")
    save(key, store, root=tmp_path / "snaps",
         cost={"ingest_ms": 937836, "items": 30, "tokens_in": 12345})

    got = saved_cost(key, root=tmp_path / "snaps")
    assert got["ingest_ms"] == 937836 and got["tokens_in"] == 12345


def test_a_snapshot_without_recorded_cost_reports_nothing(tmp_path: Path) -> None:
    """⛔ 老快照没记成本——⚠️ 那就报 None，不猜、不拿 0 冒充。"""
    from amb.runner.snapshot import SnapshotKey, save, saved_cost

    key = SnapshotKey("mem0", "2.0.19", "b", "abc123")
    store = tmp_path / "store"
    store.mkdir()
    (store / "db").write_text("x")
    save(key, store, root=tmp_path / "snaps")        # ⚠️ 不带 cost
    assert saved_cost(key, root=tmp_path / "snaps") is None


def test_the_snapshot_footnote_never_goes_silent() -> None:
    """⛔ 踩过：`ingest_snapshot` 的值加了说明之后，渲染层的精确匹配
    `== "命中"` 静默失效，† 标记整个消失。

    ⚠️ 一个「标记本身会不见」的 bug 比标错更糟：读者不会发现少了什么。
    """
    from amb.report.render import _render_cost
    from amb.report.schema import ArmResult, Score

    def _arm(name, snap):
        return ArmResult(
            arm=name, is_control=(name == "naive_rag"),
            scores={"locomo_retrieval": Score(suite="locomo_retrieval",
                                              status="scored",
                                              metrics={"evidence_recall": 0.7})},
            cost={"ingest": 900_000, "probe": 100},
            cost_profile={"items_ingested": 30, "items_probed": 17},
            ingest_snapshot=snap)

    text = "\n".join(_render_cost(
        [_arm("a_mem", "命中（摄入成本取自存快照那次实测）"),
         _arm("naive_rag", "未启用")],
        ["locomo_retrieval"], "Qwen/Qwen3-8B"))
    assert "†" in text and "存快照那次实测" in text


# ── ⑧ 坏快照必须被**验出来**，⛔ 不能静默出假分 ────────────────
def test_canary_catches_bookkeeping_left_outside_the_store(tmp_path: Path) -> None:
    """⛔ 实测最危险的一次：a_mem 把「记忆 id → doc id」的映射放在 worker
    内存里，⚠️ 快照只拷了 chroma 目录。命中快照 → 映射为空 →
    检索结果全都没有 doc_id → **recall 从 0.526 静默变成 0.000**，不报错。

    ⭐ 所以指纹**不能只数条数**：那个 bug 里 chroma 的 30 条全都在，
    错的是判分真正依赖的 doc_ids。指纹必须真跑一次检索。
    """
    from amb.runner.phases import Plan, _canary

    class _Arm:
        """模拟命中快照后的 a_mem：条数对，⛔ 但给不出 doc_ids。"""

        def __init__(self, with_doc_ids: bool) -> None:
            self._ok = with_doc_ids

        def search(self, query, k, **kw):
            from amb.core import Entry
            return [Entry(id="m1", digest="x",
                          doc_ids=["d0"] if self._ok else [])]

        def count(self):
            return 30

    plan = Plan(manifest=None, documents=[Document(doc_id="d0", text="李雷入职")])
    healthy = _canary(_Arm(True), plan)
    broken = _canary(_Arm(False), plan)

    assert healthy["count"] == broken["count"] == 30   # ⚠️ 条数一样
    assert healthy != broken                           # ⭐ 但指纹不同
    assert healthy["with_doc_ids"] == 1 and broken["with_doc_ids"] == 0


def test_a_snapshot_without_a_canary_is_not_trusted(tmp_path: Path) -> None:
    """⛔ 老快照没存指纹——⚠️ 验不了就不敢用，重新摄入。

    ⭐ 失败方向必须是「多花一次时间」，不是「出一个假分」。
    """
    from amb.runner.snapshot import SnapshotKey, save, saved_canary

    key = SnapshotKey("a_mem", "0.2.6", "b", "abc")
    store = tmp_path / "s"
    store.mkdir()
    (store / "db").write_text("x")
    save(key, store, root=tmp_path / "snaps")          # ⚠️ 不带 canary
    assert saved_canary(key, root=tmp_path / "snaps") is None


def test_a_canary_round_trips(tmp_path: Path) -> None:
    from amb.runner.snapshot import SnapshotKey, save, saved_canary

    key = SnapshotKey("a_mem", "0.2.6", "b", "abc")
    store = tmp_path / "s"
    store.mkdir()
    (store / "db").write_text("x")
    save(key, store, root=tmp_path / "snaps",
         canary={"count": 30, "hits": 3, "with_doc_ids": 3})
    assert saved_canary(key, root=tmp_path / "snaps")["with_doc_ids"] == 3
