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
