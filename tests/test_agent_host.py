"""DSH 宿主。

⚠️ 真跑 agent 的那条要钱要时间，默认跳过：
    AMB_AGENT_SMOKE=1 pytest tests/test_agent_host.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from amb.agent import Host, HostSpec, HostUnavailable, spec_from_env
from amb.world import materialize

import worlds.toy as toy

LIVE = os.environ.get("AMB_AGENT_SMOKE") == "1"


def _runtime_installed() -> bool:
    from importlib.util import find_spec

    return find_spec("deepseek_harness") is not None


def test_spec_pins_everything_that_must_be_global() -> None:
    """⛔ 宿主是受控变量：这些每一项对所有臂都必须相同。"""
    spec = spec_from_env()
    assert spec.model and spec.base_url and spec.api_key_env
    if _runtime_installed():
        # ⚠️ 版本要进报告——换 DSH 版本等于换尺子，要重跑全部基线
        assert spec.version != "unknown"


def test_settings_never_contain_the_key(tmp_path: Path) -> None:
    """⛔ 写给 DSH 的配置只引用变量名，绝不含 key 本身。"""
    spec = HostSpec(model="m", base_url="http://x/v1", api_key_env="AMB_FAKE_KEY")
    os.environ["AMB_FAKE_KEY"] = "sk-should-never-be-written"
    try:
        host = Host(spec, tmp_path / "w", tmp_path / "home")
        (tmp_path / "home").mkdir(parents=True, exist_ok=True)
        host._write_settings()  # noqa: SLF001
        body = (tmp_path / "home" / "settings.yaml").read_text()
        assert "sk-should-never-be-written" not in body
        assert json.loads(body)["llm-pi-ai"]["providers"]["amb-backbone"][
            "apiKeyEnv"
        ] == "AMB_FAKE_KEY"
    finally:
        del os.environ["AMB_FAKE_KEY"]


def test_missing_runtime_is_unavailable_not_zero(tmp_path: Path, monkeypatch) -> None:
    """⛔ 运行时装不上 → 该档记不可用，不是 0 分。"""
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "deepseek_harness":
            raise ImportError("boom")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(HostUnavailable, match="不是 0 分"):
        Host(spec_from_env(), tmp_path / "w", tmp_path / "h").start()


@pytest.mark.skipif(not LIVE, reason="需 AMB_AGENT_SMOKE=1（真跑 agent，要钱要时间）")
def test_agent_reads_the_world_we_mounted() -> None:
    """⭐ 最小闭环：世界挂成 cwd，agent 自己去读，答案只可能来自那里。"""
    with tempfile.TemporaryDirectory(prefix="amb-agent-") as tmp:
        root = materialize(toy.MANIFEST, Path(tmp) / "world")
        with Host(spec_from_env(), root, Path(tmp) / "home") as host:
            turn = host.ask(
                "读一下 notes/neocortex.md，用一个词回答：哪个脑结构学得慢？"
            )
    assert turn.finish_reason == "completed"
    assert "新皮层" in turn.text
    # ⭐ agent/* 事件流：每一步都看得到，不需要被测系统配合
    assert len(turn.events) > 0
