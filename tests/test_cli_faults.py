"""端到端：⭐ 让端点**真的坏掉**，看整条流水线怎么反应。

## ⛔ 这一批为什么必须存在

⚠️ 2026-09-10 native 真跑：`bm25` 撞上一次读超时被判「配置问题」
**整条臂跳过**——⛔ 而它是 n1/n4 三档唯一的地板臂。
⭐ 当时 844 个测试全绿，因为**没有一个让网络真的坏过**。

⚠️ 这里用真 HTTP 的 mock 端点（`mockllm.py`）跑真的 `amb.cli`，
⛔ 除了端点，别的全是真代码。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mockllm import MockLLM, env_for


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """一个指向 mock 端点、cwd 干净的 CLI。⭐ 返回 (跑一次, mock)。"""
    with MockLLM() as m:
        for k, v in env_for(m).items():
            monkeypatch.setenv(k, v)
        monkeypatch.chdir(tmp_path)

        out_path = tmp_path / "r.json"
        monkeypatch.setenv("_AMB_TEST_JSON", str(out_path))

        def run(*args: str) -> tuple[int, dict]:
            from amb.cli.main import main

            out = out_path
            code = main(["--bench", "toy", "--json", str(out),
                         "--skip-preflight", *args])
            got = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
            return code, got

        yield run, m


def _arms(report: dict) -> dict[str, dict]:
    return {a["arm"]: a for r in report.get("lanes", {}).values() for a in r}


# ── ⛔ 网络坏掉时，一条臂不该带走整跑 ────────────────────────────
def test_a_timeout_is_recorded_as_crashed_not_as_a_config_problem(cli) -> None:
    """⛔ **那条 bug 的现场**：⚠️ 读超时被判「配置问题」。

    ⭐ 两者在报告里是**不同的列**：`harness_fault` 说「我们的错」，
    `crashed` 说「这条臂没跑成」——⚠️ 而一次网络抖动**两者都不是**
    「你没配对」。⛔ 判错列的代价是：读者以为是自己环境的问题，
    而真相是端点抖了一下。
    """
    run, mock = cli
    # ⚠️ 让端点挂住，⭐ 并把读超时压到 0.5s——⛔ 否则默认 600s，
    # 这条测试等 3 秒就"通过"了而**从没超时过**（我第一版就是这么假的）。
    os.environ["AMB_LLM_TIMEOUT_S"] = "0.5"
    mock.hang(99, seconds=3.0)
    _code, rep = run("--arms", "bm25")
    a = _arms(rep).get("bm25")
    assert a is not None, "⛔ 这条臂静默消失了"
    fault = a.get("harness_fault") or ""
    assert "Timeout" not in fault and "配置" not in fault, (
        f"⛔ 超时被判成配置问题：{fault}")
    # ⭐ 它该如实记 crashed：⚠️ 「这条臂没跑成」是真的，
    # ⛔ 「你没配对」是假的
    assert a.get("crashed"), f"⛔ 超时该记 crashed，实际：{a}"


def test_one_broken_arm_does_not_take_down_the_others(cli) -> None:
    """⛔ 一条臂跑挂了，别的照跑——⚠️ 而报告要**说出来**它挂了，
    ⭐ 静默消失会被读成「没参赛」。
    """
    run, mock = cli
    code, rep = run("--arms", "null,bm25,does_not_exist")
    got = _arms(rep)
    assert "null" in got and "bm25" in got, f"⛔ 好的臂被带走了：{list(got)}"
    bad = got.get("does_not_exist")
    assert bad is not None, "⛔ 挂掉的臂静默消失了"
    assert bad.get("harness_fault") or bad.get("crashed"), "⛔ 没说它怎么了"


def test_a_bad_arm_name_is_our_fault_not_the_systems(cli) -> None:
    """⛔ 臂名打错是**评测器侧**的配置错——⚠️ 记 `crashed` 的话，
    报告那一列会被读成「这个系统不稳」。
    """
    run, _mock = cli
    _code, rep = run("--arms", "null,typo_arm")
    a = _arms(rep)["typo_arm"]
    assert a.get("harness_fault"), "⛔ 该记框架的账"
    assert not a.get("crashed"), "⛔ 不是这个系统跑挂了"


# ── ⭐ 落盘与续跑 ──────────────────────────────────────────────
def test_each_arm_is_checkpointed_before_the_next_one_starts(cli, tmp_path) -> None:
    """⛔ 一次几小时的跑不该是全有或全无。

    ⚠️ 实测踩过：跑到第 5 条臂被 OOM 杀掉，前面 4 条臂 42 分钟的结果
    **全部丢失**——⭐ 因为 CLI 只在全部跑完才写 JSON。

    ⛔ 我第一版这条是**假的**：⚠️ 它只看跑完之后的报告，
    而那是最后统一写的——⭐ 「中途落不落盘」从它底下溜过去了
    （变异「不写 checkpoint」时它照样绿）。
    ⚠️ 现在测真正的性质：**第二条臂还没开始时，第一条已经在盘上**。
    """
    run, mock = cli
    seen: list[int] = []
    out = Path(os.environ["_AMB_TEST_JSON"])

    # ⭐ 借 mock 端点当"钩子"：⚠️ 每次 chat 请求时看一眼盘上有几条臂
    orig = mock._handle

    def spy(h):
        if out.exists():
            try:
                seen.append(len(_arms(json.loads(out.read_text(encoding="utf-8")))))
            except Exception:
                pass
        return orig(h)

    mock._handle = spy
    _code, rep = run("--arms", "null,bm25")
    assert len(_arms(rep)) == 2
    assert rep.get("resume_key"), "⛔ 没有续跑键，下次接不上"
    # ⛔ 跑第二条臂的过程中，盘上必须已经有第一条
    assert max(seen, default=0) >= 1, (
        "⛔ 整跑结束才落盘——⚠️ 中途被杀就什么都不剩")


def test_a_second_run_skips_what_is_already_done(cli, capsys) -> None:
    """⭐ 续跑：⚠️ 原样再跑一次就该跳过已完成的。"""
    run, _mock = cli
    run("--arms", "null,bm25")
    capsys.readouterr()
    _code, rep = run("--arms", "null,bm25")
    err = capsys.readouterr().err
    assert "续跑" in err, f"⛔ 没走续跑：{err[-200:]}"
    assert len(_arms(rep)) == 2


def test_fresh_ignores_the_checkpoint(cli, capsys) -> None:
    """⛔ `--fresh` 要真的从头来——⚠️ 否则「重跑一次确认」是假的。"""
    run, _mock = cli
    run("--arms", "null")
    capsys.readouterr()
    run("--arms", "null", "--fresh")
    assert "续跑" not in capsys.readouterr().err


# ── ⭐ 端点抽风时的行为 ────────────────────────────────────────
def test_a_transient_5xx_is_ridden_out(cli) -> None:
    """⚠️ 端点抽一次风不该丢掉一条臂——⭐ 重试救得回来。"""
    run, mock = cli
    mock.fail(2, status=503)
    _code, rep = run("--arms", "bm25")
    a = _arms(rep).get("bm25")
    assert a is not None and not a.get("crashed"), f"⛔ 被一次 503 带走了：{a}"
    assert mock.wire.injected.count("503") == 2


def test_rate_limiting_is_not_the_systems_fault(cli) -> None:
    """⛔ 429 是供应商在节流——⚠️ 记成这条臂的失败等于把账算错人头上。"""
    run, mock = cli
    mock.rate_limit(2)
    _code, rep = run("--arms", "bm25")
    a = _arms(rep).get("bm25")
    assert a is not None and not a.get("crashed")


def test_the_report_is_still_written_when_an_arm_dies(cli, tmp_path) -> None:
    """⛔ 报告必须落盘——⚠️ 挂一条臂就什么都不留的话，
    ⭐ 那次跑的钱全白花。
    """
    run, _mock = cli
    _code, rep = run("--arms", "typo_arm")
    assert rep, "⛔ 一个字都没写"
    assert (tmp_path / "r.json").exists()
