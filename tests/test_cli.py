"""入口：参数解析 · 失败处理 · 报告落盘。

⛔ 之前 15% 覆盖——⚠️ 而它是唯一把「一条臂跑挂了」变成
**报告里一行**而不是整个进程崩掉的地方。那个行为坏了，
一次跑几十分钟的实验会在最后一条臂上全部白费。

⭐ 这里只测**我们自己的编排逻辑**，不真跑臂（真跑见 docs/runs/）。
"""

from __future__ import annotations

import json

import pytest

from amb.cli.main import main


def test_unknown_bench_is_rejected_not_scored_zero(capsys) -> None:
    """⛔ 不认识的题库要报错，⚠️ 不许静默跑出一个 0 分。"""
    with pytest.raises((KeyError, SystemExit)):
        main(["--bench", "没这个题库", "--arms", "null"])


def test_toy_run_writes_both_report_and_json(tmp_path, capsys) -> None:
    """⭐ `--json` 落盘的内容必须与 stdout 那份是同一次跑。"""
    out = tmp_path / "run.json"
    code = main(["--bench", "toy", "--arms", "null", "--no-answer",
                 "--json", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert out.is_file(), "⛔ --json 没落盘"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run_id"] in printed, "⛔ JSON 与 stdout 不是同一次跑"
    arms = [a["arm"] for a in data["lanes"]["library"]]
    assert arms == ["null"]


def test_a_crashing_arm_is_recorded_not_fatal(tmp_path, monkeypatch) -> None:
    """⛔ 一条臂挂了，其余的必须照跑完——⚠️ 而且它要在报告里留一行。

    ⭐ 这是「跑挂了」与「不支持」分开记的前提：
    静默消失会被读成「没参赛」，那尺子就废了。
    """
    # ⚠️ 不能 `import amb.cli.main as cli`——包里已经把 `main` 函数
    # 导出成了同名属性，⛔ 那样拿到的是函数不是模块。
    import importlib

    cli = importlib.import_module("amb.cli.main")
    real = cli.build

    def _explode(name, **kw):
        if name == "bm25":
            raise RuntimeError("故意炸的")
        return real(name, **kw)

    monkeypatch.setattr(cli, "build", _explode)
    out = tmp_path / "run.json"
    main(["--bench", "toy", "--arms", "bm25,null", "--no-answer",
          "--json", str(out)])
    rows = {a["arm"]: a for a in
            json.loads(out.read_text(encoding="utf-8"))["lanes"]["library"]}
    assert "故意炸的" in (rows["bm25"]["crashed"] or ""), "⛔ 挂掉的臂没留痕"
    assert rows["null"]["scores"], "⛔ 一条臂挂了把其余的也带下水了"


def test_no_answer_means_no_backbone_in_the_header(tmp_path, capsys) -> None:
    """⚠️ 检索档不挂 backbone——⛔ 报告抬头必须说清楚，否则会被当成回答档读。"""
    main(["--bench", "toy", "--arms", "null", "--no-answer"])
    assert "未跑 answer 档" in capsys.readouterr().out


def test_arms_list_ignores_empty_entries() -> None:
    """⚠️ `--arms null,,bm25` 里的空段要吞掉——⛔ 否则会去 build('')。"""
    assert main(["--bench", "toy", "--arms", "null,,null", "--no-answer"]) == 0


def test_setup_check_reports_status_without_installing(capsys) -> None:
    """⛔ `--check` 只看状态，⚠️ 绝不能顺手装东西。"""
    from amb.cli.main import _setup_cmd

    code = _setup_cmd(["--check"])
    printed = capsys.readouterr().out
    assert "声明" in printed and "实际" in printed
    assert code in (0, 1)      # ⚠️ 装没装全取决于本机，⛔ 不断言具体值
