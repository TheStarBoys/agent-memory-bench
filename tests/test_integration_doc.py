"""接入文档与代码的一致性。

⛔ 一份「照着做」的文档一旦与代码漂开，比没有更糟——
照着做的人会撞墙，然后不再信任任何一份文档。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "integrating.md").read_text(encoding="utf-8")


def test_every_cli_command_in_the_doc_actually_exists() -> None:
    """⛔ 文档里的命令必须真的能跑。"""
    from amb.cli.main import main

    # ⚠️ 排除 Markdown 表格分隔线（|---|---|）
    flags = {m for m in re.findall(r"(?<![-|])--[a-z][a-z-]*", DOC)}
    known = {
        "--arms", "--bench", "--sample", "--sample-seed", "--max-convs",
        "--max-turns", "--lane", "--no-answer", "--json", "--budget",
        "--check", "--upgrade",
    }
    unknown = flags - known
    assert not unknown, f"文档里的这些参数不存在：{sorted(unknown)}"
    assert callable(main)


def test_sampling_strategies_named_in_the_doc_all_exist() -> None:
    from amb.suites.public import Strategy

    for name in ("all", "first", "random", "stratified", "ids"):
        assert name in DOC, f"文档没提 {name} 策略"
        Strategy(name)          # ⛔ 不存在就抛


def test_files_the_doc_points_at_exist() -> None:
    """⛔ 文档里的每个路径都要真的在。"""
    missing = []
    for rel in re.findall(r"\]\(\.\./([^)#]+)\)", DOC):
        if not (ROOT / rel).exists():
            missing.append(rel)
    assert not missing, f"文档指向不存在的路径：{missing}"


def test_the_three_outcome_kinds_are_all_described() -> None:
    """⛔ 三态是这份协议的要害，接入文档必须讲全。"""
    for word in ("不支持", "Failed", "0 分"):
        assert word in DOC


def test_thinness_limits_in_the_doc_match_the_spec() -> None:
    """⚠️ 上限改了文档没改，接入的人会白写一遍。"""
    import tomllib

    spec = tomllib.loads((ROOT / "architecture.toml").read_text())["adapters"]
    assert str(spec["max_python_files"]) in DOC
    assert str(spec["max_total_lines"]) in DOC


def test_registered_examples_in_the_doc_are_real() -> None:
    """⚠️ 文档拿 mem0 和 locomo 当例子——它们得真的在清单里。"""
    from amb.setup import REGISTRY
    from amb.suites.public import REGISTRY as UPSTREAM

    assert "mem0" in REGISTRY and "locomo" in REGISTRY
    assert "beam" in UPSTREAM, "文档拿 beam 举了 caveats 的例子"


def test_setup_dependency_fields_in_the_doc_are_real() -> None:
    """⛔ 文档给的 Dependency 例子必须能构造出来。"""
    from amb.setup import Dependency, Kind

    dep = Dependency(name="x", kind=Kind.PIP, source="y", pin="1.0",
                     verify_import="y", note="n")
    for field in ("name", "kind", "source", "pin", "verify_import", "note"):
        assert field in DOC, f"文档没提 Dependency.{field}"
        assert hasattr(dep, field)


def test_the_checklist_covers_the_guarded_invariants() -> None:
    """⭐ 检查单要盖住测试真正在管的那些事。"""
    checklist = DOC[DOC.index("## 5. 检查单"):]
    for item in ("pin", "actual", "Unsupported", "薄度", "摄入成本"):
        assert item in checklist, f"检查单漏了 {item}"


@pytest.mark.parametrize("anchor", ["#2-接一个被测记忆系统",
                                    "#3-接一个公开题库", "#4-接一个运行时"])
def test_internal_jump_targets_exist(anchor: str) -> None:
    """⚠️ 顶部那张「要接什么」的跳转表不能是死链。"""
    slug = anchor.lstrip("#")
    headings = {
        re.sub(r"[^\w一-鿿-]", "", h.replace(" ", "-").replace(".", ""))
        for h in re.findall(r"^#{2,3} (.+)$", DOC, re.M)
    }
    assert slug in headings, f"跳转 {anchor} 没有对应标题；有的是 {sorted(headings)}"
