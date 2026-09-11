"""Lint 闸门——⛔ 挂在 pytest 上，⚠️ 因为这个仓库**没有 CI**。

## ⭐ 为什么不是一条独立的 `ruff check` 命令

⚠️ 技能里那句话对这个仓库是字面成立的：**未强制的约定与没有规则
无法区分**。⛔ 没有 CI 的话，`ruff check` 只在有人想起来时才跑——
⭐ 而 pytest 是这个仓库**真的每次都跑**的那条路径（957 条测试）。

⚠️ 已有的架构守卫（`test_architecture.py`）用的就是这个办法。
⭐ 沿用它，而不是另起一条没人走的路。

## ⛔ 这些闸门自己也要能失败

⚠️ 这个仓库刚清掉 **47 个死 `# noqa`**——它们是对着一个**从没运行过**的
linter 写的装饰，⭐ 让读代码的人以为「有人考虑过 linter 的意见」。
⛔ 那正是「一个不会失败的闸门比没有闸门更糟」的实例。

⭐ 所以下面每条闸门都配了**反向测试**：种一个违规，确认它真的红。
"""

from __future__ import annotations

import ast
import collections
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("src", "worlds", "tests")


def _ruff(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", *args],
        cwd=ROOT, capture_output=True, text=True, check=False)


def test_ruff_is_available_at_the_pinned_version() -> None:
    """⛔ 闸门跑不起来时必须**红**，⚠️ 不是静默跳过。

    ⭐ 「工具没装就跳过」是这一整类闸门最常见的失效方式：
    ⚠️ CI 上少装一个包，所有 lint 检查就集体消失，⛔ 而报告显示全绿。
    """
    got = subprocess.run([sys.executable, "-m", "ruff", "--version"],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    assert got.returncode == 0, (
        "⛔ ruff 跑不起来——⚠️ 装它：`.venv/bin/python -m pip install "
        "'ruff>=0.16,<0.17'`")
    # ⚠️ 版本钉在区间里：⛔ Ruff 还没到 1.0，小版本会加规则，
    # ⭐ 不钉的话同一份代码在两台机器上结论不同。
    ver = got.stdout.split()[-1]
    assert ver.startswith("0.16."), f"⛔ 版本超出钉死区间：{ver}"


def test_the_codebase_passes_lint() -> None:
    """⭐ 主闸门。⛔ 违规清单直接贴出来——⚠️ 只说「失败了」查不出是什么。"""
    got = _ruff(*TARGETS)
    assert got.returncode == 0, "⛔ lint 不过：\n" + got.stdout[-3000:]


def test_the_lint_gate_actually_fails_on_a_violation() -> None:
    """⛔ **反向测试**：⚠️ 种一个违规，确认闸门真的红。

    ⭐ 这一条守的是闸门本身：⚠️ 一份配置可以跑得很欢而什么都不检查——
    ⛔ 而那时它照样打印 `All checks passed`。

    ## ⛔ 违规必须种在**真实的目标目录下**

    ⚠️ 第一版把它种在 `tmp_path` 里——⭐ 实测：把 `per-file-ignores` 改成
    `"src/**" = ["ALL"]`（闸门对生产代码彻底变瞎）之后，
    ⛔ 这条测试**照样绿**——因为临时文件不在 `src/**` 底下，不受影响。

    ⚠️ 那验的是「配置能抓临时文件」，⛔ 不是「能抓我们的代码」。
    ⭐ 现在种进 `src/amb/` 里，跑完删掉。
    """
    planted = ROOT / "src" / "amb" / "_lint_probe_delete_me.py"
    # ⚠️ 三个不同族各一个，⛔ 免得只验到其中一族
    planted.write_text(textwrap.dedent("""
        import os
        def f():
            try:
                pass
            except Exception:
                pass
        def g():
            undefined_name_here()
    """), encoding="utf-8")
    try:
        got = _ruff(*TARGETS)
        assert got.returncode != 0, (
            "⛔ 在 src/ 里种了违规，闸门却是绿的——⚠️ 它对生产代码是瞎的")
        for code in ("F821", "S110", "F401"):
            assert code in got.stdout, f"⛔ 没抓到 {code}：\n{got.stdout[-1500:]}"
    finally:
        planted.unlink(missing_ok=True)


def test_no_dead_noqa_comments() -> None:
    """⛔ 一个不起作用的 `# noqa` 就是**装饰**。

    ⚠️ 实测：这个仓库曾有 **48 个 noqa，其中 47 个是死的**——
    ⭐ 它们是对着一个从没运行过的 linter 写的，
    ⛔ 而它们让读代码的人以为「这里有人考虑过并决定豁免」。

    ⚠️ `RUF100` 已经在主闸门里，⭐ 这一条单独留着是为了让**这句话**
    有个落点：⛔ 死 noqa 不是风格问题，是假信号。
    """
    # ⛔ **不能单独 select 这一条规则**：⚠️ 那个 flag 会**覆盖**配置里的
    # select，⭐ 于是所有针对其他规则的 noqa 全部被误判成「死的」——
    # ⚠️ 实测：两个有效的 SLF001 抑制注释被报成 `non-enabled`。
    # ⛔ 那是「闸门在检查，但检查的不是你以为的东西」的一种。
    got = _ruff(*TARGETS)
    dead = [ln for ln in got.stdout.splitlines() if "RUF100" in ln]
    assert not dead, "⛔ 有死 noqa：\n  " + "\n  ".join(dead[:10])


# ── ⭐ 循环依赖：⛔ `architecture.toml` 只管**层间方向** ────────
def _import_graph() -> dict[str, set[str]]:
    """模块 → 它 import 的本仓库模块。"""
    g: dict[str, set[str]] = collections.defaultdict(set)
    for p in (ROOT / "src" / "amb").rglob("*.py"):
        mod = (str(p.relative_to(ROOT / "src")).removesuffix(".py")
               .removesuffix("/__init__").replace("/", "."))
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("amb"):
                g[mod].add(n.module)
            elif isinstance(n, ast.Import):
                g[mod].update(a.name for a in n.names if a.name.startswith("amb"))
    return g


def _cycles(graph: dict[str, set[str]]) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()

    def walk(node: str, path: list[str], seen: set[str]) -> None:
        if node in path:
            found.add(tuple(sorted(set(path[path.index(node):]))))
            return
        if node in seen or node not in graph:
            return
        seen.add(node)
        for nxt in graph[node]:
            walk(nxt, [*path, node], seen)

    for m in list(graph):
        walk(m, [], set())
    return found


def test_no_circular_imports() -> None:
    """⛔ Python **允许**循环导入——⚠️ 而这个仓库此前没有任何东西检查它。

    ⭐ `architecture.toml` + `test_architecture.py` 管的是**层间方向**
    （`adapters` 不许 import `runner`）——⛔ 层**内**的模块互相引用它管不着。

    ⚠️ 循环导入的症状是**时好时坏**：⛔ 它取决于谁先被 import，
    ⭐ 于是同一份代码在测试里过、在 `python -m` 入口下炸。

    ⚠️ 基线是 **0 组循环**（90 个模块）——⭐ 所以这条守卫是零成本的，
    ⛔ 它只防漂移。
    """
    got = _cycles(_import_graph())
    assert not got, ("⛔ 出现循环导入：\n  "
                     + "\n  ".join(" ↔ ".join(c) for c in sorted(got)))


def test_the_cycle_detector_actually_detects_cycles() -> None:
    """⛔ **反向测试**：⚠️ 上面那条基线是 0，⭐ 而「永远返回空」的检测器
    与「没有循环」在结果上**一模一样**。
    """
    assert _cycles({"a": {"b"}, "b": {"a"}}), "⛔ 两个模块互指都测不出来"
    assert _cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}}), "⛔ 三元环测不出来"
    assert not _cycles({"a": {"b"}, "b": {"c"}}), "⚠️ 反向：无环不该误报"


def test_the_graph_is_not_empty() -> None:
    """⛔ **这条最要紧**：⚠️ 一个分析了 0 个模块的检查器会报告
    「没有循环」并**退出 0**。

    ⭐ 技能里记着这个实例：dependency-cruiser 对着 TypeScript 7 分析了
    **0 个模块**，打印 `no violations found`——⛔ 闸门死了而看起来在跑。
    """
    g = _import_graph()
    assert len(g) >= 50, f"⛔ 只扫到 {len(g)} 个模块——⚠️ 检查器多半没在工作"
