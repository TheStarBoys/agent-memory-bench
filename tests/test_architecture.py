"""架构守卫。

⛔ 这个文件的存在理由：架构文档会漂，断言不会。
每一条都对着一个在 MemoryData 上实测到的失效模式——出处见 ARCHITECTURE.md。
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "amb"
SPEC = tomllib.loads((ROOT / "architecture.toml").read_text())

LAYERS: list[str] = SPEC["layers"]["order"]
DEPS: dict[str, list[str]] = SPEC["deps"]


def _layer_of(path: Path) -> str | None:
    parts = path.relative_to(SRC).parts
    return parts[0] if parts and parts[0] in LAYERS else None


def _imported_layers(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module]
        for m in mods:
            p = m.split(".")
            if p[0] == "amb" and len(p) > 1 and p[1] in LAYERS:
                out.add(p[1])
    return out


def _py_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


# ── 1. 层依赖方向 ───────────────────────────────────────────────
# 实测失效：MemoryData 的 benchmark/longbench/loader.py:5
#   from benchmark.memoryagentbench.hf_datasets import load_from_disk
# 兄弟包反向依赖，于是 MemoryAgentBench 从「并列的基准之一」
# 变成了所有人的地基——加一个平级的新基准要先绕开这层耦合。
def test_no_upward_or_sideways_imports() -> None:
    bad: list[str] = []
    for path in _py_files():
        layer = _layer_of(path)
        if layer is None:
            continue
        allowed = set(DEPS[layer]) | {layer}
        for dep in sorted(_imported_layers(ast.parse(path.read_text())) - allowed):
            bad.append(
                f"{path.relative_to(ROOT)}: {layer} → {dep}"
                f"（{layer} 只允许依赖 {DEPS[layer] or '无'}）"
            )
    assert not bad, "层依赖被破坏：\n  " + "\n  ".join(bad)


# ── 2. 顶层包不得增生 ───────────────────────────────────────────
# 这是「写死目录结构」的实际执行点：加一个新顶层包必须先改
# architecture.toml 想清楚它的依赖边，而不是随手 mkdir。
def test_top_level_packages_match_spec() -> None:
    actual = {p.name for p in SRC.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert not (actual - set(LAYERS)), (
        f"未声明的顶层包：{sorted(actual - set(LAYERS))}；"
        "加新层要先改 architecture.toml 与 ARCHITECTURE.md"
    )
    assert not (set(LAYERS) - actual), (
        f"architecture.toml 声明了但目录不存在：{sorted(set(LAYERS) - actual)}"
    )


# ── 3. 每个包必须自述职责 ───────────────────────────────────────
# 实测失效：MemoryData 的 utils/ 共 8006 行，utils/agent.py 单文件 4569 行，
# 且 utils/locomo_utils.py（题库专有逻辑）住在公共 utils 里。
# 一个说不清「自己只干哪一件事」的包，迟早长成 utils。
def test_every_package_declares_its_job() -> None:
    missing = [
        str(d.relative_to(ROOT))
        for d in sorted(SRC.rglob("*"))
        if d.is_dir() and "__pycache__" not in d.parts and not (d / "README.md").exists()
    ]
    assert not missing, "缺 README.md（写清只干哪一件事）：\n  " + "\n  ".join(missing)


# ── 4. ⛔ 适配器不得 vendor 上游 ────────────────────────────────
# 实测失效：MemoryData 的 methods/ 共 367,673 行 / 2,358 个 py 文件 / 579 个目录。
# methods/mem0/source/mem0/ 是整份上游包，methods/MemOS/source/pyproject.toml
# 是上游的构建文件。抄进来之后「被测的到底是哪个版本」就无从谈起了。
IMPL = SRC / "adapters" / "impl"
_PKGS = [p for p in sorted(IMPL.iterdir()) if p.is_dir() and p.name != "__pycache__"]


@pytest.mark.parametrize("pkg", _PKGS, ids=lambda p: p.name)
def test_adapter_stays_thin(pkg: Path) -> None:
    cfg = SPEC["adapters"]
    files = [p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts]
    lines = sum(len(p.read_text().splitlines()) for p in files)

    smuggled = [f for f in cfg["forbidden_files"] if list(pkg.rglob(f))]
    assert not smuggled, (
        f"{pkg.name}: 出现上游构建文件 {smuggled}——上游代码不进本仓库（原则④）"
    )
    assert len(files) <= cfg["max_python_files"], (
        f"{pkg.name}: {len(files)} 个 py 文件 > 上限 {cfg['max_python_files']}"
    )
    assert lines <= cfg["max_total_lines"], (
        f"{pkg.name}: {lines} 行 > 上限 {cfg['max_total_lines']}——"
        "适配器变胖通常意味着上游被抄了进来，或判分逻辑跑错了层"
    )


# ── 5. ⛔ 自研判分层不得出现 LLM 调用 ───────────────────────────
# 约束①：自研套件确定性判分，不用 LLM 评委。
# 公开档照用上游判分（含它们的评委），那发生在 suites/public，不在这里。
_LLM_HINTS = ("openai", "anthropic", "litellm", "langchain", "transformers", "llm")


def test_scoring_is_free_of_judges() -> None:
    bad: list[str] = []
    for path in (SRC / "scoring").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            mods = (
                [a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom)
                else []
            )
            for m in mods:
                if any(h in m.lower() for h in _LLM_HINTS):
                    bad.append(f"{path.relative_to(ROOT)}: import {m}")
    assert not bad, (
        "scoring/ 里出现疑似 LLM 依赖，违反约束①（自研套件不用评委）：\n  "
        + "\n  ".join(bad)
    )


# ── ⛔ 测试必须走生产那扇门 ─────────────────────────────────────
def test_every_control_arm_is_reachable_through_build() -> None:
    """⛔ `build()` 是**唯一**认识具体臂的地方——⚠️ env 变量、`storage_dir`、
    embedding 配置、臂名分派全在这里。

    ⭐ 而它此前只被 `null` / `bm25` / `host_default` 走过，
    ⚠️ 恰好是三条**不需要外部配置**的臂。⛔ 实测后果：
    给 `hybrid` 多传一个它不认识的参数 = TypeError，
    而全套 683 个测试照样绿——**因为没有一个测试走这条路造这条臂**。
    """
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    import offline
    # ⛔ 以**注册表**为准，⚠️ 不是 `CONTROL_ARMS`——
    # ⭐ 这条守卫最早遍历的就是 CONTROL_ARMS，而 `hybrid` 不在里面，
    # 于是它对**引发这一切的那条臂本身是瞎的**。
    from amb.adapters import SYSTEMS
    from amb.adapters.registry import _REGISTRY
    from amb.runner import build

    old = dict(__import__("os").environ)
    try:
        __import__("os").environ.update(offline.ENV)
        for name in sorted(set(_REGISTRY) - set(SYSTEMS)):
            build(name)          # ⛔ 造不出来就是 TypeError / KeyError
    finally:
        __import__("os").environ.clear()
        __import__("os").environ.update(old)


def test_a_new_arm_cannot_skip_the_full_pipeline_layer() -> None:
    """⛔ **防复发装置**：加一条新臂时，它必须同时进离线全流水线那一层。

    ⚠️ 那 7 个 bug 的共同成因是「这条臂没被完整跑过」——⭐ 而防止它再发生
    的办法不是记得，是**让漏掉这一步过不了测试**。

    ⚠️ 真要豁免某条臂，就在 `test_offline_fullrun.ARMS` 旁边写清为什么，
    ⛔ 而不是让它默默不在名单里。
    """
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from amb.adapters import SYSTEMS
    from amb.adapters.registry import _REGISTRY
    from test_offline_fullrun import ARMS

    # ⛔ 同理以注册表为准：⚠️ 被测系统要装外部依赖，离线层跑不了它们。
    missing = sorted(set(_REGISTRY) - set(SYSTEMS) - set(ARMS))
    assert not missing, (
        f"⛔ 这些对照臂没进离线全流水线：{missing}——"
        f"⚠️ 加进 `test_offline_fullrun.ARMS`，或写清豁免理由")


# ── ⛔ 臂的名册只能有一份 ───────────────────────────────────────
def test_every_registered_arm_is_classified() -> None:
    """⛔ 注册了的臂必须被归类：**要么对照，要么被测**。

    ⚠️ 实测漏网：`hybrid` 注册了、`build()` 认识它、真跑在用它，
    ⛔ 但它**既不在 `CONTROL_ARMS` 也不在 `SYSTEMS`**。三个后果都是实的：

      ① `test_adapter_conformance` 的 ARMS = 对照 + 被测 → **从不测它**
      ② 真跑里 `is_control=False` → 被标成「被测系统」
      ③ 地板线只从对照臂里选 → 它进不了地板候选

    ⭐ 加一条臂要动五个地方（注册表 / CONTROL_ARMS / SYSTEMS /
    `build()` 的分派 / `ARM_DEPENDENCY`），⛔ 而此前**没有任何东西检查它们一致**。
    """
    from amb.adapters import CONTROL_ARMS, SYSTEMS
    from amb.adapters.registry import _REGISTRY

    unclassified = sorted(set(_REGISTRY) - set(CONTROL_ARMS) - set(SYSTEMS))
    assert not unclassified, (
        f"⛔ 这些臂注册了却没归类：{unclassified}——"
        f"⚠️ 进 `CONTROL_ARMS`（对照）或 `SYSTEMS`（被测），"
        f"⭐ 否则一致性测试跳过它、报告标错它、地板线选不到它")


def test_the_two_rosters_never_overlap() -> None:
    """⛔ 一条臂不能既是对照又是被测——⚠️ 那样它会与自己比。"""
    from amb.adapters import CONTROL_ARMS, SYSTEMS

    both = sorted(set(CONTROL_ARMS) & set(SYSTEMS))
    assert not both, f"⛔ 这些臂两边都在：{both}"


def test_build_dispatches_on_every_registered_arm() -> None:
    """⛔ `build()` 造得出注册表里的**每一条**臂。

    ⚠️ 它的 if/elif 链是第四份名册——⭐ 而链子漏了谁，
    只有真跑到那条臂时才会知道。
    """
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    import offline
    from amb.adapters.registry import _REGISTRY
    from amb.runner.build import build
    from amb.setup import snapshot

    installed = set(snapshot())
    old = dict(__import__("os").environ)
    try:
        __import__("os").environ.update(offline.ENV)
        for name in sorted(_REGISTRY):
            try:
                build(name)
            except Exception as exc:  # noqa: BLE001
                # ⛔ 只放过「没装外部依赖」：⚠️ 兜底 except 会把
                # **构造 bug** 一起吞掉——那正是 `hybrid` 的 TypeError
                # 在 `test_adapter_conformance` 里溜过去的方式。
                assert "未安装" in str(exc) or "not installed" in str(exc), (
                    f"⛔ build({name!r}) 挂了，⚠️ 而这不是「依赖没装」："
                    f"{type(exc).__name__}: {exc}")
    finally:
        __import__("os").environ.clear()
        __import__("os").environ.update(old)
