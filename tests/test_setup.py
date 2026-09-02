"""外部依赖的一键 setup。

⛔ 三条规矩：钉死版本 · 记录实际版本 · 源码不进仓库。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amb.setup import (
    Dependency,
    Kind,
    REGISTRY,
    SetupError,
    dependency,
    require_installed,
    status,
)
from amb.setup.spec import load_lock, save_lock


def test_every_registered_dependency_is_pinned() -> None:
    """⛔ 清单里不许有未钉死的依赖——换版本等于换了被测对象。"""
    for name, dep in REGISTRY.items():
        assert dep.pin, f"{name} 没钉死"


def test_an_unpinned_dependency_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="换了被测对象"):
        Dependency(name="x", kind=Kind.PIP, source="y", pin="")


def test_lookup_is_exact() -> None:
    """⛔ 与适配器注册表同一条纪律。"""
    with pytest.raises(KeyError):
        dependency("mem")          # mem0 的前缀


def test_not_installed_is_refused_not_zeroed(tmp_path: Path) -> None:
    """⛔ 没装就拒绝，⚠️ 不许静默跑出一个分。"""
    lock = tmp_path / "installed.json"
    with pytest.raises(SetupError, match="不是 0 分"):
        require_installed("mem0", lock)


def test_a_failed_install_is_also_refused(tmp_path: Path) -> None:
    lock = tmp_path / "installed.json"
    save_lock({"mem0": {"name": "mem0", "ok": False, "actual": "-"}}, lock)
    with pytest.raises(SetupError):
        require_installed("mem0", lock)


def test_actual_version_is_recorded_separately_from_declared() -> None:
    """⭐ git 声明分支名，实际记的是 commit sha——两者必须分开存。"""
    rows = {r.name: r for r in status()}
    locomo = rows["locomo"]
    if not locomo.ok:
        pytest.skip("locomo 还没装（python -m amb.cli setup locomo）")
    assert locomo.declared == "main"
    assert len(locomo.actual) == 40, "⭐ 实际版本应当是完整 commit sha"
    assert locomo.actual != locomo.declared


def test_external_sources_are_gitignored() -> None:
    """⛔ 源码不进本仓库（原则④）。"""
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    assert ".external/" in ignored


def test_snapshot_shape_is_report_ready() -> None:
    from amb.setup import snapshot

    for name, row in snapshot().items():
        assert {"declared", "actual", "ok"} <= set(row), f"{name} 快照缺字段"
