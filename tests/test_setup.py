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


# ── ⛔ 版本表不许说谎 ────────────────────────────────────────────
def test_a_failed_install_is_written_to_the_lockfile() -> None:
    """⛔ 只落成功那几行 = 上一次成功的 `ok: true` 原封不动留着。

    ⚠️ 于是报告那张版本表与 `require_installed()` **一起说谎**：
    ⭐ 而「没记录版本的跑不算数」是本项目的硬规矩。
    """
    import inspect

    # ⚠️ `amb.setup.install` 这个名字被同名**函数**遮住了——
    # ⛔ 这个坑在本仓库已经出现三次（amb.report.render / amb.cli.main / 这里）
    from importlib import import_module

    src = inspect.getsource(import_module("amb.setup.install"))
    assert "def _record(" in src, "⛔ 没有「失败也落锁文件」这条路"
    # ⭐ 版本对不上那条路径必须先落再抛
    body = src[src.index("if have != dep.pin:"):]
    assert body.index("_record(") < body.index("raise VersionMismatch")


def test_venv_subprocesses_do_not_inherit_a_leaky_environment() -> None:
    """⛔ 隔离必须是真的：⚠️ 版本核对与 `verify_import` 都在子进程里跑，
    而 `PYTHONPATH` 指向 venv 外面时它们照样通过——⭐「装好了」是假的。

    ⚠️ 用户的硬规矩：被测系统一律装进独立 venv，
    ⛔ 绝不往 anaconda 之类的日常环境装任何东西。
    """
    from amb.setup.venv import _LEAKY, _clean_env

    import os

    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        assert name in _LEAKY, f"⛔ {name} 会把外面的包漏进来"
    os.environ["PYTHONPATH"] = "/tmp/leak"
    try:
        env = _clean_env()
        assert "PYTHONPATH" not in env
        assert env.get("PYTHONNOUSERSITE") == "1", "⚠️ 用户 site-packages 也是一条路"
    finally:
        os.environ.pop("PYTHONPATH", None)
