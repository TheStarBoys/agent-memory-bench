"""一键 setup：装外部依赖，⭐ 并记录**实际装到的**版本。

    python -m amb.cli setup            # 装全部
    python -m amb.cli setup mem0       # 只装一个
    python -m amb.cli setup --check    # 只看状态，不装

⛔ 记录的是**实际**版本不是声明版本——两者可能不一样，
而报告里要的是实际那个。
"""

from __future__ import annotations

import shutil
import subprocess
import os
import sys
from pathlib import Path

from amb.setup.spec import (
    EXTERNAL,
    LOCKFILE,
    Dependency,
    Installed,
    Kind,
    REGISTRY,
    SetupError,
    VersionMismatch,
    dependency,
    load_lock,
    save_lock,
)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def installed_pip_version(module_or_dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(module_or_dist)
    except PackageNotFoundError:
        return None


def _is_user_daily_env() -> str:
    """当前解释器像不像**用户的日常环境**。⛔ 返回原因，空串 = 看着是隔离的。

    ⚠️ 用户有一条硬规矩：被测系统一律装进独立 venv，
    ⛔ **绝不往 anaconda 之类的日常环境装任何东西**。
    而 `Kind.PIP` 装的是 `sys.executable`——⭐ 那可能正是它。
    """
    # ⭐ 判据是**解释器本身在不在 venv 里**，⚠️ 不是外层 shell 开着什么：
    # 实测踩到——项目自己的 `.venv/bin/python` 在一个开着 conda 的 shell 里
    # 跑，早先被误判成「日常环境」并拒绝安装。
    if sys.prefix != sys.base_prefix:
        return ""                       # 在 venv / virtualenv 里，⭐ 是隔离的
    exe = sys.executable.lower()
    for mark in ("anaconda", "miniconda", "/usr/bin/python",
                 "/usr/local/bin/python"):
        if mark in exe:
            return f"当前解释器是日常环境：{sys.executable}"
    if os.environ.get("CONDA_PREFIX"):
        return f"在 conda 的 base 环境里：{os.environ['CONDA_PREFIX']}"
    return ""


def install_pip(dep: Dependency, *, upgrade: bool = False) -> Installed:
    have = installed_pip_version(dep.source)
    if have != dep.pin or upgrade:
        # ⛔ 装之前先问：这是不是用户的日常环境？
        # ⚠️ `AMB_ALLOW_SYSTEM_PIP=1` 是**明示同意**的出口，⭐ 默认拒绝。
        if (why := _is_user_daily_env()) and not os.environ.get(
                "AMB_ALLOW_SYSTEM_PIP"):
            raise SetupError(
                f"⛔ 拒绝往日常环境装 {dep.source}：{why}\n"
                f"⚠️ 被测系统一律装进独立 venv。⭐ 确实要装就先建一个 venv，"
                f"或显式设 AMB_ALLOW_SYSTEM_PIP=1")
        proc = _run([sys.executable, "-m", "pip", "install", "-q",
                     f"{dep.source}=={dep.pin}"])
        if proc.returncode != 0:
            return Installed(dep.name, dep.pin, have or "-", str(dep.kind), "",
                             ok=False, detail=proc.stderr.strip()[-400:])
        have = installed_pip_version(dep.source)

    if have != dep.pin:
        # ⛔ 装到的不是钉死的那个版本——那已经是另一个被测对象了。
        # ⚠️ 但**先把「不 ok」落进锁文件再抛**：早先直接 raise，
        # ⭐ 于是上一次成功的 `ok: true` 那一行原封不动留着——
        # 报告那张版本表和 `require_installed()` 一起说谎。
        _record(Installed(dep.name, dep.pin, have or "-", str(dep.kind), "",
                          ok=False,
                          detail=f"钉死 {dep.pin}，实际 {have}——⛔ 版本不符"))
        raise VersionMismatch(
            f"{dep.name}: 钉死 {dep.pin}，实际装到 {have}。"
            f"⛔ 拒绝——换版本等于换了被测对象"
        )

    location = ""
    if dep.verify_import:
        proc = _run([sys.executable, "-c",
                     f"import {dep.verify_import} as m; print(m.__file__)"])
        if proc.returncode != 0:
            return Installed(dep.name, dep.pin, have, str(dep.kind), "",
                             ok=False, detail=f"装上了但 import 不了：{proc.stderr[-300:]}")
        location = proc.stdout.strip()
    return Installed(dep.name, dep.pin, have, str(dep.kind), location, ok=True)


def install_git(dep: Dependency) -> Installed:
    """clone 到 .external/，⛔ 不进版本库。⭐ 记录**解析出来的 commit sha**。"""
    target = EXTERNAL / dep.name
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        cmd = ["git", "clone", "--depth", "1", "--branch", dep.pin,
               dep.source, str(target)]
        if dep.sparse:
            cmd = ["git", "clone", "--depth", "1", "--filter=blob:none",
                   "--sparse", "--branch", dep.pin, dep.source, str(target)]
        proc = _run(cmd)
        if proc.returncode != 0:
            return Installed(dep.name, dep.pin, "-", str(dep.kind), str(target),
                             ok=False, detail=proc.stderr.strip()[-400:])
        if dep.sparse:
            _run(["git", "-C", str(target), "sparse-checkout", "set", *dep.sparse])

    sha = _run(["git", "-C", str(target), "rev-parse", "HEAD"]).stdout.strip()
    if not sha:
        return Installed(dep.name, dep.pin, "-", str(dep.kind), str(target),
                         ok=False, detail="拿不到 commit sha")
    # ⭐ 实际版本 = 解析出来的完整 sha，⚠️ 不是声明的分支名
    return Installed(dep.name, dep.pin, sha, str(dep.kind), str(target), ok=True)


def install(name: str, *, upgrade: bool = False) -> Installed:
    dep = dependency(name)
    if dep.kind is Kind.VENV:
        from amb.setup.venv import install_venv

        got = install_venv(dep, upgrade=upgrade)
    elif dep.kind is Kind.PIP:
        got = install_pip(dep, upgrade=upgrade)
    else:
        got = install_git(dep)
    lock = load_lock()
    # ⛔ **「命令没成功」与「环境里没有」是两件事**：⚠️ 早先无条件覆盖，
    # 于是一次联网失败的 setup 会把好端端的装机状态改写成「未接入」。
    # ⭐ 失败时保留旧行，只把失败原因附上去。
    old = lock.get(name)
    if not got.ok and old and old.get("ok"):
        old = dict(old)
        old["detail"] = (f"⚠️ 上一次 setup 没成功（{got.detail[:120]}），"
                         f"⛔ 但环境里那份仍在——下面是它的记录")
        lock[name] = old
    else:
        lock[name] = got.as_dict()
    save_lock(lock)
    return got


def _record(got: "Installed") -> None:
    """把一行落进锁文件。⛔ **失败也要落**——⚠️ 只落成功的那几行，
    上一次成功的 `ok: true` 就会原封不动留着，⭐ 而报告那张版本表
    与 `require_installed()` 会一起说谎。"""
    lock = load_lock()
    lock[got.name] = got.as_dict()
    save_lock(lock)


def install_all(names: list[str] | None = None, *,
                upgrade: bool = False) -> list[Installed]:
    out: list[Installed] = []
    for name in names or sorted(REGISTRY):
        try:
            out.append(install(name, upgrade=upgrade))
        except (SetupError, ValueError) as exc:
            dep = REGISTRY[name]
            bad = Installed(name, dep.pin, "-", str(dep.kind), "",
                            ok=False, detail=str(exc))
            # ⛔ 失败也落锁文件：⚠️ 否则上一次成功那一行会留下来
            _record(bad)
            out.append(bad)
    return out


def status(names: list[str] | None = None) -> list[Installed]:
    """只看，不装。"""
    lock = load_lock()
    out: list[Installed] = []
    for name in names or sorted(REGISTRY):
        dep = REGISTRY[name]
        row = lock.get(name)
        if row is None:
            out.append(Installed(name, dep.pin, "-", str(dep.kind), "",
                                 ok=False, detail="未安装"))
        else:
            # ⛔ 缺键不许炸：⚠️ 锁文件是外部状态，格式变过就会缺字段，
            # ⭐ 而 `status()` 的全部意义是「告诉我现在是什么情况」
            out.append(Installed(**{
                k: row.get(k, "" if k != "ok" else False)
                for k in ("name", "declared", "actual", "kind",
                          "location", "ok", "detail")}))
    return out


def snapshot() -> dict[str, dict]:
    """⚠️ 进结果报告的那一份。**没记录版本的跑不算数。**"""
    return load_lock()
