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


def install_pip(dep: Dependency, *, upgrade: bool = False) -> Installed:
    have = installed_pip_version(dep.source)
    if have != dep.pin or upgrade:
        proc = _run([sys.executable, "-m", "pip", "install", "-q",
                     f"{dep.source}=={dep.pin}"])
        if proc.returncode != 0:
            return Installed(dep.name, dep.pin, have or "-", str(dep.kind), "",
                             ok=False, detail=proc.stderr.strip()[-400:])
        have = installed_pip_version(dep.source)

    if have != dep.pin:
        # ⛔ 装到的不是钉死的那个版本——那已经是另一个被测对象了
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
    lock[name] = got.as_dict()
    save_lock(lock)
    return got


def install_all(names: list[str] | None = None, *,
                upgrade: bool = False) -> list[Installed]:
    out: list[Installed] = []
    for name in names or sorted(REGISTRY):
        try:
            out.append(install(name, upgrade=upgrade))
        except (SetupError, ValueError) as exc:
            dep = REGISTRY[name]
            out.append(Installed(name, dep.pin, "-", str(dep.kind), "",
                                 ok=False, detail=str(exc)))
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
            out.append(Installed(**{k: row[k] for k in
                                    ("name", "declared", "actual", "kind",
                                     "location", "ok", "detail")}))
    return out


def snapshot() -> dict[str, dict]:
    """⚠️ 进结果报告的那一份。**没记录版本的跑不算数。**"""
    return load_lock()
