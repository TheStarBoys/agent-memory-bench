"""每个**被测系统**一个独立 venv。

⛔ 被测系统绝不装进我们的解释器。理由不是洁癖，是实测踩到的两次：

| 踩到的 | 后果 |
|---|---|
| MemoryOS 把 `openai` 从 2.x 降到 1.109 | 我们自己的调用全废，回滚才恢复 |
| a-mem 依赖 `litellm`，它声明 `openai>=2.20,<3.0` | 会把 3.7.0 降下来 |

⚠️ 更要命的是：本机的解释器是使用者的**日常环境**（装着别的项目），
往里塞被测对象等于拿别人的工作环境做实验台。

⭐ 所以：`Kind.VENV` 的依赖装到 `.external/venvs/<name>/`，
适配器通过子进程跟它说话（见 `amb.adapters.bridge`）。
⚠️ `Kind.PIP` 留给**宿主与工具**（如 dsh）——那些我们要在进程内直接 import。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from amb.setup.spec import EXTERNAL, Dependency, Installed, SetupError

VENVS = EXTERNAL / "venvs"
#: 装一个带 torch 的系统可能要几分钟，⚠️ 不能用默认超时掐掉
INSTALL_TIMEOUT_S = 1800


def venv_dir(name: str) -> Path:
    return VENVS / name


def venv_python(name: str) -> Path:
    """⚠️ 只给路径，不保证存在——存在性由 `require_venv` 判。"""
    base = venv_dir(name)
    exe = base / ("Scripts" if sys.platform == "win32" else "bin")
    return exe / ("python.exe" if sys.platform == "win32" else "python")


def require_venv(name: str) -> Path:
    """⛔ 没建就拒绝，不静默回退到本解释器——回退会把隔离悄悄取消掉。"""
    python = venv_python(name)
    if not python.exists():
        raise SetupError(
            f"{name} 的独立环境还没建。⛔ 该系统记「未接入」，不是 0 分。\n"
            f"    python -m amb.cli setup {name}"
        )
    return python


def _run(cmd: list[str], timeout: int = INSTALL_TIMEOUT_S):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def install_venv(dep: Dependency, *, upgrade: bool = False) -> Installed:
    """建 venv → 装钉死的版本 → ⭐ 在**那个** venv 里核对实际版本。"""
    python = venv_python(dep.name)
    if not python.exists() or upgrade:
        venv_dir(dep.name).parent.mkdir(parents=True, exist_ok=True)
        proc = _run([sys.executable, "-m", "venv", str(venv_dir(dep.name))])
        if proc.returncode != 0:
            return _failed(dep, f"建 venv 失败：{proc.stderr.strip()[-400:]}")

    proc = _run([str(python), "-m", "pip", "install", "-q",
                 f"{dep.source}=={dep.pin}"])
    if proc.returncode != 0:
        return _failed(dep, proc.stderr.strip()[-400:])

    # ⭐ 版本要从**那个 venv** 里问，⛔ 不是从我们的解释器
    actual = _version_in(python, dep.source)
    if actual != dep.pin:
        return _failed(
            dep, f"钉死 {dep.pin}，实际装到 {actual}。⛔ 换版本等于换了被测对象")

    if dep.verify_import:
        proc = _run([str(python), "-c",
                     f"import {dep.verify_import} as m; print(m.__file__)"],
                    timeout=300)
        if proc.returncode != 0:
            return _failed(dep, f"装上了但 import 不了：{proc.stderr[-300:]}")

    return Installed(dep.name, dep.pin, actual, str(dep.kind),
                     str(python), ok=True,
                     detail=_isolation_note(python))


def _version_in(python: Path, dist: str) -> str:
    proc = _run([str(python), "-c",
                 "import sys;from importlib.metadata import version,"
                 "PackageNotFoundError\n"
                 "try: print(version(sys.argv[1]))\n"
                 "except PackageNotFoundError: print('-')", dist], timeout=120)
    return proc.stdout.strip() or "-"


def _isolation_note(python: Path) -> str:
    """⭐ 把「它自己那套关键依赖是什么版本」记进锁文件。

    ⚠️ 这不是凑数：`openai` 在 venv 里是几点几，直接决定了
    这次跑的结果能不能跟别的系统并排看。
    """
    watched = ("openai", "chromadb", "litellm", "sentence-transformers")
    got = [f"{d}={_version_in(python, d)}" for d in watched]
    return "隔离环境：" + " ".join(g for g in got if not g.endswith("=-"))


def _failed(dep: Dependency, detail: str) -> Installed:
    return Installed(dep.name, dep.pin, "-", str(dep.kind),
                     str(venv_dir(dep.name)), ok=False, detail=detail)
