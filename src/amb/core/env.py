"""从仓库根的 .env 加载环境变量。stdlib 实现，不引第三方依赖。

⛔ 只是把值放进 os.environ，**已存在的变量不覆盖**——
显式导出的环境变量优先于文件，这样 CI 与本地不会互相打架。

密钥只住在 .env（已 gitignore）；配置文件里存的是变量名。
"""

from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(start: Path | None = None) -> Path | None:
    """从 start 往上找 .env，止于文件系统根。"""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """读入 .env，返回本次实际设置的键值。找不到文件就安静地返回空。"""
    path = path or find_dotenv()
    if path is None or not path.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        value = value.split(" #")[0].strip().strip("'\"")
        if not name or (not override and name in os.environ):
            continue
        os.environ[name] = value
        applied[name] = value
    return applied


def require(name: str) -> str:
    """读一个必需的环境变量，缺了就报清楚是**哪个变量名**。

    ⛔ 不返回空字符串静默降级——那会让一次跑悄悄用错配置。
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise KeyError(
            f"环境变量 {name} 未设置。"
            f"⛔ 密钥不进仓库——拷 .env.example 成 .env 填入，见 configs/README.md"
        )
    return value
