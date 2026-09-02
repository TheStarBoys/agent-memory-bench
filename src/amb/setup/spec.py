"""外部依赖的声明式清单。

⛔ 三条规矩，每一条都有测试盯着：
    ① 钉死版本/commit —— 没钉死的不许进清单
    ② 记录**实际装到的**版本 —— 声明的与装到的不一定一样
    ③ 源码不进本仓库（原则④）—— 装到 .external/ 或 site-packages

⚠️ 版本快照进结果报告。**没记录版本的跑不算数**：
换一个被测系统的版本等于换了被测对象。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Kind(StrEnum):
    #: ⚠️ 装进**我们的**解释器。只给宿主与工具（如 dsh）——
    #: 那些我们要在进程内直接 import。
    PIP = "pip"
    #: ⭐ 装进 `.external/venvs/<name>/` 的**独立** venv，走子进程说话。
    #: ⛔ 所有被测系统都必须是这一种：它们的依赖跟我们的会打架，
    #: 实测 MemoryOS 降过 openai、a-mem 的 litellm 卡 openai<3。
    VENV = "venv"
    GIT = "git"      # clone 到 .external/，⛔ 不进版本库


class SetupError(RuntimeError):
    """装不上。⛔ 该系统记「未接入」，不是 0 分。"""


class VersionMismatch(SetupError):
    """装到的版本与钉死的不一致。⛔ 直接拒绝——那已经是另一个被测对象了。"""


@dataclass(frozen=True, slots=True)
class Dependency:
    name: str
    kind: Kind
    #: pip 用包名，git 用仓库 URL
    source: str
    #: ⛔ pip 用精确版本号，git 用完整 commit sha
    pin: str
    #: 装完之后拿什么来验证它真的在
    verify_import: str | None = None
    #: git 专用：只要这几个子路径（省磁盘）
    sparse: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not self.pin:
            raise ValueError(
                f"{self.name} 没钉死版本。⛔ 清单里不许有未钉死的依赖——"
                f"换一个版本等于换了被测对象"
            )


@dataclass
class Installed:
    """一次实际安装的结果。⚠️ 这份要进报告。"""

    name: str
    declared: str
    #: ⭐ 实际装到的——与 declared 不一定一样
    actual: str
    kind: str
    location: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "declared": self.declared, "actual": self.actual,
            "kind": self.kind, "location": self.location, "ok": self.ok,
            "detail": self.detail,
        }


#: 装到哪。⛔ 已 gitignore——源码不进本仓库。
EXTERNAL = Path(".external")
#: 版本快照落在哪，⚠️ 报告从这里读
LOCKFILE = Path(".external/installed.json")


REGISTRY: dict[str, Dependency] = {
    "mem0": Dependency(
        name="mem0",
        kind=Kind.VENV,
        source="mem0ai",
        pin="2.0.19",
        verify_import="mem0",
        note="被测系统。⚠️ 需要 OPENAI 兼容端点，配置见 configs/",
    ),
    "a_mem": Dependency(
        name="a_mem",
        kind=Kind.VENV,
        source="a-mem",   # ⚠️ 发行名是 a-mem，import 名才是 agentic_memory
        pin="0.2.6",
        verify_import="agentic_memory",
        note="被测系统。⭐ embedding 本地跑不花钱；"
             "⚠️ 它写死 OpenAI 官方端点，适配器用 OPENAI_BASE_URL 搭桥",
    ),
    "locomo": Dependency(
        name="locomo",
        kind=Kind.GIT,
        source="https://github.com/snap-research/locomo",
        # ⛔ 接入时由 setup 解析出来并写回；空值会被 __post_init__ 拒绝，
        # 所以这里放一个占位分支名，实际 commit 记在 lockfile 里
        pin="main",
        sparse=("data",),
        note="公开题库。⚠️ NOASSERTION 许可——只读数据，代码不复制进本仓库",
    ),
    "dsh": Dependency(
        name="dsh",
        kind=Kind.PIP,
        source="deepseek-harness-sdk",
        pin="0.1.2a3",
        verify_import="deepseek_harness",
        note="agent 宿主。⛔ 受控变量——换版本要重跑全部基线",
    ),
}


def dependency(name: str) -> Dependency:
    """⛔ 精确查找。"""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"未登记的外部依赖 {name!r}。已登记：{sorted(REGISTRY)}") from None


def load_lock(path: Path = LOCKFILE) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_lock(entries: dict[str, dict], path: Path = LOCKFILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")


def require_installed(name: str, path: Path = LOCKFILE) -> dict:
    """⛔ 没装过就拒绝——不许在缺依赖的情况下静默跑出一个分。"""
    entry = load_lock(path).get(name)
    if entry is None or not entry.get("ok"):
        raise SetupError(
            f"{name} 还没装。⛔ 该系统记「未接入」，不是 0 分。\n"
            f"    python -m amb.cli setup {name}"
        )
    return entry
