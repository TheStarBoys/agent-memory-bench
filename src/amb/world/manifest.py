"""世界清单：声明式，可复现。

⛔ 同一份清单 + 同一个种子必须产出可复现的世界——
否则两次跑的差可能全部来自世界本身，而不是被测系统。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FileSpec:
    path: str          # 相对 root
    text: str
    mode: int = 0o444  # ⛔ 默认只读


@dataclass(frozen=True, slots=True)
class WorldManifest:
    """⚠️ clock_start 同时是所有文件的 mtime——见 materialize。"""

    name: str
    seed: int
    clock_start: str                                  # RFC3339
    files: tuple[FileSpec, ...] = ()
    facts: dict[str, str] = field(default_factory=dict)

    def file(self, path: str) -> FileSpec | None:
        return next((f for f in self.files if f.path == path), None)
