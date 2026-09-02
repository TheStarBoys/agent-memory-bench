"""摄入快照：⭐ 贵的那一步只做一次。

实测摄入占总耗时 **86%**（mem0_raw：730 秒 vs 问答 115 秒）。
而摄入对**同一份语料 + 同一个系统 + 同一个 backbone** 是确定的——
⭐ 存下来，重跑问答就免费。

⛔ 快照键必须覆盖所有影响摄入结果的东西：
系统名 + 版本 + backbone + 语料指纹。⚠️ 漏一个就会拿错快照，
而那比慢更糟——**它会静默给出别的系统的分**。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from amb.core import Document

ROOT = Path(".external/snapshots")


@dataclass(frozen=True, slots=True)
class SnapshotKey:
    arm: str
    #: ⚠️ 被测系统的**实际**版本，⛔ 不是声明的
    arm_version: str
    backbone: str
    corpus_digest: str

    @property
    def digest(self) -> str:
        raw = json.dumps({
            "arm": self.arm, "arm_version": self.arm_version,
            "backbone": self.backbone, "corpus": self.corpus_digest,
        }, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def path(self, root: Path = ROOT) -> Path:
        return root / f"{self.arm}-{self.digest}"

    def as_dict(self) -> dict[str, str]:
        return {"arm": self.arm, "arm_version": self.arm_version,
                "backbone": self.backbone, "corpus": self.corpus_digest,
                "digest": self.digest}


def corpus_digest(documents: list[Document]) -> str:
    """语料指纹。⛔ 顺序也算进去——摄入顺序会影响归并型系统的结果。"""
    h = hashlib.sha256()
    for doc in documents:
        h.update(f"{doc.doc_id}\0{doc.text}\0{doc.principal or ''}\n".encode())
    return h.hexdigest()[:16]


def restore(key: SnapshotKey, into: Path, root: Path = ROOT) -> bool:
    """有快照就恢复。⭐ 命中即跳过摄入。"""
    src = key.path(root)
    if not (src / ".complete").is_file():
        return False
    shutil.rmtree(into, ignore_errors=True)
    into.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "store", into)
    return True


def save(key: SnapshotKey, store: Path, root: Path = ROOT) -> None:
    """摄入完存一份。⚠️ 写完才落 .complete——⛔ 半截快照比没有更糟。"""
    if not store.is_dir():
        return
    dst = key.path(root)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store, dst / "store")
    (dst / "meta.json").write_text(
        json.dumps(key.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    # ⛔ 最后一步：只有它在，快照才算数
    (dst / ".complete").write_text("", encoding="utf-8")
