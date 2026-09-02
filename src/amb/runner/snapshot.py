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


def saved_cost(key: SnapshotKey, root: Path = ROOT) -> dict | None:
    """存快照那次**实测**的摄入成本。⛔ 命中快照时它才是真数字。

    ⚠️ 不带这个的话，快照命中的臂摄入耗时显示成 ~0，
    而那正好是成本对比里最要紧的一格——⭐ 省了时间不该连测量结果一起丢掉。

    ⚠️ 这个数字**可以**跨跑复用：快照键已经锁死了臂 + 版本 + 摄入身份 +
    语料指纹，⛔ 四项全同才会命中，所以它测的就是这份语料上的这个系统。
    ⚠️ 但它是**上一次**的墙钟，机器负载不同会有出入——报告里要标出来。
    """
    if not (key.path(root) / ".complete").is_file():
        return None
    try:
        meta = json.loads((key.path(root) / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    got = meta.get("ingest_cost")
    return got if isinstance(got, dict) else None


def saved_canary(key: SnapshotKey, root: Path = ROOT) -> dict | None:
    """存快照那次摄入完的**行为**指纹。⛔ 没有就返回 None——验不了就别用。"""
    if not (key.path(root) / ".complete").is_file():
        return None
    try:
        meta = json.loads((key.path(root) / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    got = meta.get("canary")
    return got if isinstance(got, dict) and got else None


def save(key: SnapshotKey, store: Path, root: Path = ROOT,
         cost: dict | None = None, canary: dict | None = None) -> None:
    """摄入完存一份。⚠️ 写完才落 .complete——⛔ 半截快照比没有更糟。

    `cost`：⭐ 这一次**实测**的摄入成本，一起存进去，
    ⛔ 让后面命中快照的跑还能报出真数字。
    """
    if not store.is_dir():
        return
    dst = key.path(root)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store, dst / "store")
    meta = key.as_dict()
    if cost:
        meta["ingest_cost"] = cost
    if canary:
        # ⭐ 恢复之后拿它对账：⛔ 行为对不上就丢弃快照，不出假分
        meta["canary"] = canary
    (dst / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # ⛔ 最后一步：只有它在，快照才算数
    (dst / ".complete").write_text("", encoding="utf-8")
