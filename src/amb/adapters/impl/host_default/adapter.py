"""裸宿主：⭐ 真实地板。

不挂任何记忆插件，只用 agent 宿主自带的上下文管理与压缩
（DSH 的工作记忆 · ctx.compaction · 文件引用）。

⭐ 这才是「不装记忆系统」真实的样子，也是最该报的一条线：
    真正要回答的不是「你比什么都没有强吗」，
    而是「你比让宿主自己压缩上下文强吗」。

⚠️ 从评测器这一面看，它的 search 返回空——因为**没有记忆层可查**。
它与 null 的差别只在装进 agent 跑的时候才显现：
宿主的上下文管理会替它记住最近发生的事。
"""

from __future__ import annotations

from amb.adapters.worldcheck import WorldReader, looks_like_world_ref
from amb.adapters.answerable import Answerable
from amb.core import (
    BASELINE, AdapterBase, Capability, Claim, Document, Entry, Failed, Verdict,
    WorldHandle,
)


class HostDefaultAdapter(Answerable, AdapterBase):
    name = "host_default"

    def __init__(self) -> None:
        self._seen = 0
        self._reader: WorldReader | None = None

    def capabilities(self) -> set[Capability]:
        # ⭐ 声明 REALITY：它没有记忆，所以每次都去重读世界。
        # baselines.md 预测这条线在 N1 上会反直觉地高——正确但昂贵。
        return set(BASELINE) | self._answer_caps() | {Capability.REALITY}

    def setup(self, world: WorldHandle) -> None:
        super().setup(world)
        self._reader = WorldReader(world)

    def audit(self, claims: list[Claim]) -> list[Verdict] | Failed:
        """重读世界。⚠️ 没存过旧值，所以只判「还在不在」，判不了「变没变」。"""
        if self._reader is None:
            # ⛔ 有能力但这次没做成 = Failed，不是 Unsupported——
            #    混淆两者就给了「声明全部能力、次次失败」一个免罚位置。
            return Failed("setup() 未调用，拿不到世界句柄")
        out: list[Verdict] = []
        for c in claims:
            grounds = [f"clock:{self._reader.now()}"]
            missing = False
            unverifiable = False
            for ref in c.doc_ids:
                if not looks_like_world_ref(ref):
                    unverifiable = True       # ⛔ 内部 id，核不了
                    continue
                r = (self._reader.file(ref) if "/" in ref
                     else self._reader.fact(ref))
                grounds.append(r.ground)
                if not r.exists:
                    missing = True
                else:
                    # ⛔ 从没存过原值，无从判断内容变没变——诚实地说不知道，
                    #    ⚠️ 不许猜成 holds：那是把「没检查」当成「检查通过」。
                    unverifiable = True
            if missing:
                out.append(Verdict(c.claim_id, "broken", grounds))
            elif unverifiable:
                out.append(Verdict(c.claim_id, "unknown", grounds,
                                   note="没有存过原值，判不了内容是否变化"))
            else:
                out.append(Verdict(c.claim_id, "holds", grounds))
        return out

    def ingest(self, doc: Document) -> None:
        """不建索引。语料由宿主的上下文管理消化，⛔ 这里不插手。"""
        self._seen += 1

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        return []  # ⛔ 没有记忆层——不是「查了但没找到」

    def count(self) -> int:
        return 0  # 记忆层里确实是 0 条；_seen 只用于冒烟核对

    def reset(self) -> None:
        self._seen = 0
