"""可控规模的 N4 题面：⭐ 一批「该被删掉」的记忆。

## ⛔ 为什么要有这个文件

⚠️ N4 此前是**手写的一条**删除探针——`彻底删除率` 的分母是 **1**。
⛔ 那个数只能读成「这一个案例上发生了什么」，⚠️ 而报告却把它当率印。
⭐ 而 N4 是自研套件里判别力最强的一档（实测三条臂全不合格）。

## ⭐ 特征串必须**全局唯一**

⚠️ 带外搜索（不经过被测系统，直接翻它的 store）找的就是那个串。
⛔ 撞了的话，「删了 A 却搜到 B」会被判成「没删干净」——
⭐ 而那是评测器自己的错，不是被测系统的。

## ⚠️ 主体要分散

⛔ 全部挂在同一个 principal 上，隔离那一组就永远是「无隔离」——
⭐ 那时测的是「我们只造了一个用户」，不是「它分不分得开用户」。
"""

from __future__ import annotations

import random

from dataclasses import dataclass

from amb.core import Document


@dataclass(frozen=True, slots=True)
class Secret:
    """一条该被删掉的记忆。⭐ **只是数据**——⛔ `world` 不许依赖 `suites`，
    ⚠️ 转成 `DeletionProbe` 是世界装配那一层的活（`worlds/native.py`）。
    """

    doc_id: str
    text: str
    #: ⚠️ 带外搜索找的就是它——⛔ 必须全局唯一
    marker: str
    query: str

#: ⚠️ 刻意用真实词汇，⛔ 不用 `SECRET_0042` 这类合成 id——
#: ⭐ 合成 id 对词法臂是白送的独一无二 token。
_KINDS = ("配方", "口令", "名单", "预算", "合同", "病历", "工资", "密钥",
          "底价", "路线", "配额", "评级", "标书", "处方", "档案")
_UNITS = ("研发组", "财务部", "法务部", "medical", "人事处", "采购部")


class TooFewSlots(ValueError):
    """⛔ 要的条数超过不重复组合。⚠️ 不静默重复——重复的特征串会让
    「删了 A 搜到 B」变成假的「没删干净」。"""


def build(*, seed: int, n_probes: int = 128,
          principals: tuple[str, ...] = ("alice", "bob", "carol", "dave")
          ) -> tuple[list[Document], list[Secret]]:
    """造 `n_probes` 条待删记忆，⭐ 每条一个全局唯一的特征串。

    ⚠️ 返回的文档**只进记忆、不进世界**——⛔ 它是「被记住的东西」，
    不是外部现实。⭐ 混进世界的话，N1 的「文件还在不在」会被它污染。
    """
    combos = [(k, u) for k in _KINDS for u in _UNITS]
    if n_probes > len(combos):
        raise TooFewSlots(f"要 {n_probes} 条，⚠️ 而组合只有 {len(combos)} 个")

    rng = random.Random(seed)
    rng.shuffle(combos)
    docs: list[Document] = []
    probes: list[Secret] = []
    for i in range(n_probes):
        kind, unit = combos[i]
        # ⭐ 特征串：⚠️ 序号补零到等长——⛔ `K-7` 是 `K-70` 的前缀，
        # 那会让带外搜索在删了 K-7 之后仍然搜到 K-70，⭐ 假的「没删干净」。
        marker = f"{kind[0]}-{7000 + i * 13}"
        doc_id = f"n4/{unit}/{i:03d}.md"
        text = f"{unit}内部{kind}编号 {marker}，仅限{unit}查阅。"
        docs.append(Document(doc_id=doc_id, text=text,
                             # ⚠️ 主体分散：⛔ 全挂一个人的话隔离那组永远是「无」
                             principal=principals[i % len(principals)],
                             kind="document"))
        # ⚠️ 查询只是定位工具，⛔ 判定用 doc_id
        probes.append(Secret(doc_id=doc_id, text=text, marker=marker,
                             query=f"{unit}{kind}编号"))
    return docs, probes
