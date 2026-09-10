"""可控规模的 N1 题面：⭐ 文档 · 命题 · 变更 · 真值，一起生成。

## ⛔ 为什么要有这个文件

⚠️ N1 此前是**手写的三条命题**——`检出率` 的分母因此只有 2。
⛔ 那不是「测不出差别」，是**结构上不可能测出差别**：
⭐ 按仓库自己的样本量公式（`required_n`），配对口径下要辨 0.05 需要 121 题。

## ⭐ 四种变更必须成比例出现

⚠️ 只造「被删掉的文件」那一类，检出率就退化成「会不会查文件存不存在」。
⛔ 四类各有各的失效方式：

    VANISH     文件没了      —— 最容易发现
    REVALUE    值变了        —— 要比对内容，不能只看存在性
    ADVANCE    过期了        —— 要读时钟，⚠️ 而多数系统不读
    IRRELEVANT 无关变更      —— ⛔ **反方向**：这些命题仍然成立，
                                 谁把它们也报成 broken 就是误报

⭐ 没有第四类，一条「什么都报 broken」的臂能拿满分。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from amb.core import Claim, Document
from amb.world.mutate import Change, ChangeKind


@dataclass
class RealityWorld:
    """⚠️ 四样东西必须同源生成——⛔ 分开写迟早对不上。"""

    documents: list[Document] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    #: `claim_id → "holds" | "broken"`
    truth: dict[str, str] = field(default_factory=dict)
    #: ⭐ 事实表的初值：⚠️ REVALUE 那一类要同时改文件与事实表，
    #: ⛔ 只改一边世界就自相矛盾，命题没有确定的真值（第一次跑就这么错的）。
    facts: dict[str, str] = field(default_factory=dict)


#: ⚠️ 主题池：⛔ 刻意用真实词汇而不是 `对象V064` 这类合成 id——
#: 合成 id 对 BM25 是独一无二的 token，对 embedding 却是噪声，
#: ⭐ 那会让两类臂在**不同的题**上比。
_TOPICS = (
    "巡检记录", "备份策略", "值班安排", "采购清单", "培训计划",
    "机房温度", "带宽配额", "证书有效期", "灰度比例", "告警阈值",
    "留存天数", "缓存容量", "重试次数", "超时上限", "并发上限",
    "日志级别", "副本数量", "分片大小", "心跳间隔", "回滚窗口",
    "巡检周期", "密钥轮换", "灾备演练", "限流阈值", "熔断比例",
    "预热时长", "抓取频率", "清洗规则", "对齐基准", "派发策略",
)
_OWNERS = ("研发一组", "运维值班", "数据平台", "安全合规", "质量保障",
           "网络组", "存储组", "调度组", "风控组", "交付组",
           "监控组", "基础架构", "测试中台", "发布委员会", "容量规划")


class TooFewTopics(ValueError):
    """⛔ 要的题数超过主题池能给的不重复组合。⚠️ 不静默重复——
    重复的主题会让不同命题指向同一份语料，⭐ 那时真值会互相污染。"""


def build(*, seed: int, n_claims: int = 128) -> RealityWorld:
    """造 `n_claims` 条命题，⭐ 四类变更各占四分之一。

    ⚠️ 每条命题**独占**一份文档：⛔ 共用的话一次变更会同时改掉几条命题的
    真值，⭐ 而那时「检出一条」与「检出三条」在判分上分不开。
    """
    kinds = (ChangeKind.VANISH, ChangeKind.REVALUE,
             ChangeKind.ADVANCE, ChangeKind.IRRELEVANT)
    combos = [(t, o) for t in _TOPICS for o in _OWNERS]
    if n_claims > len(combos):
        raise TooFewTopics(
            f"要 {n_claims} 条，⚠️ 而主题池只有 {len(combos)} 个不重复组合")

    rng = random.Random(seed)
    rng.shuffle(combos)
    w = RealityWorld()
    for i in range(n_claims):
        topic, owner = combos[i]
        kind = kinds[i % len(kinds)]
        doc_id = f"n1/{kind.value}/{i:03d}.md"
        value = str(100 + i)
        # ⚠️ 正文里带上**键名**：⛔ 纯中文问句对纯 ASCII 的键共享 token 为零，
        # ⭐ 那对任何词法臂是结构性不可答，与被测系统无关。
        key = f"item_{i:03d}"
        text = f"{topic}由{owner}负责，{key}={value}。"
        w.documents.append(Document(doc_id=doc_id, text=text,
                                    principal="alice", kind="document"))
        w.facts[key] = value

        cid = f"c{i:03d}"
        if kind is ChangeKind.VANISH:
            w.claims.append(Claim(cid, f"{topic}的记录存在", [doc_id]))
            w.changes.append(Change(ChangeKind.VANISH, doc_id))
            w.truth[cid] = "broken"
        elif kind is ChangeKind.REVALUE:
            w.claims.append(Claim(cid, f"{key} 是 {value}", [doc_id]))
            # ⛔ 文件与事实表**一起改**，⚠️ 只改一边世界就自相矛盾
            w.changes.append(Change(ChangeKind.REVALUE, key, str(int(value) + 7)))
            w.changes.append(Change(
                ChangeKind.REVALUE, doc_id,
                f"{topic}由{owner}负责，{key}={int(value) + 7}。"))
            w.truth[cid] = "broken"
        elif kind is ChangeKind.ADVANCE:
            # ⚠️ 带有效期的命题：⛔ 时钟推过去就不成立了
            w.claims.append(Claim(cid, f"{topic}的{key}仍在有效期内", [doc_id]))
            w.truth[cid] = "broken"
        else:
            # ⭐ **反方向**：⛔ 这条命题在变更之后仍然成立
            w.claims.append(Claim(cid, f"{topic}由{owner}负责", [doc_id]))
            w.changes.append(Change(ChangeKind.IRRELEVANT, doc_id,
                                    f"{topic}由{owner}负责，{key}={value}。补充说明。"))
            w.truth[cid] = "holds"

    # ⭐ 时钟前推一次，⚠️ 让 ADVANCE 那一类真的过期
    w.changes.append(Change(ChangeKind.ADVANCE, "2026-12-31T00:00:00Z"))
    return w
