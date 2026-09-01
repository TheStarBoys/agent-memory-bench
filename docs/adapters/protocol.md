# 接口定义

> **状态：草案。** 原则见 [`README.md`](README.md)，世界模型见 [`world.md`](world.md)，
> 报告格式见 [`../report.md`](../report.md)。
> 签名用 Python 写，因为被测生态是 Python；语义与语言无关。

## 三态返回

贯穿全协议。**任何非基线方法都可以回 `Unsupported`**，这是一等结果不是错误。

```python
@dataclass(frozen=True)
class Unsupported:
    reason: str        # 这个系统没有这个能力

@dataclass(frozen=True)
class Failed:
    reason: str        # 声明了这个能力，但这次没做成

Outcome = T | Unsupported | Failed
```

⛔ 判分时三者**永不合并**，且三者进报告的方式各不相同：

| 结果 | 计入分母 | 报告位置 |
|---|---|---|
| 正常返回 | 是 | 分数 |
| `Unsupported` | **否** | 独立的「不支持」列 |
| `Failed` | **是**，记为未答对 | 分数，**并单独报 Failed 率** |

**`Unsupported` 与 `Failed` 的区别是这套协议的要害。** 前者是诚实的能力缺失，
后者是声明了却没做到。把 `Failed` 也挪出分母，等于开一个后门：
声明全部能力、次次返回 `Failed`，就能换到一个永远不掉分的位置。

⚠️ 单次跑里某套件的 `Failed` 率超过 **20%**，该套件结果标记为**不可信**，
不进对比表（[`../report.md`](../report.md)）。一个总是失败的能力声明，
和没有这个能力之间的差别只剩下声明本身。

### 基线方法不得返回三态

`setup` `ingest` `finalize` `search` `count` `reset` `close` **必须实现**，
不得返回 `Unsupported` 或 `Failed`。接不进来的系统就是接不进来，
不要用三态掩盖——那会让「不支持」这一列失去意义。

## 能力

```python
class Capability(StrEnum):
    INGEST     = "ingest"      # 基线，必需
    SEARCH     = "search"      # 基线，必需
    ANSWER     = "answer"      # 端到端答题（含生成器）
    REALITY    = "reality"     # N1
    PROVENANCE = "provenance"  # N2
    REASONING  = "reasoning"   # N3
    GOVERNANCE = "governance"  # N4
    RETENTION  = "retention"   # N5 系统自报（外部观察不需要声明）
    CONFIDENCE = "confidence"  # N7
    INDUCTION  = "induction"   # N8 推导链一档（三个问题不需要声明）
    ACCOUNTING = "accounting"  # 成本计量（原则⑥）
```

⚠️ **三个套件不在这张表里，因为它们不需要声明任何能力**：
[N5 外部观察](../suites/n5-consolidation.md)与
[N6 关联结构](../suites/n6-structure.md)只用 `search`，
[N8 三个问题](../suites/n8-induction.md)只用 `answer`。
**能从外部观察到的东西，不要求系统配合**——这是[原则②](README.md#p2)的直接后果，
也让这三种能全员参赛。

`capabilities()` 返回的集合决定评测器跑哪些套件。**未声明的不跑，记不支持。**

⚠️ **声明有代价，不声明也有代价。** 报告并列给出**分数**与**声明与参与**
（声明了几项 / 实际参与了几题），少声明会在声明与参与上直接可见。
没有这一列的话，「不支持不计 0 分」就变成了鼓励少声明——见
[原则①](README.md#p1) 与 [`../report.md`](../report.md)。

## 世界句柄

`setup()` 把世界交给被测系统。三样东西全部是**只读、进程外、不推送**的
（[`world.md`](world.md)）：

```python
@dataclass(frozen=True)
class WorldHandle:
    root: str          # 只读文件树的绝对路径
    clock_url: str     # GET → {"now": "<RFC3339>"}，评测器可随时拨动
    facts_url: str     # GET {facts_url}/{key} → {"key":…, "value":…} | 404
```

⛔ **时钟与事实表是端点，不是快照值。** 把一个当前时间的字符串发过去，
系统就永远察觉不到时间流逝——而时间流逝是 [N1 的五种变更之一](world.md#变更的类型)。

⛔ 三个端点**都不会通知任何人**。世界变了没有回调、没有事件、没有版本号跳变。
这是 N1 的全部要害。

## 数据类型

```python
@dataclass
class Document:
    """摄入单元。doc_id 与字符偏移由评测器分配并永久稳定——N2 靠它对账。"""
    doc_id: str
    text: str                      # 已做 NFC 规范化
    timestamp: str | None          # RFC3339
    principal: str | None          # 谁写的（N4）
    kind: Literal["turn", "document"]


@dataclass
class Span:
    """原文区间。⛔ 起止都要，只给起点不算。"""
    doc_id: str
    start: int                     # Unicode 码点偏移（Python str 索引），闭
    end: int                       # 同上，开
```

⛔ **偏移单位是 Unicode 码点，不是字节，也不是 token。**
中文语料下字节口径会让所有区间偏三倍。评测器在 `ingest` 之前把每份
`Document.text` 做 NFC 规范化，此后偏移永久固定。

```python
@dataclass
class Entry:
    """检索返回的一条。正文可选——多数系统不返回正文是对的。"""
    id: str
    digest: str
    score: float | None
    doc_ids: list[str]             # 源自哪些 Document。给不出就空列表（见下）
    spans: list[Span]              # N2 精确区间；不支持就空列表
    principal: str | None          # N4
    state: Literal["holds", "broken", "unknown"] | None   # N1 无提示；不表态就 None
    confidence: float | None       # N7：这条记忆正确的概率 [0,1]。⛔ 不是 score
    supersedes: list[str]          # 这条归并/取代了哪些旧 Entry.id
```

⛔ **`confidence` 不是 `score`。** `score` 是相关性排序分，量纲与语义由系统自定；
`confidence` 是有外部含义的概率——「这条是对的概率」。
拿 `score` 顶替会让 [N7](../suites/n7-calibration.md) 的校准分退化成
对排序分的重新缩放，⚠️ **那比返回 `None` 更糟**：`None` 是诚实的，顶替是伪造了一个维度。

⚠️ `doc_ids` 比 `spans` 弱得多——只说「来自哪份文档」，不说哪一段。
多数系统给得出（它们通常记了来源会话）。但**给不出就意味着
[N1 无提示](../suites/n1-reality.md#两种)无法对账**，该系统在无提示的情况下记不支持：
评测器无从知道被标 `broken` 的那条对应哪个被破坏的事实。

```python
@dataclass
class Claim:
    """N1：评测器出的一条待检命题。
    ⛔ 不问系统内部存了什么，只问这句话对当前世界还成不成立。"""
    claim_id: str
    text: str
    doc_ids: list[str]             # 这条命题当初由哪些 Document 陈述


@dataclass
class Verdict:
    """N1：一条命题对**当前世界**还成不成立。"""
    claim_id: str
    state: Literal["holds", "broken", "unknown"]
    grounds: list[str]             # 依据：world 内路径 / 事实表键 / Entry.id
    note: str | None               # 散文说明，判分不读
```

⛔ **`grounds` 不得为空，`unknown` 尤其不得为空**——它要列出「没能核到的是哪些东西」。
空 `grounds` 判为 `Failed`，不是 `unknown`。

这样「说清为什么」才是可确定性判分的：评测器只检查 `grounds` 非空、
且每一项都解析得到（路径存在于世界清单 / 键存在于事实表 / id 存在于该系统），
**从不读 `note` 里的散文**。散文没法确定性判分，所以它不参与判分。

```python
class Rule(StrEnum):
    """N3 推导链的封闭规则词表。⛔ 不在表内的一律判该步不成立。"""
    ASSERT     = "assert"       # 前提直接是一条记忆，零步推导
    CONJOIN    = "conjoin"      # 合取多个前提
    SUBSTITUTE = "substitute"   # 等值替换：a=b, P(a) ⊢ P(b)
    TRANSITIVE = "transitive"   # 传递：a→b, b→c ⊢ a→c
    COMPARE    = "compare"      # 数值 / 时间比较
    EXCLUDE    = "exclude"      # 候选集合减去被否定的项
    NEGATE     = "negate"       # 封闭世界否定：找不到满足 X 的记忆 ⊢ ¬X


@dataclass
class Premise:
    kind: Literal["entry", "step"]
    ref: str                       # Entry.id 或 Step.step_id


@dataclass
class Step:
    step_id: str
    claim: str                     # ⛔ 规范化三元组 "subject|relation|object"
    premises: list[Premise]
    rule: Rule


@dataclass
class Answer:
    text: str
    derivation: list[Step]         # N3；不支持就空列表
    used: list[str]                # 用到的 Entry.id
    missing: list[str]             # 未决时缺哪些前提，引用评测器的 claim_id
    confidence: float | None       # N7：这个回答正确的概率 [0,1]
```

⛔ **`Step.claim` 是规范化三元组不是散文，`rule` 取自封闭词表，`Step` 有自己的 id。**
这三条是 [N3 能确定性判分的前提](../suites/n3-reasoning.md)：
自由文本的推理步骤只有评委判得了，那就违反了[约束①](../suites/README.md)。

⛔ **`Premise` 带 `kind` 是必需的，不是冗余。** 前提要么引用一条记忆，
要么引用链上前一步的结论；不区分的话，接链只能靠 claim 字符串匹配，
而字符串匹配在有重复 claim 的链上会接错。

```python
@dataclass
class RecallVerdict:
    """N5 系统自报：一条命题现在还留着吗、多强。
    ⛔ 与 Verdict 一样由评测器出命题——不问系统内部存了什么。"""
    claim_id: str
    state: Literal["retained", "dropped", "unknown"]
    strength: float | None         # 保留强度 [0,1]，给不出就 None


class DefeasibleRule(StrEnum):
    """N8 的扩展规则词表。⛔ 与 N3 的 Rule 分开，不合并——
    Rule 全是单调的，混进非单调规则会让 N3 的判分变得可争议。"""
    DEFAULT = "default"    # 默认成立的规律：通常 A → P
    EXCEPT  = "except"     # 明确的例外：a 是 A，但 ¬P(a)


@dataclass
class Regularity:
    """N8 推导链一档（可选）：系统归纳到的一条规律。"""
    claim: str                     # 规范化三元组，同 Step.claim
    kind: DefeasibleRule
    strength: float | None         # 系统认为这条规律的成立率
    exceptions: list[str]          # 已知例外，引用 Entry.id


@dataclass
class DeleteResult:
    deleted: list[str]             # 确认删除的 Entry.id
    refused: dict[str, str]        # entry_id → 拒绝原因


@dataclass
class AuditEvent:
    event_id: str
    action: Literal["ingest", "update", "delete", "read"]
    entry_ids: list[str]
    principal: str | None          # 谁做的
    at: str                        # RFC3339
    detail: str | None


@dataclass
class Usage:
    """原则⑥。⚠️ 墙钟评测器自己能测，token 只有适配器报得出来。"""
    phase: Literal["ingest", "probe"]
    tokens_in: int
    tokens_out: int
    llm_calls: int
    wall_ms: int
```

## id 契约

四类题全部靠 id 对账，所以稳定性必须成文。

| id / 偏移 | 谁分配 | 稳定期 |
|---|---|---|
| `Document.doc_id`、字符偏移 | 评测器 | 永久 |
| `Claim.claim_id` | 评测器 | 永久 |
| `Entry.id` | 被测系统 | 从 `finalize()` 到下一次 `reset()` |
| `Step.step_id` | 被测系统 | 单次 `answer()` 调用内 |

三条硬规则：

1. ⛔ **id 不得回收。** 退役的 `Entry.id` 永远不得再指向别的内容。
2. ⛔ **归并必须留下继承链。** 系统在 `finalize()` 之后仍会归并条目——
   mem0 的 ADD/UPDATE/DELETE、ReMe 的四态裁决都会。新条目须在 `supersedes` 里
   列出它取代的旧 id。旧 id 此后可以不再被 `search` 返回，
   但 `delete(旧id)` 与 `audit_log` 仍须认得它。
3. ⛔ **`mutate` 阶段不得改变 id。** 适配器在该阶段不被调用也不被通知，
   任何后台线程导致的 id 漂移都会让 N1 的对账失效——**检测到即判本次跑作废**，
   不是扣分。

## 方法

```python
class Adapter(Protocol):
    def capabilities(self) -> set[Capability]: ...

    # ── 生命周期 ──────────────────────────────────────────────
    def setup(self, world: WorldHandle) -> None: ...   # 交出世界
    def reset(self) -> None: ...                       # 清空，套件之间隔离
    def close(self) -> None: ...

    # ── 基线：公开基准与自研套件都要，不得回三态 ─────────────────
    def ingest(self, doc: Document) -> None: ...
    def finalize(self) -> None: ...
    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]: ...
    def count(self) -> int: ...

    # ── ANSWER：端到端。报分时必须写成「<系统> + <backbone>」──
    def answer(self, query: str, *,
               principal: str | None = None) -> Answer | Unsupported | Failed: ...

    # ── N1 REALITY ────────────────────────────────────────────
    # 评测器改完世界之后调用，并**把命题交给系统**——不问它内部存了什么。
    # 无提示不调这个方法，只看 search() 顺带返回的 Entry.state。
    def audit(self, claims: list[Claim]) -> list[Verdict] | Unsupported | Failed: ...

    # ── N4 GOVERNANCE ─────────────────────────────────────────
    def delete(self, entry_ids: list[str]) -> DeleteResult | Unsupported | Failed: ...
    def audit_log(self) -> list[AuditEvent] | Unsupported | Failed: ...
    # 供评测器做带外只读取证，区分「过滤」与「真删」（原则④的唯一例外）
    def storage_locations(self) -> list[str] | Unsupported: ...

    # ── N5 RETENTION（系统自报；外部观察只用 search，无需此方法）──
    def recall(self, claims: list[Claim]) -> list[RecallVerdict] | Unsupported | Failed: ...

    # ── N8 INDUCTION（推导链一档；三个问题只用 answer）─────────────
    def regularities(self) -> list[Regularity] | Unsupported | Failed: ...

    # ── ACCOUNTING（原则⑥）────────────────────────────────────
    def usage(self) -> list[Usage] | Unsupported: ...
```

<a id="why-evaluator-issues-claims"></a>

### 为什么 `audit()` 由评测器出命题

一个看起来更自然的设计是让 `audit()` 不带参数，由系统返回
「我的哪些记忆不再成立」。⛔ **那个设计是错的**，因为它预设了
**系统持有可枚举、有稳定 id 的离散条目集合**——
MemoryLLM（参数化记忆）、raptor（递归摘要树）根本没有这种集合。
**这就是[原则②](README.md#p2)要挡的形状偏心**：接口挑实现，
而落选的理由是形状不合，不是能力不够。

评测器出命题之后，系统只需回答「这句话现在还成不成立」，
内部怎么存不影响它能不能参赛。

**「主动发现」这一维没有丢，它由[两种](world.md#两种)承载**：
有提示给命题，无提示什么都不给、只看 `search()` 顺带返回的 `Entry.state`。

## 阶段

评测器固定按这个顺序驱动，**适配器不得跨阶段留后门**：

```
setup     建世界 → reset() → setup(world)
ingest    逐条 ingest(doc) → finalize()
mutate    ⚠️ 只有评测器动世界。适配器不参与，也不被通知
probe     search / answer / audit / audit_log / storage_locations
score     确定性判分
```

**每个阶段边界都校验一次世界哈希**，不只是 `probe` 前后——
`ingest` 与 `finalize` 期间系统同样在运行，同样够得着世界。
校验方式与只读的强制手段见 [`world.md`](world.md#归属与只读)。

`mutate` 阶段不通知适配器是刻意的：被告知"世界变了"再去查，测的是执行；
没被告知还能发现，测的才是 N1。

## 每个套件用哪些方法

| 套件 | 必需能力 | 判分读什么 |
|---|---|---|
| 公开基准 | `INGEST` `SEARCH`（+ `ANSWER` 跑端到端档） | 交给上游评测框架 的判分代码 |
| N1 有提示 | `REALITY` | `Verdict.state` 三态 + `grounds` 可解析性 |
| N1 无提示 | `REALITY` + `Entry.doc_ids` 非空 | `Entry.state` 对 ground truth 的 3×2 混淆矩阵 |
| N2 原文回链 | `PROVENANCE` | `Entry.spans` 对 ground-truth 区间 |
| N3 推理链 | `REASONING` | `Answer.derivation` 逐步校验 + `missing` |
| N4 治理 | `GOVERNANCE` | 跨 principal 的 `search`、`delete`、`audit_log`、带外取证 |
| N5 外部观察 | **无** | 多个时间点的 `search` 结果对需求概率 |
| N5 系统自报 | `RETENTION` | `RecallVerdict.state` + `strength` |
| N6 关联结构 | **无** | 可达性与精确检索两条曲线（都来自 `search`） |
| N7 校准 | `CONFIDENCE` | `Answer.confidence` / `Entry.confidence` 的 ECE · Brier · 可靠性图 |
| N8 三个问题 | **无**（需 `ANSWER`） | 泛化 / 例外 / 规律存活三问的行为 |
| N8 推导链一档 | `INDUCTION` | `Regularity` 的 `kind` 与 `strength` |
| 成本与延迟 | `ACCOUNTING` | `Usage` 逐阶段（墙钟评测器另行独立测一份） |

## 两个钩子分开报，不许合并

| 钩子 | 量的是 | 报分写法 |
|---|---|---|
| `search()` | **记忆层的召回质量**——正是记忆系统的职责 | `<系统>` |
| `answer()` | 端到端准确率，**含答案生成器** | `<系统> + <backbone>` |

只报 `answer` 是把别人的生成器算进自己的成绩；只报 `search` 是回避端到端可用性。
**两个都跑，分开报，说明差别。**

⚠️ 跑 `answer` 档时**所有系统必须用同一个 backbone**。
一家用 GPT-4o、一家用本地 7b，分数没有可比性——那时候"公平"就成了自说自话。
