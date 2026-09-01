# 接口定义

> **状态：草案。** 原则见 [`README.md`](README.md)，世界模型见 [`world.md`](world.md)。
> 签名用 Python 写，因为被测生态是 Python；语义与语言无关。

## 三态返回

贯穿全协议。**任何方法都可以回 `Unsupported`**，这是一等结果不是错误。

```python
Supported[T]  = T                  # 做了，这是结果
Unsupported   = ("unsupported", reason: str)   # 这个系统没有这个能力
Failed        = ("failed", reason: str)        # 有能力但这次没做成
```

⛔ 判分时三者**永不合并**。`Unsupported` 进独立的列，不计入分数也不记 0。

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
```

`capabilities()` 返回的集合决定 harness 跑哪些套件。**未声明的不跑，记不支持。**

## 数据类型

```python
@dataclass
class Document:
    """摄入单元。id 与 offset 由 harness 分配并保持稳定——N2 靠它对账。"""
    doc_id: str
    text: str
    timestamp: str | None          # RFC3339
    principal: str | None          # 谁写的（N4）
    kind: Literal["turn", "document"]

@dataclass
class Span:
    """原文区间。⛔ 起止都要，只给起点不算。"""
    doc_id: str
    start: int                     # 字符偏移，闭
    end: int                       # 字符偏移，开

@dataclass
class Entry:
    """检索返回的一条。正文可选——多数系统不返回正文是对的。"""
    id: str
    digest: str
    score: float | None
    spans: list[Span]              # N2；不支持就空列表
    principal: str | None          # N4

@dataclass
class Step:
    """推导链的一步（N3）。"""
    claim: str
    premises: list[str]            # 引用 Entry.id 或前面 Step 的结论
    rule: str | None

@dataclass
class Answer:
    text: str
    derivation: list[Step]         # N3；不支持就空列表
    used: list[str]                # 用到的 Entry.id

@dataclass
class Verdict:
    """N1：一条记忆对当前世界还成不成立。"""
    entry_id: str
    state: Literal["pass", "fail", "unknown"]
    reason: str                    # ⛔ unknown 必须说明为什么没能判定
```

## 方法

```python
class Adapter(Protocol):
    def capabilities(self) -> set[Capability]: ...

    # ── 基线：公开基准与自研套件都要 ──────────────────────────
    def ingest(self, doc: Document) -> None: ...
    def finalize(self) -> None: ...
    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]: ...
    def count(self) -> int: ...

    # ── ANSWER：端到端。报分时必须写成「<系统> + <backbone>」──
    def answer(self, query: str, *,
               principal: str | None = None) -> Answer | Unsupported: ...

    # ── N1 REALITY ────────────────────────────────────────────
    # harness 改完世界之后调用。问的是「你的哪些记忆不再成立」，
    # 不问你怎么知道的（原则②）。
    def audit(self) -> list[Verdict] | Unsupported: ...

    # ── N4 GOVERNANCE ─────────────────────────────────────────
    def delete(self, entry_ids: list[str]) -> DeleteResult | Unsupported: ...
    def audit_log(self) -> list[AuditEvent] | Unsupported: ...

    # ── 生命周期 ──────────────────────────────────────────────
    def reset(self) -> None: ...      # 清空，用于套件之间隔离
    def close(self) -> None: ...
```

## 阶段

harness 固定按这个顺序驱动，**适配器不得跨阶段留后门**：

```
setup     建世界（world.md）、reset() 适配器
ingest    逐条 ingest(doc) → finalize()
mutate    ⚠️ 只有 harness 动世界。适配器不参与，也不被通知
probe     search / answer / audit / audit_log
score     确定性判分
```

`mutate` 阶段**不通知适配器**是刻意的：被告知"世界变了"再去查，
测的是执行；没被告知还能发现，测的才是 N1。

## 每个套件用哪些方法

| 套件 | 必需能力 | 判分读什么 |
|---|---|---|
| 公开基准 | `INGEST` `SEARCH`（+ `ANSWER` 跑端到端档） | 交给上游 harness 的判分代码 |
| N1 对现实求值 | `REALITY` | `Verdict.state` 三态 + `reason` |
| N2 原文回链 | `PROVENANCE` | `Entry.spans` 对 ground-truth 区间 |
| N3 推理链 | `REASONING` | `Answer.derivation` 逐步校验 |
| N4 治理 | `GOVERNANCE` | 跨 principal 的 `search`、`delete`、`audit_log` |

## 两个钩子分开报，不许合并

| 钩子 | 量的是 | 报分写法 |
|---|---|---|
| `search()` | **记忆层的召回质量**——正是记忆系统的职责 | `<系统>` |
| `answer()` | 端到端准确率，**含答案生成器** | `<系统> + <backbone>` |

只报 `answer` 是把别人的生成器算进自己的成绩；只报 `search` 是回避端到端可用性。
**两个都跑，分开报，说明差别。**

⚠️ 跑 `answer` 档时**所有系统必须用同一个 backbone**。
一家用 GPT-4o、一家用本地 7b，分数没有可比性——那时候"公平"就成了自说自话。
