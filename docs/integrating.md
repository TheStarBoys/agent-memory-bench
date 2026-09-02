# 接入外部东西：照着做

> 三类外部依赖，**同一套流程**：钉死版本 → 一键装 → 记录实际装到的 →
> 薄壳接入 → 如实声明 → 真跑一次。
>
> ⛔ 每一步都有测试盯着，跳过哪一步会当场红。

| 要接什么 | 跳到 |
|---|---|
| 一个**被测记忆系统**（mem0、letta、cognee…） | [§2](#2-接一个被测记忆系统) |
| 一个**公开题库**（LoCoMo、BEAM…） | [§3](#3-接一个公开题库) |
| 一个 **agent 宿主**或别的运行时 | [§4](#4-接一个运行时) |

---

## 1. 共同的第一步：进 setup 清单

⛔ **所有外部东西都从这里进来**，没有例外。

在 [`src/amb/setup/spec.py`](../src/amb/setup/spec.py) 的 `REGISTRY` 加一条：

```python
"letta": Dependency(
    name="letta",
    kind=Kind.PIP,               # PIP 或 GIT
    source="letta",              # pip 包名 / git URL
    pin="0.6.4",                 # ⛔ 精确版本；git 用分支名，实际记 commit sha
    verify_import="letta",       # 装完拿它验一下真的在
    note="被测系统。⚠️ 需要 …",
),
```

然后：

```sh
python -m amb.cli setup letta        # 装
python -m amb.cli setup --check      # 只看状态
```

### ⛔ 三条规矩

| | 为什么 |
|---|---|
| **钉死版本** | ⚠️ 换一个被测系统的版本**等于换了被测对象**。空 `pin` 在构造时就被拒绝 |
| **记录实际装到的** | ⭐ 声明的与装到的**不一定一样**：git 声明分支名，实际记完整 commit sha。pip 装错版本直接抛 `VersionMismatch` |
| **源码不进本仓库** | pip 进 site-packages，git 进 `.external/`（已 gitignore）。[原则④](adapters/README.md#p4) |

⚠️ 实际版本进[结果报告](report.md)：**没记录版本的跑不算数**。

⛔ 别的层要用它之前先 `require_installed(name)`——
没装就抛异常，**不是给 0 分**。一个缺依赖的跑不该悄悄产出一个分数。

---

## 2. 接一个被测记忆系统

### 2.1 写一个薄壳

`src/amb/adapters/impl/<名字>/adapter.py`，从
[`_template/`](../src/amb/adapters/impl/_template/README.md) 拷。

⛔ **只走公开接口**：包顶层导出的 SDK、HTTP、CLI。
不 import 内部子模块、不复制它的代码。

⚠️ 薄度有上限，测试强制：**≤10 个 py 文件 / ≤1000 行**。
超限通常意味着上游被抄了进来，或判分逻辑跑错了层。

```python
class LettaAdapter(Answerable, AdapterBase):
    name = "letta"

    def capabilities(self) -> set[Capability]:
        return set(BASELINE) | self._answer_caps() | {…}

    def ingest(self, doc: Document) -> None: ...
    def search(self, query, k, *, principal=None) -> list[Entry]: ...
    def count(self) -> int: ...
```

### 2.2 ⭐ 如实声明能力——这一步最容易做错

**声明了什么就必须做到什么，做不到的就别声明。**

| 情形 | 该怎么记 |
|---|---|
| 没这个能力 | ⛔ **不声明** → 记「不支持」，不计分母，⛔ **不是 0 分** |
| 有能力但这次没做成 | 返回 `Failed` → **计入分母**记为未答对 |
| 做了但答案不对 | 正常返回 → 计入分数 |

⛔ 三者任意两个合并，这把尺子就废了。
把 `Failed` 挪出分母 = 开后门：声明全部能力、次次失败，就换到一个永不掉分的位置。

⚠️ **少声明也有代价**：报告里并列「声明了几项 / 参与了几题」，
沉默本身可见。

⭐ 拿不准就看 mem0 是怎么填的
（[`impl/mem0/README.md`](../src/amb/adapters/impl/mem0/README.md)）：

| | |
|---|---|
| ⛔ 不声明 `PROVENANCE` | 它返回**抽出来的事实**，不是原文片段，给不出区间 |
| ⛔ 不声明 `REALITY` | 它不对外部世界求值 |
| ⚠️ 声明 `GOVERNANCE` | 有 `user_id` 过滤 + `history()`——⭐ **但那是过滤不是授权**，四步探针会把这一点测出来 |

### 2.3 注册 + 构造

```python
# src/amb/adapters/registry.py
SYSTEMS = ("mem0", "letta")          # ⚠️ 与 CONTROL_ARMS 分开
register("letta", LettaAdapter)      # ⛔ 精确名，绝不子串匹配
```

⚠️ 构造参数在 [`runner/build.py`](../src/amb/runner/build.py)——
⛔ 那是**唯一一处**认识具体系统的地方，cli 不许认识。

⚠️ 被测系统的依赖**延迟 import**：没装它的机器也要能跑对照组。

### 2.4 ⭐ 先量成本，再跑题库

⛔ **别一上来就跑全量。** 先测摄入速率：

```python
t = time.perf_counter(); arm.ingest(doc); print(time.perf_counter() - t)
```

⚠️ 实测 mem0 在 Qwen3-8B 上 **36.7 秒/轮**（15–86 秒波动）——
它每次 `add()` 要多轮 LLM 调用（抽取 → 比对 → 裁决）。
外推 LoCoMo 全量 5882 轮要 ⛔ **60 小时**。

⭐ 所以对这类系统，`--max-turns` 不是可选项，是**跑得动的前提**。
这也正是[原则⑥](adapters/README.md#p6)存在的理由：
**一个赢 2 个百分点但贵 100 倍的系统，不写下来就看不出来。**

### 2.5 真跑一次，冒烟确认四件事

```sh
python -m amb.cli --arms letta,bm25,null --no-answer
```

| ⛔ 必须确认 |
|---|
| 「不支持」被记成不支持，**不是 0** |
| 「失败」被记成 `Failed`，**不是不支持** |
| 适配器全程**没写过世界**（哈希在四个阶段边界都一致） |
| id 不回收、归并留 `supersedes`、`mutate` 期间不漂 |

---

## 3. 接一个公开题库

⛔ **唯一的纪律：用它们的判分代码，不自己重写。**
重写只会引入「我们的判分与别人不同」这个不可比性。

### 3.1 进 setup 清单，登记已知缺陷

```python
# src/amb/suites/public/spec.py
"beam": Pin(
    repo="https://github.com/mohammadtavakoli78/BEAM",
    commit="<40 位 sha>",
    caveats=(
        "判分是 nugget + LLM 评委，⚠️ 带评委漂移，"
        "不可与自研套件的确定性分数并列比较",
        "⚠️ 只报准确率，不报成本与延迟",
    ),
),
```

⚠️ `caveats` 逐条进报告的 `upstream_notes`。
⛔ **上游判分有缺陷时照旧调用，不静默修好**——
悄悄修了，我们的数就和所有引用上游成绩的论文对不上，而读者不知道差在哪。

⛔ 上游判分不可用时 `require_scorer()` 抛异常，
**不许自己写一个顶上**。

### 3.2 写 loader

`src/amb/suites/public/<名字>.py`。三件事分开：
**取数据** · **喂进去** · **判分**，⛔ 第三件永远交给上游。

⭐ 有 ground-truth 证据字段的题库（LoCoMo 的 `evidence`）
可以**不生成答案就判检索**——那一档量的是记忆层本身，
没把别人的生成器算进成绩。

⛔ 数据不在就抛 `DatasetMissing`，**记「未接入」不是 0 分**。

### 3.3 ⭐ 抽题：两个维度，别混

| 控什么 | 参数 | ⚠️ |
|---|---|---|
| **题数** | `--sample stratified:50` | 五种策略：`all` `first:N` `random:N` **`stratified:N`** `ids:a,b` |
| **语料量** | `--max-convs` / `--max-turns` | ⛔ **与题数是两件事** |

⭐ **分层是推荐的**：LoCoMo 五类占比 42/22/16/14/5%，
简单随机抽 50 题，占 5% 的那类很可能一道都没抽到——
而那一类往往正是最该看的。

⛔ **截了语料就必须丢掉证据被截掉的题**，
留着它们必然全错，**那个分数是假的**，还会让系统看起来比实际差。
丢了几道进报告。

⛔ 抽样方式与种子进报告：不记的话，两次跑的差可能全来自抽到了不同的题。

### 3.4 逐类分开报

⛔ **总分会把结构糊掉。** bm25 在 LoCoMo 上的实测：

| 类 | recall |
|---|---:|
| 单跳事实 | 0.783 |
| **多跳** | **0.136** ⛔ |
| *总分* | *0.522* ⚠️ 完全看不出多跳塌了 |

---

## 4. 接一个运行时

比如 agent 宿主。同样进 setup 清单钉死版本。

⛔ **宿主是受控变量，不是被测对象**：
换版本等于换尺子，**要重跑全部基线**。版本进报告。

⚠️ 需要钉死的 seam（以 DSH 为例）：
`ctx.llm`（所有系统同一 backbone）· `ctx.agentLoop` · `ctx.tools` · `ctx.systemPrompt`。
细节见 [`src/amb/agent/README.md`](../src/amb/agent/README.md)。

---

## 5. 检查单

接完之后逐条过：

- [ ] 进了 setup 清单，`pin` 非空
- [ ] `python -m amb.cli setup <名字>` 装得上
- [ ] `setup --check` 里 `actual` 有值
- [ ] ⛔ 源码没进版本库（`git status` 干净）
- [ ] 能力自述**如实**——做不到的没声明
- [ ] ⛔ 没这能力时返回 `Unsupported`，不是空列表
- [ ] 声明了的能力**真的实现了**（不返回 `Unsupported`）
- [ ] 薄度在上限内（测试会告诉你）
- [ ] 量过摄入成本，知道该用多大语料
- [ ] 真跑过一次，四条冒烟确认都过
- [ ] `python -m pytest tests/ -q` 全绿

⚠️ 最后一条最要紧：**架构守卫会拦下大部分做错的方式**——
层依赖、包增生、适配器变胖、判分层混进评委，它都管。
