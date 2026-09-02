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
    kind=Kind.VENV,              # ⛔ 被测系统一律 VENV，见下
    source="letta",              # pip 包名 / git URL
    pin="0.6.4",                 # ⛔ 精确版本；git 用分支名，实际记 commit sha
    verify_import="letta",       # 装完拿它验一下真的在
    note="被测系统。⚠️ 需要 …",
),
```

<a id="isolation"></a>

### ⛔ 被测系统一律装进**它自己的 venv**

| `kind` | 装到哪 | 给谁用 |
|---|---|---|
| `Kind.VENV` | `.external/venvs/<名字>/`，⭐ 走子进程说话 | **所有被测系统** |
| `Kind.PIP` | 我们的解释器 | 只给**宿主与工具**（如 dsh）——那些我们要在进程内直接 `import` |
| `Kind.GIT` | clone 到 `.external/` | 题库这类只读数据 |

这不是洁癖，是两次实测：

| 踩到的 | 后果 |
|---|---|
| MemoryOS 把 `openai` 从 2.x 降到 1.109 | 我们自己的调用全废，回滚才恢复 |
| a-mem 依赖 `litellm`，它声明 `openai>=2.20,<3.0` | 会把 3.7 降下来 |

⚠️ 而且跑评测的那台机器上，解释器往往是使用者的**日常环境**。
往里塞被测对象，等于拿别人的工作环境做实验台。

⭐ 隔离还顺手解决了一件事：**每个系统自己那套依赖的版本也被记进锁文件**。
a_mem 那条记的是 `openai=2.54.0 chromadb=1.5.9 litellm=1.99.0`——
⚠️ 这些数字跟被测系统的版本号一样，是结果可复现的一部分。

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

⛔ 被测系统在**另一个解释器**里，所以适配器不能 `import` 它。
分两个文件写：

| 文件 | 跑在哪 | 能 import 什么 |
|---|---|---|
| `adapter.py` | 我们的进程 | `amb.*`，⛔ **不许** import 被测系统 |
| `worker.py` | `.external/venvs/<名字>/bin/python` | 标准库 + 被测系统，⛔ **不许** import `amb` |

中间靠 [`bridge`](../src/amb/adapters/bridge.py)：stdout 上一行一个 JSON。
四个约定，⛔ 别加别的：

1. 一行一个 JSON——多行没法在流上切开
2. worker 的日志一律走 **stderr**，⛔ stdout 只放协议
3. 出错回 `{"ok": false, "error": …}`，⛔ 不靠退出码——进程还要接着用
4. ⛔ 子进程死了**不许重启**：重启会把已摄入的状态悄悄清空，
   而调用方毫无察觉——那样跑出来的是「半个语料的记忆系统」的分数，
   ⚠️ 比直接崩掉更有害。抛 `BridgeError`，这条臂记「跑挂了」，**不是 0 分**。

⭐ `llm_cache` 只依赖标准库 + openai，worker 按路径直接加载宿主那一份，
所以「缓存」和「temperature 钉 0」在隔离环境里**仍然只有一份实现**。

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

### 3.4 ⭐ 分数必须带区间

⛔ 抽样跑出来的分**不许只给一个点估计**——
[抽样方法论](sampling.md)：小样本给的是带区间的无偏估计，
一个不带区间的抽样分是在骗人。

接题库时要提供的：

| | |
|---|---|
| 每题的对错（或命中/未命中） | ⭐ 区间要靠逐题结果算，⛔ 不能只给汇总 |
| 每题属于哪一层 | 分层估计要按层加权还原总体 |
| 全量里每层有多少条 | 有限总体校正要用 |

⚠️ 先想清楚**要分辨多大的差异**，再决定抽多少题：
20 个百分点要 92 题，10 个要 384 题，5 个要 1556 题。

### 3.5 逐类分开报

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
