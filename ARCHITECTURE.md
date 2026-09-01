# 架构

> **目录结构是写死的。** 依赖边的唯一权威是 [`architecture.toml`](architecture.toml)，
> 由 [`tests/test_architecture.py`](tests/test_architecture.py) 强制执行。
> **这份文档解释为什么，那个文件说了算。**

## 从 MemoryData 学到的

[MemoryData](docs/harnesses.md) 是覆盖面最广的现成评测框架，我们**调用它但不 fork**。
下面每一条都在本地 clone 上实测过，它们直接决定了本项目的模块边界——
**这套架构主要是照着这些失效模式反着设计的。**

| 实测 | 数字 | 我们的对策 |
|---|---|---|
| `methods/` 把上游整包抄进仓库<br>（`methods/mem0/source/mem0/`、`methods/MemOS/source/pyproject.toml`） | **367,673 行 / 2,358 py / 579 目录** | 适配器薄度上限写进 `architecture.toml`，守卫测试拒收上游构建文件 |
| 分发靠子串匹配 `if "mem0" in agent_name`，约 25 处 | `"mem0" in "amem0"` → **真**，两个方法会撞 | 注册表**只做精确键查找**，查不到即报错 |
| 兄弟包反向依赖<br>`benchmark/longbench/loader.py:5` → `benchmark.memoryagentbench.hf_datasets` | 1 处，但足以让 MemoryAgentBench 从「并列基准之一」变成地基 | **分层 + 禁止同级互 import**，守卫测试用 AST 检查 |
| `utils/` god-module，含题库专有的 `locomo_utils.py` | **8,006 行**，`agent.py` 单文件 **4,569 行** | ⛔ **没有 `utils/`**。每个包一份 README 写清「只干哪一件事」，缺 README 测试即红 |
| 入口文件承载题库专有知识<br>（模块顶层的 `BACKFILL_LONGMEMEVAL_RECALL_METHODS`） | `main.py` **925 行** | `cli/` 只解析参数并交给 `runner`，题库/系统知识不得出现 |
| 函数内部 lazy import `evaluation.longmemeval…`，3 处 | 循环依赖的绕行痕迹 | 分层本身消除循环；lazy import 不再是必需品 |
| `config/` 按「系统 × 变体」平铺 | **30 个 yaml** | `configs/` 按维度分目录，运行时组合 |
| `methods/raptor/raptor.py` 是用 langchain **重写**的一版 raptor | 314 行，不是上游 | 只走上游公开接口；**重写就不是在测那个系统了** |

⚠️ 一处更正：MemoryData 的 `results/` **没有**进版本库（`git ls-files` 为 0）。

## 分层

```
core      协议与类型，零依赖
 ├── world      世界：清单 · 物化 · 时钟 · 事实表 · 事件流
 │    └── agent     DSH 宿主：钉死 seam · 把世界挂到 ctx.fs · 驱动会话
 └── adapters   被测系统 = DSH 记忆插件（双面，见下）
      ├── suites    出题：native/ N1–N8 · public/ 调上游
      └── scoring   判分：确定性，⛔ 无评委
           └── report    报告：schema · 序列化 · 渲染
                └── runner    编排五阶段 · 装配插件 · 成本记账
                     └── cli       入口，薄
```

| 层 | 只干一件事 | 允许依赖 |
|---|---|---|
| `core` | 定义协议里的类型与契约 | ⛔ **无** |
| `world` | 造一个评测器拥有、系统只读、可复现的世界 | `core` |
| `adapters` | 把被测系统包装成 `Adapter` 协议 | `core` |
| `suites` | 决定在哪个阶段问什么，产出观测记录 | `core` `world` `agent` |
| `scoring` | 观测 + ground truth → 指标 | `core` |
| `report` | 指标 → 报告结构与渲染 | `core` `scoring` |
| `runner` | 驱动五阶段 | 以上全部 |
| `cli` | 解析参数，交给 runner | `core` `runner` `report` |

### 三条边值得单独解释

**`adapters` 不依赖 `world`，也不依赖 `agent`。** 插件契约在 `core`，
装配由 `runner` 做。⛔ 适配器一旦能 import `agent`，
它就能改宿主配置——而宿主是受控变量，改了分数就不可比。

**`suites` 不依赖 `adapters`。** 出题只面向 `core.Adapter` 协议，
永远不认识任何具体系统。⛔ **一旦某个套件 import 了某家的适配器，
[机制中立](docs/adapters/README.md#p2)就破了**——那意味着题目是照着某个实现写的。

**`scoring` 只依赖 `core`。** 判分吃观测记录，不碰适配器、不碰世界。
出题与判分分开，是为了堵住「改题面顺手改判分」。

## 守卫

`tests/test_architecture.py` 五条断言，各对着上表的一行：

| 断言 | 挡什么 |
|---|---|
| `test_no_upward_or_sideways_imports` | AST 检查跨层 import，挡 longbench↔memoryagentbench 那类反向依赖 |
| `test_top_level_packages_match_spec` | ⭐ **目录结构写死的执行点**——加新顶层包必须先改 `architecture.toml` |
| `test_every_package_declares_its_job` | 每个包必须有 README 说清只干一件事，挡 `utils/` 增生 |
| `test_adapter_stays_thin` | 文件数 / 行数上限 + 拒收上游构建文件，挡 vendoring |
| `test_scoring_is_free_of_judges` | `scoring/` 里出现 LLM 依赖即红，守住[约束①](docs/suites/README.md) |

## agent 层：DSH 是被测对象的宿主

**不装进 agent，我们测的就只是记忆库**——喂文档、检索、答题。那不是 agent memory。

装进去之后 **agent 成为受控变量**：同一个循环、同一套工具、同一个 backbone，
只换记忆插件。⭐ 这才是「统一宿主内比较」。

[DSH](https://github.com/deepseek-ai/deepseek-harness)（★207k · MIT）适合当这个宿主，
因为它是 Cordis 插件树——按其架构文档的原话：

> Every part of the product is a plugin, including the model adapter, the tool registry,
> the session log, and the agent loop itself, so each is replaceable from configuration.
> **There is no privileged core to patch.**

于是「钉死什么」和「替换什么」都能从配置做到：

| | seam | 为什么 |
|---|---|---|
| **钉死** | `ctx.llm` | ⛔ 所有系统同一 backbone——我们本来就要求，DSH 让它可执行 |
| | `ctx.agentLoop` `ctx.tools` `ctx.systemPrompt` | 循环、工具、提示装配不能是变量 |
| **我们提供** | ⭐ `ctx.fs` | 世界经由这个 seam 交付 |
| | `ctx.tokenMeter` `ctx.sessionTelemetry` | [原则⑥](docs/adapters/README.md#p6)——**没有任何公开题库做这件事，DSH 自带** |
| **可复现** | `llm-replay` 包 | 重放，服务确定性 |
| **观测** | `agent/*` 事件 · `session/event` 日志 | 每一步都看得到，**不需要被测系统配合** |

⚠️ DSH 有第一方 Python SDK（`dsh --profile sdk`），跨语言成本可控。

### 适配器是双面的

被测系统接进来的形态是**一个 DSH 记忆插件**，但它同时对评测器暴露我们的协议：

| 面向 | 形态 |
|---|---|
| **DSH** | 贡献 `ctx.systemPrompt`（注入检索到的记忆）· 订阅 `session/event`（摄入）· 可选接管 `ctx.compaction` |
| **评测器** | [协议](docs/adapters/protocol.md)：`Span` `principal` `confidence` `Verdict` … |

⭐ **DSH 当宿主并不约束我们的协议**——DSH 不需要知道 span 是什么。
[机制中立](docs/adapters/README.md#p2)因此不受影响：宿主固定的是**agent**，
不是记忆系统的内部形状。

### 两种运行模式

| 模式 | 记忆系统怎么被用 | 用于 |
|---|---|---|
| **直接调库** | 喂一段语料，问几个问题，比对答案 | 公开题库——LoCoMo 这些是**对话日志，不是 agent 会话** |
| **装进 agent** | agent 真的干活，世界真的变，再看它记住了什么 | 自研八类——世界与主体都是真的 |

⛔ **两种模式分开报，数不可互比。** 同一个系统在两边的分不是一回事：
直接调库喂的是干净的语料，装进 agent 喂的是 agent 自己搅出来的现场。

### ⭐ `ctx.fs` 可替换，世界的归属反而更强

早先我以为「agent 会写文件」和「世界只读挂载 + 哈希校验」冲突，
因此把 DSH 排除在运行时之外。**那是错的**：`ctx.fs` 是个 seam，
世界不必放在 DSH 外面——**我们实现 `ctx.fs`，agent 的每次写都经过我们**。

于是能做到原来做不到的事：**区分「评测器改的」与「agent 改的」**。
[N1](docs/suites/n1-reality.md) 因此更准确，世界变化可归因，
而不是只能整体哈希对比。

### ⛔ 宿主是受控变量

- 宿主**钉死版本**，源码不进本仓库（[原则④](docs/adapters/README.md#p4)）
- ⛔ 换 DSH 版本等于换尺子，**要重跑全部基线**；版本号进[报告](docs/report.md)
- ⚠️ 核实 2026-09-01：DSH 生态里**没有**现成的被测系统插件
  （逐个搜 16 个系统零命中，记忆走 MCP）。**插件要我们自己写**——
  但那本来就是适配器的工作量，不是额外的。

⭐ 净收获：DSH 生态里那些**自研的**记忆实现
（`Autonomous-Long-Term-Memory-System` ★92 · `dsh-memory` ★89 · Engram · Memorix）
本身是被测对象，已进 [`docs/systems.md`](docs/systems.md)。

## 仓库布局

```
ARCHITECTURE.md      本文
architecture.toml    ⭐ 依赖边的唯一权威，守卫测试读它
docs/                设计（先于实现，且是实现的依据）
src/amb/             实现，八层
tests/               含架构守卫
worlds/              世界清单（数据）
corpora/             真实语料（数据）——只用于拟合需求概率曲线与真实性对照
configs/             运行配置，按维度组合
tools/               离线工具，⛔ 不被 src/amb 依赖
out/                 运行产物，gitignore
```

## 改架构的流程

1. 先改 [`docs/`](docs/)——设计是依据，不是事后补的说明
2. 改 `architecture.toml`（依赖边）与本文（为什么）
3. 再动代码

⛔ **反过来做会漂。** 守卫测试挡得住 import 方向和包增生，
挡不住「代码先跑起来、文档以后再说」——那个只能靠这条流程。
