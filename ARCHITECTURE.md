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
                          ┌──────────┐
                          │   core   │  零依赖 · 协议与类型
                          └────┬─────┘  Adapter / Entry / Verdict / 三态
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
      ┌────────────┐   ┌──────────────┐   ┌────────────┐
      │   world    │   │   adapters   │   │  scoring   │
      │ 清单·物化   │   │ 注册表·对照组 │   │ 确定性判分  │
      │ 哈希·变更   │   │ impl/ 薄壳    │   │ ⛔ 无评委   │
      │ 只读端点    │   └──────────────┘   └─────┬──────┘
      └─────┬──────┘    ⛔ 不依赖 world/agent      ▼
            ▼                                ┌──────────┐
      ┌───────────┐                          │  report  │
      │   agent   │  DSH 宿主                 │ 地板线·Δ  │
      └─────┬─────┘                          └────┬─────┘
            ▼                                     │
      ┌───────────┐  出题，⛔ 不判分                │
      │  suites   │  ⛔ 不认识任何具体系统           │
      └─────┬─────┘                               │
            └──────────────┬─────────────────────┘
                           ▼
                    ┌─────────────┐
                    │   runner    │  编排五阶段 · 哈希守卫 · 记账
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │     cli     │  ⛔ 薄
                    └─────────────┘
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

## 一次跑的数据流

```
worlds/*.py ──清单+种子──▶ world.materialize
                             │  ⛔ mtime 钉死在时钟起点
                             ▼
                       ┌───────────┐   只读 HTTP
                       │   世界     │◀─ /clock     GET
                       │ 文件树     │   /facts/:k  GET
                       │ 时钟       │   ⛔ 其余方法一律 405
                       │ 事实表     │
                       └─────┬─────┘
                             │ WorldHandle
  ┌──────────────────────────┼───────────────────────────┐
  │  runner 五阶段            ▼                           │
  │                                                       │
  │  setup ─── reset() → setup(world) ────────── ✓ 哈希    │
  │  ingest ── ingest(doc)×N → finalize() ────── ✓ 哈希    │
  │  mutate ── ⚠️ 只有评测器动手，不通知适配器 ── 重设基线    │
  │  probe ─── search / audit / answer / … ───── ✓ 哈希    │
  │  score ─── 确定性判分                                  │
  │                                                       │
  │  ⛔ 任一边界哈希不一致 → WorldTampered，本次跑作废        │
  └───────────────────────────┬───────────────────────────┘
                              ▼
       suites ──Observation──▶ scoring ──Score──▶ report
                                                    │
                                    地板线 = 对照组里**最强**的那条
                                    Δ = 被测 − 地板 ⛔ 只对被测系统算
```

⛔ **`mutate` 不通知适配器是刻意的**：被告知"世界变了"再去查测的是执行，
没被告知还能发现测的才是 [N1](docs/suites/n1-reality.md)。

## 三样东西怎么进来

### 题库

| | 谁读 | 判分归谁 |
|---|---|---|
| **公开题库** | `suites/public/` 调上游包，钉死 commit | ⛔ **上游的判分代码**，我们不重写 |
| **自研八类** | `suites/native/` 从 `worlds/` 的清单生成 | `scoring/`，确定性，⛔ 无评委 |

公开题库是**现成的语料+题目+答案**，我们只负责喂进去、把回答收上来、交给上游判分。
自研八类没有现成语料——[世界](docs/adapters/world.md)由清单和种子生成，
真值随之而来（哪条命题失效了、需求概率多少、规律的例外是哪个）。

⭐ 差别在**真值从哪来**：公开题库的真值是别人标的，自研八类的真值是世界生成时就知道的。

### 被测的记忆系统

```
            ┌──────────── adapters/impl/ ────────────┐
            │                                        │
 五条对照组   │  null · host_default · bm25            │  被测系统
 （参照系）   │  naive_rag · full_context              │  （慢慢接）
            │                                        │
            └────────────────┬───────────────────────┘
                             │ 同一个 core.Adapter 协议
                             │ ⛔ 同一条代码路径
                             ▼
       对宿主那一面                    对评测器那一面
       ctx.systemPrompt 注入记忆        Span / principal
       session/event 摄入               confidence / Verdict
```

一个目录 `adapters/impl/<系统>/`，实现两面：

```
对宿主   贡献 ctx.systemPrompt（把检索到的记忆注入）
         订阅 session/event（摄入）
对评测器 core.Adapter 协议（Span / principal / confidence / Verdict）
```

⛔ 上游代码不进本仓库，只走 HTTP / CLI / 包顶层导出的 SDK。
薄度由 `architecture.toml` 强制：≤10 文件 / ≤1000 行。

### 对照组

⭐ **[五条对照组](docs/baselines.md)也是适配器**，住在同一个 `impl/` 里、走同一条路径。

| | 是什么 |
|---|---|
| `null` | 只给当前轮，什么都不留 |
| `host_default` | ⭐ 不挂记忆插件，只用 DSH 自带的上下文管理与压缩——**真实地板** |
| `naive_rag` | chunk + embedding + top-k |
| `bm25` | 纯词频，连 embedding 都不要 |
| `full_context` | 全部语料塞进窗口——天花板参照 |

⛔ **绝对分不单独报**，每个数跟着地板线与 Δ。
一个跑不赢「让宿主自己压缩上下文」的记忆系统，是在给 agent 添麻烦。

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

## 端到端已经通了

```sh
python -m amb.cli                    # 五条对照组，跑完出报告
python -m amb.cli --arms bm25        # 只跑一条
python -m amb.cli --json out/r.json  # 附带机读结果
```

⚠️ 目前跑的是 `worlds/toy.py`——**它不是够格的题库**，题量太小，
统计上说明不了任何事。它存在的唯一目的是让五阶段先通起来，
好尽早发现协议哪里不好用。

### 通到哪了

| | 状态 |
|---|---|
| `core` `world` `adapters` `suites` `scoring` `report` `runner` `cli` | ✅ 已通 |
| 五阶段端到端 · 哈希守卫 · 五条对照组 · 地板线与 Δ | ✅ 已通 |
| N1 有提示 · N2 · 检索档 | ✅ 已通 |
| ⚠️ **`agent`（DSH 宿主）** | **未实现——最大的空缺** |
| ⚠️ `suites/public`（公开题库接入） | 未实现 |
| N1 无提示 · `answer()` 端到端 · N4 删除四步探针 | 🔜 机制缺口 |

⛔ **`agent/` 一天不实现，「装进 agent 跑」就只是设计**——
现在跑的全是[「直接调库」那一档](docs/adapters/README.md)。

⭐ 第一次跑就抓到三个设计问题，都已修：

| 跑出来才发现的 | 修法 |
|---|---|
| `scoring` import 了 `suites` —— 判分反过来依赖出题 | `Observation`/`SuiteRun` 移进 `core`，它们是两边的共享词汇 |
| `cli` import 了 `adapters` —— 入口认识了具体系统 | 构造挪进 `runner/build.py`，cli 只传名字 |
| `null` 在 N2 上得 **0 分**而不是不支持 | N2 要求 `PROVENANCE`；⭐ 而 bm25/naive_rag 切块边界就是真区间，如实声明 |

第三条正是这个项目要防的那个错，**被自己的流水线抓了个正着**。

## 改架构的流程

1. 先改 [`docs/`](docs/)——设计是依据，不是事后补的说明
2. 改 `architecture.toml`（依赖边）与本文（为什么）
3. 再动代码

⛔ **反过来做会漂。** 守卫测试挡得住 import 方向和包增生，
挡不住「代码先跑起来、文档以后再说」——那个只能靠这条流程。
