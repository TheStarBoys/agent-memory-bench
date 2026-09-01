# 架构

> **目录结构是写死的。** 依赖边的唯一权威是 [`architecture.toml`](architecture.toml)，
> 由 [`tests/test_architecture.py`](tests/test_architecture.py) 强制执行。
> **这份文档解释为什么，那个文件说了算。**

## 从 MemoryData 学到的

[MemoryData](docs/harnesses.md) 是覆盖面最广的现成 harness，我们**调用它但不 fork**。
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
 ├── world      世界：清单 · 物化 · 哈希 · 时钟 · 事实表 · 事件流
 └── adapters   接入：注册表 · 能力自述 · impl/ 每系统一个薄包
      ├── suites    出题：native/ N1–N8 · public/ 调上游
      └── scoring   判分：确定性，⛔ 无评委
           └── report    报告：schema · 序列化 · 渲染
                └── runner    编排五阶段 · 世界哈希校验 · 成本记账
                     └── cli       入口，薄
```

| 层 | 只干一件事 | 允许依赖 |
|---|---|---|
| `core` | 定义协议里的类型与契约 | ⛔ **无** |
| `world` | 造一个 harness 拥有、系统只读、可复现的世界 | `core` |
| `adapters` | 把被测系统包装成 `Adapter` 协议 | `core` |
| `suites` | 决定在哪个阶段问什么，产出观测记录 | `core` `world` |
| `scoring` | 观测 + ground truth → 指标 | `core` |
| `report` | 指标 → 报告结构与渲染 | `core` `scoring` |
| `runner` | 驱动五阶段 | 以上全部 |
| `cli` | 解析参数，交给 runner | `core` `runner` `report` |

### 三条边值得单独解释

**`adapters` 不依赖 `world`。** 适配器**收到**的是 `core` 里的 `WorldHandle`
（路径 + 两个只读端点），它不需要知道世界怎么造出来的。
这条边一旦连上，被测系统就离「能写世界」只差一步。

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

## ⛔ 为什么没有「agent 宿主」这一层

一度想把 [DSH](https://github.com/deepseek-ai/deepseek-harness) 放进 `world/` 当负载后端。
**那是错的**，因为它把三样东西混成了一样：

| | 是什么 | 谁拥有 |
|---|---|---|
| **世界** | 外部现实：文件树 · 时钟 · 事实表 | harness 拥有，被测系统**只读** |
| **语料** | 被 `ingest()` 进去的文档 | 由世界的生成过程派生 |
| **agent 宿主** | 一个**会动手**的东西 | —— |

**一个在世界里动手的东西，不可能是世界的一部分。**
DSH 是 agent，它写文件、跑工具；而世界是
[只读挂载 + 每个阶段边界校验哈希](docs/adapters/world.md#归属与只读)的。
把它放进去，要么被只读挂载挡住，要么把哈希校验搞崩、判本次跑作废。

而且录下来的轨迹**没有 ground truth**——N1 要知道哪条命题已失效，
N5 要需求概率，N8 要种下的规律与例外，真实轨迹一样都没有。
所以它服务不了那四类。

⭐ **它真正的位置文档里早就写了**：拟合经验需求概率曲线需要真实语料
（[world.md](docs/adapters/world.md#need-probability)：真实的 agent 会话日志 / 工单流 / 提交历史）。
DSH 是产生这种语料的一个办法。

**结论：DSH 不是模块，是数据来源。**
离线跑（[`tools/`](tools/README.md)），产出到 [`corpora/`](corpora/README.md)，
⛔ 不进运行时依赖图。

⚠️ 核实于 2026-09-01：DSH 生态里**没有**通往被测系统的插件——
14k★ 精选列表记忆类仅 2 条，逐个搜我们的 16 个系统零命中，
DSH 也没有 `packages/memory`（记忆走 MCP）。**它省不掉任何一个适配器。**

⭐ 净收获：DSH 生态里那些**自研的**记忆实现
（`Autonomous-Long-Term-Memory-System` ★92 · `dsh-memory` ★89 · Engram · Memorix）
本身是 agent memory 系统，属于**被测对象**，已进 [`docs/systems.md`](docs/systems.md)。

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
