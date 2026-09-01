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
 ├── world      世界：清单 · 物化 · 哈希 · 时钟 · 事实表 · 事件流 · 负载后端
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

## DeepSeek Harness 放在哪

[DSH](https://github.com/deepseek-ai/deepseek-harness)（★207k，MIT，TypeScript）
是 **`world/workload/` 下的一个可选负载后端**，用来产生真实的
工具调用 / 文件编辑 / 会话轨迹——这正是 N1 的世界与 N4 的 principal 想要的东西。

⛔ **它不是 adapters 层，也不做 submodule。** 三条理由：

1. ⚠️ **核实于 2026-09-01：DSH 生态里没有通往被测系统的插件。**
   14k★ 的 `awesome-dsh-plugin` 精选列表里记忆类只有 2 条，
   逐个搜我们的 16 个系统**零命中**；DSH 也没有 `packages/memory`，
   它的记忆是走 MCP 接的（`apps/cli/config/examples/mcp-memory/`：engram · memorix · mcp-reference-memory）。
   **所以拿它当接入层，22 个适配器一个都省不掉。**
2. 我们需要的是 DSH **跑起来**，不是它的源码。钉版本 + 进程外调用
   比 submodule 更符合[原则④](docs/adapters/README.md#p4)，
   也不用把一个 pnpm monorepo 拖进仓库。
3. ⛔ **适配器协议不得跑在 DSH 的插件模型里。**
   那等于把机制中立的裁判权交给上游，DSH 的形状会变成我们的形状——
   而那正是我们不 fork 现成 harness 的理由。

⭐ 顺带一个净收获：DSH 生态里那些**自研的**记忆插件
（`Autonomous-Long-Term-Memory-System` ★92 Apache-2.0 · `dsh-memory` · engram · memorix）
本身就是 agent memory 系统，属于**被测对象**，见 [`docs/systems.md`](docs/systems.md)。

## 仓库布局

```
ARCHITECTURE.md      本文
architecture.toml    ⭐ 依赖边的唯一权威，守卫测试读它
docs/                设计（先于实现，且是实现的依据）
src/amb/             实现，八层
tests/               含架构守卫
worlds/              世界清单（数据）
configs/             运行配置，按维度组合
out/                 运行产物，gitignore
```

## 改架构的流程

1. 先改 [`docs/`](docs/)——设计是依据，不是事后补的说明
2. 改 `architecture.toml`（依赖边）与本文（为什么）
3. 再动代码

⛔ **反过来做会漂。** 守卫测试挡得住 import 方向和包增生，
挡不住「代码先跑起来、文档以后再说」——那个只能靠这条流程。
