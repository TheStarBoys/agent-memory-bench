# 被测系统

> 核实 **2026-09-01**，stars 与推送日期取自 GitHub API 当日快照。数字会变，本文允许整节重写。
>
> **日期约定**：不带年份的（`08-31`）一律指 **2026 年**；跨年的写全（`2025-12-12`）。
> `—` 表示**本次未核实**，不表示"没有"。
>
> 「解决什么」写的是**它把哪件事当主要问题**，不是功能罗列。接入状态见
> [`adapters/README.md`](adapters/README.md)。

## 通用记忆层

| 系统 | ★ | 许可 | 最后推送 | 解决什么 |
|---|---:|---|---|---|
| [mem0](https://github.com/mem0ai/mem0) ⭐已接入 | 64.5k | Apache-2.0 | 08-31 | **事实抽取 + 增量归并**。LLM 抽事实，与已有比对后 ADD/UPDATE/DELETE。生态最大。⚠️ 实测摄入 **36.7 秒/条**；⭐ `infer=False` 关掉抽取后 **1.7 秒/条**，两种配置并排跑 |
| [graphrag](https://github.com/microsoft/graphrag) | 35.8k | MIT | 08-31 | **图化索引**。抽实体关系建图，社区检测出层级摘要。严格说是 RAG，常被当对照 |
| [graphiti (Zep)](https://github.com/getzep/graphiti) | 30.5k | Apache-2.0 | 09-01 | **时序知识图谱**。四时间戳双时态；裁决两阶段——LLM 出候选、**代码用时间区间确定性校验** |
| [cognee](https://github.com/topoteretes/cognee) | 30.4k | Apache-2.0 | 09-01 | **ECL 管道**（Extract-Cognify-Load）。把记忆当数据工程问题，图+向量混合 |
| [supermemory](https://github.com/supermemoryai/supermemory) | 29.2k | MIT | 09-01 | **面向应用的托管记忆**。产品化程度高，自带 memorybench |
| [TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) | 25.5k | NOASSERTION | 08-31 | **四类资产**：Chat Memory（原文→原子事实→场景块→画像）· Skill · Wiki · CodeGraph。团队级四档可见性 |
| [letta](https://github.com/letta-ai/letta) | 24.5k | Apache-2.0 | 08-23 | **上下文分层**（前身 MemGPT）：Main / Archival / Recall + sleep-time compute。`.af` 是业内**唯一真实的导出格式** |
| [Memori](https://github.com/MemoriLabs/Memori) | 16.3k | NOASSERTION | 08-21 | **SQL 原生记忆**。不用向量库，赌关系数据库够用 |
| [MemOS](https://github.com/MemTensor/MemOS) | 11.1k | Apache-2.0 | 09-01 | **把记忆当 OS 资源调度**：明文/激活/参数三种形态可互转 |
| [ReMe](https://github.com/agentscope-ai/ReMe) | 3.4k | Apache-2.0 | 09-01 | **文件即真源**。四态裁决 `CREATE/CORROBORATE/REFINE/CORRECT`，提示词明写 *no SKIP*；冲突只加内联标注，**从不删旧内容** |
| [MIRIX](https://github.com/Mirix-AI/MIRIX) | 3.4k | Apache-2.0 | 08-20 | **多智能体记忆**，含屏幕活动等多模态来源 |
| [langmem](https://github.com/langchain-ai/langmem) | 1.6k | MIT | 08-11 | **按更新语义分类**：语义→替换（Profiles 幂等折叠）· 情节→追加（Collections 只增不减）· 程序→人工审核 |
| [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | 1.6k | Apache-2.0 | 07-07 | **分页调度**，短期/中期/长期三级，仿 OS 换页 |
| [A-mem](https://github.com/agiresearch/A-mem) | 1.2k | MIT | 2025-12-12 | **Zettelkasten 式自组织**，记忆间自动生成链接并演化 |
| [MemoryScope](https://github.com/modelscope/MemoryScope) ⚠️未核实 | — | Apache-2.0 | — | 阿里系，长期记忆 + 反思巩固。**stars 与推送日期本次未取到，接入前须补核** |

## 检索 / 压缩类（对照基线）

| 系统 | ★ | 许可 | 最后推送 | 解决什么 |
|---|---:|---|---|---|
| [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | 4.0k | MIT | 08-23 | **海马体索引理论**，个性化 PageRank 做多跳。MAB 冲突消解**全场第一（54%）** |
| [self-rag](https://github.com/AkariAsai/self-rag) | 2.4k | MIT | 2024-05-25 | 自反思检索，模型自判要不要检索、检得好不好 |
| [MemoRAG](https://github.com/qhjqhj00/MemoRAG) | 2.3k | Apache-2.0 | 2025-09-11 | 全局记忆模型先出线索，再去精确检索 |
| [raptor](https://github.com/parthsarthi03/raptor) | 1.8k | MIT | 2024-09-03 | **递归摘要树**，多粒度检索 |
| [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent) | 1.1k | Apache-2.0 | 05-12 | RL 训练出来的记忆读写策略 |
| [MemoryLLM](https://github.com/wangyu-ustc/MemoryLLM) | 322 | MIT | 2025-07-28 | **参数化记忆**，把记忆塞进权重而非外部库 |
| [MemoChat](https://github.com/LuJunru/MemoChat) | 30 | MIT | 2024-04-18 | 早期对话记忆分片检索 |

## ⭐ 接入成本调研（2026-09-02）

⚠️ 挑「便宜的先接」时逐个查过。**结论是个负面发现，但很清楚**：

| 系统 | 摄入要 LLM 吗 | 接入障碍 |
|---|---|---|
| **mem0** | ⛔ 要（36.7 秒/条）| ⭐ 已接。⚠️ 有 `infer=False` 开关 → 1.7 秒/条 |
| A-mem | ⛔ 要 | LLM 生成笔记与链接 |
| MemoryOS | ⛔ 要 | LLM 摘要 |
| cognee · langmem | ⛔ 要 | 管道里就有 LLM |
| **Memori** | ⛔ 要 | ⛔ 还要**云端 API key**——外部服务不可复现 |
| graphiti / Zep | ⛔ 要 | 还要 Neo4j |
| txtai | ✅ 不要 | ⚠️ 242 个依赖，而且它是**框架不是记忆系统** |

⭐ **这不是巧合——「LLM 抽取」就是这类系统的定义特征。**
所以「便宜的记忆系统」基本等于「RAG 变体」，
而那正是[对照组](baselines.md)已经覆盖的。

⚠️ 对评测的两个含义：

1. ⛔ **成本不是可选维度。** 一整类系统的定义特征就是「贵」，
   不把成本纳入判定，等于假装那个代价不存在（[原则⑥](adapters/README.md#p6)）。
2. ⭐ **同一系统的开关比换系统更干净。** `mem0` vs `mem0_raw` 差别**只在**
   那一个开关上——backbone、embedding、存储全一样。
   换第三方系统时，差异里会混进实现质量、依赖版本、适配器写法一堆东西。

## Agent 宿主生态（新增，核实 2026-09-01）

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（★207k · MIT）
生态里的记忆实现。它们与上面两类的差别是**长在 agent 运行时里**，
看得到工具调用与会话上下文，而不是被当作库调用。

| 系统 | ★ | 许可 | 最后推送 | 解决什么 |
|---|---:|---|---|---|
| [Autonomous-Long-Term-Memory-System](https://github.com/cuiyuestar/Autonomous-Long-Term-Memory-System) | 92 | Apache-2.0 | 08-31 | DSH 的自主长期记忆系统 |
| [dsh-memory](https://github.com/FuRongJun-1999/dsh-memory) | 89 | MIT | 08-31 | **白箱 AGI 架构探索**：元认知 · 知识飞轮 · 语义时空图 · **零 LLM 白箱管线** |
| Engram | — | — | — | ⚠️ **未核实**。DSH 以 MCP 方式接入（`engram mcp`，存储与项目选择由它自己拥有） |
| Memorix | — | — | — | ⚠️ **未核实**。同上，走 MCP |

⚠️ **DSH 生态里没有通往上面两类系统的插件**——逐个搜过 16 个系统，零命中；
DSH 也没有 `packages/memory`，记忆一律走 MCP。
所以它对本项目的价值是**两条**：这里的新被测对象，以及
[作为离线的真实语料来源](../ARCHITECTURE.md)——录下轨迹用来拟合需求概率曲线。
⛔ **它既不是接入层，也不是世界的一部分**：一个会动手的 agent 不可能是被观察的世界。

## ⚠️ 已停更（超过 8 个月无推送）

判据：最后推送早于 **2026-01-01**（核实基准日 2026-09-01 往前 8 个月）。

| 系统 | 最后推送 | 距今 |
|---|---|---:|
| `MemoChat` | 2024-04-18 | ~28 个月 |
| `self-rag` | 2024-05-25 | ~27 个月 |
| `raptor` | 2024-09-03 | ~24 个月 |
| **`MemoryLLM`** | 2025-07-28 | ~13 个月 |
| `MemoRAG` | 2025-09-11 | ~12 个月 |
| `A-mem` | 2025-12-12 | ~9 个月 |

按**冻结的参照实现**对待：可以接、可以跑，别期待上游修 bug。接入时钉死版本号。

⚠️ `MemoryLLM` 是照判据补上的——它满足 8 个月阈值却不在原先的清单里。
清单改成表格就是为了让判据和结果并排，下次核实时对不上一眼看得见。
