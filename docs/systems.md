# 被测系统

> 核实 **2026-09-01**，stars 与推送日期取自 GitHub API 当日快照。数字会变，本文允许整节重写。
>
> 「解决什么」写的是**它把哪件事当主要问题**，不是功能罗列。接入状态见
> [`../adapters/README.md`](../adapters/README.md)。

## 通用记忆层

| 系统 | ★ | 许可 | 最后推送 | 解决什么 |
|---|---:|---|---|---|
| [mem0](https://github.com/mem0ai/mem0) | 64.5k | Apache-2.0 | 08-31 | **事实抽取 + 增量归并**。LLM 抽事实，与已有比对后 ADD/UPDATE/DELETE。生态最大 |
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
| [MemoryScope](https://github.com/modelscope/MemoryScope) | — | Apache-2.0 | — | 阿里系，长期记忆 + 反思巩固 |

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

## ⚠️ 已停更（超过 8 个月无推送）

`A-mem` · `raptor` · `self-rag` · `MemoRAG` · `MemoChat`

按**冻结的参照实现**对待：可以接、可以跑，别期待上游修 bug。接入时钉死版本号。
