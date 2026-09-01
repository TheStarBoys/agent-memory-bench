# 公开题库

> 核实 **2026-09-01**。这里只写**别人的题**；我们自己出的题在
> [`suites/README.md`](suites/README.md)。
>
> 纪律：公开题库覆盖到的能力，**一律调用它们、用它们的判分代码**，不自己重写一份。
> 重写只会引入「我们的判分与别人不同」这个不可比性。

## 怎么获取

| 途径 | 适用 | 说明 |
|---|---|---|
| 仓库直接给 JSON | LoCoMo | clone 就有，最省事 |
| HuggingFace Datasets | MemoryAgentBench · LongMemEval · MemBench | `load_dataset(...)`，首次拉取要网络与磁盘 |
| 需自行合成 | RULER · InfiniteBench | 按目标上下文长度现场生成 |

## 题库表

| 题库 | ★ | 许可 | 最后推送 | 规模 | 题型 |
|---|---:|---|---|---|---|
| [RULER](https://github.com/NVIDIA/RULER) | 1.6k | Apache-2.0 | 07-22 | 可合成任意长度 | 大海捞针 · 变量追踪 · 高频词提取 · 多跳聚合 |
| [LongBench](https://github.com/THUDM/LongBench) | 1.2k | MIT | 2025-01-15 | 双语多任务 | 单/多文档 QA · 摘要 · few-shot · 代码补全 |
| [LoCoMo](https://github.com/snap-research/locomo) | 1.1k | NOASSERTION | 2024-08-13 | 10 组对话 / **1986 题** | 见下方细分 |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 1.0k | MIT | 05-11 | 500 题 | 抽取 · 多会话推理 · 时间 · **知识更新** · **弃权**。V2 面向 agent 轨迹 |
| [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) | 444 | MIT | 08-20 | 2071 题 / 103K–1.44M token | 四能力，见下 |
| [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) | 391 | MIT | 2024-09-25 | 平均 100K+ token | 12 个超长上下文任务 |
| [memorybench](https://github.com/supermemoryai/memorybench) | 312 | MIT | 08-24 | 多数据集 | 对话记忆 + RAG 混合 |
| [HaluMem](https://github.com/MemTensor/HaluMem) | 159 | NOASSERTION | 08-28 | — | **记忆幻觉**：抽取 / 更新 / 问答三处各自的幻觉率 |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | 135 | MIT | 08-31 | 100 对话 / 2000 题 / **128K–10M token** | **十项能力**，见下。**唯一同时报 token 消耗与延迟** |
| [LifelongAgentBench](https://github.com/caixd-220529/LifelongAgentBench) | 98 | ❌无 | 2025-05-30 | — | 终身学习：跨任务技能累积 |
| [stream-bench](https://github.com/stream-bench/stream-bench) | 85 | Apache-2.0 | 2024-10-28 | — | 流式持续学习，从反馈中改进 |
| [MemBench](https://github.com/import-myself/Membench) | 59 | ❌无 | 2025-11-27 | — | 简单 / 高级 / **知识更新** / **噪声** / 多会话推荐 |
| [MemGUI-Bench](https://github.com/lgy0404/MemGUI-Bench) | 49 | MIT | 08-29 | — | GUI agent 的记忆 |
| [MemSim](https://github.com/nuster1128/MemSim) | 18 | ❌无 | 2024-10-10 | — | 合成用户画像 + 自动出题 |
| [MADial-Bench](https://github.com/hejunqing/MADial-Bench) | 4 | MIT | 2025-04-30 | — | 记忆增强对话，情感与共情维度 |

## 三个重点题库的细分

### LoCoMo —— 1986 题

**每题带 `evidence`（ground-truth 轮次 ID），所以可以不生成答案就判分**——
这是它对记忆层最友好的地方，也是唯一能纯粹量检索质量的公开题库。

| 类 | 题数 | 占比 | 考什么 |
|---|---:|---:|---|
| 4 | 841 | 42% | 单跳事实 |
| **5** | **446** | **22%** | **不可答 / 弃权**——无 `answer` 字段，只有 `adversarial_answer` |
| 2 | 321 | 16% | 时间推理 |
| 1 | 282 | 14% | 多跳 |
| 3 | 96 | 5% | 开放域 / 推断 |

⚠️ **22% 是弃权题，比多跳还多。** 只会返回 top-k 的系统在这一类上必然全错——
**拒答不是加分项，是四分之一的卷面。**

### MemoryAgentBench —— 四能力

精确检索（LongMemEval / EventQA / RULER-QA）· **冲突消解**（FactConsolidation）·
测试时学习（ICL）· 长程理解。

⚠️ **FactConsolidation 的题面直接给出了消解规则**——原文：
*"newer facts have larger serial numbers. Resolve conflicts by using the newest relevant fact only."*
所以它考的是「能不能捞到序号最大的那条」，**不是「能不能发现两条在打架」**。
真实冲突不带序号。

### BEAM —— 十项能力

抽取 · 多跳 · 知识更新 · 时间 · **弃权** · **矛盾消解** · 事件排序 · 指令遵循 ·
偏好遵循 · 摘要。**同时报 token 消耗与延迟**，是唯一把成本当一等指标的。

⚠️ 10M token 档的摄入成本对任何走 LLM 抽取的系统都不可接受，分档接入。

## ❌ 无 LICENSE 的

`MemBench` · `LifelongAgentBench` · `MemSim`

**可以读、可以当外部依赖跑，代码不得复制进本仓库。**
