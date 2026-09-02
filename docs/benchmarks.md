# 公开题库

> 核实 **2026-09-01**。这里只写**别人的题**；我们自己出的题在
> [`suites/README.md`](suites/README.md)。
>
> **日期约定**：不带年份的（`08-31`）一律指 **2026 年**；跨年的写全（`2025-01-15`）。
>
> 纪律：公开题库覆盖到的能力，**一律调用它们、用它们的判分代码**，不自己重写一份。
> 重写只会引入「我们的判分与别人不同」这个不可比性。
> 上游判分本身有缺陷时怎么办，见
> [`harnesses.md`](harnesses.md#上游判分坏了怎么办)——**照旧调，但在报告里写明**。

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
| [memorybench](https://github.com/supermemoryai/memorybench) ⚑ | 312 | MIT | 08-24 | 多数据集 | 对话记忆 + RAG 混合。**⚑ 由 supermemory 维护，而 supermemory 是被测系统**——见下 |
| [HaluMem](https://github.com/MemTensor/HaluMem) | 159 | NOASSERTION | 08-28 | ~15k 记忆点 / 3.5k 题 | **记忆幻觉**：抽取 / 更新 / 问答三处各自的幻觉率。六种题型，见下。arXiv 2511.03506 |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | 135 | MIT | 08-31 | 100 对话 / 2000 题 / **128K–10M token** | **十项能力**，见下。ICLR 2026，arXiv 2510.27246 |
| [LifelongAgentBench](https://github.com/caixd-220529/LifelongAgentBench) | 98 | ❌无 | 2025-05-30 | — | 终身学习：跨任务技能累积 |
| [stream-bench](https://github.com/stream-bench/stream-bench) | 85 | Apache-2.0 | 2024-10-28 | — | 流式持续学习，从反馈中改进 |
| [MemBench](https://github.com/import-myself/Membench) | 59 | ❌无 | 2025-11-27 | — | 简单 / 高级 / **知识更新** / **噪声** / 多会话推荐 |
| [MemGUI-Bench](https://github.com/lgy0404/MemGUI-Bench) | 49 | MIT | 08-29 | — | GUI agent 的记忆 |
| [MemSim](https://github.com/nuster1128/MemSim) | 18 | ❌无 | 2024-10-10 | — | 合成用户画像 + 自动出题 |
| [MADial-Bench](https://github.com/hejunqing/MADial-Bench) | 4 | MIT | 2025-04-30 | — | 记忆增强对话，情感与共情维度 |

## 三个重点题库的细分

### LoCoMo —— 1986 题　⭐ 已接入

```sh
python -m amb.cli setup locomo
python -m amb.cli --bench locomo --sample stratified:50 --arms bm25
```

抽题四种：`all` · `first:N` · `random:N` · **`stratified:N`** · `ids:a,b`。
⭐ **分层是推荐的**：五类占比悬殊，简单随机抽 50 题，
占 5% 的开放域推断类很可能一道都没抽到。
⛔ 抽样方式与种子进报告——不记的话两次跑的差可能全来自抽到了不同的题。

**⭐ bm25 首跑（分层 50 题，evidence 判检索，不生成答案）：**

| 类 | recall | |
|---|---:|---|
| 单跳事实 | **0.783** | BM25 的主场 |
| 时间推理 | 0.750 | |
| 开放域推断 | 0.667 | ⚠️ n=2，不可当真 |
| 弃权 | 0.545 | |
| **多跳** | **0.136** | ⛔ 塌了——词频接不上多跳 |
| *总分* | *0.522* | ⚠️ **把多跳那件事完全糊掉了** |

⚠️ **实测发现的上游瑕疵**：446 条弃权题里有 **2 条带 `answer`**
（`conv-26#167` 与 `#178`，答案都是 "No"）。
⛔ 我们不改上游数据，把它钉死在测试里——数量变了就说明上游动过，
那时候分数也不再可比。



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

#### 回答档：⛔ 检索到证据 ≠ 答得对

```sh
# 检索档：不挂 backbone，只问「证据捞到没有」
python -m amb.cli --bench locomo --convs conv-30 --max-turns 30 --no-answer
# 回答档：挂 backbone，问「捞到之后答对没有」
python -m amb.cli --bench locomo --convs conv-30 --max-turns 30
```

⛔ **两档的数不可互比**，报分时回答档必须写成「<系统> + <backbone>」——
这一档含答案生成器，只报系统名等于把别人的生成器算进自己的成绩。

| 这一档报什么 | |
|---|---|
| `准确率` | ⭐ 主指标。⛔ 逐字比对，[不用评委](adapters/README.md#p5) |
| `宽松准确率` | ⚠️ **判分上界，不是分数**——它与准确率的差就是这把尺的不确定度 |
| `该答却弃权` | 单列。⛔ 不算错——⚠️ 拒答与答错是两件事 |
| `正确弃权率` / `编造率` | ⭐ 必须与准确率同屏，否则见题就编的系统会更好看 |
| 逐类 | ⛔ 22% 是弃权题，总分会把那一类糊掉 |

⚠️ **LoCoMo 官方判分是 LLM 评委，我们不用**（[原则⑤](adapters/README.md#p5)）。
⛔ 所以这一档的数**不可与已发表的 LoCoMo 分数并列**——尺子不同。

##### ⛔ 两个已知的、会压低这一档绝对分的东西

**① 逐字比对判不了自由文本的 gold。** LoCoMo 的 gold 中位 20 字符，
但有 35% 长过 25 字符。实测：

| gold | 系统答的 | 严格 | 宽松 |
|---|---|---|---|
| `September, 2023` | `September` | ✗ | ✓ |
| `Winning first place at a regionals dance competition` | `… her team won first place at a regionals at age fifteen` | ✗ | ✓ |
| `They are performing at the festival` | `Festival performers.` | ✗ | ✗ |

⭐ 所以看这一档要**同时看两个数**，⛔ 排名一律看严格那个：
宁可漏判成错，不靠判分的宽松度刷分。

**② ⛔ 时间推理那一类是构造性不可答的。**
[`documents_for()`](../src/amb/suites/public/locomo.py) 把每一轮变成
`"<说话人>: <正文>"`，⚠️ **会话日期没有进摄入单元**。
于是问 `When did Jon go to a fair?`（gold `24 April, 2023`），
所有臂拿到的资料里根本没有日期，只能答 `Yesterday`——
⛔ **这一类所有臂都是 0.000，那是语料的性质，不是系统的**。

⚠️ 要修就得把日期拼进摄入单元，⛔ 代价是**语料指纹变了**：
全部摄入快照作废，已有存档的检索分不可再与新跑并列
（`mem0` 那条重摄一次 73 分钟 / $0.22）。⭐ 没做，留着这条记录。

### MemoryAgentBench —— 四能力

精确检索（LongMemEval / EventQA / RULER-QA）· **冲突消解**（FactConsolidation）·
测试时学习（ICL）· 长程理解。

⚠️ **FactConsolidation 的题面直接给出了消解规则**——原文：
*"newer facts have larger serial numbers. Resolve conflicts by using the newest relevant fact only."*
所以它考的是「能不能捞到序号最大的那条」，**不是「能不能发现两条在打架」**。
真实冲突不带序号。

### BEAM —— 十项能力

抽取 · 多跳 · 知识更新 · 时间 · **弃权** · **矛盾消解** · 事件排序 · 指令遵循 ·
偏好遵循 · 摘要。其中后三项（指令遵循 · 事件排序 · 矛盾消解）是它新提的，
其余七项取自更早的基准。

⚠️ **BEAM 只报准确率，不报成本与延迟**（核对论文全文 arXiv 2510.27246 与项目页，
核实 2026-09-01）。论文里出现的 "cost" 全部是作者自己生成数据集时的开销
（用 LLaMA-3.3 70B 出题以省钱），不是被测系统的成本指标。

⛔ **没有任何公开题库把成本当一等指标。**
[原则⑥](adapters/README.md#p6)因此是本项目独有的要求，没有现成的可照抄。

⚠️ **BEAM 的判分是 LLM 评委**：把参考答案拆成原子 nugget，
再由 LLM judge 逐条判系统回答是否覆盖。这正是
[⑤ 确定性判分](adapters/README.md#p5)警告的那类判分——
接入时照用它的判分代码（纪律如此），但**它的分数带评委漂移，
不可与自研套件的确定性分数并列比较**。

⚠️ **弃权是二元的**（"withholds answers when evidence is missing"），
不含置信度分级，也没有 ECE / Brier 之类的校准指标。

⚠️ 10M token 档的摄入成本对任何走 LLM 抽取的系统都不可接受，分档接入。

⚠️ 论文的配套方法 **LIGHT** 是认知驱动的（情节记忆 + 工作记忆 + scratchpad）——
见 [`cognition.md`](cognition.md)。**方法有认知学依据，而它用来验证的题库测不了那些依据**，
这个错位本身是个信号。

### HaluMem —— 六种题型

按论文附录 A.2 的原文定义：

| 题型 | 考什么 |
|---|---|
| Basic Fact Recall | 对话里明确出现过的单个事实或偏好 |
| Multi-hop Inference | 需要综合多个片段，靠逻辑或时间推理才能得出 |
| Dynamic Update | 追踪信息随时间的变化，识别最新状态 |
| **Memory Boundary** | **问输入里没提过的细节，看系统会不会编** |
| Generalization & Application | 基于已知偏好，在新场景下给出合理建议 |
| Memory Conflict | 题面故意含与已知记忆矛盾的错误前提，看能否纠正 |

⚠️ **`Memory Boundary` 比预想的更接近「虚构」这一维**——
它已经在测「没说过的东西你会不会编」。这直接影响自研套件的取舍，
见 [`cognition.md`](cognition.md#gaps)。

⚠️ HaluMem **不测置信度校准**，判分是二元对错。

## ⚑ 利益关系

**`memorybench` 由 [supermemory](systems.md) 维护，而 supermemory 是被测系统之一。**

这不是排除它的理由——它确实是一个覆盖面不错的题库，纪律说该用就用。
但「用别人的判分代码」这条纪律在这里会让**一个被测方出的题**进入公开档，
而这正是[原则②](adapters/README.md#p2)
要防的主场优势。

做法：**照用，但标注。** supermemory 在 memorybench 上的成绩在报告里标
`⚑ 利益关系`，读者自己判断。同一条规则适用于「适配器由被测方自己提交」
和「被测系统是本项目作者的」——细则见 [`report.md`](report.md#利益关系必须标注)。

标注不是指控。一个不标注利益关系的评测框架，和一个伸手进自己人内部的评测框架，
失去公信力的方式是同一种。

## ❌ 无 LICENSE 的

`MemBench` · `LifelongAgentBench` · `MemSim`

**可以读、可以当外部依赖跑，代码不得复制进本仓库。**
