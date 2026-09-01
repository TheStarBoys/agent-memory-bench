# 文档索引

| 文档 | 回答什么 |
|---|---|
| [`cognition.md`](cognition.md) | **凭什么考这些**：从人的记忆反推评测维度，以及四个尚未覆盖的缺口 |
| [`systems.md`](systems.md) | **被测对象**：22 个记忆系统，各自把哪件事当主要问题，stars 与活跃度 |
| [`benchmarks.md`](benchmarks.md) | **别人的题**：15 个公开题库，怎么获取，什么题型 |
| [`harnesses.md`](harnesses.md) | **现成框架**：有哪些，为什么一个都不 fork，上游判分坏了怎么办 |
| [`suites/README.md`](suites/README.md) | **我们的题**：四类，以及为什么只有四类 |
| [`adapters/README.md`](adapters/README.md) | **接入协议**：为什么要新接口，六条原则 |
| [`report.md`](report.md) | **报告格式**：三态怎么落到表上，利益关系怎么标 |

**实现的架构不在 docs/**，在仓库根的 [`ARCHITECTURE.md`](../ARCHITECTURE.md)——
模块划分、依赖法则，以及它们各自对着 MemoryData 上的哪个实测失效模式。
依赖边由 `tests/test_architecture.py` 强制执行，不是文档。

## 自研套件

| | 考什么 |
|---|---|
| [`suites/n1-reality.md`](suites/n1-reality.md) | 对现实求值——世界变了，你发现没有 |
| [`suites/n2-provenance.md`](suites/n2-provenance.md) | 原文回链——你说这条来自 X，X 真是它的来源吗 |
| [`suites/n3-reasoning.md`](suites/n3-reasoning.md) | 推理链——结论对之外，每一步成立吗 |
| [`suites/n4-governance.md`](suites/n4-governance.md) | 治理——谁写的、谁能读、删了还留不留痕 |
| [`suites/n5-consolidation.md`](suites/n5-consolidation.md) | 巩固与遗忘——该留的留了，**该丢的丢了吗** |
| [`suites/n6-structure.md`](suites/n6-structure.md) | 关联结构——可达性与扇形干扰，两条曲线一起报 |
| [`suites/n7-calibration.md`](suites/n7-calibration.md) | 置信度校准——它知不知道自己什么时候会错 |
| [`suites/n8-induction.md`](suites/n8-induction.md) | 归纳与可废止推理——例外不推翻规律 |

## 一条贯穿的纪律

**公开题库覆盖到的能力，一律调用它们、用它们的判分代码。**
自己重写一份只会引入「我们的判分与别人不同」这个不可比性。

自研题只补它们**结构上测不到**的那四类。

## 接入协议

| | |
|---|---|
| [`adapters/README.md`](adapters/README.md) | 为什么现成接口不够用；**六条原则**，其中②机制中立是防主场优势的唯一手段 |
| [`adapters/protocol.md`](adapters/protocol.md) | 接口定义：三态返回、能力自述、id 契约、五个阶段 |
| [`adapters/world.md`](adapters/world.md) | 有状态的世界——N1 的地基 |

## 两条推导

题目从哪来，有两条独立的推导路线。**它们对上的地方最可信。**

```
防守性推导   benchmarks.md → suites/   「公开题库漏了什么」
进攻性推导   cognition.md  → suites/   「记忆本该是什么样，所以该考什么」
```

N1–N4 来自防守性推导，N5–N8 来自进攻性推导。
⭐ 两条推导**独立地指向了 N1–N4 中的全部四个**——这是对现有设计最强的支持。

⚠️ 进攻性推导还指出过第五个缺口「图式虚构」，
核实 HaluMem 题面后发现**已被 `Memory Boundary` 覆盖，因此没有立类**。
[「能用别人的尺子就用别人的」](benchmarks.md)在那里省掉了一整个套件。

## 四份东西一起看才成立

```
cognition.md   凭什么考这些          —— 决定该测的是不是那件事
adapters/      系统怎么接进来        —— 决定什么测得了
suites/        我们具体问什么        —— 决定题面是否真的问到了
report.md      结果怎么呈现          —— 决定前三者的小心是否白费
```

⚠️ **`report.md` 最容易被当成收尾工作，其实不是。**
协议里把「不支持 / 失败 / 做错」分了三态，报告里压成一列就全白做了。
