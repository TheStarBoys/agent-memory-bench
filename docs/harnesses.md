# 现成的评测框架

> 为什么本项目存在、以及为什么**不 fork** 任何一个。

## 现有的

| 框架 | ★ | 许可 | 最后推送 | 覆盖 |
|---|---:|---|---|---|
| [MemoryData](https://github.com/OpenDataBox/MemoryData) | 143 | ❌ **无** | 2026-07-05 | 22 个方法 × 4 个基准族（MemoryAgentBench · LoCoMo · LongBench · MemBench） |
| [memorybench](https://github.com/supermemoryai/memorybench) | 312 | MIT | 2026-08-24 | 对话记忆 + RAG 多数据集 |
| [awesome-agent-memory](https://github.com/OpenDataBox/awesome-agent-memory) | 56 | ❌无 | 2026-07-22 | 清单，不是 harness |

## MemoryData：可以用，不能 fork

它是目前覆盖面最广的一个，我们**把它当外部依赖调用**跑公开基准。但不作为地基，三条理由：

**① 无 LICENSE。** 默认保留全部权利——fork 进本仓库等于分发未授权代码。
一个自身法律地位不清的评测框架，没法要求别人采信。

**② 不是维护中的项目。** `git log` 只有 **1 个 squash 提交**，2026-07-05 之后没动过。
fork 一个不会更新的上游 = 第一天就接管全部维护。

**③ 接口形状不对。** 它的适配器只有五个方法：

```python
add_chunk(content, timestamp)   finalize()
ask(question) -> str            retrieve_entries(question)   memory_count()
```

**这是「喂对话、问问题」的形状，假设记忆系统是个纯函数。**
而 [自研套件](suites/README.md) 的四类题需要一个**有状态的世界**——
外部现实会变，问的是"你发现没有"。四类题没有一类塞得进这个接口。

**这不是加数据集能补的，是接口形状不同。**

### 另外两个已知质量问题

- `utils/eval_other_utils.py:1068` —— `"judge": bool(metrics["exact_match"])`。
  这个字段叫 judge 但**不是 LLM 评委**，比 f1 还严。**公开表里的 J 分在这套 harness 里复现不出来。**
- `benchmark/longbench/loader.py` 反过来 import `benchmark.memoryagentbench.hf_datasets`。
  MemoryAgentBench 不是并列的基准之一，是别人都依赖的地基——加平级的新基准要先绕开这层耦合。

## 本项目的定位

```
suites/public/   调 MemoryData 等跑公开基准 —— 用别人的判分代码，避免自证
suites/native/   自研四类题 —— 有状态世界，别人没有的接口
```

**两条路分开，理由不是洁癖。** 公开基准要的是「和别人同一把尺子」，那就该用别人的尺子；
自研题要的是「考别人没考的」，那必须是新接口，借不到。
