# 选 backbone：一张可以直接查的表

> ⭐ backbone 是**受控变量**——所有臂共用同一个，否则分数不可比。
> ⛔ 换它意味着已发布的分数**全部要重跑**。所以：要换就趁早，且要有依据。

| | |
|---|---|
| ⭐ **默认** | `Qwen/Qwen3-8B` + **关思考** |
| ⚠️ **当前阶段临时用** | `inclusionAI/Ling-flash-2.0`——先用最快的把能力跑通 |

<a id="why-default"></a>

## 为什么默认是 Qwen3-8B 而不是最快的那个

不是因为它最快，是因为它**跟 MemoryData 的对照配置同款**
（Qwen3-8B + Qwen3-Embedding-4B）。两边的地板线能互相印证——
⭐ 对「公正可比」而言，这比再快一倍更值钱。

⚠️ 当前阶段还在「把八类题在两个档上都跑通」，速度优先于可比性，
所以临时手动指定 `Ling-flash-2.0`。⛔ **一旦开始出可发布的分数就换回默认**，
并且报告抬头会写着用的是哪个。

<a id="thinking"></a>

## ⭐ 第一件事：关思考。这是 15 倍，不是 15%

同一条抽取调用，同一个模型，只切思考开关：

| | 墙钟 | 输出 token |
|---|---:|---:|
| Qwen3-8B 思考开 | 24.2s | 539（外加 1846 字符思考） |
| **Qwen3-8B 思考关** | **4.3s** | **74** |

端到端到 A-mem（摄入 3 篇一句话文档，经隔离 venv + 子进程桥）：

| | 总耗时 | 最贵那次调用的输出 |
|---|---:|---:|
| 思考开 | 663.3s | 5341 token |
| **思考关** | **42.7s** | **191 token** |

⭐ **15.5 倍，而抽取质量没降**——⚠️ 反而少造了一条对不回原文的记忆
（A-mem 的 `count` 从 4 回到 3）。

⛔ 所以「记忆系统贵得没法用」这个印象，有一大半是**我们让 backbone
边想边抽**造成的，不是记忆机制本身的成本。

默认已是 `thinking = false`。要开：`AMB_LLM_THINKING=1`，
⚠️ 报告抬头会写「⚠️ 思考开」。

<a id="table"></a>

## 实测表（2026-09-02，siliconflow，temperature=0，真实 `json_schema` 调用）

| 模型 | 秒 | 输出 token | JSON | 备注 |
|---|---:|---:|:--:|---|
| **inclusionAI/Ling-flash-2.0** | **3.3** | 124 | ✓ | ⭐ 最快 |
| **Qwen/Qwen3-30B-A3B-Instruct-2507** | **7.1** | 124 | ✓ | ⭐ 关键词最全，唯一抓到「韩梅梅」 |
| Qwen/Qwen3-8B（关思考） | 8.1 | 109 | ✓ | ⭐ 默认；与 MemoryData 对照同款 |
| deepseek-ai/DeepSeek-V4-Flash | 11.9 | 258 | ✓ | |
| Qwen/Qwen3.5-9B | 23.3 | 2642 | ✓ | ⚠️ 关不掉思考 |
| Qwen/Qwen2.5-7B-Instruct | 111.1 | 4096 | ⛔ | 退化成 `on on on` 死循环 |
| zai-org/GLM-4.5-Air | — | — | ⛔ | 不支持 json mode |

<a id="single-call-vs-end-to-end"></a>

### ⚠️ 单次调用快，不等于端到端快

上表是**单次调用**。同样 3 篇文档端到端跑 A-mem：

| | D1 | D2 | D3 | 总 |
|---|---:|---:|---:|---:|
| Qwen3-8B（关思考） | 20.6 | 8.9 | 12.0 | 42.7s |
| Ling-flash-2.0 | 23.1 | 8.9 | 6.3 | 39.5s |

单次 3.3s vs 8.1s 的差距**没有出现在端到端上**——D1 里有一大截是本地
embedder 的加载时间，跟 backbone 无关。

⛔ **n=3，这个差距在噪声里，不足以声称谁更快**
（同[抽样纪律](sampling.md)：区间重叠不许排名）。
⚠️ 选 Ling-flash 是「单次调用最快且 JSON 可用」，
⛔ 不是「实测端到端更快」——那要更大的语料才看得出来。

自己跑一遍（各家上下架很快，⚠️ 上表是当天的数）：

```sh
python tools/compare_backbones.py                     # 内置候选
python tools/compare_backbones.py 模型A 模型B          # 指定
```

<a id="real-shape"></a>

## ⛔ 不能用裸提示量——那会给出**反的**结论

`Qwen2.5-7B-Instruct` 在裸提示上 **2.0 秒**，看着是全场最快。
加上抽取型记忆系统普遍要的 `response_format={"type":"json_schema"}` 之后，
它退化成 4096 token 的 `on on on` 死循环，**111 秒**。

⭐ 所以 `tools/compare_backbones.py` 量的是 A-mem `analyze_content` 的
**真实调用形状**：带 schema、要 JSON、要抽关键词。

⚠️ 但这张表量的是**延迟与可用性**，⛔ 不是抽取质量——
JSON 合法不等于抽得对。质量得靠跑题库。

<a id="controlled"></a>

## ⛔ 我们在被测系统的调用里钉死了什么

被测系统自己发的 LLM 调用，有两项由我们统一覆盖
（[`backbone_overrides()`](../src/amb/adapters/llm_cache.py)）：

| 钉什么 | 为什么 |
|---|---|
| `temperature = 0.0` | 判分要可复现。⚠️ mem0 默认 0.1、A-mem 默认 1.0，两个都踩过 |
| `enable_thinking = False` | ⭐ 上面那 15 倍 |

⛔ 这是 backbone 的受控变量，不是被测系统的设置——所有臂必须一致。
⚠️ 每一项都在各系统的适配器 README 里登记，并进报告。

⚠️ 不认 `enable_thinking` 的服务端会忽略它，⛔ 不是错误。

<a id="switch"></a>

## 怎么换

```sh
# 临时：只影响这一次跑
AMB_LLM_MODEL=inclusionAI/Ling-flash-2.0 python -m amb.cli run …

# 本机长期：写进 .env（已 gitignore）
```

配置文件在 [`configs/backbone/`](../configs/backbone/)，
⛔ 里面存的是**环境变量名**，不是 key 本身。

⚠️ 换完之后：**已有的分数作废**。报告抬头会印出用的是哪个模型、
思考开没开——⭐ 读者据此判断两份结果能不能并排看。

## ⚠️ 还没量的一项：钱

上表只有**时间**。单价（尤其 MoE 模型）没查，⛔ 不编。
[原则⑥](adapters/README.md#p6)要求成本是一等维度，
⚠️ 时间和钱都算——钱这一栏要补。
