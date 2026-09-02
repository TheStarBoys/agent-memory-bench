# `setup` —— 外部依赖的一键安装

**只干一件事**：按[清单](spec.py)把外部依赖装好，⭐ 并记录**实际装到的**版本。

```sh
python -m amb.cli setup            # 全部
python -m amb.cli setup mem0       # 一个
python -m amb.cli setup --check    # 只看状态
```

## ⛔ 三条规矩，都有测试盯着

| | |
|---|---|
| **钉死版本** | 清单里不许有未钉死的依赖——⚠️ 换版本等于换了被测对象 |
| **记录实际版本** | ⭐ 声明的与装到的不一定一样（git 声明分支名，实际记 commit sha） |
| **源码不进仓库** | pip 进 site-packages，git 进 `.external/`（已 gitignore），[原则④](../../../docs/adapters/README.md#p4) |

## ⛔ 没装就拒绝，不给 0 分

`require_installed()` 抛异常而不是静默降级——
一个缺依赖的跑不该悄悄产出一个分数。

⚠️ 版本快照进[结果报告](../../../docs/report.md)：**没记录版本的跑不算数**。
