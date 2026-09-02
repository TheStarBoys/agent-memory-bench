"""公开题库的接入形状。

⛔ **纪律：用它们的判分代码，不自己重写。**
自己重写一份只会引入「我们的判分与别人不同」这个不可比性。

⛔ 上游一律**钉死 commit**，不 fork、不复制进本仓库（原则④）。
上游判分有已知缺陷时**照旧调用**，把缺陷写进报告的 `upstream_notes`
（见 docs/harnesses.md）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from amb.core import Document


class DatasetMissing(RuntimeError):
    """题库数据还没取下来。⛔ 该档记「未接入」，不是 0 分。"""


class UpstreamScorerMissing(RuntimeError):
    """上游判分代码不可用。

    ⛔ 这时候**不许自己写一个顶上**——那正是纪律要防的事。
    该题库记「未接入」。
    """


@dataclass(frozen=True, slots=True)
class Pin:
    """一个上游依赖，⛔ 钉死到 commit。"""

    repo: str
    commit: str
    #: ⚠️ 已知缺陷，逐条进报告的 upstream_notes
    caveats: tuple[str, ...] = ()

    def provenance(self) -> dict[str, object]:
        return {"repo": self.repo, "commit": self.commit,
                "caveats": list(self.caveats)}


@dataclass
class PublicSuite:
    """一个公开题库的接入。

    ⚠️ 三件事分开：**取数据** · **喂进去** · **判分**。
    ⛔ 第三件永远交给上游。
    """

    name: str
    pin: Pin
    #: 数据放在哪（corpora/ 或 datasets/），⛔ 不进版本库
    data_dir: Path
    documents: list[Document] = field(default_factory=list)

    def available(self) -> bool:
        return self.data_dir.is_dir() and any(self.data_dir.iterdir())

    def require_available(self) -> None:
        if not self.available():
            raise DatasetMissing(
                f"{self.name} 的数据不在 {self.data_dir}。"
                f"⛔ 该档记「未接入」，不是 0 分。"
                f"取数据：见 docs/benchmarks.md#怎么获取"
            )


class UpstreamScorer(Protocol):
    """上游的判分。⛔ 我们只调用，不实现。"""

    def score(self, predictions: dict[str, str]) -> dict[str, float]: ...


#: 已登记的上游，⛔ 每一个都钉死 commit。
#: ⚠️ caveats 里的东西**照旧调用**，只是要写进报告。
REGISTRY: dict[str, Pin] = {
    "locomo": Pin(
        repo="https://github.com/snap-research/locomo",
        commit="",          # ⛔ 接入时填
        caveats=("22% 是弃权题，只会返回 top-k 的系统在这一类上必然全错",),
    ),
    "memoryagentbench": Pin(
        repo="https://github.com/HUST-AI-HYZ/MemoryAgentBench",
        commit="",
        caveats=("FactConsolidation 的题面直接给出了消解规则，"
                 "考的是「能不能捞到序号最大的那条」，不是「能不能发现两条在打架」",),
    ),
    "memorydata": Pin(
        repo="https://github.com/OpenDataBox/MemoryData",
        commit="",
        caveats=("⛔ 无 LICENSE，只能当外部依赖调用，代码不得复制进本仓库",
                 "utils/eval_other_utils.py:1068 的 judge 字段不是 LLM 评委，"
                 "比 f1 还严——公开表里的 J 分在这套框架里复现不出来",
                 "⚑ 利益关系：memorybench 由 supermemory 维护，"
                 "而 supermemory 是被测系统之一"),
    ),
    "beam": Pin(
        repo="https://github.com/mohammadtavakoli78/BEAM",
        commit="",
        caveats=("判分是 nugget + LLM 评委，⚠️ 带评委漂移，"
                 "不可与自研套件的确定性分数并列比较",
                 "⚠️ 只报准确率，不报成本与延迟",),
    ),
}


def pin_for(name: str) -> Pin:
    """⛔ 精确查找。"""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"未登记的公开题库 {name!r}。已登记：{sorted(REGISTRY)}") from None
