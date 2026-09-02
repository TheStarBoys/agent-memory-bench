"""mem0：第一个真被测系统。

⛔ 它**不在我们的解释器里**跑：被测系统一律隔离在
`.external/venvs/mem0/`，这个适配器通过 [`bridge`](../../bridge.py)
起子进程跟它说话，真正干活的是同目录的 `worker.py`。
（为什么必须隔离，见 [`setup/venv.py`](../../../setup/venv.py) 里那两次实测。）

⛔ 只走公开接口（原则④）：worker 里 `from mem0 import Memory` 是它包顶层的
导出，⚠️ 不 import 任何内部子模块，不复制它的代码。

⭐ 它把「事实抽取 + 增量归并」当主要问题：LLM 抽事实，
与已有比对后 ADD/UPDATE/DELETE。所以：
    ingest 会调 LLM —— ⚠️ 摄入成本远高于纯索引，成本那一栏要看
    search 返回的是**抽出来的事实**，不是原文片段 —— ⛔ 所以 N2 报不支持
"""

from __future__ import annotations

from typing import Any

from amb.adapters.answerable import Answerable
from amb.core import (
    BASELINE,
    AdapterBase,
    AuditEvent,
    Capability,
    DeleteResult,
    Document,
    Entry,
    Span,
    Unsupported,
    Usage,
)


class Mem0Adapter(Answerable, AdapterBase):
    name = "mem0"

    def __init__(self, *, llm_model: str, llm_base_url: str,
                 embed_model: str, embed_base_url: str,
                 embed_dims: int, storage_dir: str,
                 api_key_env: str = "SILICONFLOW_API_KEY",
                 infer: bool = True,
                 default_principal: str = "amb") -> None:
        self._api_key_env = api_key_env
        # ⭐ infer=False 跳过 LLM 抽取，只存原文 + 向量。
        # ⚠️ 实测快 21 倍（1.7s/条 vs 36.7s/条）——
        # 两者并排跑，量的就是「LLM 抽取买到了什么、花了多少」。
        self._infer = infer
        self._cfg = {
            "llm": {"provider": "openai", "config": {
                "model": llm_model, "openai_base_url": llm_base_url,
                # ⛔ mem0 默认 temperature=0.1 —— 判分要可复现，采样温度不该 >0。
                # ⚠️ 这也是缓存能生效的前提：temperature>0 时不缓存，
                # 否则会把随机性冻成一个固定答案。
                "temperature": 0.0,
                "top_p": 1.0}},
            "embedder": {"provider": "openai", "config": {
                "model": embed_model, "embedding_dims": embed_dims,
                "openai_base_url": embed_base_url}},
            "vector_store": {"provider": "qdrant", "config": {
                "path": f"{storage_dir}/qdrant", "on_disk": True,
                "collection_name": "amb", "embedding_model_dims": embed_dims}},
            "history_db_path": f"{storage_dir}/history.db",
        }
        self._storage_dir = storage_dir
        self._default = default_principal
        self._bridge: Any = None
        #: ⭐ infer=False 时留一份原文，N2 的区间靠它算
        self._raw: dict[str, str] = {}

    # ── 能力自述 ────────────────────────────────────────────────
    def capabilities(self) -> set[Capability]:
        """⛔ 不声明 REALITY：它不对外部世界求值。
        ⚠️ 声明 GOVERNANCE 是因为有 user_id 过滤 + history()——
        ⭐ 但那是过滤不是授权，四步探针会把这一点测出来。
        """
        caps = set(BASELINE) | self._answer_caps() | {Capability.GOVERNANCE}
        if not self._infer:
            # ⭐ 不抽取时它存的**就是原文**，所以给得出来源区间——
            # ⚠️ 这是关掉抽取换来的一个真实能力差异。
            caps |= {Capability.PROVENANCE}
        return caps

    # ── 生命周期 ────────────────────────────────────────────────
    def _talk(self) -> Any:
        if self._bridge is None:
            from amb.adapters.bridge import Bridge, worker_script
            from amb.core import require
            from amb.setup import require_venv

            self._bridge = Bridge(
                python=require_venv("mem0"),
                script=worker_script(__package__),
                # ⛔ key 只在内存里传给子进程，不落配置、不进命令行
                config={"config": self._cfg, "storage_dir": self._storage_dir,
                        "api_key": require(self._api_key_env),
                        "infer": self._infer,
                        "default_principal": self._default},
            )
        return self._bridge

    def reset(self) -> None:
        """⛔ 必须真的把盘上的库清空——⚠️ 不只是「桥起着的时候清一下」。

        ⛔ 踩过（这就是「同一条臂两个分数」的成因）：桥是**懒起**的，
        `reset()` 在 setup 阶段调用时 `self._bridge is None`，于是它
        什么都没做——⚠️ 而 `.external/mem0_raw-store/qdrant` 里
        **上一跑摄入的 30 条还在盘上**。这一跑再摄入 30 条，
        库里就是**每条两份共 60 条**。

        ⭐ 实测后果：top-10 里一半是重复条目，去重后只剩 5 篇不同文档，
        evidence_recall **0.789 → 0.471**（离线复算）／**0.474**（真跑）。
        ⛔ 全程无异常、无告警，分数看上去完全正常。
        """
        # ⛔ 先关子进程再删目录：qdrant 还开着的时候删，删不干净且会留锁
        self.close()
        import shutil

        shutil.rmtree(self._storage_dir, ignore_errors=True)
        self._raw.clear()

    def close(self) -> None:
        """⛔ 必须真的释放——不释放的话下一条臂会撞 Qdrant 的存储锁。

        ⭐ 隔离之后这件事简单多了：⚠️ 子进程一退，锁一定没了。
        """
        if self._bridge is None:
            return
        try:
            self._bridge.call("shutdown")
        except Exception:  # noqa: BLE001 —— 关不上就直接杀进程
            pass
        self._bridge.close()
        self._bridge = None

    # ── 基线 ────────────────────────────────────────────────────
    def ingest(self, doc: Document) -> None:
        """⚠️ 这一步会调 LLM 抽事实——摄入成本远高于纯索引。"""
        self._talk().call("ingest", doc_id=doc.doc_id, text=doc.text,
                          principal=doc.principal)
        if not self._infer:
            self._raw[doc.doc_id] = doc.text

    def finalize(self) -> None:
        """⛔ 原文表必须落进 store——⚠️ 它是判 `spans` 的唯一依据。

        ⛔ 踩过（a_mem 那次的同一类）：记账放在进程内存里，
        命中快照时恢复的只有向量库，原文表是空的 → `spans` 全没了 →
        ⚠️ 行为指纹对不上 → 快照被判不可信 → **每一跑都重新摄入**。
        ⭐ 放进 store，快照才真的能用。
        """
        self._dump_raw()

    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]:
        got = self._talk().call("search", query=query, k=k, principal=principal)
        return [
            Entry(
                id=row["id"],
                digest=row["memory"],
                score=row.get("score"),
                # ⭐ 靠我们塞进 metadata 的 doc_id 对账；
                # ⚠️ 它归并之后一条记忆可能来自多个文档，这里只拿得到一个
                doc_ids=[row["doc_id"]] if row.get("doc_id") else [],
                # ⭐ 不抽取时存的就是原文，区间对得上；
                # ⛔ 抽取模式给不出——所以那时候不声明 PROVENANCE
                spans=self._span_for(row.get("doc_id"), row["memory"]),
                principal=row.get("principal"),
            )
            for row in got["entries"]
        ]

    def count(self) -> int:
        return int(self._talk().call("count")["count"])

    # ── N4 ──────────────────────────────────────────────────────
    def delete(self, entry_ids: list[str]) -> DeleteResult:
        got = self._talk().call("delete", entry_ids=entry_ids)
        return DeleteResult(deleted=got["deleted"], refused=got["refused"])

    def audit_log(self) -> list[AuditEvent] | Unsupported:
        """⚠️ 逐条拼出来的——⛔ 删掉的条目查不到了，
        所以这份日志**删除之后不完整**。四步探针会把这一点暴露出来。
        """
        got = self._talk().call("audit_log")
        return [
            AuditEvent(
                event_id=e["event_id"], action=e["action"],
                entry_ids=e["entry_ids"], principal=e.get("principal"),
                at=e.get("at", ""),
                detail=None,      # ⛔ 不放正文——藏进审计日志不算删除
            )
            for e in got["events"]
        ]

    def usage(self) -> list[Usage] | Unsupported:
        """⭐ 成本实测。⚠️ 数字来自**我们的**包装层，⛔ 不是它自报的——
        它没有报 token 的接口，但每一次 openai 调用都经过我们手里。

        ⛔ 分不出 ingest / probe：我们的计量器是按进程累计的，
        ⚠️ 硬拆会造出一个假的分配。全部记在 ingest——
        这一档（`--no-answer`）本来就只有摄入调 LLM。
        """
        if self._bridge is None:
            return Unsupported("没跑过，无从计量")
        got = self._talk().call("meter")
        if not got:
            return Unsupported("worker 拿不到计量器")
        return [Usage(phase="ingest", tokens_in=got["tokens_in"],
                      tokens_out=got["tokens_out"], llm_calls=got["llm_calls"])]

    def storage_locations(self) -> list[str]:
        """⭐ 申报持久层，让带外取证那一步能做。"""
        return [self._storage_dir]

    #: 原文表落盘的位置。⛔ 必须在 store 里——快照拷的就是这个目录。
    _RAW_FILE = "amb-raw.json"

    def _raw_path(self):
        from pathlib import Path

        return Path(self._storage_dir) / self._RAW_FILE

    def _dump_raw(self) -> None:
        if self._infer or not self._raw:
            return
        import json

        path = self._raw_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._raw, ensure_ascii=False),
                        encoding="utf-8")

    def _raw_table(self) -> dict[str, str]:
        """内存里没有就从 store 读——⭐ 那正是「命中快照」的情形。"""
        if self._raw or self._infer:
            return self._raw
        import json

        path = self._raw_path()
        if not path.is_file():
            return self._raw
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # ⚠️ 读不出来就当没有——⛔ 不猜区间
            return self._raw
        if isinstance(got, dict):
            self._raw = {str(k): str(v) for k, v in got.items()}
        return self._raw

    def _span_for(self, doc_id: str | None, memory: str) -> list:
        """⭐ 只在 infer=False 时给得出区间——存的就是原文。"""
        if self._infer or not doc_id:
            return []
        original = self._raw_table().get(doc_id)
        if not original:
            return []
        start = original.find(memory)
        if start < 0:
            # ⛔ 对不上就不给——⚠️ 猜一个区间比不给更糟
            return []
        return [Span(doc_id=doc_id, start=start, end=start + len(memory))]
