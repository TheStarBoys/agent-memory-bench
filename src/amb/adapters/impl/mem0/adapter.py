"""mem0：第一个真被测系统。

⛔ 只走公开接口（原则④）：`from mem0 import Memory` 是它包顶层的导出，
⚠️ 不 import 任何内部子模块，不复制它的代码。

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
    Unsupported,
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
                "model": llm_model, "openai_base_url": llm_base_url}},
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
        self._memory: Any = None
        self._ids: list[str] = []
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
    def _client(self) -> Any:
        if self._memory is None:
            import os

            from amb.core import require

            from mem0 import Memory        # ⛔ 只用包顶层导出

            # ⚠️ mem0 内部用 openai SDK，它只认 OPENAI_API_KEY。
            # ⛔ 不让使用者手动设——适配器自己搭这座桥，
            # 配置里存的仍然只是**变量名**。
            os.environ.setdefault("OPENAI_API_KEY", require(self._api_key_env))
            # ⭐ 给 mem0 内部的 openai 客户端套一层内容寻址缓存。
            # ⛔ 只在 AMB_LLM_CACHE=1 时生效，⚠️ 且缓存命中的跑不是独立的延迟测量。
            _install_openai_cache()
            self._memory = Memory.from_config(self._cfg)
        return self._memory

    def reset(self) -> None:
        if self._memory is not None:
            self._memory.reset()
        self._ids = []

    def close(self) -> None:
        if self._memory is not None and hasattr(self._memory, "close"):
            self._memory.close()

    # ── 基线 ────────────────────────────────────────────────────
    def ingest(self, doc: Document) -> None:
        """⚠️ 这一步会调 LLM 抽事实——摄入成本远高于纯索引。"""
        got = self._client().add(
            doc.text,
            user_id=doc.principal or self._default,
            metadata={"doc_id": doc.doc_id},
            infer=self._infer,
        )
        if not self._infer:
            self._raw[doc.doc_id] = doc.text
        for row in (got or {}).get("results", []):
            if row.get("id"):
                self._ids.append(row["id"])

    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]:
        got = self._client().search(
            query, top_k=k,
            filters={"user_id": principal or self._default},
        )
        out: list[Entry] = []
        for row in (got or {}).get("results", []):
            meta = row.get("metadata") or {}
            doc_id = meta.get("doc_id")
            out.append(Entry(
                id=str(row.get("id", "")),
                digest=str(row.get("memory", "")),
                score=row.get("score"),
                # ⭐ 靠我们塞进 metadata 的 doc_id 对账；
                # ⚠️ 它归并之后一条记忆可能来自多个文档，这里只拿得到一个
                doc_ids=[doc_id] if doc_id else [],
                # ⭐ 不抽取时存的就是原文，区间对得上；
                # ⛔ 抽取模式给不出——所以那时候不声明 PROVENANCE
                spans=self._span_for(doc_id, row.get("memory", "")),
                principal=(row.get("user_id") or principal or self._default),
            ))
        return out

    def count(self) -> int:
        got = self._client().get_all(filters={"user_id": self._default}, top_k=1000)
        return len((got or {}).get("results", []))

    # ── N4 ──────────────────────────────────────────────────────
    def delete(self, entry_ids: list[str]) -> DeleteResult:
        deleted, refused = [], {}
        for eid in entry_ids:
            try:
                self._client().delete(eid)
                deleted.append(eid)
            except Exception as exc:  # noqa: BLE001 —— 逐条记原因，⛔ 不整批失败
                refused[eid] = f"{type(exc).__name__}: {exc}"[:200]
        return DeleteResult(deleted=deleted, refused=refused)

    def audit_log(self) -> list[AuditEvent] | Unsupported:
        """mem0 的 history() 是按条目查的，没有全局日志。

        ⚠️ 我们逐条拼出来——⛔ 但删掉的条目查不到了，
        所以这份日志**删除之后不完整**。四步探针会把这一点暴露出来。
        """
        rows: list[AuditEvent] = []
        for eid in self._ids:
            try:
                history = self._client().history(eid) or []
            except Exception:  # noqa: BLE001 —— 删掉的查不到，跳过
                continue
            for i, h in enumerate(history):
                rows.append(AuditEvent(
                    event_id=f"{eid}:{i}",
                    action=str(h.get("event", "update")).lower()[:6] or "update",
                    entry_ids=[eid],
                    principal=h.get("actor_id"),
                    at=str(h.get("created_at") or ""),
                    detail=None,      # ⛔ 不放正文——藏进审计日志不算删除
                ))
        return rows

    def storage_locations(self) -> list[str]:
        """⭐ 申报持久层，让带外取证那一步能做。"""
        return [self._storage_dir]

    def _span_for(self, doc_id: str | None, memory: str) -> list:
        """⭐ 只在 infer=False 时给得出区间——存的就是原文。"""
        from amb.core import Span

        if self._infer or not doc_id:
            return []
        original = self._raw.get(doc_id)
        if not original:
            return []
        start = original.find(memory)
        if start < 0:
            # ⛔ 对不上就不给——⚠️ 猜一个区间比不给更糟
            return []
        return [Span(doc_id=doc_id, start=start, end=start + len(memory))]


def _install_openai_cache() -> None:
    """给 openai 的 chat.completions.create 套缓存。

    ⚠️ 打补丁是唯一的办法——mem0 不暴露注入点。
    ⛔ 只包一层，不改它的行为：同样的请求返回同样的响应。
    """
    import time

    from amb.adapters.llm_cache import global_cache

    cache = global_cache()
    if not cache.enabled:
        return

    from openai.resources.chat import completions as _mod

    if getattr(_mod.Completions.create, "_amb_cached", False):
        return
    original = _mod.Completions.create

    def cached(self, **kwargs):  # noqa: ANN001
        payload = {k: v for k, v in kwargs.items() if k != "extra_headers"}
        try:
            hit = cache.get(_jsonable(payload))
        except Exception:  # noqa: BLE001 —— ⛔ 缓存出问题就退回真调用
            hit = None
        if hit is not None:
            from openai.types.chat import ChatCompletion

            return ChatCompletion.model_validate(hit)
        t0 = time.perf_counter()
        got = original(self, **kwargs)
        try:
            cache.put(_jsonable(payload), got.model_dump(),
                      int((time.perf_counter() - t0) * 1000))
        except Exception:  # noqa: BLE001
            pass
        return got

    cached._amb_cached = True          # noqa: SLF001
    _mod.Completions.create = cached


def _jsonable(payload: dict) -> dict:
    import json

    return json.loads(json.dumps(payload, default=str, sort_keys=True))
