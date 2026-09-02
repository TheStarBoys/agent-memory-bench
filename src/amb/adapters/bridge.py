"""跟**独立 venv 里**的被测系统说话。

⛔ 被测系统不在我们的解释器里（见 `amb.setup.venv` 里那两次实测踩坑），
所以适配器不能 `import` 它，只能起一个子进程、用 stdout 上的 JSON 行对话。

    我们 ──{"op":"ingest","doc_id":…,"text":…}──▶ worker（跑在它自己的 venv）
        ◀─────{"ok":true,"result":…}────────────

⚠️ worker 脚本跑在**别人的解释器**里，⛔ 只准 import 标准库 + 被测系统本身。
不能 import `amb` —— 那个包在那个 venv 里不存在。

⭐ 协议只有四个约定，别的都别加：
    ① 一行一个 JSON，⛔ 不许多行——多行没法在流上切开
    ② worker 的日志一律走 **stderr**，stdout 只放协议
    ③ 出错回 `{"ok":false,"error":…}`，⛔ 不靠退出码——进程还要接着用
    ④ 崩了就是崩了，⚠️ 抛 `BridgeError` 让这条臂记「跑挂了」，不是 0 分
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


class BridgeError(RuntimeError):
    """子进程说不上话。⛔ 这条臂记「跑挂了」，不是 0 分。"""


class Bridge:
    """一个 worker 子进程。⚠️ 不是线程安全的——一条臂一个。"""

    def __init__(self, python: Path, script: Path, config: dict[str, Any], *,
                 env: dict[str, str] | None = None,
                 timeout_s: float = 1800.0) -> None:
        self._python = python
        self._script = script
        self._config = config
        self._env = env or {}
        self._timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._stderr: list[str] = []
        #: ⛔ 一旦判定死了就**永久**死了。⚠️ 不能只靠 poll()——
        #: 进程刚退出时 poll() 可能还返回 None（还没被回收），
        #: 那一瞬 _start() 会把死进程当活的交出去，
        #: 「绝不静默重启」这条保证就漏了。实测在负载下会偶发。
        self._dead: str | None = None

    def _start(self) -> subprocess.Popen:
        if self._dead is not None:
            raise BridgeError(self._dead)
        if self._proc is not None:
            if self._proc.poll() is None:
                return self._proc
            # ⛔ 死了就是死了，**绝不静默重启**。
            # ⚠️ 重启会丢掉它摄入的全部状态，而调用方毫无察觉——
            # 那样跑出来的分数是「半个语料的记忆系统」的分数，比崩掉更糟。
            raise BridgeError(self._mark_dead("子进程中途死了"))
        # ⭐ `llm_cache` 只 import 标准库 + openai，所以 worker 能按路径直接加载它。
        # ⚠️ 这样「缓存」和「temperature 钉 0」只有**一份**实现，
        # ⛔ 不在隔离环境里复制一遍——复制就会漂移。
        env = {**os.environ, **self._env, "PYTHONUNBUFFERED": "1",
               "AMB_CACHE_MODULE_DIR": str(Path(__file__).parent)}
        self._proc = subprocess.Popen(
            [str(self._python), str(self._script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env, bufsize=1)
        # ⭐ stderr 单独抽干：⛔ 不读会把管道灌满，子进程直接卡死（踩过）
        self._stderr.clear()
        threading.Thread(target=self._drain, daemon=True).start()
        self.call("init", **self._config)
        return self._proc

    def _drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                # ⚠️ 只留最后 200 行——崩了要看得见，但不能撑爆内存
                self._stderr.append(line)
                del self._stderr[:-200]

    def call(self, op: str, **payload: Any) -> Any:
        proc = self._proc if op == "init" else self._start()
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise BridgeError(f"{self._script.name}: 子进程起不来")
        try:
            proc.stdin.write(json.dumps({"op": op, **payload},
                                        ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise BridgeError(self._mark_dead(f"写不进去：{exc}")) from None

        line = proc.stdout.readline()
        if not line:
            raise BridgeError(self._mark_dead("子进程没回话就退了"))
        try:
            got = json.loads(line)
        except json.JSONDecodeError:
            # ⚠️ 常见原因：被测系统往 stdout 打了日志，把协议冲了
            raise BridgeError(
                self._died(f"回的不是 JSON（多半是往 stdout 打日志了）："
                           f"{line[:200]!r}")) from None
        if not got.get("ok"):
            # ⛔ 带上子进程的 stderr——worker 已经把 traceback 打在那儿了。
            # ⚠️ 只报一行 error 会让「它为什么失败」查不出来（踩过：
            # 一个 sqlite 错误码，看不到是哪一行、什么路径触发的）。
            tail = "\n    ".join(self._stderr[-20:])
            raise BridgeError(
                f"{self._script.name} {op}: {got.get('error')}"
                + (f"\n  子进程 stderr 末尾：\n    {tail}" if tail else ""))
        return got.get("result")

    def _mark_dead(self, why: str) -> str:
        """记下死因并**封死**这座桥。

        ⚠️ 之后每次调用都抛同一条消息，而它必须同时说清两件事：
        ⭐ **原始死因**（否则查不出为什么挂）和
        ⛔ **不会重启**（否则调用方会以为再试一次就好）。
        """
        if self._dead is None:
            self._dead = (self._died(why)
                          + "\n  ⛔ 这座桥已封死，不重启——"
                            "重启会把已摄入的状态悄悄清空。")
        return self._dead

    def _died(self, why: str) -> str:
        code = self._proc.poll() if self._proc else None
        tail = "\n    ".join(self._stderr[-15:])
        return (f"{self._script.name}（退出码 {code}）：{why}"
                + (f"\n  子进程 stderr 末尾：\n    {tail}" if tail else ""))

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 —— 关不掉就杀，⛔ 不能留僵尸
            proc.kill()
            proc.wait(timeout=5)


def worker_script(package: str) -> Path:
    """worker 跟适配器住在一起。⚠️ 用文件路径找，⛔ 不 import——
    它要被**另一个解释器**执行。"""
    return Path(sys.modules[package].__file__).with_name("worker.py")
