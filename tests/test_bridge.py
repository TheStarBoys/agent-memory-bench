"""子进程桥。⛔ 被测系统跑在别的解释器里，这层出错必须**看得见**。

每条断言都对着一个真踩过或真会踩的坑：
    ① 崩了不许静默重启 —— 重启会把已摄入的状态悄悄清空
    ② 往 stdout 打日志会冲掉协议 —— 报错要直接点出这一点
    ③ 子进程的 stderr 要带进异常 —— 否则「它为什么挂」查不出来
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from amb.adapters.bridge import Bridge, BridgeError


def _worker(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(body))
    return script


ECHO = '''
    import json, sys
    for line in sys.stdin:
        msg = json.loads(line)
        op = msg.pop("op")
        if op == "boom":
            sys.exit(9)
        if op == "noisy":
            print("我往 stdout 打了日志")     # ⛔ 这会冲掉协议
            sys.stdout.flush()
        if op == "fail":
            sys.stdout.write(json.dumps({"ok": False, "error": "我不干"}) + "\\n")
            sys.stdout.flush(); continue
        sys.stdout.write(json.dumps({"ok": True, "result": msg}) + "\\n")
        sys.stdout.flush()
'''


def _bridge(tmp_path: Path, body: str = ECHO) -> Bridge:
    return Bridge(Path(sys.executable), _worker(tmp_path, body), {"hello": 1})


def test_round_trip(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    assert bridge.call("ping", x=7) == {"x": 7}
    bridge.close()


def test_worker_error_is_not_swallowed(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    with pytest.raises(BridgeError, match="我不干"):
        bridge.call("fail")
    bridge.close()


def test_dead_worker_is_never_silently_restarted(tmp_path: Path) -> None:
    """⛔ 最要紧的一条。

    静默重启的话，调用方会以为一切正常，而被测系统已经把摄入的东西全丢了——
    ⚠️ 那样跑出来的是「半个语料的记忆系统」的分数，比直接崩掉更有害。
    """
    bridge = _bridge(tmp_path)
    assert bridge.call("ping") == {}
    with pytest.raises(BridgeError):
        bridge.call("boom")          # 子进程 exit(9)
    with pytest.raises(BridgeError, match="不重启"):
        bridge.call("ping")          # ⛔ 这一次必须还是报错，不许悄悄复活
    bridge.close()


def test_stdout_pollution_names_the_cause(tmp_path: Path) -> None:
    """被测系统往 stdout 打日志是最常见的接入故障，⚠️ 报错要直说。"""
    bridge = _bridge(tmp_path)
    with pytest.raises(BridgeError, match="stdout"):
        bridge.call("noisy")
    bridge.close()


def test_stderr_reaches_the_exception(tmp_path: Path) -> None:
    """⛔ 子进程死前说的话必须带上来——否则查不出它为什么死。"""
    bridge = Bridge(Path(sys.executable), _worker(tmp_path, '''
        import sys
        print("我死之前想说：依赖装错了", file=sys.stderr, flush=True)
        sys.exit(3)
    '''), {})
    with pytest.raises(BridgeError, match="依赖装错了"):
        bridge.call("anything")
    bridge.close()


def test_init_config_reaches_the_worker(tmp_path: Path) -> None:
    """⭐ 配置（含 key）只经 stdin 递进去，⛔ 不落盘、不进命令行。"""
    marker = tmp_path / "seen.txt"
    bridge = Bridge(Path(sys.executable), _worker(tmp_path, f'''
        import json, sys, pathlib
        for line in sys.stdin:
            msg = json.loads(line)
            if msg["op"] == "init":
                pathlib.Path({str(marker)!r}).write_text(json.dumps(msg))
            sys.stdout.write(json.dumps({{"ok": True, "result": None}}) + "\\n")
            sys.stdout.flush()
    '''), {"api_key": "秘密", "model": "M"})
    bridge.call("ping")
    import json

    assert json.loads(marker.read_text())["api_key"] == "秘密"
    bridge.close()
