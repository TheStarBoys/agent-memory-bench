"""DSH 宿主：把被测对象装进一个**固定的** agent。

⛔ 宿主是受控变量，不是被测对象：
    同一个循环、同一套工具、同一个 backbone，**只换记忆插件**。
换 DSH 版本等于换尺子，要重跑全部基线。

⚠️ 与「直接调库」那一档的数**不可互比**——那一档喂的是干净语料，
这一档喂的是 agent 自己搅出来的现场。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from amb.core import load_dotenv, require


class HostUnavailable(RuntimeError):
    """DSH 运行时装不上或起不来。⛔ 该档记不可用，不是 0 分。"""


@dataclass(frozen=True, slots=True)
class HostSpec:
    """一次跑里全局唯一的宿主配置。

    ⛔ 每一项都必须对所有臂相同——只要有一项不同，
    分数的差就可能来自那一项，而不是记忆层。
    """

    model: str
    base_url: str
    api_key_env: str
    #: ⚠️ 自定义 provider 的 id，写进 DSH_HOME/settings.yaml。
    #: 内置的 deepseek-official 只认 DeepSeek 自己的端点。
    provider: str = "amb-backbone"
    profile: str = "sdk"
    #: 挂到宿主上的 cordis patch。⭐ 记忆插件从这里进来。
    patches: tuple[str, ...] = ()
    max_tokens: int = 512
    request_timeout_s: float = 180.0
    #: ⭐ **上下文窗口**（token）。⚠️ 这是 agent 档最要紧的受控变量：
    #: ⛔ 整段会话装得下的话，记忆插件就是摆设——那时测的是**模型自己**。
    #:
    #: ⭐ 把它调小，几十轮就进入压缩/滑窗区间：⚠️ 于是「记忆有没有用」
    #: 与**会话多长**解耦了，⛔ 而会话长度正是成本的主要驱动。
    #:
    #: ⚠️ `None` = 用 DSH 的默认（128k）。⛔ 那时这一档测不到记忆的价值。
    context_window: int | None = None

    @property
    def version(self) -> str:
        """⚠️ 进报告：换版本等于换尺子。"""
        from importlib.metadata import PackageNotFoundError, version

        try:
            got = version("deepseek-harness-sdk")
        except PackageNotFoundError:  # pragma: no cover
            return "⛔ 未安装"
        # ⛔ **与钉死的版本对账**：⚠️ 早先自己查完就印，
        # ⭐ 「换 DSH 版本等于换尺子」这句话没有任何一处在执行
        try:
            from amb.setup.spec import REGISTRY

            pin = REGISTRY["dsh"].pin
        except Exception:  # noqa: BLE001
            return got
        return got if got == pin else f"{got} ⚠️与钉死的 {pin} 不同"


def spec_from_env(patches: tuple[str, ...] = ()) -> HostSpec:
    """⚠️ `AMB_CONTEXT_WINDOW`：⭐ agent 档的上下文窗口（token）。

    ⛔ 不设就用 DSH 的默认 128k——⚠️ 那时几十轮的会话整段装得下，
    ⭐ 记忆插件是摆设，这一档测的是**模型自己**。
    """
    import os

    load_dotenv()
    raw = os.environ.get("AMB_CONTEXT_WINDOW", "").strip()
    return HostSpec(
        model=require("AMB_LLM_MODEL"),
        base_url=require("AMB_LLM_BASE_URL"),
        api_key_env=require("AMB_LLM_API_KEY_ENV"),
        patches=patches,
        # ⛔ 空串不是 0：⚠️ `AMB_CONTEXT_WINDOW=` 该当成「没设」
        context_window=int(raw) if raw else None,
    )


@dataclass
class AgentTurn:
    """一轮会话的结果。"""

    text: str
    finish_reason: str | None
    #: ⭐ agent/* 事件流——每一步都看得到，**不需要被测系统配合**。
    events: list[dict] = field(default_factory=list)


class Host:
    """一个跑起来的 DSH。

    ⚠️ 世界通过 `cwd` 交给 agent——它在里面读写文件，
    而每一次写都被评测器的哈希守卫看着（world.md）。
    """

    def __init__(self, spec: HostSpec, world_root: Path, home: Path) -> None:
        self._spec = spec
        self._world = world_root
        self._home = home
        self._harness = None
        #: ⭐ 这条臂的会话。⛔ 跨轮复用——探针的「上一轮」全靠它。
        self._session = None

    def _write_settings(self) -> None:
        """把 backbone 声明成 DSH 的自定义 provider。

        ⛔ key 只用 apiKeyEnv 引用变量名，不写进文件。
        """
        import json

        settings = {
            "llm-pi-ai": {
                "providers": {
                    self._spec.provider: {
                        "apiKeyEnv": self._spec.api_key_env,
                        "api": "openai-completions",
                        "baseURL": self._spec.base_url,
                        # ⭐ `contextWindow` 是 DSH 模型条目的合法字段
                        # （schema: `{id, name?, contextWindow?, maxTokens?}`）。
                        # ⚠️ 不给就用它的默认 128k——⛔ 那时整段会话装得下，
                        # 记忆插件是摆设。
                        "models": [{
                            "id": self._spec.model,
                            **({} if self._spec.context_window is None
                               else {"contextWindow": self._spec.context_window}),
                        }],
                    }
                }
            }
        }
        # YAML 是 JSON 的超集，⚠️ 免掉一个依赖
        (self._home / "settings.yaml").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def start(self) -> None:
        try:
            from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig
        except ImportError as exc:  # pragma: no cover
            raise HostUnavailable(
                "缺 deepseek-harness-sdk。⛔ 该档记不可用，不是 0 分。"
                "装法：pip install deepseek-harness-sdk"
            ) from exc

        self._home.mkdir(parents=True, exist_ok=True)
        self._write_settings()
        cfg = DeepSeekHarnessConfig(
            provider=self._spec.provider,
            model=self._spec.model,
            base_url=self._spec.base_url,
            api_key=require(self._spec.api_key_env),
            # ⚠️ 运行时按 apiKeyEnv 从环境读，这里也传一份到子进程环境
            env={self._spec.api_key_env: require(self._spec.api_key_env)},
            max_tokens=self._spec.max_tokens,
            cwd=str(self._world),          # ⭐ 世界即 agent 的工作目录
            dsh_home=str(self._home),      # ⚠️ 隔离，⛔ 绝不用 ~/.dsh
            profile=self._spec.profile,
            patches=list(self._spec.patches),
            request_timeout_seconds=self._spec.request_timeout_s,
        )
        harness = DeepSeekHarness(cfg)
        harness.start()
        self._harness = harness
        self._session = None

    def ask(self, prompt: str) -> AgentTurn:
        """问一轮。⭐ **同一个会话贯穿整条臂**——⛔ 不是每轮新开。

        ⚠️ 早先这里是 `start_session().run(prompt)`，每次都 uuid4 新开一个
        会话，于是宿主**没有任何跨轮上下文**。所有依赖「上一轮」的探针
        测的都不是它们声称的东西：

        | 探针 | 它以为在测 | 实际 |
        |---|---|---|
        | N1 有提示第二轮 | 「现在提交你对 c1 的判定」 | ⛔ 新会话里没见过 c1 |
        | N4 忘记探针 | 记住 → 问 → 忘掉 → 再问 | ⛔ 四个互不相干的会话 |
        | N8 规律存活 | 「⭐ 见过例外**之后**」 | ⛔ 新会话没见过例外 |
        | `host_default` | 「只用 DSH 自带的工作记忆」 | ⛔ 那份工作记忆永远是空的 |

        ⭐ 会话即那条臂的一生：`reset_session()` 由 runner 在**换臂时**调。
        """
        if self._harness is None:
            raise HostUnavailable("宿主未启动")
        if self._session is None:
            self._session = self._harness.start_session()
        # ⛔ **瞬时故障要重试**：⚠️ `adapters/llm.py` 那条腿改过了，
        # 这一条早先没有——⭐ 同一次网络抖动，直接调库那档被救回来，
        # agent 档整条臂记 crashed。**框架的缺陷记成被测系统的失败**。
        result = None
        last: Exception | None = None
        for attempt in range(3):
            try:
                result = self._session.run(prompt)
                break
            except Exception as exc:  # noqa: BLE001 —— 宿主的异常类型不归我们管
                last = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if result is None:
            raise HostUnavailable(f"宿主连续 3 次没跑成：{last}") from last
        return AgentTurn(
            text=result.final_response,
            finish_reason=result.finish_reason,
            events=list(result.events),
        )

    def reset_session(self) -> None:
        """开一个新会话。⛔ **只在换臂时调**——⚠️ 套件之间共用同一个会话
        是刻意的：探针依赖「上一轮」，而那正是这一档要测的东西。"""
        self._session = None

    def close(self) -> None:
        self._session = None
        if self._harness is not None:
            self._harness.close()
            self._harness = None

    def __enter__(self) -> "Host":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
