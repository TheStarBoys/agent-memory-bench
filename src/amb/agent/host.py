"""DSH 宿主：把被测对象装进一个**固定的** agent。

⛔ 宿主是受控变量，不是被测对象：
    同一个循环、同一套工具、同一个 backbone，**只换记忆插件**。
换 DSH 版本等于换尺子，要重跑全部基线。

⚠️ 与「直接调库」那一档的数**不可互比**——那一档喂的是干净语料，
这一档喂的是 agent 自己搅出来的现场。
"""

from __future__ import annotations

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

    @property
    def version(self) -> str:
        """⚠️ 进报告：换版本等于换尺子。"""
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("deepseek-harness-sdk")
        except PackageNotFoundError:  # pragma: no cover
            return "unknown"


def spec_from_env(patches: tuple[str, ...] = ()) -> HostSpec:
    load_dotenv()
    return HostSpec(
        model=require("AMB_LLM_MODEL"),
        base_url=require("AMB_LLM_BASE_URL"),
        api_key_env=require("AMB_LLM_API_KEY_ENV"),
        patches=patches,
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
                        "models": [{"id": self._spec.model}],
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

    def ask(self, prompt: str) -> AgentTurn:
        if self._harness is None:
            raise HostUnavailable("宿主未启动")
        result = self._harness.start_session().run(prompt)
        return AgentTurn(
            text=result.final_response,
            finish_reason=result.finish_reason,
            events=list(result.events),
        )

    def close(self) -> None:
        if self._harness is not None:
            self._harness.close()
            self._harness = None

    def __enter__(self) -> "Host":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
