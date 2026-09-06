"""入口：解析参数、组装、交给 runner。

⛔ 保持薄——任何题库专有或系统专有的知识都不属于这一层。
实测失效：MemoryData 的 main.py 有 925 行，且在模块顶层写死了
「哪些方法在某个题库上要特殊处理」的清单。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from amb.core import AnswerStyle, HarnessFault, load_dotenv
from amb.report import ArmResult, Report, render
from amb.runner import (
    answer_prompt, backbone, build, build_plan, cache_report, context_overflow,
    corpus_fingerprint,
    control_arms,
    host_unavailable, ingest_identity, now_rfc3339, run_one,
)


def _setup_cmd(argv: list[str]) -> int:
    """一键装外部依赖。⭐ 记录**实际装到的**版本。"""
    from amb.setup import install_all, status

    ap = argparse.ArgumentParser(prog="amb setup")
    ap.add_argument("names", nargs="*", help="不给就装全部")
    ap.add_argument("--check", action="store_true", help="只看状态，不装")
    ap.add_argument("--upgrade", action="store_true")
    args = ap.parse_args(argv)

    rows = (status(args.names or None) if args.check
            else install_all(args.names or None, upgrade=args.upgrade))
    width = max((len(r.name) for r in rows), default=4)
    for r in rows:
        mark = "✓" if r.ok else "✗"
        same = "" if r.actual == r.declared else "  ⚠️ 与声明不同"
        print(f"  {mark} {r.name:<{width}}  声明 {r.declared}  实际 {r.actual}{same}")
        if r.detail:
            print(f"      {r.detail[:200]}")
    return 0 if all(r.ok for r in rows) else 1


def _show_preflight(report, *, where: str) -> None:
    """把自检结果打到 stderr。⛔ stdout 留给报告本身。

    ⭐ **样本一定要打**：自动检查覆盖不到「摄入单元读起来像不像真的」，
    ⚠️ 而那正是最贵的那次浪费（2h45m 全废）的成因——
    ⛔ 唯一挡得住它的就是人眼看这一下。
    """
    print(f"\n── 跑前自检（{where}）" + "─" * 30, file=sys.stderr)
    for f in report.findings:
        print(f"  {f}", file=sys.stderr)
    if not report.findings:
        print("  · 无发现", file=sys.stderr)
    print("\n  ⭐ 语料与题面样本——⚠️ **自动检查看不出「像不像真的」，请人眼过一遍**：",
          file=sys.stderr)
    for line in report.samples:
        print(f"    {line[:160]}", file=sys.stderr)
    if report.budget:
        print(f"\n  · 预算：{report.budget}", file=sys.stderr)
    print("─" * 46 + "\n", file=sys.stderr)


def _preflight_cmd(argv: list[str]) -> int:
    """⛔ 零网络调用的跑前自检。⭐ 单独跑一次比跑完再看便宜几个数量级。"""
    from amb.runner.preflight import estimate, inspect

    ap = argparse.ArgumentParser(prog="amb preflight")
    ap.add_argument("--bench", choices=("toy", "locomo", "dialogue"), default="toy")
    ap.add_argument("--condition", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--convs", default="")
    args = ap.parse_args(argv)

    plan, _, name = build_plan(
        args.bench, condition=args.condition, max_turns=args.max_turns,
        conversations=tuple(c for c in args.convs.split(",") if c))
    report = inspect(plan)
    arms = tuple(a for a in args.arms.split(",") if a)
    if arms:
        mins = estimate(len(plan.documents), arms)
        report.budget["摄入分钟"] = {k: round(v, 1) for k, v in mins.items()}
        report.budget["合计小时"] = round(sum(mins.values()) / 60, 2)
    _show_preflight(report, where=name)
    return 1 if report.fatal else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "setup":
        return _setup_cmd(argv[1:])
    if argv and argv[0] == "preflight":
        return _preflight_cmd(argv[1:])

    ap = argparse.ArgumentParser(prog="amb")
    ap.add_argument("--arms", default=",".join(control_arms()),
                    help="逗号分隔；默认跑全部五条对照组")
    ap.add_argument("--budget", type=int, default=24000, help="full_context 的上下文预算")
    ap.add_argument("--json", type=Path, help="同时写一份 JSON")
    ap.add_argument("--bench", choices=("toy", "locomo", "dialogue"),
                    default="toy", help="跑哪个题库")
    ap.add_argument("--condition", default="",
                    choices=("", "dense", "diluted", "repeated", "revised"),
                    help="⚠️ 只对 --bench dialogue 有意义：同一批事实的四种讲法。"
                         "⛔ 四个条件的数不可互比，必须显式给")
    ap.add_argument("--sample", default="all",
                    help="抽题：all | first:N | random:N | stratified:N | ids:a,b")
    ap.add_argument("--max-convs", type=int, default=None,
                    help="限几个对话——⛔ 控的是语料量，与题数是两件事")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="每个对话只留前 N 轮——⛔ evidence 落在被截部分的题会被丢掉")
    ap.add_argument("--convs", default="",
                    help="点名跑哪些对话（逗号分隔）。⭐ 各对话题目产出差 2.5 倍，"
                         "⚠️ 随机抽会白付摄入成本；⛔ 覆盖 --max-convs")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="⚠️ 随机抽样的种子——⛔ 进报告，不记就不可复现")
    ap.add_argument("--lane", choices=("library", "agent", "both"),
                    default="library", help="跑哪一档。⛔ 两档的数不可互比")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="⛔ 跳过跑前自检。⚠️ 只在自检自己坏了的时候用——"
                         "它花几秒，而一次跑要几小时")
    ap.add_argument("--no-answer", action="store_true",
                    help="不挂 backbone，只跑检索档（省钱、离线可跑）")
    args = ap.parse_args(argv)

    # ⛔ 全局唯一的 backbone——所有臂必须同一个，否则 answer 档不可比。
    # ⚠️ 要在造 plan 之前定下来：没挂 backbone 就不放回答档进去。
    llm = None if args.no_answer else backbone()

    plan, sampling, world_name = build_plan(
        args.bench, sample=args.sample, seed=args.sample_seed,
        max_conversations=args.max_convs, max_turns=args.max_turns,
        conversations=tuple(c for c in args.convs.split(",") if c),
        with_answer=llm is not None, condition=args.condition)

    # ⛔ 答题口径的语言必须跟题库走。⚠️ 实测踩过：中文提示 + 英文题库，
    # 模型一律用中文答，逐字比对全判错——⭐ 那不是记忆层不行，是尺子在量语言。
    prompt = answer_prompt(args.bench)

    from amb.setup import snapshot

    report = Report(
        run_id=f"{world_name}-{now_rfc3339()}",
        at=now_rfc3339(),
        # ⛔ `digest` 是**世界状态**哈希（守卫每阶段比对），它盖不住
        # 喂给被测系统的语料——⚠️ 实测：172 篇与 434 篇两份「公认不可比」的
        # 存档 `digest` 完全相同。⭐ 所以语料指纹必须单独进存档，
        # 否则任何自动对账（tools/compare_runs.py）都漏这一类。
        world={"name": world_name, "seed": args.sample_seed, "digest": "",
               "corpus": corpus_fingerprint(plan.documents),
               "documents": len(plan.documents)},
        backbone={"model": llm.model if llm else "—（未跑 answer 档）",
                  "temperature": llm.temperature if llm else None,
                  # ⛔ 受控变量，必须进报告：思考型 backbone 输出 token
                  # 大 6～8 倍，实测 A-mem 摄入 3 条 663s → 43s
                  "thinking": llm.thinking if llm else None,
                  # ⭐ 摄入用的那个模型。⛔ 与回答 backbone 是两件事：
                  # `--no-answer` 时没有回答 backbone，⚠️ 但被测系统摄入时
                  # 照样调 LLM——钱那一列要靠它才算得出来。
                  "ingest_model": os.environ.get("AMB_LLM_MODEL", ""),
                  # ⛔ 换提示等于换尺子，两次跑不可比——必须进报告
                  "answer_prompt": prompt.system if llm else None,
                  # ⭐ 跟**套件**走的那一层变体：⚠️ 默认口径要求「资料里没有
                  # 就弃权」，而 N8 问的是故意没进语料的个体——⛔ 两者相反。
                  # ⚠️ 变体同样是尺子的一部分，一并进报告。
                  "answer_prompt_styles": (
                      {v.value: prompt.styled(v).system
                       for v in AnswerStyle if v is not AnswerStyle.STRICT}
                      if llm else None)},
        # ⭐ 外部依赖的实际版本，⛔ 没有它这次跑不算数
        externals=snapshot(),
        # ⚠️ 抽样方式进报告——⛔ 抽样变了分数就不可比
        sampling=sampling,
    )

    names = [a for a in args.arms.split(",") if a]

    # ⛔ 先自检再花钱。⚠️ 它零网络调用、几秒钟，⭐ 而一次跑要几小时——
    # 实测两次教训：一次报告印出假话，一次语料造错整跑作废，
    # **两次的成因都不需要真跑就能发现**。
    if not args.skip_preflight:
        from amb.runner.preflight import estimate, inspect

        pre = inspect(plan, arms=tuple(names))
        mins = estimate(len(plan.documents), tuple(names))
        if mins:
            pre.budget["摄入分钟"] = {k: round(v, 1) for k, v in mins.items()}
            pre.budget["合计小时"] = round(sum(mins.values()) / 60, 2)
        _show_preflight(pre, where=world_name)
        if pre.fatal:
            print("⛔ 自检有致命问题，**不开跑**——修掉，或 --skip-preflight 强跑",
                  file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory(prefix="amb-world-") as tmp:
        if args.lane in ("agent", "both"):
            _run_agent_lane(report, names, Path(tmp) / "agent")
        if args.lane == "agent":
            _emit(report, args)
            return 0
        for i, name in enumerate(names, 1):
            root = Path(tmp) / name
            # ⚠️ 一条臂可能跑一小时（实测 a_mem 55s/条且随库变贵）。
            # ⛔ 全跑完才出声的话，中途崩了就什么都看不到——进度走 stderr，
            # ⭐ stdout 留给报告本身，管道用法不受影响。
            print(f"▶ [{i}/{len(names)}] {name} …", file=sys.stderr, flush=True)
            started = time.perf_counter()
            try:
                result, world_digest = run_one(
                    name, build(name, context_budget=args.budget, llm=llm,
                                prompt=prompt), plan, root,
                    is_control=name in control_arms(),
                    # ⚠️ N4 第 3 步要重开一个同样的适配器
                    rebuild=lambda n=name: build(n, context_budget=args.budget,
                                                 llm=llm, prompt=prompt),
                    # ⭐ 摄入快照的键之一：**影响摄入的**那套 LLM 配置。
                    # ⛔ 不是 llm.model——`--no-answer` 时它是 None，
                    # 但被测系统摄入时照样调自己配的 LLM。
                    backbone=ingest_identity(),
                )
            except context_overflow() as exc:
                # ⭐ 这不是故障，是**这条臂在这个语料上不适用**。
                # ⛔ 按 docs/baselines.md 记 N/A——⚠️ 记成 crashed 就把
                # 「不适用 / 失败」两态压成了一态。
                why = str(exc)[:200]
                print(f"— [{i}/{len(names)}] {name}: N/A（{why}）",
                      file=sys.stderr, flush=True)
                report.lanes.setdefault("library", []).append(
                    ArmResult(arm=name, is_control=name in control_arms(),
                              not_applicable=why))
                continue
            except HarnessFault as exc:
                # ⛔ **评测器自己**没跑成——⚠️ 记成 crashed 就是拿我们的 bug
                # 去记它的账，那一列一混，读者只会以为这个系统不稳。
                why = str(exc)[:200]
                print(f"⛔ [{i}/{len(names)}] {name}: 框架自己的问题（{why}）",
                      file=sys.stderr, flush=True)
                report.lanes.setdefault("library", []).append(
                    ArmResult(arm=name, is_control=name in control_arms(),
                              harness_fault=why))
                continue
            except (KeyError, EnvironmentError) as exc:
                # ⛔ **评测器侧的配置错**：臂名打错、必需环境变量没设、
                # 依赖没装——⚠️ 那不是「这个系统跑挂了」，
                # ⭐ 而报告里 crashed 那一列会被读成「这个系统不稳」。
                why = f"{type(exc).__name__}: {exc}"[:200]
                print(f"⛔ [{i}/{len(names)}] {name}: 配置问题（{why}）",
                      file=sys.stderr, flush=True)
                report.lanes.setdefault("library", []).append(
                    ArmResult(arm=name, is_control=name in control_arms(),
                              harness_fault=why))
                continue
            except Exception as exc:  # noqa: BLE001
                # ⛔ 不只打到 stderr——静默消失会被读成「没参赛」
                msg = f"{type(exc).__name__}: {exc}"[:200]
                print(f"✗ [{i}/{len(names)}] {name}: {msg}",
                      file=sys.stderr, flush=True)
                report.lanes.setdefault("library", []).append(
                    ArmResult(arm=name, is_control=name in control_arms(),
                              crashed=msg))
                continue
            report.world["digest"] = world_digest
            report.lanes.setdefault('library', []).append(result)
            took = time.perf_counter() - started
            snap = ("　⭐ 摄入快照命中"
                    if result.ingest_snapshot == "命中" else "")
            print(f"✓ [{i}/{len(names)}] {name}　{took:.0f}s{snap}",
                  file=sys.stderr, flush=True)

    # ⭐ 缓存状况进报告——⚠️ 包括「为什么没生效」
    report.cache = cache_report()

    _emit(report, args)
    return 0


def _emit(report: Report, args) -> None:
    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                        indent=2, default=str), encoding="utf-8")


def _run_agent_lane(report: Report, names: list[str], workdir: Path) -> None:
    """⭐ 装进 agent 那一档。⛔ 与直接调库的数不可互比。"""
    from amb.runner import AgentPlan, agent_arms, host_spec, run_one_agent

    from worlds import toy

    spec = host_spec()
    report.host = {"version": spec.version, "profile": spec.profile}
    # ⛔ **喂全部语料**，⚠️ 不是只喂 5 篇世界文件：
    # 早先这里是 `toy.DOCUMENTS`（4 个世界文件 + 1 条 N4 探针语料），
    # 而 agent 档的探针有 46/57 题问的是 `extra_documents()`（618 篇）里的内容——
    # ⭐ 那些内容既没进记忆、也不在世界文件里，于是 qa / n3 / n5 / n6 / n8
    # 对**每一条臂**恒为 0.000，五条臂无从区分。⛔ 一个记忆层再好也拿不到分。
    plan = AgentPlan(manifest=toy.MANIFEST, documents=toy.all_documents(),
                     changes=toy.CHANGES, suites_for=toy.agent_suites)
    for name in names:
        try:
            result, digest = run_one_agent(name, spec, plan, workdir / name,
                                           is_control=name in agent_arms())
        except host_unavailable() as exc:
            # ⛔ **宿主装不上是框架这一侧的事**，⚠️ 不是被测系统跑挂了。
            # 早先它落进 `except Exception` → 记 `crashed` → 报告印
            # 「它们不是不支持，也不是 0 分——是跑挂了」，⭐ 而那句
            # 「记不可用，不是 0 分」原样躺在 crashed 列里。
            why = str(exc)[:200]
            print(f"⛔ agent/{name}: 宿主不可用（{why}）", file=sys.stderr)
            report.lanes.setdefault("agent", []).append(
                ArmResult(arm=name, is_control=name in agent_arms(),
                          harness_fault=why))
            continue
        except HarnessFault as exc:
            why = str(exc)[:200]
            print(f"⛔ agent/{name}: 框架自己的问题（{why}）", file=sys.stderr)
            report.lanes.setdefault("agent", []).append(
                ArmResult(arm=name, is_control=name in agent_arms(),
                          harness_fault=why))
            continue
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"[:200]
            print(f"✗ agent/{name}: {msg}", file=sys.stderr)
            report.lanes.setdefault("agent", []).append(
                ArmResult(arm=name, is_control=name in agent_arms(), crashed=msg))
            continue
        report.world["digest"] = digest
        report.lanes.setdefault("agent", []).append(result)

