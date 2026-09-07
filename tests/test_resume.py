"""中断之后接着跑。⛔ **一次几小时的跑不该是全有或全无。**

⚠️ 实测踩到：跑到第 5 条臂被系统 OOM 杀掉，前面 4 条臂 42 分钟的结果
（`naive_rag` 1386s（整条臂） + `mem0_raw` 1142s）全部丢失。

⭐ 这个文件守的**不是**「续跑好用」，而是「⛔ 该拒绝时真的拒绝」——
一个会静默混用两套口径的续跑，比没有续跑糟得多。
"""

from __future__ import annotations

import json

from amb.report import ArmResult, Report
from amb.runner.resume import KEY_FIELDS, code_digest, key_of, load, mismatches, restore
from amb.scoring import Score


def _report(**over) -> Report:
    r = Report(run_id="r", at="t",
               world={"name": "toy", "seed": 42, "digest": "d", "corpus": "c0"},
               backbone={"model": "m", "thinking": False, "ingest_model": "m",
                         "answer_prompt": "p", "context_budget": 24000},
               externals={}, sampling={"strategy": "all"})
    for k, v in over.items():
        setattr(r, k, v)
    r.resume_key = key_of(r)
    return r


def _finished(name: str) -> ArmResult:
    a = ArmResult(arm=name, is_control=True)
    sc = Score("retrieval", "scored", denominator=40, metrics={"top1": 0.9})
    sc.denominators = {"top1": 40}
    a.scores["retrieval"] = sc
    a.cost = {"probe": 100}
    return a


def _write(tmp_path, report, arms):
    report.lanes["library"] = arms
    p = tmp_path / "run.json"
    p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p


def test_a_finished_arm_is_skipped_on_the_next_run(tmp_path) -> None:
    """⭐ 这就是那 42 分钟本该被救回来的路径。"""
    src = _report()
    path = _write(tmp_path, src, [_finished("bm25"), _finished("null")])

    fresh = _report()
    done, notes = load(path, fresh.resume_key)
    names = restore(fresh, done)
    assert names == {"bm25", "null"}
    assert any("续跑" in n for n in notes)
    # ⭐ 分与分母都要原样回来——⛔ 否则续跑等于换了一把尺
    got = fresh.lanes["library"][0].scores["retrieval"]
    assert got.metrics["top1"] == 0.9 and got.denominators["top1"] == 40


def test_changed_code_forbids_resume(tmp_path) -> None:
    """⛔ **判分改了就不许续**：⚠️ 新口径的分与旧口径的分并排放，
    就是这个仓库刚治过的「新表格配旧数」。

    ⚠️ 严格到「动一个注释也不给续」是刻意的——⭐ 分辨「这次改动影响不影响
    判分」需要判断，而判断会出错；从头跑只是慢，混用是错。
    """
    src = _report()
    path = _write(tmp_path, src, [_finished("bm25")])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["resume_key"]["code"] = "别的版本"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    done, notes = load(path, _report().resume_key)
    assert done == {}
    assert any("不能续跑" in n and "code" in n for n in notes)


def test_every_key_field_blocks_resume(tmp_path) -> None:
    """⛔ 键里**每一项**都要真的拦得住——⚠️ 少拦一项就可能混用不可比的分。"""
    src = _report()
    path = _write(tmp_path, src, [_finished("bm25")])
    base = json.loads(path.read_text(encoding="utf-8"))

    for field in KEY_FIELDS:
        raw = json.loads(json.dumps(base))
        raw["resume_key"][field] = "变了"
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        done, notes = load(path, _report().resume_key)
        assert done == {}, f"⛔ {field} 变了却仍然续跑"
        assert any(field in n for n in notes), f"⚠️ 没说清是 {field} 变了"


def test_an_archive_without_a_key_is_not_resumed(tmp_path) -> None:
    """⛔ 旧格式存档没有续跑键——⚠️ 不许猜它是不是同一跑。"""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"lanes": {"library": [{"arm": "bm25"}]}}),
                    encoding="utf-8")
    done, notes = load(path, _report().resume_key)
    assert done == {} and any("旧格式" in n for n in notes)


def test_a_corrupt_archive_does_not_take_down_the_run(tmp_path) -> None:
    """⚠️ 存档坏了就当没有，⛔ 不炸。"""
    path = tmp_path / "bad.json"
    path.write_text("{ 半截", encoding="utf-8")
    done, notes = load(path, _report().resume_key)
    assert done == {} and any("读不动" in n for n in notes)


def test_no_archive_is_not_an_error() -> None:
    from pathlib import Path

    assert load(None, {}) == ({}, [])
    assert load(Path("/tmp/不存在的存档.json"), {}) == ({}, [])


def test_the_code_digest_moves_when_source_moves(tmp_path) -> None:
    """⭐ 指纹要真的跟着代码走——⛔ 不跟着走的键等于没有键。"""
    root = tmp_path / "repo"
    (root / "src" / "amb").mkdir(parents=True)
    f = root / "src" / "amb" / "x.py"
    f.write_text("a = 1", encoding="utf-8")
    before = code_digest(root)
    f.write_text("a = 2", encoding="utf-8")
    assert code_digest(root) != before
    # ⚠️ 也要**稳定**：没动就不变
    assert code_digest(root) == code_digest(root)


def test_mismatches_lists_every_difference() -> None:
    a = {f: "x" for f in KEY_FIELDS}
    b = dict(a, code="y", corpus="z")
    assert set(mismatches(a, b)) == {"code", "corpus"}
