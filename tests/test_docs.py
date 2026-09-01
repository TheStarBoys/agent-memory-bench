"""文档守卫。

手工跑了八遍、每次都抓到真 bug 的检查，提成测试。
⛔ 尤其是重复标题：字符串拼接时把 `## X` 接在以 `## X` 开头的残段前面，
同一个错误在本项目里已经犯过两次。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "node_modules", "__pycache__", "out"}
DOCS = sorted(p for p in ROOT.rglob("*.md") if not SKIP & set(p.relative_to(ROOT).parts))


def _slug(text: str) -> str:
    text = re.sub(r"[`*]", "", text.strip().lower())
    text = "".join(c for c in text if c.isalnum() or c in " -_" or "一" <= c <= "鿿")
    return re.sub(r"\s+", "-", text.strip())


def _anchors(path: Path) -> set[str]:
    body = path.read_text()
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", body, re.M)} | set(
        re.findall(r'<a id="([^"]+)"></a>', body)
    )


ANCHORS = {p: _anchors(p) for p in DOCS}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_internal_links_resolve(doc: Path) -> None:
    bad: list[str] = []
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", doc.read_text()):
        target = m.group(2)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        rel, _, anchor = target.partition("#")
        dest = (doc.parent / rel).resolve() if rel else doc.resolve()
        try:
            dest.relative_to(ROOT)
        except ValueError:
            bad.append(f"{target} → 跳出仓库")
            continue
        if not dest.exists():
            bad.append(f"{target} → 文件不存在")
        elif anchor and dest in ANCHORS and anchor not in ANCHORS[dest]:
            bad.append(f"{target} → 锚点不存在")
    assert not bad, f"{doc.relative_to(ROOT)} 里的坏链接：\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_duplicated_heading_text_on_one_line(doc: Path) -> None:
    # `## 复现## 复现` —— 拼接残留，肉眼几乎看不出来，锚点却会失效
    bad = [
        f"{i}: {line}"
        for i, line in enumerate(doc.read_text().splitlines(), 1)
        if re.match(r"^#{1,6}\s+.*#{1,6}\s", line)
    ]
    assert not bad, f"{doc.relative_to(ROOT)} 疑似重复标题：\n  " + "\n  ".join(bad)


def test_headings_are_unique_within_a_doc() -> None:
    bad: list[str] = []
    for doc in DOCS:
        seen: dict[str, int] = {}
        for m in re.finditer(r"^#{1,6}\s+(.+)$", doc.read_text(), re.M):
            s = _slug(m.group(1))
            seen[s] = seen.get(s, 0) + 1
        for s, n in seen.items():
            if n > 1:
                bad.append(f"{doc.relative_to(ROOT)}: 「{s}」出现 {n} 次——锚点会指向第一个")
    assert not bad, "标题重名：\n  " + "\n  ".join(bad)
