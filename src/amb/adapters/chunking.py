"""切块。对照组共用，⛔ 对照组之间必须用同一套切法。

切法不同，「差别只来自记忆层」就不成立了——那时候比的是谁切得好。
"""

from __future__ import annotations

from dataclasses import dataclass

from amb.core import Span


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    text: str
    start: int  # ⛔ Unicode 码点偏移，与 Span 同一口径
    end: int

    def to_span(self) -> Span:
        return Span(doc_id=self.doc_id, start=self.start, end=self.end)


def chunk(doc_id: str, text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """定长滑窗切块。

    朴素是刻意的：这是**地板线**，它的作用是回答「不动脑子能到多少分」。
    地板线一旦开始调优，它就不再是地板线了。
    """
    if size <= 0:
        raise ValueError("size 必须为正")
    if not 0 <= overlap < size:
        raise ValueError("overlap 必须在 [0, size) 内")

    out: list[Chunk] = []
    step = size - overlap
    for start in range(0, max(len(text), 1), step):
        piece = text[start : start + size]
        if not piece:
            break
        out.append(Chunk(doc_id, piece, start, start + len(piece)))
        if start + size >= len(text):
            break
    return out
