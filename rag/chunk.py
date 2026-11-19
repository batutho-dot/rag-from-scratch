"""Chunking: cut each long text into short overlapping pieces.

A search engine cannot hand a 100,000-word regulation to a reader. It hands back a few short
pieces, so the text is cut up first. Here a chunk is a window of CHUNK_WORDS words; the next
window starts STRIDE words later, so neighbouring chunks share OVERLAP words and a sentence
cut at one edge is still whole in the neighbour.
"""

from dataclasses import dataclass
from pathlib import Path

CHUNK_WORDS = 200
OVERLAP = 50
STRIDE = CHUNK_WORDS - OVERLAP


@dataclass
class Chunk:
    id: str  # e.g. "csrd#0042"
    doc: str
    text: str


def chunk_text(doc: str, text: str) -> list[Chunk]:
    words = text.split()
    chunks = []
    for n, start in enumerate(range(0, max(len(words) - OVERLAP, 1), STRIDE)):
        piece = " ".join(words[start : start + CHUNK_WORDS])
        chunks.append(Chunk(id=f"{doc}#{n:04d}", doc=doc, text=piece))
    return chunks


def load_chunks(text_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(text_dir.glob("*.txt")):
        chunks.extend(chunk_text(path.stem, path.read_text(encoding="utf-8")))
    return chunks
