"""Download the EUR-Lex HTML texts into data/raw/ and turn them into plain text in data/text/.

Run: python -m rag.download
"""

import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from rag.sources import SOURCES, url

DATA = Path(__file__).resolve().parent.parent / "data"
BLOCK_TAGS = {"p", "td", "th", "li", "h1", "h2", "h3", "h4", "div", "tr", "br"}


class _TextOnly(HTMLParser):
    """Keeps the visible text, puts a line break at the end of every block element."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextOnly()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def main() -> None:
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    (DATA / "text").mkdir(parents=True, exist_ok=True)
    for key, src in SOURCES.items():
        raw = DATA / "raw" / f"{key}.html"
        if not raw.exists():
            req = urllib.request.Request(url(src["celex"]), headers={"User-Agent": "rag-from-scratch/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw.write_bytes(resp.read())
        text = html_to_text(raw.read_text(encoding="utf-8"))
        (DATA / "text" / f"{key}.txt").write_text(text, encoding="utf-8")
        print(f"{key:9s} {src['celex']}  {len(text.split()):>7,} words")


if __name__ == "__main__":
    main()
