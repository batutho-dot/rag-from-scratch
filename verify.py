"""Check this repository after a change. Prints one line per step and ends with
`verify: PASS` or `verify: FAIL at <step>: <reason>`. Changes nothing in the repo:
all output goes to a temporary folder.

    python verify.py          fast checks plus a real run of version A (a few seconds)
    python verify.py --full   also runs version B (embedding model, about 2 minutes)

Uses the downloaded HTML in data/raw/ when present; otherwise downloads it (network).
"""

import io
import json
import py_compile
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class Fail(Exception):
    pass


def check(cond, reason):
    if not cond:
        raise Fail(reason)


def step_compile(tmp):
    files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "rag").glob("*.py"))
    for f in files:
        try:
            py_compile.compile(str(f), cfile=str(tmp / "pyc" / (f.name + "c")), doraise=True)
        except py_compile.PyCompileError as e:
            raise Fail(f"{f.name}: {e.msg.strip().splitlines()[-1]}")
    return f"{len(files)} files compile"


def step_units(tmp):
    from rag.chunk import OVERLAP, chunk_text
    from rag.download import html_to_text
    from rag.evaluate import normalise
    from rag.version_a import TfidfIndex

    words = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text("t", words)
    check(len(chunks) == 3, f"500 words gave {len(chunks)} chunks, expected 3")
    check(chunks[0].text.split()[-OVERLAP:] == chunks[1].text.split()[:OVERLAP], "chunks do not overlap by OVERLAP words")
    check(chunks[-1].text.split()[-1] == "w499", "last word lost")
    check(html_to_text("<p>a  b</p><script>x</script><td>c</td>") == "a b\nc", "html_to_text output changed")
    check(normalise("It’s – A B") == "it's - a b", "normalise output changed")
    ix = TfidfIndex(["solar panels on roofs", "wind turbines at sea", "taxes on income"])
    check(ix.search("offshore wind turbines", 1)[0][0] == 1, "TF-IDF top-1 wrong on toy corpus")
    return "chunking, html_to_text, normalise, TF-IDF toy search"


def step_questions(tmp):
    qs = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    check(len(qs) == 10, f"{len(qs)} questions, expected 10")
    check(len({q["id"] for q in qs}) == 10, "duplicate question ids")
    for q in qs:
        check(all(q.get(k) for k in ("id", "question", "doc", "where", "evidence")), f"{q.get('id')} missing a field")
        check(len(q["evidence"].split()) <= 50, f"{q['id']} evidence longer than the chunk overlap")
    return "10 questions, all fields, evidence fits in the overlap"


def build_corpus(tmp):
    from rag import download

    data = tmp / "data"
    if (ROOT / "data" / "raw").is_dir():
        shutil.copytree(ROOT / "data" / "raw", data / "raw")
    with redirect_stdout(io.StringIO()):
        download.main(data)
    return data / "text"


def ranks_for(index_cls, text_dir):
    from rag.chunk import load_chunks
    from rag.evaluate import rank_of_first_hit

    chunks = load_chunks(text_dir)
    qs = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    index = index_cls([c.text for c in chunks])
    return len(chunks), {q["id"]: rank_of_first_hit(index, chunks, q) for q in qs}


def known_good(key):
    return json.loads((ROOT / "results" / "latest.json").read_text(encoding="utf-8"))[key]["rank_of_answer"]


def step_real_a(tmp):
    from rag.version_a import TfidfIndex

    text_dir = build_corpus(tmp)
    n, ranks = ranks_for(TfidfIndex, text_dir)
    check(n == 1257, f"{n} chunks, expected 1257")
    good = known_good("A_tfidf_numpy")
    diff = {k: (good[k], ranks[k]) for k in good if good[k] != ranks[k]}
    check(not diff, f"ranks differ from results/latest.json (known, now): {diff}")
    hits = sum(r is not None and r <= 5 for r in ranks.values())
    return f"version A on the real corpus: {hits}/10 at k=5, ranks match results/latest.json"


def step_real_b(tmp):
    from rag.version_b import EmbeddingIndex

    _, ranks = ranks_for(EmbeddingIndex, tmp / "data" / "text")
    good = known_good("B_embedding")
    diff = {k: (good[k], ranks[k]) for k in good if good[k] != ranks[k]}
    check(not diff, f"ranks differ from results/latest.json (known, now): {diff}")
    hits = sum(r is not None and r <= 5 for r in ranks.values())
    return f"version B on the real corpus: {hits}/10 at k=5, ranks match results/latest.json"


def main():
    full = "--full" in sys.argv
    print(f"verify-repo: {ROOT}")
    steps = [("compile", step_compile), ("units", step_units), ("questions", step_questions), ("real-run-A", step_real_a)]
    if full:
        steps.append(("real-run-B", step_real_b))
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, fn in steps:
            try:
                print(f"{name}: ok ({fn(tmp)})")
            except Fail as e:
                print(f"verify: FAIL at {name}: {e}")
                return 1
            except Exception as e:  # a crash is a failure of that step, never a pass
                print(f"verify: FAIL at {name}: {type(e).__name__}: {e}")
                return 1
    if not full:
        print("real-run-B: skipped (run with --full)")
    print("verify: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
