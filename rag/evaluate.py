"""Score retrieval: for each question, is the passage that answers it inside the top-k chunks?

A question counts as a hit when any of the top-k chunks contains its evidence phrase (the
exact words from the law that answer it, compared after normalising spaces, quotes and
dashes). The score is hits out of the number of questions.

Run: python -m rag.evaluate            (both versions, k = 5)
     python -m rag.evaluate --only a   (numpy version only, no model download)
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rag.chunk import CHUNK_WORDS, OVERLAP, load_chunks

ROOT = Path(__file__).resolve().parent.parent
K = 5
REPORT_KS = (1, 3, 5, 10)


def normalise(s: str) -> str:
    for a, b in {"\u2019": "'", "\u2018": "'", "\u2013": "-", "\u2014": "-", "\u00a0": " "}.items():
        s = s.replace(a, b)
    return " ".join(s.lower().split())


def rank_of_first_hit(index, chunks, q, depth=max(REPORT_KS)) -> int | None:
    evidence = normalise(q["evidence"])
    for rank, (i, _) in enumerate(index.search(q["question"], depth), start=1):
        if evidence in normalise(chunks[i].text):
            return rank
    return None


def run(name: str, index, chunks, questions) -> dict:
    ranks = {q["id"]: rank_of_first_hit(index, chunks, q) for q in questions}
    hits = {k: sum(r is not None and r <= k for r in ranks.values()) for k in REPORT_KS}
    print(f"\n{name}: hit rate at k={K}: {hits[K]}/{len(questions)}")
    print("  " + "  ".join(f"k={k}: {h}/{len(questions)}" for k, h in hits.items()))
    for q in questions:
        r = ranks[q["id"]]
        mark = "hit " if r is not None and r <= K else "MISS"
        print(f"  {mark} {q['id']} rank={r if r else '>' + str(max(REPORT_KS))}  {q['question']}")
    return {"hits_at_k": hits, "rank_of_answer": ranks}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["a", "b"])
    args = p.parse_args()

    chunks = load_chunks(ROOT / "data" / "text")
    texts = [c.text for c in chunks]
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    for q in questions:  # every evidence phrase must exist in the corpus, or the test is broken
        assert any(normalise(q["evidence"]) in normalise(c.text) for c in chunks), q["id"]
    print(f"{len(chunks)} chunks of {CHUNK_WORDS} words (overlap {OVERLAP}), {len(questions)} questions")

    log = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "k": K,
           "chunks": len(chunks), "chunk_words": CHUNK_WORDS, "overlap": OVERLAP}
    if args.only in (None, "a"):
        from rag.version_a import TfidfIndex

        t = time.perf_counter()
        index = TfidfIndex(texts)
        log["A_tfidf_numpy"] = run("Version A (TF-IDF, numpy)", index, chunks, questions)
        log["A_tfidf_numpy"]["seconds"] = round(time.perf_counter() - t, 1)
    if args.only in (None, "b"):
        from rag.version_b import MODEL, EmbeddingIndex

        t = time.perf_counter()
        index = EmbeddingIndex(texts)
        log["B_embedding"] = run(f"Version B ({MODEL})", index, chunks, questions)
        log["B_embedding"]["model"] = MODEL
        log["B_embedding"]["seconds"] = round(time.perf_counter() - t, 1)

    out = ROOT / "results" / ("latest.json" if args.only is None else f"latest_{args.only}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nlogged to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
