# Scoring against the 10 questions

- **For the user:** for each question in `questions.json`, finds the rank where a chunk
  containing its evidence phrase first appears, and reports hits at k = 1, 3, 5, 10.
- **How to reach it:** `python -m rag.evaluate` (both versions) or `python run.py`. Writes
  `results/latest.json`; a run with `--only` writes `results/latest_a.json` or
  `latest_b.json` instead (git-ignored).
- **How verify drives it:** `questions` checks the file's shape and that every evidence
  phrase is at most 50 words, so it always fits inside one chunk. `real-run-A` and
  `real-run-B` call `rank_of_first_hit` for every question.
- **End state that proves it worked:** ranks equal the committed `results/latest.json`.
- **Gotchas:** one accepted passage per question, so a chunk that answers from a different
  paragraph (q07's recital) counts as a miss. Changing the question set means rerunning
  `python run.py` and committing the new log.
