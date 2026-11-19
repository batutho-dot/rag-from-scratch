# Ask Claude a question (ask.py)

- **For the user:** answers a question in plain words from the top 5 retrieved chunks, with
  chunk ids as citations, or "not in the documents"; prints tokens and cost per call.
- **How to reach it:** `python ask.py "question"`, `-i` for a loop, `--model sonnet`,
  `--retriever a`. Needs `ANTHROPIC_API_KEY` and the texts from `python run.py`.
- **How verify drives it:** `compile` only. verify makes no API call, because every call
  costs money and needs a key.
- **End state that proves it worked:** checked by hand on 2026-11-19: 3 Haiku calls, answers
  and costs in the README and `results/ask_log.jsonl`.
- **Gotchas:** the embedding cache in `data/cache/` is keyed by a hash of the model name and
  all chunk texts, so a change to chunking rebuilds it (about 3 minutes). The prices in
  `ask.py` are list prices typed in by hand; update them when Anthropic changes prices.
