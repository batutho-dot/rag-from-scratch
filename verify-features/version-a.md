# Version A: TF-IDF search in numpy

- **For the user:** returns the k chunks most similar to a question, using word counts
  weighted by rarity (TF-IDF) and cosine similarity, written by hand in numpy.
- **How to reach it:** `rag.version_a.TfidfIndex(texts).search(question, k)`;
  `python -m rag.evaluate --only a`.
- **How verify drives it:** `units` searches a 3-sentence toy corpus for "offshore wind
  turbines". `real-run-A` indexes the real corpus and asks the 10 questions.
- **End state that proves it worked:** toy top-1 is the wind sentence; the rank of every
  question's answer equals `results/latest.json` (6/10 at k=5 today).
- **Gotchas:** chunking constants (200 words, 50 overlap) feed every rank; changing them
  changes the known-good file, which must then be regenerated with `python run.py` and
  committed together.
