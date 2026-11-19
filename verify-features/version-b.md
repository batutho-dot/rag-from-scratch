# Version B: embedding search

- **For the user:** the same search, but each chunk becomes 384 numbers from the
  `BAAI/bge-small-en-v1.5` model (quantised ONNX, via fastembed), run on the CPU.
- **How to reach it:** `rag.version_b.EmbeddingIndex(texts).search(question, k)`;
  `python -m rag.evaluate --only b`.
- **How verify drives it:** `python verify.py --full` runs `real-run-B` on the real corpus.
- **End state that proves it worked:** the rank of every answer equals
  `results/latest.json` (6/10 at k=5 today).
- **Gotchas:** about 2 minutes on a laptop CPU, so it is behind `--full`. The first run
  downloads about 65 MB of model files. A new fastembed or onnxruntime version could shift
  ranks slightly; requirements are pinned for that reason.
