"""Version B: same chunks, same cosine top-k, but the vectors come from a small open
embedding model (BAAI/bge-small-en-v1.5, 384 numbers per text) run on the CPU through
fastembed (ONNX, no PyTorch). The model places texts with similar meaning close together,
so "green investments" can match "environmentally sustainable economic activities" even
with no word in common.
"""

import hashlib
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained to see this prefix on the question side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingIndex:
    def __init__(self, texts: list[str], cache: Path | None = None):
        """cache: optional .npy file. Embedding 1,257 chunks takes about 2 minutes on a
        laptop CPU, so ask.py stores the result and reuses it while the chunks are unchanged."""
        from fastembed import TextEmbedding

        self.model = TextEmbedding(MODEL)
        key = hashlib.sha256((MODEL + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()
        stamp = cache.with_suffix(".sha256") if cache else None
        if cache and cache.exists() and stamp.exists() and stamp.read_text() == key:
            self.matrix = np.load(cache)
            return
        self.matrix = self._unit(np.array(list(self.model.embed(texts, batch_size=64))))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.matrix)
            stamp.write_text(key)

    @staticmethod
    def _unit(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._unit(np.array(list(self.model.embed([QUERY_PREFIX + query]))))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]
