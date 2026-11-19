"""Version B: same chunks, same cosine top-k, but the vectors come from a small open
embedding model (BAAI/bge-small-en-v1.5, 384 numbers per text) run on the CPU through
fastembed (ONNX, no PyTorch). The model places texts with similar meaning close together,
so "green investments" can match "environmentally sustainable economic activities" even
with no word in common.
"""

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained to see this prefix on the question side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingIndex:
    def __init__(self, texts: list[str]):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(MODEL)
        self.matrix = self._unit(np.array(list(self.model.embed(texts, batch_size=64))))

    @staticmethod
    def _unit(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._unit(np.array(list(self.model.embed([QUERY_PREFIX + query]))))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]
