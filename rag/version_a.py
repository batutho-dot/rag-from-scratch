"""Version A: TF-IDF vectors and cosine top-k, written by hand in numpy.

A vector here is one number per word of the vocabulary. TF-IDF sets that number high when a
word is frequent in this chunk (term frequency) and rare across all chunks (inverse document
frequency), so "the" counts for almost nothing and "taxonomy" counts for a lot.
"""

import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "which with shall be been were was not any such their these those other where when".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class TfidfIndex:
    def __init__(self, texts: list[str]):
        docs = [tokenize(t) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        counts = self._counts(docs)  # chunks x vocab, raw word counts
        df = (counts > 0).sum(axis=0)  # in how many chunks each word appears
        n = counts.shape[0]
        self.idf = np.log((1 + n) / (1 + df)) + 1.0
        self.matrix = self._weigh(counts)

    def _counts(self, docs: list[list[str]]) -> np.ndarray:
        m = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for row, d in enumerate(docs):
            for w in d:
                if w in self.vocab:
                    m[row, self.vocab[w]] += 1
        return m

    def _weigh(self, counts: np.ndarray) -> np.ndarray:
        tfidf = np.log1p(counts) * self.idf  # log1p damps a word repeated 20 times
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        return tfidf / np.maximum(norms, 1e-12)  # unit length, so dot product = cosine

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._weigh(self._counts([tokenize(query)]))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]
