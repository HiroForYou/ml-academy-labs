"""Índice TF-IDF local con filtro opcional de metadatos. Local TF-IDF index with an optional metadata filter.

Sustituye al embedding + vector store de una base de conocimiento. Replaces a knowledge base's embedding + vector store.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Palabras vacías mínimas en español (scikit-learn solo trae inglés). Minimal Spanish stop words (scikit-learn ships English).
STOP_ES = (
    "a al con cual de del el en es la las lo los me mi mis para por que se su sus te tengo un una y o cuanto cuantos "
    "cuanta debo hay puedo como donde cuando"
).split()


class Retriever:
    def __init__(self, chunks: list[dict], lang: str = "es"):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            stop_words=STOP_ES if lang == "es" else "english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in chunks)
        self.tenants = np.array([c["tenant"] for c in chunks])

    def search(self, question: str, k: int = 5, tenant: str | None = None) -> list[dict]:
        """Top-k chunks por similitud coseno; con tenant, filtra antes de ordenar (como un filtro `equals`).

        Top-k chunks by cosine similarity; with tenant, filters before ranking (like an `equals` filter).
        """
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        if tenant is not None:
            scores = np.where(self.tenants == tenant, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order if np.isfinite(scores[i])]
