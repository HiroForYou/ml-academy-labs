"""Métricas de recuperación, un generador extractivo de juguete y el diagnóstico por componente.

Retrieval metrics, a toy extractive generator, and per-component diagnosis.

Un chunk es relevante si se solapa con la frase de evidencia del documento correcto (mismo tenant).
A chunk is relevant if it overlaps the evidence sentence in the correct document (same tenant).
"""

import re
import unicodedata

import pandas as pd

from retrieval import STOP_ES, Retriever


def relevant_ids(query: dict, chunks: list[dict]) -> set[str]:
    start, end = query["evidence"]
    return {c["chunk_id"] for c in chunks if c["doc_id"] == query["doc_id"] and c["start"] < end and c["end"] > start}


def coverage(query: dict, hits: list[dict]) -> float:
    """Fracción de la frase de evidencia contenida en los chunks recuperados. Share of the evidence sentence retrieved."""
    start, end = query["evidence"]
    covered = set()
    for c in hits:
        if c["doc_id"] == query["doc_id"]:
            covered.update(range(max(start, c["start"]), min(end, c["end"])))
    return len(covered) / (end - start)


def retrieval_metrics(retriever: Retriever, queries: list[dict], ks=(1, 2, 3, 4, 5), use_filter: bool = False) -> pd.DataFrame:
    """Una fila por pregunta y k: recall@k, precision@k, coverage@k, MRR y fuga de tenant.

    One row per question and k: recall@k, precision@k, coverage@k, MRR, and tenant leak.
    """
    rows = []
    for q in queries:
        hits = retriever.search(q["question"], k=max(ks), tenant=q["tenant"] if use_filter else None)
        rel = relevant_ids(q, retriever.chunks)
        ranks = [i + 1 for i, h in enumerate(hits) if h["chunk_id"] in rel]
        for k in ks:
            top = hits[:k]
            found = sum(h["chunk_id"] in rel for h in top)
            rows.append(
                {
                    "query_id": q["query_id"],
                    "tenant": q["tenant"],
                    "k": k,
                    "recall": found / len(rel),
                    "precision": found / k,
                    "coverage": coverage(q, top),
                    "mrr": 1 / ranks[0] if ranks and ranks[0] <= k else 0.0,
                    "leak": any(h["tenant"] != q["tenant"] for h in top),
                }
            )
    return pd.DataFrame(rows)


def summary(per_query: pd.DataFrame) -> pd.DataFrame:
    return per_query.groupby("k")[["recall", "precision", "coverage", "mrr", "leak"]].mean().round(3)


def _words(text: str, lang: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    stop = set(STOP_ES) if lang == "es" else {"the", "a", "an", "of", "to", "do", "i", "is", "my", "how", "what", "for"}
    return {w for w in re.findall(r"[a-z0-9@.]+", text) if w not in stop}


def generate(question: str, context: list[dict], lang: str = "es") -> dict:
    """Generador extractivo de juguete: devuelve la frase del contexto con más palabras en común con la pregunta.

    Toy extractive generator: returns the context sentence that shares the most words with the question.
    Stands in for the LLM so the lab stays local and deterministic.
    """
    q_words = _words(question, lang)
    best, best_score = {"answer": "", "chunk": None}, -1
    for chunk in context:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            score = len(q_words & _words(sentence, lang))
            if score > best_score:
                best, best_score = {"answer": sentence.strip(), "chunk": chunk}, score
    return best


def end_to_end(retriever: Retriever, queries: list[dict], k: int = 3, use_filter: bool = False, lang: str = "es") -> pd.DataFrame:
    """Respuesta generada y diagnóstico: correct, wrong_tenant, retrieval_miss o generation_miss.

    Generated answer and diagnosis: correct, wrong_tenant, retrieval_miss, or generation_miss.
    """
    rows = []
    for q in queries:
        hits = retriever.search(q["question"], k=k, tenant=q["tenant"] if use_filter else None)
        out = generate(q["question"], hits, lang)
        retrieved = bool(relevant_ids(q, retriever.chunks) & {h["chunk_id"] for h in hits})
        if q["answer"] in out["answer"]:
            outcome = "correct"
        elif out["chunk"] is not None and out["chunk"]["tenant"] != q["tenant"]:
            outcome = "wrong_tenant"
        elif not retrieved:
            outcome = "retrieval_miss"
        else:
            outcome = "generation_miss"
        rows.append(
            {
                "query_id": q["query_id"],
                "question": q["question"],
                "expected": q["answer"],
                "generated": out["answer"],
                "source": out["chunk"]["chunk_id"] if out["chunk"] else None,
                "retrieved_evidence": retrieved,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)
