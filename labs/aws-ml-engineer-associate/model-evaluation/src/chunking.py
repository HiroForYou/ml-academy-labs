"""Dos estrategias de chunking que conservan offsets y metadatos. Two chunking strategies that keep offsets and metadata.

- fixed: ventanas de N palabras, sin respetar frases (como fixed-size chunking con tamaño pequeño).
  N-word windows that ignore sentence boundaries (like fixed-size chunking with a small size).
- paragraph: un chunk por párrafo; respeta los límites lógicos del documento.
  One chunk per paragraph; respects the document's logical boundaries.
"""

import re


def _chunk(doc: dict, strategy: str, i: int, start: int, end: int) -> dict:
    return {
        "chunk_id": f"{doc['doc_id']}#{strategy}-{i}",
        "doc_id": doc["doc_id"],
        "tenant": doc["tenant"],
        "start": start,
        "end": end,
        "text": doc["text"][start:end],
    }


def fixed(doc: dict, words: int = 30, overlap: int = 0) -> list[dict]:
    spans = [m.span() for m in re.finditer(r"\S+", doc["text"])]
    step = words - overlap
    chunks = []
    for i, first in enumerate(range(0, len(spans), step)):
        last = min(first + words, len(spans)) - 1
        chunks.append(_chunk(doc, "fixed", i, spans[first][0], spans[last][1]))
        if last == len(spans) - 1:
            break
    return chunks


def paragraph(doc: dict) -> list[dict]:
    chunks, start = [], 0
    for i, part in enumerate(doc["text"].split("\n\n")):
        chunks.append(_chunk(doc, "paragraph", i, start, start + len(part)))
        start += len(part) + 2
    return chunks


STRATEGIES = {"fixed-30": lambda d: fixed(d, 30), "paragraph": paragraph}


def chunk_corpus(docs: list[dict], strategy: str) -> list[dict]:
    return [c for d in docs for c in STRATEGIES[strategy](d)]
