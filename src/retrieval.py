import json
from pathlib import Path

import numpy as np

from src.embedding import embed_texts


def load_chunks(chunks_path: Path) -> list[dict]:
    records = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def retrieve(query: str, chunks: list[dict], embeddings: np.ndarray, top_k: int = 5) -> list[dict]:
    """Return the top_k chunks most similar to the query, each tagged with its score."""
    query_vector = embed_texts([query])[0]

    # embeddings: (num_chunks, dim), query_vector: (dim,)
    # embeddings @ query_vector gives one similarity score per chunk (dot product,
    # which equals cosine similarity because every vector is already normalized).
    scores = embeddings @ query_vector

    # argsort sorts ascending; flip it and take the first top_k for highest-first.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        result = dict(chunks[i])
        result["score"] = float(scores[i])
        results.append(result)
    return results
