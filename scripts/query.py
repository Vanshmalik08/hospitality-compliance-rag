import sys
from pathlib import Path

import numpy as np

from src.retrieval import load_chunks, retrieve

CHUNKS_PATH = Path("data/processed/fssai_licensing_registration_regulations.jsonl")
EMBEDDINGS_PATH = Path("data/processed/fssai_licensing_registration_regulations.embeddings.npy")


def main():
    query = " ".join(sys.argv[1:])
    if not query:
        print("Usage: python -m scripts.query <your question>")
        return

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    results = retrieve(query, chunks, embeddings, top_k=5)

    print(f'Query: "{query}"\n')
    for rank, result in enumerate(results, start=1):
        print(f"#{rank}  score={result['score']:.3f}  page={result['page']}")
        print(result["text"][:300].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
