import sys
from pathlib import Path

import numpy as np

from src.generation import generate_answer
from src.retrieval import load_chunks, retrieve

CHUNKS_PATH = Path("data/processed/fssai_licensing_registration_regulations.jsonl")
EMBEDDINGS_PATH = Path("data/processed/fssai_licensing_registration_regulations.embeddings.npy")


def main():
    query = " ".join(sys.argv[1:])
    if not query:
        print("Usage: python -m scripts.answer <your question>")
        return

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    top_chunks = retrieve(query, chunks, embeddings, top_k=5)
    answer = generate_answer(query, top_chunks)

    print(f'Question: "{query}"\n')
    print(answer)
    print("\n--- Retrieved pages used as context:", sorted({c["page"] for c in top_chunks}))


if __name__ == "__main__":
    main()
