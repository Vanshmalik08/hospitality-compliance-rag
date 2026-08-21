import json
from pathlib import Path

import numpy as np

from src.embedding import embed_texts

CHUNKS_PATH = Path("data/processed/fssai_licensing_registration_regulations.jsonl")
EMBEDDINGS_PATH = Path("data/processed/fssai_licensing_registration_regulations.embeddings.npy")


def main():
    records = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    texts = [record["text"] for record in records]
    embeddings = embed_texts(texts)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embedded {len(texts)} chunks into {embeddings.shape[1]}-dimensional vectors")
    print(f"Saved to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
