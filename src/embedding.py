from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model_name: str = MODEL_NAME):
    """Turn a list of strings into a 2D array of embedding vectors, one row per string."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
