def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks
