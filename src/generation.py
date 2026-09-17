import ollama

MODEL_NAME = "llama3.1:8b"

SYSTEM_PROMPT = """You are a compliance assistant that answers questions about Indian \
hospitality regulations using ONLY the excerpts provided in the context below.

Rules:
- Only use information present in the context. Do not use any outside knowledge, \
even if you believe you know the answer.
- Every claim you make must be followed by a citation to the source page, like (page 35).
- If the context does not contain enough information to answer the question, say so \
explicitly instead of guessing. Do not fill gaps with assumptions.
"""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labeled context block the model can cite from."""
    blocks = []
    for chunk in chunks:
        blocks.append(f"[page {chunk['page']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]
