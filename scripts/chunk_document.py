import json
from pathlib import Path

from pypdf import PdfReader

from src.chunking import chunk_text

RAW_PDF = Path("data/raw/fssai_licensing_registration_regulations.pdf")
OUT_PATH = Path("data/processed/fssai_licensing_registration_regulations.jsonl")


def main():
    reader = PdfReader(RAW_PDF)
    chunk_records = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text.strip():
            continue

        for chunk in chunk_text(page_text, chunk_size=1000, overlap=200):
            chunk_records.append({
                "chunk_id": len(chunk_records),
                "source": RAW_PDF.name,
                "page": page_number,
                "text": chunk,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for record in chunk_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunk_records)} chunks from {len(reader.pages)} pages to {OUT_PATH}")


if __name__ == "__main__":
    main()
