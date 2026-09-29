import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest import load_faq_chunks
from app.vectorstore import index_chunks


def main():
    chunks = load_faq_chunks()
    print(f"Loaded {len(chunks)} FAQ chunks")
    index_chunks(chunks)
    print("Indexed into resources/vectorstore/")


if __name__ == "__main__":
    main()