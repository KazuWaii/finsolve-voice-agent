from app.ingest import Chunk
from app.vectorstore import index_chunks, query


def test_index_and_query_returns_relevant_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vectorstore.PERSIST_DIR", tmp_path)

    chunks = [
        Chunk(
            id="hours-1",
            text="Our support line is open Monday to Friday, 8am to 8pm.",
            metadata={"source": "faq.md", "section": "Business Hours", "part": 0},
        ),
        Chunk(
            id="card-1",
            text="If your card is lost or stolen, freeze it immediately in the app.",
            metadata={"source": "faq.md", "section": "Lost or Stolen Card", "part": 0},
        ),
    ]
    index_chunks(chunks)

    results = query("What are your business hours?", n_results=1)

    assert len(results) == 1
    assert results[0]["metadata"]["section"] == "Business Hours"


def test_index_chunks_is_idempotent_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vectorstore.PERSIST_DIR", tmp_path)

    chunk = Chunk(
        id="hours-1",
        text="Business hours content.",
        metadata={"source": "faq.md", "section": "Business Hours", "part": 0},
    )
    index_chunks([chunk])
    index_chunks([chunk])  # re-running must not duplicate

    results = query("business hours", n_results=10)
    assert len(results) == 1


def test_query_respects_n_results(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vectorstore.PERSIST_DIR", tmp_path)

    chunks = [
        Chunk(id=f"c{i}", text=f"Some FAQ content number {i}.", metadata={"source": "faq.md", "section": f"Section {i}", "part": 0})
        for i in range(5)
    ]
    index_chunks(chunks)

    results = query("FAQ content", n_results=2)
    assert len(results) == 2