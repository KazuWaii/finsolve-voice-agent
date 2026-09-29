from app import vectorstore
from app.ingest import Chunk


def test_query_finds_the_relevant_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "PERSIST_DIR", tmp_path)

    chunks = [
        Chunk(id="a", text="Our support line is open 8am to 8pm on weekdays.", metadata={"source": "faq.md", "section": "Business Hours"}),
        Chunk(id="b", text="High-yield savings earns 4.25% APY.", metadata={"source": "faq.md", "section": "Interest Rates"}),
    ]
    vectorstore.index_chunks(chunks)

    results = vectorstore.query("What time are you open?", n_results=1)

    assert len(results) == 1
    assert results[0]["metadata"]["section"] == "Business Hours"


def test_index_chunks_is_idempotent_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "PERSIST_DIR", tmp_path)

    chunk = Chunk(id="a", text="Some FAQ content.", metadata={"source": "faq.md", "section": "s"})
    vectorstore.index_chunks([chunk])
    vectorstore.index_chunks([chunk])

    results = vectorstore.query("FAQ content", n_results=10)
    assert len(results) == 1
