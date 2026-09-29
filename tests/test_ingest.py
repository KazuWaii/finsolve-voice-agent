from app import ingest


def test_split_by_headers_splits_on_h1_and_h2():
    text = "# Title\nIntro.\n## Section One\nContent one.\n## Section Two\nContent two."
    sections = ingest._split_by_headers(text)
    assert [h for h, _ in sections] == ["Title", "Section One", "Section Two"]


def test_split_by_size_terminates_and_covers_long_text():
    text = "word " * 1000
    pieces = ingest._split_by_size(text, chunk_size=900, overlap=150)
    assert 1 < len(pieces) < 100
    assert all(len(p) <= 900 for p in pieces)


def test_load_faq_chunks_on_real_data():
    chunks = ingest.load_faq_chunks()
    # One chunk per top-level section of faq.md (title + 8 FAQ topics),
    # since each section is short enough to not need further size-splitting.
    assert len(chunks) == 9
    sections = {c.metadata["section"] for c in chunks}
    assert "Business Hours" in sections
    assert "Lost or Stolen Card" in sections
    assert all(c.metadata["source"] == "faq.md" for c in chunks)
