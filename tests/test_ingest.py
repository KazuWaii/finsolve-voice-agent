from app.ingest import _split_by_headers, _nearest_whitespace, _split_by_size, load_faq_chunks, DATA_DIR


def test_split_by_headers_splits_on_h1_and_h2():
    text = "# Title\nIntro text.\n## Section One\nContent one.\n## Section Two\nContent two."
    sections = _split_by_headers(text)
    assert [header for header, _ in sections] == ["Title", "Section One", "Section Two"]


def test_split_by_headers_no_headers_returns_whole_text():
    text = "Just a plain paragraph, no headers at all."
    assert _split_by_headers(text) == [("", text)]


def test_nearest_whitespace_finds_space_in_window():
    text = "hello world this is a test"
    index = text.index("world") + 3  # lands mid-word, inside "world"
    result = _nearest_whitespace(text, index, search_window=10)
    assert text[:result] == "hello "


def test_split_by_size_short_text_is_a_single_piece():
    text = "short text"
    assert _split_by_size(text, chunk_size=900, overlap=150) == [text]


def test_split_by_size_terminates_and_covers_long_text():
    text = "word " * 1000  # 5000 chars, well over chunk_size
    pieces = _split_by_size(text, chunk_size=900, overlap=150)
    assert 1 < len(pieces) < 100
    assert all(len(p) <= 900 for p in pieces)


def test_load_markdown_file_one_chunk_per_short_section(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ingest.DATA_DIR", tmp_path)
    md_path = tmp_path / "doc.md"
    md_path.write_text(
        "# Title\nIntro.\n## Section A\nShort content A.\n## Section B\nShort content B.",
        encoding="utf-8",
    )

    chunks = load_faq_chunks(faq_path=md_path)

    assert len(chunks) == 3  # Title, Section A, Section B -- none need size-splitting
    assert {c.metadata["section"] for c in chunks} == {"Title", "Section A", "Section B"}
    assert all("part" in c.metadata for c in chunks)


def test_load_faq_chunks_on_real_project_data():
    # Regression test against the actual resources/data/faq.md fixture.
    chunks = load_faq_chunks()
    assert len(chunks) == 9