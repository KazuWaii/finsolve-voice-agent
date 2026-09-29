from pathlib import Path
from dataclasses import dataclass
import re # (expressions régulières)


DATA_DIR = Path(__file__).resolve().parents[1] / "resources" / "data"
# Regex to cut md files into sections based on headers
_HEADER_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
# Chunk size and overlap for splitting text into smaller pieces
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

class Chunk:
    def __init__(self, id, text, metadata):
        self.id = id
        self.text = text
        self.metadata = metadata

def load_faq_chunks(faq_path=DATA_DIR / "faq.md"):
    return _load_markdown_file(faq_path)


def _split_by_headers(text):
    matches = list(_HEADER_RE.finditer(text))

    # cas ou le texte ne contient pas de headers
    if not matches:
        return [("", text)]

    sections = []
    for i, m in enumerate(matches):
        # debut du texte de la section
        start = m.start()
        # fin du texte de la section
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group(2).strip()
        sections.append((header, text[start:end].strip()))
    return sections


def _nearest_whitespace(text, index, search_window=50):
    window_start = max(0, index - search_window)
    space_pos = text.rfind(" ", window_start, index)
    return space_pos + 1 if space_pos != -1 else index

def _split_by_size(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(text) <= chunk_size:
        return [text]
    else:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                end = _nearest_whitespace(text, end)
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = end - overlap
        return chunks

def _load_markdown_file(file_path):
    text = file_path.read_text(encoding="utf-8")
    chunks = []

    for header, section_text in _split_by_headers(text):
        for i, piece in enumerate(_split_by_size(section_text)):
            chunk = Chunk(
                id=f"{file_path.stem}_{header}_part{i}",
                text=piece,
                metadata={
                    "source": file_path.relative_to(DATA_DIR).as_posix(),
                    "section": header or file_path.stem,
                    "part": i,
                },
            )
            chunks.append(chunk)

    return chunks