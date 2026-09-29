"""Chunk resources/data/faq.md the same way ds-rpc-01 chunks markdown:
header-aware splitting first, with a sliding-window fallback for any
section still too long. No RBAC/department tagging here -- this
knowledge base is public-facing by design (it's what a voice agent
answers anonymous callers with).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "resources" / "data"
FAQ_PATH = DATA_DIR / "faq.md"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

_HEADER_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group(2).strip()
        sections.append((header, text[start:end].strip()))
    return sections


def _nearest_whitespace(text: str, index: int, search_window: int = 50) -> int:
    window_start = max(0, index - search_window)
    space_pos = text.rfind(" ", window_start, index)
    return space_pos + 1 if space_pos != -1 else index


def _split_by_size(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _nearest_whitespace(text, end)
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def load_faq_chunks(faq_path: Path = FAQ_PATH) -> list[Chunk]:
    text = faq_path.read_text(encoding="utf-8")
    chunks = []
    for section_idx, (header, section_text) in enumerate(_split_by_headers(text)):
        if not section_text.strip():
            continue
        for piece_idx, piece in enumerate(_split_by_size(section_text)):
            chunks.append(
                Chunk(
                    id=f"faq-{section_idx}-{piece_idx}",
                    text=piece,
                    metadata={"source": faq_path.name, "section": header or "FAQ"},
                )
            )
    return chunks
