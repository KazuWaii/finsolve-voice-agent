"""RAG for the FAQ tool: retrieve -> augment -> generate.

One deliberate difference from a text chatbot's prompt: the answer gets
read aloud by a TTS engine, so the system prompt explicitly forbids
markdown/bullet formatting -- "**4.25% APY**" or a bulleted list sounds
wrong (or gets read literally as asterisks) when spoken.
"""

from __future__ import annotations

from app import vectorstore
from app.llm import chat

# FAQ chunks are short and varied in topic, so this is looser than
# ds-rpc-01's 0.65 -- a tighter threshold rejected too many legitimate
# matches during testing.
MAX_RELEVANT_DISTANCE = 0.9

FALLBACK_ANSWER = (
    "I don't have information about that. Would you like me to connect you with a human agent?"
)

SYSTEM_PROMPT = (
    "You are FinSolve's customer support voice assistant, speaking to a caller "
    "on the phone. Answer using ONLY the context provided below. "
    "Keep it short (1-3 sentences), natural, and conversational -- this will "
    "be read aloud by a text-to-speech system, so never use markdown, bullet "
    "points, or asterisks. If the context doesn't answer the question, say so "
    "and offer to connect them with a human agent."
)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(c["text"] for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def answer_faq(question: str, n_results: int = 3) -> str:
    chunks = vectorstore.query(question, n_results=n_results)
    chunks = [c for c in chunks if c["distance"] <= MAX_RELEVANT_DISTANCE]

    if not chunks:
        return FALLBACK_ANSWER

    prompt = _build_prompt(question, chunks)
    result = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return result.content
