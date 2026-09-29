from app import vectorstore
from app.llm import chat

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


def _build_prompt(question, chunks):
    context = "\n\n".join(c["text"] for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def answer_faq(question, n_results=3):
    chunks = vectorstore.query(question, n_results=n_results)
    chunks = [c for c in chunks if c["distance"] <= MAX_RELEVANT_DISTANCE]

    if not chunks:
        return FALLBACK_ANSWER

    prompt = _build_prompt(question, chunks)
    result = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return result.content