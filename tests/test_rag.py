from unittest.mock import patch

from app import rag
from app.llm import ChatResult


def _chunk(distance, text="some FAQ text"):
    return {"text": text, "metadata": {"source": "faq.md", "section": "s"}, "distance": distance}


def test_answer_faq_returns_fallback_when_nothing_relevant():
    with patch("app.rag.vectorstore.query", return_value=[]):
        answer = rag.answer_faq("some obscure question")

    assert answer == rag.FALLBACK_ANSWER


def test_answer_faq_filters_chunks_past_threshold():
    chunks = [_chunk(distance=0.2), _chunk(distance=0.95)]  # second is past MAX_RELEVANT_DISTANCE
    fake_result = ChatResult(content="a spoken answer", prompt_tokens=10, completion_tokens=5)

    with patch("app.rag.vectorstore.query", return_value=chunks), \
         patch("app.rag.chat", return_value=fake_result) as mock_chat:
        answer = rag.answer_faq("a question")

    assert answer == "a spoken answer"
    # only the in-threshold chunk's text should have made it into the prompt
    prompt_sent = mock_chat.call_args[0][0][1]["content"]
    assert prompt_sent.count("some FAQ text") == 1


def test_answer_faq_system_prompt_forbids_markdown():
    # Regression guard: this is a voice agent, its answers get read aloud by
    # TTS -- markdown/bullets must never be allowed back into the prompt.
    assert "markdown" in rag.SYSTEM_PROMPT.lower()
