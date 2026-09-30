from unittest.mock import patch

from app import rag
from app.llm import ChatResult


def _sample_chunk(distance, text="some retrieved text", section="Section"):
    return {
        "text": text,
        "metadata": {"source": "faq.md", "section": section, "part": 0},
        "distance": distance,
    }


def test_answer_faq_no_relevant_chunks_returns_fallback():
    with patch("app.rag.vectorstore.query", return_value=[]):
        result = rag.answer_faq("some obscure question")

    assert result == rag.FALLBACK_ANSWER


def test_answer_faq_filters_out_chunks_past_distance_threshold():
    chunks = [_sample_chunk(distance=1.5)]  # past MAX_RELEVANT_DISTANCE (0.9)
    with patch("app.rag.vectorstore.query", return_value=chunks):
        result = rag.answer_faq("a question")

    assert result == rag.FALLBACK_ANSWER


def test_answer_faq_returns_llm_answer_when_relevant_chunks_found():
    chunks = [_sample_chunk(distance=0.2, text="Business hours are 8am-8pm.")]
    fake_result = ChatResult(content="We're open 8am to 8pm.", prompt_tokens=10, completion_tokens=5)

    with patch("app.rag.vectorstore.query", return_value=chunks), \
         patch("app.rag.chat", return_value=fake_result) as mock_chat:
        result = rag.answer_faq("What are your business hours?")

    assert result == "We're open 8am to 8pm."
    mock_chat.assert_called_once()
    messages = mock_chat.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "Business hours are 8am-8pm." in messages[1]["content"]
