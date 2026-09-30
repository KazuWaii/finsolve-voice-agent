import json
from unittest.mock import patch

from app import agent
from app.llm import ChatResult


def test_classify_intent_returns_recognized_intent():
    fake_result = ChatResult(content="faq", prompt_tokens=5, completion_tokens=1)
    with patch("app.agent.chat", return_value=fake_result):
        result = agent.classify_intent({"message": "What are your business hours?"})

    assert result == {"intent": "faq"}


def test_classify_intent_normalizes_case_and_whitespace():
    fake_result = ChatResult(content="  FAQ  \n", prompt_tokens=5, completion_tokens=1)
    with patch("app.agent.chat", return_value=fake_result):
        result = agent.classify_intent({"message": "hi"})

    assert result == {"intent": "faq"}


def test_classify_intent_falls_back_to_other_for_unrecognized_output():
    fake_result = ChatResult(content="something_unexpected", prompt_tokens=5, completion_tokens=1)
    with patch("app.agent.chat", return_value=fake_result):
        result = agent.classify_intent({"message": "asdf"})

    assert result == {"intent": "other"}


def test_handle_faq_delegates_to_answer_faq():
    with patch("app.agent.answer_faq", return_value="We're open 8-8.") as mock_answer:
        result = agent.handle_faq({"message": "business hours?"})

    mock_answer.assert_called_once_with("business hours?")
    assert result == {"answer": "We're open 8-8."}


def test_handle_transaction_no_id_in_message_asks_for_one():
    result = agent.handle_transaction({"message": "what's my balance"})

    assert "transaction ID" in result["answer"]


def test_handle_transaction_extracts_id_case_insensitively():
    with patch("app.agent.lookup_transaction", return_value={
        "status": "completed", "amount": 100.0, "date": "2026-01-01", "description": "Test",
    }) as mock_lookup:
        result = agent.handle_transaction({"message": "status of txn10005 please"})

    # re.IGNORECASE means the regex still finds a lowercase id; the id text itself
    # is passed through as found -- lookup_transaction is what upper()s it.
    mock_lookup.assert_called_once_with("txn10005")
    assert "completed" in result["answer"]


def test_handle_transaction_not_found():
    with patch("app.agent.lookup_transaction", return_value=None):
        result = agent.handle_transaction({"message": "status of TXN00000"})

    assert "couldn't find" in result["answer"]


def test_handle_callback_extracts_details_and_schedules():
    fake_result = ChatResult(
        content=json.dumps({"name": "Alice", "phone_number": "555-1234", "preferred_time": "tomorrow at 3pm"}),
        prompt_tokens=10, completion_tokens=5,
    )
    with patch("app.agent.chat", return_value=fake_result), \
         patch("app.agent.schedule_callback", return_value="Thanks Alice, we've scheduled a callback.") as mock_schedule:
        result = agent.handle_callback({"message": "I'm Alice, call me at 555-1234 tomorrow at 3pm"})

    mock_schedule.assert_called_once_with("Alice", "555-1234", "tomorrow at 3pm")
    assert result == {"answer": "Thanks Alice, we've scheduled a callback."}


def test_handle_callback_missing_field_asks_again():
    fake_result = ChatResult(
        content=json.dumps({"name": "Alice", "phone_number": None, "preferred_time": "tomorrow"}),
        prompt_tokens=10, completion_tokens=5,
    )
    with patch("app.agent.chat", return_value=fake_result):
        result = agent.handle_callback({"message": "I'm Alice, call tomorrow"})

    assert "phone number" in result["answer"]


def test_handle_callback_malformed_json_asks_again():
    fake_result = ChatResult(content="not valid json", prompt_tokens=10, completion_tokens=5)
    with patch("app.agent.chat", return_value=fake_result):
        result = agent.handle_callback({"message": "call me sometime"})

    assert "phone number" in result["answer"]


def test_handle_fallback_returns_static_redirect():
    result = agent.handle_fallback({"message": "anything"})

    assert "rephrase" in result["answer"]


def test_run_agent_routes_faq_intent_end_to_end():
    classify_result = ChatResult(content="faq", prompt_tokens=1, completion_tokens=1)
    with patch("app.agent.chat", return_value=classify_result), \
         patch("app.agent.answer_faq", return_value="We're open 8-8."):
        answer = agent.run_agent("What are your business hours?")

    assert answer == "We're open 8-8."


def test_run_agent_routes_other_intent_end_to_end():
    classify_result = ChatResult(content="other", prompt_tokens=1, completion_tokens=1)
    with patch("app.agent.chat", return_value=classify_result):
        answer = agent.run_agent("Can you help me file my taxes?")

    assert "rephrase" in answer
