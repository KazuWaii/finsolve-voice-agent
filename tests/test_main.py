from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_completions_streams_sse_with_run_agent_answer():
    with patch("app.main.run_agent", return_value="We're open 8-8.") as mock_run_agent:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "business hours?"}], "stream": True},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    mock_run_agent.assert_called_once_with("business hours?")

    body = resp.text
    assert "We're open 8-8." in body
    assert '"finish_reason": "stop"' in body
    assert body.strip().endswith("data: [DONE]")


def test_chat_completions_picks_latest_user_message_from_history():
    with patch("app.main.run_agent", return_value="answer") as mock_run_agent:
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "assistant", "content": "Hi, how can I help?"},
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "some reply"},
                    {"role": "user", "content": "second question"},
                ],
                "stream": True,
            },
        )

    mock_run_agent.assert_called_once_with("second question")


def test_chat_completions_rejects_malformed_body():
    resp = client.post("/v1/chat/completions", json={"not_messages": "oops"})

    assert resp.status_code == 422
