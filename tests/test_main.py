from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_faq_endpoint_calls_rag():
    with patch("app.main.answer_faq", return_value="a spoken answer") as mock_answer:
        resp = client.post("/faq", json={"question": "What are your hours?"})

    assert resp.status_code == 200
    assert resp.json() == {"answer": "a spoken answer"}
    mock_answer.assert_called_once_with("What are your hours?")


def test_transaction_status_found():
    resp = client.post("/transaction-status", json={"transaction_id": "TXN10005"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["status"] == "completed"


def test_transaction_status_is_case_insensitive():
    resp = client.post("/transaction-status", json={"transaction_id": "txn10005"})
    assert resp.json()["found"] is True


def test_transaction_status_not_found():
    resp = client.post("/transaction-status", json={"transaction_id": "TXN00000"})
    assert resp.status_code == 200
    assert resp.json() == {"found": False, "status": None, "amount": None, "date": None, "description": None}


def test_schedule_callback():
    resp = client.post(
        "/schedule-callback",
        json={"name": "Alex", "phone_number": "555-0100", "preferred_time": "tomorrow at 2pm"},
    )
    assert resp.status_code == 200
    assert "Alex" in resp.json()["confirmation"]
    assert "555-0100" in resp.json()["confirmation"]


def test_faq_rejects_malformed_body():
    resp = client.post("/faq", json={"not_question": "oops"})
    assert resp.status_code == 422
