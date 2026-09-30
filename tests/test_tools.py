from app.tools import lookup_transaction, schedule_callback


def test_lookup_transaction_found():
    result = lookup_transaction("TXN10001")

    assert result["status"] == "completed"
    assert result["amount"] == 1250.00
    assert result["date"] == "2026-09-20"
    assert "Alice Martin" in result["description"]


def test_lookup_transaction_is_case_insensitive():
    result = lookup_transaction("txn10001")

    assert result is not None
    assert result["status"] == "completed"


def test_lookup_transaction_not_found_returns_none():
    assert lookup_transaction("TXN99999") is None


def test_schedule_callback_returns_confirmation_with_details():
    confirmation = schedule_callback("Alice", "555-1234", "tomorrow at 3pm")

    assert "Alice" in confirmation
    assert "555-1234" in confirmation
    assert "tomorrow at 3pm" in confirmation
