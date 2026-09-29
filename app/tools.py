from pathlib import Path

import pandas as pd

TRANSACTIONS_PATH = Path(__file__).resolve().parents[1] / "resources" / "data" / "transactions.csv"
_transactions_df = pd.read_csv(TRANSACTIONS_PATH)

_scheduled_callbacks = []


def lookup_transaction(transaction_id):
    match = _transactions_df[_transactions_df["transaction_id"] == transaction_id.upper()]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "status": row["status"],
        "amount": float(row["amount"]),
        "date": row["date"],
        "description": row["description"],
    }


def schedule_callback(name, phone_number, preferred_time):
    _scheduled_callbacks.append({"name": name, "phone_number": phone_number, "preferred_time": preferred_time})
    return f"Thanks {name}, we've scheduled a callback to {phone_number} at {preferred_time}."