from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import answer_faq

app = FastAPI(title="FinSolve Voice Agent Tools")

TRANSACTIONS_PATH = Path(__file__).resolve().parents[1] / "resources" / "data" / "transactions.csv"
_transactions_df = pd.read_csv(TRANSACTIONS_PATH)

_scheduled_callbacks = []


class FaqRequest(BaseModel):
    question: str

class FaqResponse(BaseModel):
    answer: str

class TransactionStatusRequest(BaseModel):
    transaction_id: str

class TransactionStatusResponse(BaseModel):
    found: bool
    status: str | None = None
    amount: float | None = None
    date: str | None = None
    description: str | None = None

class ScheduleCallbackRequest(BaseModel):
    name: str
    phone_number: str
    preferred_time: str

class ScheduleCallbackResponse(BaseModel):
    confirmation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/faq", response_model=FaqResponse)
def faq(request: FaqRequest):
    return FaqResponse(answer=answer_faq(request.question))


@app.post("/transaction-status", response_model=TransactionStatusResponse)
def transaction_status(request: TransactionStatusRequest):
    match = _transactions_df[_transactions_df["transaction_id"] == request.transaction_id.upper()]
    if match.empty:
        return TransactionStatusResponse(found=False)

    row = match.iloc[0]
    return TransactionStatusResponse(
        found=True,
        status=row["status"],
        amount=float(row["amount"]),
        date=row["date"],
        description=row["description"],
    )


@app.post("/schedule-callback", response_model=ScheduleCallbackResponse)
def schedule_callback(request: ScheduleCallbackRequest):
    _scheduled_callbacks.append(request.model_dump())
    return ScheduleCallbackResponse(
        confirmation=f"Thanks {request.name}, we've scheduled a callback to {request.phone_number} at {request.preferred_time}."
    )