from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import answer_faq
from app.tools import lookup_transaction, schedule_callback as schedule_callback_tool


app = FastAPI(title="FinSolve Voice Agent Tools")


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
    result = lookup_transaction(request.transaction_id)
    if result is None:
        return TransactionStatusResponse(found=False)
    return TransactionStatusResponse(found=True, **result)


@app.post("/schedule-callback", response_model=ScheduleCallbackResponse)
def schedule_callback(request: ScheduleCallbackRequest):
    confirmation = schedule_callback_tool(request.name, request.phone_number, request.preferred_time)
    return ScheduleCallbackResponse(confirmation=confirmation)