from fastapi import FastAPI
from pydantic import BaseModel

import json, time, uuid

from fastapi.responses import StreamingResponse

from app.rag import answer_faq
from app.tools import lookup_transaction, schedule_callback as schedule_callback_tool
from app.agent import run_agent

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="FinSolve Voice Agent Tools")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kazuwaii.github.io"],
    allow_methods=["GET"],
)


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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    stream: bool | None = None


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


def _sse_chunk(content=None, finish_reason=None):
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "finsolve-agent",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content is not None else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    message = next(m.content for m in reversed(request.messages) if m.role == "user")
    answer = run_agent(message)

    def event_stream():
        yield _sse_chunk(content=answer)
        yield _sse_chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")