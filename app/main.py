from fastapi import FastAPI
from pydantic import BaseModel

import json, time, uuid

from fastapi.responses import StreamingResponse

from app.agent import run_agent

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="FinSolve Voice Agent Tools")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kazuwaii.github.io"],
    allow_methods=["GET"],
)


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