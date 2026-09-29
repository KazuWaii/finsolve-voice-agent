"""Pluggable LLM backend: Ollama for local dev, Groq for the deployed
webhook the voice agent calls (Ollama isn't reachable from a container).
Same design as ds-rpc-01/app/services/llm.py -- see that project's README
for the reasoning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ollama
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = "llama3.2"
GROQ_MODEL = "openai/gpt-oss-20b"

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if LLM_PROVIDER == "groq" else None


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int


def chat(messages: list[dict]) -> ChatResult:
    if LLM_PROVIDER == "groq":
        response = _groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages)
        return ChatResult(
            content=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return ChatResult(
        content=response["message"]["content"],
        prompt_tokens=response.get("prompt_eval_count", 0),
        completion_tokens=response.get("eval_count", 0),
    )
