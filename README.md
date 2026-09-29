# FinSolve Voice Agent

A voice AI customer support agent for a fictional fintech (FinSolve Technologies): speech in, RAG-grounded answer out, speech back — built around [ElevenLabs Conversational AI](https://elevenlabs.io/) for the speech-to-text / text-to-speech / call orchestration layer, with a small custom FastAPI backend providing the "tools" the agent calls mid-conversation.

This is a standalone project, independent from [ds-rpc-01](https://github.com/KazuWaii/ds-rpc-01) (the RAG/RBAC chatbot) — no shared code or deployment, just the same author and, for consistency, the same fictional company.

## Architecture

```
Caller's voice (browser mic)
  -> ElevenLabs Conversational AI (STT, turn-taking, TTS)
      -> LLM decides which tool to call, if any:
          - POST /faq                 -- RAG over resources/data/faq.md
          - POST /transaction-status  -- mock lookup in resources/data/transactions.csv
          - POST /schedule-callback   -- mock booking (in-memory)
      -> tool's JSON response feeds back into the conversation
  -> ElevenLabs speaks the agent's reply back to the caller
```

The three tools are a plain FastAPI backend (`app/main.py`) — no auth, since a phone caller can't log in, and none of this data is sensitive (public FAQ, a caller's own transaction, a callback request).

| File | Role |
|---|---|
| `app/main.py` | FastAPI app: `/faq`, `/transaction-status`, `/schedule-callback`, `/health` |
| `app/ingest.py` | Chunks `resources/data/faq.md` (header-aware, same technique as ds-rpc-01) |
| `app/vectorstore.py` | Embeds chunks into Chroma (no RBAC filtering needed -- single-tier public KB) |
| `app/rag.py` | Retrieve -> augment -> generate, with a system prompt tuned for speech (no markdown/bullets) |
| `app/llm.py` | Pluggable LLM backend: Ollama (local dev) or Groq (deployed) |

## Local setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Pick an LLM backend via `LLM_PROVIDER` (defaults to `ollama`):

- **Ollama**: install [Ollama](https://ollama.com/download), `ollama pull llama3.2`.
- **Groq**: get a key at [console.groq.com](https://console.groq.com/), put it in a `.env` file:
  ```
  GROQ_API_KEY=your_key_here
  ```
  and set `LLM_PROVIDER=groq`.

Build the FAQ index (rerun whenever `resources/data/faq.md` changes):

```bash
uv run python scripts/build_index.py
```

Run the API:

```bash
uv run fastapi dev app/main.py --port 8000
```

## Tests

```bash
uv run pytest
```

## Connecting ElevenLabs

1. Create a free [ElevenLabs](https://elevenlabs.io/) account and open Conversational AI -> Agents.
2. Create an agent with a system prompt describing it as FinSolve's customer support assistant.
3. Add three **Tools** (webhook/server tools), each pointing at this backend's deployed URL:
   - `answer_faq` -> `POST /faq` `{"question": "..."}`
   - `check_transaction_status` -> `POST /transaction-status` `{"transaction_id": "..."}`
   - `schedule_callback` -> `POST /schedule-callback` `{"name": "...", "phone_number": "...", "preferred_time": "..."}`
4. Embed the agent's widget on a page (ElevenLabs provides the snippet) for a browser-based voice demo -- no telephony/Twilio needed.
