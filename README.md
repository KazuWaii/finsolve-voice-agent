# FinSolve Voice Agent

A voice AI customer support agent for a fictional fintech (FinSolve Technologies): speech in, RAG-grounded answer out, speech back — built around [ElevenLabs Conversational AI](https://elevenlabs.io/) for the speech-to-text / text-to-speech / call orchestration layer, with a custom [LangGraph](https://www.langchain.com/langgraph) agent making the actual decisions behind an OpenAI-compatible "Custom LLM" endpoint.

This is a standalone project, independent from [ds-rpc-01](https://github.com/KazuWaii/ds-rpc-01) (the RAG/RBAC chatbot) — no shared code or deployment, just the same author and, for consistency, the same fictional company.

## Architecture

```
Caller's voice (browser widget mic)
  -> ElevenLabs Conversational AI (STT, turn-taking, TTS)
      -> POST /v1/chat/completions (SSE, OpenAI-compatible "Custom LLM")
          -> LangGraph agent (app/agent.py):
               classify_intent -> one of:
                 - handle_faq              -- RAG over resources/data/faq.md
                 - handle_transaction      -- regex-extracted TXN id, lookup in transactions.csv
                 - handle_callback         -- LLM-extracted name/phone/time, mock booking (in-memory)
                 - handle_fallback         -- out-of-scope redirect
          -> answer streamed back as a single SSE chunk + [DONE]
  -> ElevenLabs speaks the agent's reply back to the caller
```

Unlike ElevenLabs' built-in tool-calling, the agent's routing and tool logic live entirely in our own LangGraph graph -- ElevenLabs only handles speech and forwards the conversation to `/v1/chat/completions` as if it were talking to any OpenAI-compatible model.

| File | Role |
|---|---|
| `app/main.py` | FastAPI app: `/v1/chat/completions` (SSE, Custom LLM contract), `/health` |
| `app/agent.py` | LangGraph `StateGraph`: intent classification + 4 handler nodes, `run_agent()` entry point |
| `app/ingest.py` | Chunks `resources/data/faq.md` (header-aware, same technique as ds-rpc-01) |
| `app/vectorstore.py` | Embeds chunks into Chroma (no RBAC filtering needed -- single-tier public KB) |
| `app/rag.py` | Retrieve -> augment -> generate, with a system prompt tuned for speech (no markdown/bullets) |
| `app/llm.py` | Pluggable LLM backend: Ollama (local dev) or Groq (deployed) |
| `app/tools.py` | Plain Python functions: transaction lookup, callback scheduling (in-memory) |

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

Try the agent directly, without ElevenLabs in the loop:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the status of transaction TXN10005?"}], "stream": true}'
```

## Deployment

Built as a Docker image and deployed to Azure Container Apps with `--min-replicas 0` (scale-to-zero -- no cost while idle):

```bash
docker build -t finsolve-voice-agent:latest .
docker push <your-dockerhub-user>/finsolve-voice-agent:latest

az containerapp create \
  --name finsolve-voice-agent-api \
  --resource-group finsolve-voice-agent-rg \
  --environment finsolve-voice-agent-env \
  --image <your-dockerhub-user>/finsolve-voice-agent:latest \
  --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 1 \
  --secrets groq-api-key=<your-groq-key> \
  --env-vars GROQ_API_KEY=secretref:groq-api-key LLM_PROVIDER=groq
```

The embedding model is pre-downloaded and the FAQ index is pre-built at *image build time*, so the container serves immediately once started, without needing network access to Hugging Face.

**Cold-start caveat:** scaling from zero takes roughly 48-80s (Azure allocating a replica + starting FastAPI), but ElevenLabs' Custom LLM integration times out a call after ~13-15s. The very first message of a conversation after the container has been idle can therefore fail. Workaround: hit `GET /health` to warm the container a few seconds before starting a live demo. The alternative -- `--min-replicas 1` -- removes the cold start entirely but keeps the container (and its cost) running continuously.

## Connecting ElevenLabs

1. Create a free [ElevenLabs](https://elevenlabs.io/) account, open **Agents**, and create a blank agent.
2. Under **Agent -> LLM**, select **Custom LLM** as the provider:
   - **Server URL**: `https://<your-deployed-host>/v1` (ElevenLabs appends `/chat/completions`)
   - **Model ID**: any string (the backend ignores it) -- e.g. `finsolve-agent`
   - **API Key**: not required, this endpoint has no auth
3. Publish the agent.
4. Test it from the dashboard's built-in chat simulator (text, no microphone needed) before going further.
5. Embed the widget on a page for a browser-based voice demo -- no telephony/Twilio needed. ElevenLabs provides the snippet under **Channels -> Widget**:
   ```html
   <elevenlabs-convai agent-id="your-agent-id"></elevenlabs-convai>
   <script src="https://unpkg.com/@elevenlabs/convai-widget-embed" async type="text/javascript"></script>
   ```
   See [demo.html](demo.html) for a minimal working page.
