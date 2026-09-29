# Container image for the FastAPI backend that ElevenLabs' Conversational
# AI agent calls as "tools" mid-conversation. Needs a public URL -- that's
# the only reason this is deployed at all.

FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app/ app/
COPY resources/data/ resources/data/
COPY scripts/ scripts/

# Pre-download the embedding model and pre-build the FAQ index at build
# time, so the container serves immediately instead of doing this work
# (and needing network access to Hugging Face) on the first request.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN uv run python scripts/build_index.py

# Ollama isn't reachable from inside this container -- Groq is the cloud
# LLM backend. GROQ_API_KEY must be supplied at run time, never baked in.
ENV LLM_PROVIDER=groq

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
