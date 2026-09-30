# Container image for the Agentic RAG app.
# Works on Hugging Face Spaces, Render, Railway, Fly.io, Cloud Run, or locally.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Build deps for the PDF/crypto wheels, removed again in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Hugging Face Spaces expects 7860; most other platforms inject $PORT.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/_stcore/health" || exit 1

CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT} --server.address=0.0.0.0"]
