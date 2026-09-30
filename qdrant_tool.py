"""Qdrant-backed PDF knowledge base for the Agentic RAG crew.

Responsibilities:
  1. Extract text from an uploaded PDF with ``pdfplumber``.
  2. Chunk that text and embed it with OpenAI ``text-embedding-3-large`` (3072 dims).
  3. Create/refresh a Qdrant collection and upsert the resulting vectors.
  4. Expose ``QdrantSearchTool`` -- a CrewAI tool the DB Search Agent uses to run
     semantic search against that collection.
"""

from __future__ import annotations

import uuid
from typing import Any

import pdfplumber
from crewai.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 64


# --------------------------------------------------------------------------- #
# PDF extraction + chunking
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(pdf_file: Any) -> list[dict[str, Any]]:
    """Extract per-page text from a PDF file object or path.

    Returns a list of ``{"text": ..., "page": ...}`` for every page that
    actually contained extractable text.
    """
    pages: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"text": text, "page": page_number})
    return pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring to break on whitespace."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Back off to the last space so we do not split a word in half.
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(pages: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Turn extracted pages into flat, page-attributed chunks."""
    documents: list[dict[str, Any]] = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            documents.append({"text": chunk, "page": page["page"], "source": source})
    return documents


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed_texts(
    texts: list[str],
    openai_api_key: str,
    model: str = EMBEDDING_MODEL,
) -> list[list[float]]:
    """Embed a list of strings, batching to stay within request limits."""
    client = OpenAI(api_key=openai_api_key)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
def get_qdrant_client(qdrant_url: str, qdrant_api_key: str) -> QdrantClient:
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    recreate: bool = False,
) -> None:
    """Create the collection if missing, optionally wiping an existing one."""
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )


def load_pdf_into_qdrant(
    pdf_file: Any,
    source_name: str,
    collection_name: str,
    openai_api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    recreate: bool = True,
    progress_callback: Any = None,
) -> int:
    """Full ingestion pipeline: PDF -> chunks -> embeddings -> Qdrant.

    Returns the number of chunks stored.
    """

    def report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    report("Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_file)
    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned document that needs OCR."
        )

    documents = build_chunks(pages, source=source_name)
    report(f"Created {len(documents)} chunks from {len(pages)} pages.")

    report("Generating embeddings with OpenAI...")
    vectors = embed_texts([doc["text"] for doc in documents], openai_api_key)

    report("Writing vectors to Qdrant...")
    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    ensure_collection(client, collection_name, recreate=recreate)

    points = [
        models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=document)
        for document, vector in zip(documents, vectors)
    ]
    for i in range(0, len(points), 100):
        client.upsert(collection_name=collection_name, points=points[i : i + 100])

    report(f"Stored {len(points)} chunks in collection '{collection_name}'.")
    return len(points)


def search_qdrant(
    query: str,
    collection_name: str,
    openai_api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Embed the query and return the closest chunks from the collection."""
    query_vector = embed_texts([query], openai_api_key)[0]
    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "page": payload.get("page"),
                "source": payload.get("source"),
                "score": point.score,
            }
        )
    return results


# --------------------------------------------------------------------------- #
# CrewAI tool
# --------------------------------------------------------------------------- #
class QdrantSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The natural-language question to search the PDF knowledge base for.",
    )
    limit: int = Field(5, description="How many matching passages to return (1-10).")


class QdrantSearchTool(BaseTool):
    """Semantic search over the PDF collection stored in Qdrant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Qdrant PDF Search"
    description: str = (
        "Search the uploaded PDF's vector database for passages relevant to a "
        "question. Returns the most semantically similar excerpts, each with its "
        "page number and similarity score. Use this to ground answers in the "
        "user's own document."
    )
    args_schema: type[BaseModel] = QdrantSearchInput

    collection_name: str
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str

    def _run(self, query: str, limit: int = 5) -> str:
        try:
            limit = max(1, min(int(limit), 10))
            results = search_qdrant(
                query=query,
                collection_name=self.collection_name,
                openai_api_key=self.openai_api_key,
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                limit=limit,
            )
        except Exception as exc:  # surfaced to the agent so it can adapt
            return f"Qdrant search failed: {exc}"

        if not results:
            return "No relevant passages were found in the uploaded PDF."

        blocks = []
        for i, result in enumerate(results, start=1):
            blocks.append(
                f"[Excerpt {i} | page {result['page']} | "
                f"relevance {result['score']:.3f}]\n{result['text']}"
            )
        return "\n\n".join(blocks)
