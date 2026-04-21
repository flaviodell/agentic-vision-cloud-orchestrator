"""
Embedder — wraps OpenAI text-embedding-3-small to produce dense vectors.

Used by the memory store to convert text (conversation turns, breed facts)
into vectors for semantic similarity search in Pinecone.

Environment variable:
    OPENAI_API_KEY  (shared with the LLM in nodes.py)
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Model and dimensionality — text-embedding-3-small is cheap and fast.
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536

_client = None


def _get_client():
    """Lazy singleton for the OpenAI client."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_text(text: str) -> List[float]:
    """
    Embed a single string and return a 1536-dim float vector.

    Args:
        text: The text to embed (max ~8192 tokens).

    Returns:
        List of 1536 floats representing the semantic vector.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If the OpenAI API call fails.
    """
    text = text.strip()
    if not text:
        raise ValueError("embed_text: input text is empty.")

    client = _get_client()

    try:
        response = client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=text,
        )
        vector = response.data[0].embedding
        logger.debug(f"[embedder] Embedded {len(text)} chars → {len(vector)}-dim vector.")
        return vector
    except Exception as e:
        logger.error(f"[embedder] OpenAI embedding error: {e}")
        raise RuntimeError(f"Embedding failed: {e}") from e


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple strings in a single API call (more efficient than looping).

    Args:
        texts: List of strings to embed.

    Returns:
        List of 1536-dim float vectors, in the same order as input.
    """
    if not texts:
        return []

    client = _get_client()

    try:
        response = client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=texts,
        )
        # OpenAI returns results sorted by index, so order is preserved.
        vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        logger.debug(f"[embedder] Batch embedded {len(texts)} texts.")
        return vectors
    except Exception as e:
        logger.error(f"[embedder] Batch embedding error: {e}")
        raise RuntimeError(f"Batch embedding failed: {e}") from e


def get_embedding_dim() -> int:
    """Return the fixed dimensionality of the embedding model."""
    return _EMBEDDING_DIM
