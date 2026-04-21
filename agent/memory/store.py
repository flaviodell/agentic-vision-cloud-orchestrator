"""
Pinecone vector store — upsert and query operations.

Manages a single Pinecone index used for two namespaces:
    - "conversations": past agent conversation turns
    - "breeds": static breed knowledge (pre-populated at startup)

Environment variables:
    PINECONE_API_KEY    — Pinecone API key
    PINECONE_INDEX      — index name (default: "agentic-vet-memory")
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default index name — can be overridden via env var.
_DEFAULT_INDEX = "agentic-vet-memory"

# Namespace constants.
NS_CONVERSATIONS = "conversations"
NS_BREEDS = "breeds"

_index = None  # Pinecone Index singleton.


def _get_index():
    """
    Lazy singleton: connect to Pinecone and return the Index object.

    Creates the index if it does not already exist (serverless, AWS us-east-1).
    """
    global _index
    if _index is not None:
        return _index

    from pinecone import Pinecone, ServerlessSpec
    from agent.memory.embedder import get_embedding_dim

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env file to enable vector memory."
        )

    index_name = os.getenv("PINECONE_INDEX", _DEFAULT_INDEX)
    pc = Pinecone(api_key=api_key)

    # Create index only if it doesn't exist yet.
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info(f"[store] Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=get_embedding_dim(),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(f"[store] Index '{index_name}' created.")
    else:
        logger.debug(f"[store] Index '{index_name}' already exists.")

    _index = pc.Index(index_name)
    return _index


def upsert_vector(
    vector: List[float],
    metadata: Dict[str, Any],
    namespace: str = NS_CONVERSATIONS,
    doc_id: Optional[str] = None,
) -> str:
    """
    Insert or update a single vector in Pinecone.

    Args:
        vector:    The embedding vector.
        metadata:  Arbitrary dict attached to the vector (text, breed, timestamp, etc.).
        namespace: Pinecone namespace ("conversations" or "breeds").
        doc_id:    Optional ID; auto-generated (UUID4) if not provided.

    Returns:
        The ID of the upserted vector.
    """
    doc_id = doc_id or str(uuid.uuid4())
    index = _get_index()

    index.upsert(
        vectors=[{"id": doc_id, "values": vector, "metadata": metadata}],
        namespace=namespace,
    )
    logger.debug(f"[store] Upserted vector id={doc_id} ns={namespace}")
    return doc_id


def upsert_batch(
    vectors: List[List[float]],
    metadatas: List[Dict[str, Any]],
    namespace: str = NS_CONVERSATIONS,
    doc_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Insert or update multiple vectors in a single Pinecone call.

    Args:
        vectors:   List of embedding vectors.
        metadatas: List of metadata dicts (same length as vectors).
        namespace: Pinecone namespace.
        doc_ids:   Optional IDs list; auto-generated if not provided.

    Returns:
        List of IDs for the upserted vectors.
    """
    if not vectors:
        return []

    if doc_ids is None:
        doc_ids = [str(uuid.uuid4()) for _ in vectors]

    records = [
        {"id": doc_id, "values": vec, "metadata": meta}
        for doc_id, vec, meta in zip(doc_ids, vectors, metadatas)
    ]

    index = _get_index()
    index.upsert(vectors=records, namespace=namespace)
    logger.debug(f"[store] Batch upserted {len(records)} vectors ns={namespace}")
    return doc_ids


def query_similar(
    vector: List[float],
    top_k: int = 5,
    namespace: str = NS_CONVERSATIONS,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Find the top-k most similar vectors in Pinecone.

    Args:
        vector:    Query embedding vector.
        top_k:     Number of results to return.
        namespace: Pinecone namespace to search.
        filter:    Optional metadata filter dict (Pinecone filter syntax).

    Returns:
        List of dicts, each with: id (str), score (float), metadata (dict).
        Sorted by descending similarity score.
    """
    index = _get_index()

    kwargs = {
        "vector": vector,
        "top_k": top_k,
        "namespace": namespace,
        "include_metadata": True,
    }
    if filter:
        kwargs["filter"] = filter

    response = index.query(**kwargs)

    results = [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata or {},
        }
        for match in response.matches
    ]

    top_score = f"{results[0]['score']:.4f}" if results else "N/A"
    logger.debug(
        f"[store] Query returned {len(results)} matches "
        f"(top score: {top_score})"
    )
    return results


def index_stats() -> Dict[str, Any]:
    """
    Return index statistics (total vectors, namespaces, dimension).
    Useful for debugging and the monitoring dashboard.
    """
    try:
        return _get_index().describe_index_stats()
    except Exception as e:
        logger.warning(f"[store] Could not get index stats: {e}")
        return {}
