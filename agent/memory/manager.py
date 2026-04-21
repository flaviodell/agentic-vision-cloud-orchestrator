"""
Memory manager — high-level interface for the agent's semantic memory.

Provides two main capabilities:

1. save_conversation_turn(role, text, breed, session_id)
   → embeds the text and upserts it into the "conversations" namespace.

2. retrieve_relevant_context(query, top_k, breed_filter, session_id)
   → embeds the query and returns the most similar past turns.

3. populate_breed_knowledge()
   → pre-populates the "breeds" namespace with the static breed DB,
     so the agent can do semantic search over breed facts (not just exact match).

All functions gracefully degrade (log a warning, return empty results)
if Pinecone is unavailable — so the agent keeps working without memory.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_embed(text: str) -> Optional[List[float]]:
    """Embed text, returning None on any error (graceful degradation)."""
    try:
        from agent.memory.embedder import embed_text
        return embed_text(text)
    except Exception as e:
        logger.warning(f"[memory_manager] Embedding failed: {e}")
        return None


def _safe_upsert(vector, metadata, namespace, doc_id=None):
    """Upsert vector, returning None on any error."""
    try:
        from agent.memory.store import upsert_vector
        return upsert_vector(vector, metadata, namespace, doc_id)
    except Exception as e:
        logger.warning(f"[memory_manager] Upsert failed: {e}")
        return None


def _safe_query(vector, top_k, namespace, filter=None):
    """Query Pinecone, returning [] on any error."""
    try:
        from agent.memory.store import query_similar
        return query_similar(vector, top_k, namespace, filter)
    except Exception as e:
        logger.warning(f"[memory_manager] Query failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_conversation_turn(
    role: str,
    text: str,
    session_id: str,
    breed: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Optional[str]:
    """
    Embed and store a single conversation turn in the "conversations" namespace.

    Args:
        role:       "user" or "assistant".
        text:       The message text.
        session_id: Unique ID for this conversation session (groups turns together).
        breed:      Optional breed name identified in this turn (for filtering).
        timestamp:  Unix timestamp; defaults to now.

    Returns:
        The Pinecone vector ID, or None if storage failed.
    """
    from agent.memory.store import NS_CONVERSATIONS

    text = text.strip()
    if not text:
        return None

    vector = _safe_embed(text)
    if vector is None:
        return None

    metadata: Dict[str, Any] = {
        "role": role,
        "text": text[:1000],          # Pinecone metadata has a size limit
        "session_id": session_id,
        "timestamp": timestamp or time.time(),
    }
    if breed:
        metadata["breed"] = breed.lower()

    doc_id = _safe_upsert(vector, metadata, NS_CONVERSATIONS)
    logger.info(f"[memory_manager] Saved turn: role={role}, session={session_id}, id={doc_id}")
    return doc_id


def retrieve_relevant_context(
    query: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
    breed_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve past conversation turns semantically similar to the query.

    Args:
        query:        The user's current question (used as the search vector).
        top_k:        Max number of results to return.
        session_id:   If provided, restrict results to this session only.
        breed_filter: If provided, restrict results to turns mentioning this breed.

    Returns:
        List of dicts with: role, text, score, breed (optional), timestamp.
        Empty list if nothing relevant is found or Pinecone is unavailable.
    """
    from agent.memory.store import NS_CONVERSATIONS

    vector = _safe_embed(query)
    if vector is None:
        return []

    # Build Pinecone metadata filter.
    pinecone_filter: Dict[str, Any] = {}
    if session_id:
        pinecone_filter["session_id"] = {"$eq": session_id}
    if breed_filter:
        pinecone_filter["breed"] = {"$eq": breed_filter.lower()}

    results = _safe_query(
        vector,
        top_k=top_k,
        namespace=NS_CONVERSATIONS,
        filter=pinecone_filter if pinecone_filter else None,
    )

    # Flatten metadata into the result dicts for easy consumption.
    formatted = []
    for r in results:
        meta = r.get("metadata", {})
        formatted.append({
            "role": meta.get("role", "unknown"),
            "text": meta.get("text", ""),
            "score": r.get("score", 0.0),
            "breed": meta.get("breed"),
            "timestamp": meta.get("timestamp"),
            "session_id": meta.get("session_id"),
        })

    logger.info(
        f"[memory_manager] Retrieved {len(formatted)} results "
        f"for query={query[:60]!r}"
    )
    return formatted


def populate_breed_knowledge(force: bool = False) -> int:
    """
    Pre-populate the "breeds" namespace with static breed facts from db_tool.

    Each breed is stored as a single vector whose text is a concatenation of
    all its fields. This allows semantic search over breed characteristics
    (e.g. "which breed is good with children?") without exact keyword matching.

    Args:
        force: If True, re-upsert all breeds even if they already exist.

    Returns:
        Number of breeds upserted (0 if already populated and force=False).
    """
    from agent.memory.store import NS_BREEDS, index_stats
    from agent.memory.embedder import embed_batch

    # Check if breeds namespace is already populated.
    if not force:
        stats = index_stats()
        namespaces = stats.get("namespaces", {})
        if NS_BREEDS in namespaces and namespaces[NS_BREEDS].get("vector_count", 0) > 0:
            count = namespaces[NS_BREEDS]["vector_count"]
            logger.info(f"[memory_manager] Breeds namespace already has {count} vectors. Skipping.")
            return 0

    # Import breed data from the existing static DB.
    from agent.tools.db_tool import _BREED_DB

    texts = []
    metadatas = []
    doc_ids = []

    for breed_key, data in _BREED_DB.items():
        # Build a rich text representation of the breed for semantic embedding.
        text_parts = [
            f"Breed: {breed_key.replace('_', ' ').title()}",
            f"Type: {data.get('type', '')}",
            f"Description: {data.get('description', '')}",
            f"Origin: {data.get('origin', '')}",
            f"Temperament: {data.get('temperament', '')}",
            f"Health notes: {data.get('health_notes', '')}",
            f"Lifespan: {data.get('lifespan', '')}",
            f"Size: {data.get('size', '')}",
        ]
        full_text = " | ".join(text_parts)

        texts.append(full_text)
        metadatas.append({
            "breed": breed_key,
            "type": data.get("type", ""),
            "text": full_text[:1000],
            "size": data.get("size", ""),
            "origin": data.get("origin", ""),
        })
        doc_ids.append(f"breed_{breed_key}")

    # Embed all breeds in a single batch call.
    try:
        vectors = embed_batch(texts)
    except Exception as e:
        logger.error(f"[memory_manager] Failed to embed breed knowledge: {e}")
        return 0

    # Upsert all at once.
    from agent.memory.store import upsert_batch
    upsert_batch(vectors, metadatas, namespace=NS_BREEDS, doc_ids=doc_ids)

    logger.info(f"[memory_manager] Upserted {len(texts)} breeds into '{NS_BREEDS}' namespace.")
    return len(texts)


def search_breed_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Semantic search over the breed knowledge base.

    Unlike db_query (exact match), this finds breeds that are
    semantically related to the query — e.g. "good family dog" or
    "hypoallergenic cat" even without exact breed names.

    Args:
        query:  Natural language query.
        top_k:  Max results to return.

    Returns:
        List of dicts with: breed, type, text, score.
    """
    from agent.memory.store import NS_BREEDS

    vector = _safe_embed(query)
    if vector is None:
        return []

    results = _safe_query(vector, top_k=top_k, namespace=NS_BREEDS)

    formatted = []
    for r in results:
        meta = r.get("metadata", {})
        formatted.append({
            "breed": meta.get("breed", ""),
            "type": meta.get("type", ""),
            "text": meta.get("text", ""),
            "score": r.get("score", 0.0),
        })

    return formatted
