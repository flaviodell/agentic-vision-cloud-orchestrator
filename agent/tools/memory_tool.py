"""
Memory search tool — semantic retrieval from past conversations and breed knowledge.

Exposes two LangChain tools to the agent:

1. memory_search(query)
   → searches past conversation turns for semantically similar content.

2. breed_semantic_search(query)
   → searches the breed knowledge base by meaning, not just exact name.
      Useful for queries like "which breed is good with children?" or
      "hypoallergenic cat breed".

Environment variables required:
    PINECONE_API_KEY
    OPENAI_API_KEY
"""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def memory_search(query: str) -> str:
    """
    Search past conversation history for information relevant to the current query.

    Use this tool when the user refers to something discussed earlier
    (e.g. "What did you say about that breed before?" or "Remind me of the
    last breed we identified."). Returns the most semantically similar
    past conversation turns.

    Args:
        query: The topic or question to search for in past conversations.

    Returns:
        JSON string with a list of relevant past messages, each containing:
        role (str), text (str), score (float), breed (str or null).
        Returns an "error" field if retrieval fails.
    """
    logger.info(f"[memory_search] Query: {query!r}")

    try:
        from agent.memory.manager import retrieve_relevant_context
        results = retrieve_relevant_context(query=query, top_k=5)

        if not results:
            return json.dumps({
                "message": "No relevant past conversations found.",
                "results": [],
            })

        return json.dumps({"results": results})

    except Exception as e:
        logger.error(f"[memory_search] Error: {e}")
        return json.dumps({"error": str(e)})


@tool
def breed_semantic_search(query: str) -> str:
    """
    Search the breed knowledge base by meaning, not just exact breed name.

    Use this tool when the user asks a general question about breed
    characteristics without naming a specific breed — for example:
    "Which breed is best for families with children?",
    "Recommend a low-energy cat", or "Tell me about a large guard dog".

    This performs semantic similarity search over all 37 Oxford Pets breeds.

    Args:
        query: A natural language description of the desired breed characteristics.

    Returns:
        JSON string with the top matching breeds, each containing:
        breed (str), type (str), text (str), score (float).
        Returns an "error" field if the search fails.
    """
    logger.info(f"[breed_semantic_search] Query: {query!r}")

    try:
        from agent.memory.manager import search_breed_knowledge
        results = search_breed_knowledge(query=query, top_k=3)

        if not results:
            return json.dumps({
                "message": "No breed matches found. Try a more specific description.",
                "results": [],
            })

        return json.dumps({"results": results})

    except Exception as e:
        logger.error(f"[breed_semantic_search] Error: {e}")
        return json.dumps({"error": str(e)})
