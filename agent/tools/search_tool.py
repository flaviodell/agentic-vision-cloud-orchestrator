"""
Web search tool — DuckDuckGo text search, no API key required.

Used by the agent to retrieve up-to-date veterinary and breed information
that may not be present in the static knowledge base.

Dependency: duckduckgo-search (added to requirements.txt)
"""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maximum number of search results to return to the LLM.
_MAX_RESULTS = 5


@tool
def web_search(query: str) -> str:
    """
    Search the web for veterinary or pet breed information.

    Use this tool when the user asks about breed-specific health issues,
    care tips, dietary needs, or any topic that requires current or
    detailed information beyond what you already know.

    Args:
        query: A concise search query (e.g. "Siamese cat health problems").

    Returns:
        JSON string with a list of results, each containing:
        title (str), url (str), snippet (str).
        On error, returns a JSON string with an "error" field.
    """
    logger.info(f"[web_search] Query: {query!r}")

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=_MAX_RESULTS))

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]

        logger.info(f"[web_search] Returned {len(results)} results.")
        return json.dumps({"results": results})

    except ImportError:
        msg = "duckduckgo-search is not installed. Run: pip install duckduckgo-search"
        logger.error(f"[web_search] {msg}")
        return json.dumps({"error": msg})
    except Exception as e:
        logger.error(f"[web_search] Error: {e}")
        return json.dumps({"error": str(e)})
