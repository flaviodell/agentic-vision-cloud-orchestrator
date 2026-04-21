"""
agent.memory — semantic memory layer backed by Pinecone.

Public interface:
    from agent.memory import save_turn, retrieve_context, search_breeds, populate_breeds
"""

from agent.memory.manager import (
    populate_breed_knowledge as populate_breeds,
    retrieve_relevant_context as retrieve_context,
    save_conversation_turn as save_turn,
    search_breed_knowledge as search_breeds,
)

__all__ = [
    "save_turn",
    "retrieve_context",
    "search_breeds",
    "populate_breeds",
]
