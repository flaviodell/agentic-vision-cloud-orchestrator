"""
Agent tools — public exports.

Usage:
    from agent.tools import ALL_TOOLS
    graph = build_graph(tools=ALL_TOOLS)
"""

from agent.tools.cv_tool import cv_predict
from agent.tools.db_tool import db_query
from agent.tools.memory_tool import breed_semantic_search, memory_search
from agent.tools.search_tool import web_search

# Full tool list passed to build_graph() in production.
ALL_TOOLS = [cv_predict, web_search, db_query, memory_search, breed_semantic_search]

__all__ = [
    "cv_predict",
    "web_search",
    "db_query",
    "memory_search",
    "breed_semantic_search",
    "ALL_TOOLS",
]
