"""
Agent state definition for LangGraph.

The state is passed between nodes at every step of the graph.
LangGraph merges partial updates returned by each node into the full state.
"""

import operator
from typing import Annotated, List, Optional
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Full conversation history.
    # Annotated with operator.add so LangGraph APPENDS new messages
    # instead of overwriting the list — required for multi-turn memory.
    messages: Annotated[List[BaseMessage], operator.add]

    # Safety counter: incremented at every agent node call.
    # Prevents runaway loops (checked in should_continue).
    turn_count: int

    # Raw result of the last tool execution (dict or None).
    # Used for routing logic and downstream nodes.
    last_tool_result: Optional[dict]

    # Breed string set when the CV tool returns a successful prediction.
    # Persisted across turns so the agent remembers what it identified.
    breed_identified: Optional[str]
