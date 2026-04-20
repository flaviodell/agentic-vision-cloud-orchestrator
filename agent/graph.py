"""
LangGraph graph definition for the agentic system.

Graph topology (ReAct pattern):

    [START] → agent ──(has tool_calls)──→ tools → extract_breed → agent
                    └──(no tool_calls)──→ [END]

The agent node reasons; tools executes; extract_breed persists structured
data back into state; then control returns to agent for the next reasoning step.
"""

import logging

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import extract_breed_from_tool_messages, make_agent_node, should_continue
from agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph(tools: list = []):
    """
    Compile and return the LangGraph StateGraph.

    Args:
        tools: List of LangChain-compatible tool objects to bind to the LLM.
               Pass an empty list during Phase 4 (no tools yet).
               Phase 5 will pass the real tools here.

    Returns:
        A compiled LangGraph runnable (supports .invoke() and .stream()).
    """
    # --- Nodes ----------------------------------------------------------
    agent_node = make_agent_node(tools)
    tool_node = ToolNode(tools) if tools else ToolNode([])

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract_breed", extract_breed_from_tool_messages)

    # --- Entry point ----------------------------------------------------
    graph.set_entry_point("agent")

    # --- Edges ----------------------------------------------------------
    # After agent: route based on whether tool_calls are present.
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools: always extract structured data, then return to agent.
    graph.add_edge("tools", "extract_breed")
    graph.add_edge("extract_breed", "agent")

    compiled = graph.compile()
    logger.info("[build_graph] Graph compiled successfully.")
    return compiled


# Singleton compiled graph (no tools — will be replaced in Phase 5).
_graph = None


def get_graph(tools: list = []) -> object:
    """
    Return a compiled graph. Rebuilds if tools list changes.
    For Phase 4, always returns the no-tool graph singleton.
    """
    global _graph
    if _graph is None or tools:
        _graph = build_graph(tools)
    return _graph
