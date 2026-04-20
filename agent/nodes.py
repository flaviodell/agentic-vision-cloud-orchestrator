"""
LangGraph node functions for the agentic system.

Each node receives the full AgentState and returns a dict with the fields
it wants to update. LangGraph merges these partial updates into the state.
"""

import logging
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent.state import AgentState

logger = logging.getLogger(__name__)

# Max turns before the agent is forced to stop (safety guard).
MAX_TURNS = 10

# ---------------------------------------------------------------------------
# LLM — lazy singleton, created on first use so tests can import this
# module without OPENAI_API_KEY being set at import time.
# ---------------------------------------------------------------------------
_base_llm = None


def _get_base_llm():
    """Return the shared ChatOpenAI instance, creating it on first call."""
    global _base_llm
    if _base_llm is None:
        from langchain_openai import ChatOpenAI
        _base_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1024,
        )
    return _base_llm


def get_llm(tools: list = []):
    """Return the LLM, optionally with tools bound."""
    llm = _get_base_llm()
    if tools:
        return llm.bind_tools(tools)
    return llm


# ---------------------------------------------------------------------------
# Node: agent
# ---------------------------------------------------------------------------
def make_agent_node(tools: list = []):
    """
    Factory that returns an agent_node closure with tools already bound.
    This pattern avoids global state and makes the graph easy to reconfigure.
    """

    def agent_node(state: AgentState) -> dict:
        """
        Core reasoning node.

        Calls the LLM with the current message history. The model either:
        - Returns a plain text response -> graph routes to END.
        - Returns tool_calls in the AIMessage -> graph routes to 'tools'.
        """
        messages: List[BaseMessage] = state["messages"]
        turn = state["turn_count"]

        logger.info(f"[agent_node] turn={turn}, messages={len(messages)}")

        # Hard stop if we have exceeded the max turn budget.
        if turn >= MAX_TURNS:
            logger.warning("[agent_node] MAX_TURNS reached — forcing stop.")
            stop_msg = AIMessage(
                content=(
                    "I have reached the maximum number of reasoning steps. "
                    "Please rephrase your request or start a new conversation."
                )
            )
            return {
                "messages": [stop_msg],
                "turn_count": turn + 1,
            }

        llm = get_llm(tools)
        response: AIMessage = llm.invoke(messages)
        logger.info(
            f"[agent_node] response type={'tool_call' if response.tool_calls else 'text'}"
        )

        breed = state.get("breed_identified")

        return {
            "messages": [response],
            "turn_count": turn + 1,
            "breed_identified": breed,
        }

    return agent_node


# ---------------------------------------------------------------------------
# Edge: should_continue (conditional routing function)
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    """
    Routing function used as a conditional edge.

    Returns:
        "tools"  -> if the last AI message contains tool_calls.
        "end"    -> otherwise (plain text answer, conversation complete).
    """
    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        logger.info("[should_continue] → tools")
        return "tools"

    logger.info("[should_continue] → end")
    return "end"


# ---------------------------------------------------------------------------
# Post-tool hook: extract structured info from ToolMessage results
# ---------------------------------------------------------------------------
def extract_breed_from_tool_messages(state: AgentState) -> dict:
    """
    Called after ToolNode executes. Scans the latest ToolMessages for a
    breed prediction and persists it in the state.

    This node sits between 'tools' and 'agent' in the graph.
    """
    import json

    breed = state.get("breed_identified")
    last_tool_result = state.get("last_tool_result")

    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        try:
            data = json.loads(msg.content)
            if "breed" in data:
                breed = data["breed"]
                last_tool_result = data
                logger.info(f"[extract_breed] identified breed: {breed}")
                break
        except (json.JSONDecodeError, TypeError):
            continue

    return {
        "breed_identified": breed,
        "last_tool_result": last_tool_result,
    }
