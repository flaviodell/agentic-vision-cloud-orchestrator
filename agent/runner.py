"""
Agent runner — entry point for single-turn and multi-turn conversations.

Usage (single turn):
    from agent.runner import run_agent
    result = run_agent("What breed is in this image?")
    print(result["messages"][-1].content)

Usage (multi-turn, preserving history):
    from agent.runner import AgentSession
    session = AgentSession()
    session.chat("Tell me about Siamese cats.")
    session.chat("What about their health issues?")
"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.graph import build_graph
from agent.state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

# System prompt — describes the agent's role and available capabilities.
SYSTEM_PROMPT = """You are an expert veterinary AI assistant specialized in cat and dog breeds.

You have access to tools that can:
- Identify the breed of a pet from an image URL (CV tool)
- Search the web for up-to-date veterinary information
- Query a knowledge base of breed-specific health and care data

When the user provides an image URL containing a pet, use the breed identification tool first,
then provide detailed expert information about that breed.

Always be concise, accurate, and cite your reasoning when using tools.
If you are not confident, say so — do not hallucinate breed names or medical facts."""


def run_agent(
    user_input: str,
    history: Optional[List[BaseMessage]] = None,
    tools: list = [],
    include_system_prompt: bool = True,
) -> AgentState:
    """
    Run the agent for a single user turn.

    Args:
        user_input:           The user's message as a plain string.
        history:              Previous messages (for multi-turn sessions).
                              Pass [] or None for a fresh conversation.
        tools:                LangChain tool objects to make available.
        include_system_prompt: Prepend the system prompt if True and history is empty.

    Returns:
        The final AgentState after the graph has finished.
        Access the last AI response via: result["messages"][-1].content
    """
    graph = build_graph(tools)

    messages: List[BaseMessage] = []

    # Inject system prompt only at the start of a new conversation.
    if include_system_prompt and not history:
        messages.append(SystemMessage(content=SYSTEM_PROMPT))

    if history:
        messages.extend(history)

    messages.append(HumanMessage(content=user_input))

    initial_state: AgentState = {
        "messages": messages,
        "turn_count": 0,
        "last_tool_result": None,
        "breed_identified": None,
    }

    logger.info(f"[run_agent] Starting graph. Input: {user_input[:80]}...")
    final_state = graph.invoke(initial_state)
    logger.info("[run_agent] Graph finished.")
    return final_state


class AgentSession:
    """
    Stateful wrapper for multi-turn conversations.

    Keeps message history between calls so the agent remembers
    what was said earlier in the session.

    Example:
        session = AgentSession()
        session.chat("Identify the breed in https://example.com/cat.jpg")
        session.chat("What are the common health issues for that breed?")
    """

    def __init__(self, tools: list = []):
        self.tools = tools
        self.history: List[BaseMessage] = []
        self.breed_identified: Optional[str] = None
        logger.info("[AgentSession] New session started.")

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """
        Send a message, run the agent, update history, return the AI reply.

        Args:
            user_input: The user's message.
            verbose:    If True, print the AI response to stdout.

        Returns:
            The AI's response as a plain string.
        """
        result = run_agent(
            user_input=user_input,
            history=self.history,
            tools=self.tools,
            # System prompt is already in history after first turn.
            include_system_prompt=(len(self.history) == 0),
        )

        # Persist full updated history for the next turn.
        self.history = result["messages"]

        # Track identified breed across turns.
        if result.get("breed_identified"):
            self.breed_identified = result["breed_identified"]

        reply = result["messages"][-1].content

        if verbose:
            print(f"\n[Agent]: {reply}\n")

        return reply

    def reset(self):
        """Clear session history and start fresh."""
        self.history = []
        self.breed_identified = None
        logger.info("[AgentSession] Session reset.")
