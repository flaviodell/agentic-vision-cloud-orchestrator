"""
Unit tests - LangGraph agent (no tools, no real LLM calls).

All LLM calls are mocked so these tests run offline without API keys.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.state import AgentState
from agent.nodes import should_continue, extract_breed_from_tool_messages, MAX_TURNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**kwargs) -> AgentState:
    """Build a minimal AgentState with sensible defaults."""
    defaults = {
        "messages": [],
        "turn_count": 0,
        "last_tool_result": None,
        "breed_identified": None,
    }
    defaults.update(kwargs)
    return defaults  # type: ignore


def make_mock_llm(reply_content: str = "Hello from AI!") -> MagicMock:
    """Return a mock that behaves like a bound ChatOpenAI."""
    mock = MagicMock()
    ai_msg = AIMessage(content=reply_content)
    ai_msg.tool_calls = []
    mock.invoke.return_value = ai_msg
    mock.bind_tools.return_value = mock
    return mock


# ---------------------------------------------------------------------------
# Tests: should_continue
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_routes_to_end_on_plain_text(self):
        state = make_state(messages=[AIMessage(content="Hello there!")])
        assert should_continue(state) == "end"

    def test_routes_to_tools_when_tool_calls_present(self):
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "cv_predict", "args": {}, "id": "123"}]
        state = make_state(messages=[ai_msg])
        assert should_continue(state) == "tools"

    def test_routes_to_end_when_tool_calls_empty_list(self):
        ai_msg = AIMessage(content="Done.")
        ai_msg.tool_calls = []
        state = make_state(messages=[ai_msg])
        assert should_continue(state) == "end"


# ---------------------------------------------------------------------------
# Tests: extract_breed_from_tool_messages
# ---------------------------------------------------------------------------

class TestExtractBreed:
    def test_extracts_breed_from_tool_message(self):
        payload = json.dumps({"breed": "Siamese", "confidence": 0.95, "top5": []})
        tool_msg = ToolMessage(content=payload, tool_call_id="abc")
        state = make_state(messages=[tool_msg])
        result = extract_breed_from_tool_messages(state)
        assert result["breed_identified"] == "Siamese"
        assert result["last_tool_result"]["confidence"] == 0.95

    def test_no_breed_key_leaves_breed_unchanged(self):
        payload = json.dumps({"results": ["something"], "count": 1})
        tool_msg = ToolMessage(content=payload, tool_call_id="abc")
        state = make_state(messages=[tool_msg], breed_identified="Beagle")
        result = extract_breed_from_tool_messages(state)
        assert result["breed_identified"] == "Beagle"

    def test_invalid_json_is_handled_gracefully(self):
        tool_msg = ToolMessage(content="not json at all", tool_call_id="abc")
        state = make_state(messages=[tool_msg])
        result = extract_breed_from_tool_messages(state)
        assert result["breed_identified"] is None


# ---------------------------------------------------------------------------
# Tests: agent_node (mocked LLM via patch on module-level _get_base_llm)
# ---------------------------------------------------------------------------

class TestAgentNode:
    def test_agent_node_increments_turn_count(self):
        import agent.nodes as nodes_module
        mock_llm = make_mock_llm("Hi!")
        original = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm
        try:
            from agent.nodes import make_agent_node
            node = make_agent_node([])
            state = make_state(
                messages=[HumanMessage(content="Hello")],
                turn_count=2,
            )
            result = node(state)
            assert result["turn_count"] == 3
        finally:
            nodes_module._get_base_llm = original

    def test_agent_node_stops_at_max_turns(self):
        import agent.nodes as nodes_module
        mock_llm = make_mock_llm()
        original = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm
        try:
            from agent.nodes import make_agent_node
            node = make_agent_node([])
            state = make_state(
                messages=[HumanMessage(content="loop?")],
                turn_count=MAX_TURNS,
            )
            result = node(state)
            mock_llm.invoke.assert_not_called()
            assert "maximum" in result["messages"][0].content.lower()
        finally:
            nodes_module._get_base_llm = original


# ---------------------------------------------------------------------------
# Tests: build_graph (smoke — no real invocation)
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_graph_compiles_without_tools(self):
        from agent.graph import build_graph
        graph = build_graph([])
        assert graph is not None

    def test_graph_has_invoke_method(self):
        from agent.graph import build_graph
        graph = build_graph([])
        assert callable(getattr(graph, "invoke", None))


# ---------------------------------------------------------------------------
# Tests: AgentSession (mocked LLM)
# ---------------------------------------------------------------------------

class TestAgentSession:
    def test_session_accumulates_history(self):
        import agent.nodes as nodes_module
        mock_llm = make_mock_llm("Hello from AI!")
        original = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm
        try:
            from agent.runner import AgentSession
            session = AgentSession()
            session.chat("Hi", verbose=False)
            # SystemMessage + HumanMessage + AIMessage = 3
            assert len(session.history) == 3
        finally:
            nodes_module._get_base_llm = original

    def test_session_reset_clears_history(self):
        import agent.nodes as nodes_module
        mock_llm = make_mock_llm("reply")
        original = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm
        try:
            from agent.runner import AgentSession
            session = AgentSession()
            session.chat("msg", verbose=False)
            session.reset()
            assert session.history == []
            assert session.breed_identified is None
        finally:
            nodes_module._get_base_llm = original

    def test_second_turn_has_no_duplicate_system_prompt(self):
        import agent.nodes as nodes_module
        mock_llm = make_mock_llm("reply")
        original = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm
        try:
            from agent.runner import AgentSession
            session = AgentSession()
            session.chat("first", verbose=False)
            session.chat("second", verbose=False)
            system_msgs = [m for m in session.history if isinstance(m, SystemMessage)]
            assert len(system_msgs) == 1
        finally:
            nodes_module._get_base_llm = original
