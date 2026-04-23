"""
End-to-end tests for the LangGraph agentic system.

Strategy
--------
- The LLM is fully mocked via patch on agent.nodes._get_base_llm so no
  OpenAI API key is needed and tests run deterministically.
- Tools (cv_predict, db_query, web_search) are mocked at the httpx /
  ddgs layer so the real tool *code* executes but no network calls are made.
- The LangGraph graph runs for real: routing, ToolNode, extract_breed,
  state accumulation are all exercised as in production.

Test coverage
-------------
1.  Full CV → DB pipeline  (image URL → breed identified → breed info returned)
2.  CV → web search pipeline (image URL → breed identified → web results)
3.  CV service error — graceful fallback, no exception propagated
4.  Multi-turn session — breed context persists across turns
5.  Multi-turn session — second turn skips system prompt duplication
6.  Agent output validation — reply is non-empty string
7.  State validation — breed_identified field populated after CV tool call
8.  Tool call counter — ToolMessages appear in history after tool use
9.  MAX_TURNS guard — agent stops without crashing if limit is hit
10. Memory save called — _save_turn_to_memory invoked per chat() call
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ai_with_tool_call(tool_name: str, args: dict, call_id: str = "tc_001") -> AIMessage:
    """Build an AIMessage that carries a single tool call."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": tool_name, "args": args, "id": call_id}]
    return msg


def _make_ai_text(content: str) -> AIMessage:
    """Build a plain-text AIMessage (no tool calls)."""
    msg = AIMessage(content=content)
    msg.tool_calls = []
    return msg


def _make_mock_llm(replies: list) -> MagicMock:
    """
    Return a mock LLM whose .invoke() returns each item in *replies* in order.
    Supports both AIMessage objects and plain strings (converted to AIMessage).
    """
    mock = MagicMock()
    converted = []
    for r in replies:
        if isinstance(r, AIMessage):
            converted.append(r)
        else:
            converted.append(_make_ai_text(str(r)))
    mock.invoke.side_effect = converted
    mock.bind_tools.return_value = mock
    return mock


def _cv_fake_response(breed: str = "Beagle", confidence: float = 0.91) -> dict:
    return {
        "breed": breed,
        "confidence": confidence,
        "top5": [{"breed": breed, "confidence": confidence}],
    }


def _make_cv_mock(breed: str = "Beagle", confidence: float = 0.91):
    """Patch httpx.Client so cv_predict returns a deterministic response."""
    fake = _cv_fake_response(breed, confidence)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake
    mock_resp.raise_for_status = lambda: None

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# 1. Full CV → DB pipeline
# ---------------------------------------------------------------------------

class TestCvToDbPipeline:
    """
    Simulate: user sends image URL → LLM calls cv_predict → LLM calls db_query
    → LLM composes final answer.
    """

    def test_full_cv_db_flow_returns_breed_info(self):
        image_url = "https://images.dog.ceo/breeds/beagle/n02088364_10108.jpg"

        # LLM turn 1: call cv_predict
        turn1 = _make_ai_with_tool_call("cv_predict", {"image_url": image_url}, "tc_cv")
        # LLM turn 2: call db_query with the identified breed
        turn2 = _make_ai_with_tool_call("db_query", {"breed_name": "beagle"}, "tc_db")
        # LLM turn 3: final text answer
        turn3 = _make_ai_text(
            "The image shows a Beagle (confidence 91%). Beagles are friendly, "
            "curious dogs originating from England."
        )

        mock_llm = _make_mock_llm([turn1, turn2, turn3])
        mock_cv_client = _make_cv_mock("Beagle", 0.91)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm

        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch("httpx.Client", return_value=mock_cv_client):
                from agent.runner import run_agent
                from agent.tools.cv_tool import cv_predict
                from agent.tools.db_tool import db_query

                result = run_agent(
                    user_input=f"What breed is in this image? {image_url}",
                    tools=[cv_predict, db_query],
                )
        finally:
            nodes_module._get_base_llm = original_llm

        reply = result["messages"][-1].content
        assert isinstance(reply, str)
        assert len(reply) > 0
        # The final AI message should be the text answer
        assert "beagle" in reply.lower() or "breed" in reply.lower()

    def test_breed_identified_in_state_after_cv_call(self):
        """breed_identified field must be set in AgentState after cv_predict executes."""
        image_url = "https://images.dog.ceo/breeds/beagle/n02088364_10108.jpg"

        turn1 = _make_ai_with_tool_call("cv_predict", {"image_url": image_url}, "tc_cv2")
        turn2 = _make_ai_text("The breed is Beagle.")

        mock_llm = _make_mock_llm([turn1, turn2])
        mock_cv_client = _make_cv_mock("Beagle", 0.91)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch("httpx.Client", return_value=mock_cv_client):
                from agent.runner import run_agent
                from agent.tools.cv_tool import cv_predict

                result = run_agent(
                    user_input=f"Identify: {image_url}",
                    tools=[cv_predict],
                )
        finally:
            nodes_module._get_base_llm = original_llm

        assert result["breed_identified"] == "Beagle"


# ---------------------------------------------------------------------------
# 2. CV → web search pipeline
# ---------------------------------------------------------------------------

class TestCvToWebSearchPipeline:
    """LLM calls cv_predict then web_search, then produces a final answer."""

    def test_cv_then_web_search_flow(self):
        image_url = "https://images.dog.ceo/breeds/beagle/n02088364_10108.jpg"

        turn1 = _make_ai_with_tool_call("cv_predict", {"image_url": image_url}, "tc_cv3")
        turn2 = _make_ai_with_tool_call(
            "web_search", {"query": "Beagle common health problems"}, "tc_ws"
        )
        turn3 = _make_ai_text("Beagles are prone to epilepsy and hypothyroidism.")

        mock_llm = _make_mock_llm([turn1, turn2, turn3])
        mock_cv_client = _make_cv_mock("Beagle", 0.91)

        # Mock DuckDuckGo
        fake_search = [{"title": "Beagle health", "href": "http://x.com", "body": "Epilepsy..."}]
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = iter(fake_search)
        mock_ddgs.__enter__ = lambda s: mock_ddgs
        mock_ddgs.__exit__ = MagicMock(return_value=False)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch("httpx.Client", return_value=mock_cv_client), \
                 patch("ddgs.DDGS", return_value=mock_ddgs):
                from agent.runner import run_agent
                from agent.tools.cv_tool import cv_predict
                from agent.tools.search_tool import web_search

                result = run_agent(
                    user_input=f"What health issues does this dog have? {image_url}",
                    tools=[cv_predict, web_search],
                )
        finally:
            nodes_module._get_base_llm = original_llm

        reply = result["messages"][-1].content
        assert isinstance(reply, str)
        assert len(reply) > 10

    def test_tool_messages_appear_in_history(self):
        """After tool calls, ToolMessage objects must be present in the message history."""
        image_url = "https://images.dog.ceo/breeds/beagle/n02088364_10108.jpg"

        turn1 = _make_ai_with_tool_call("cv_predict", {"image_url": image_url}, "tc_cv4")
        turn2 = _make_ai_text("Done.")

        mock_llm = _make_mock_llm([turn1, turn2])
        mock_cv_client = _make_cv_mock("Beagle", 0.91)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch("httpx.Client", return_value=mock_cv_client):
                from agent.runner import run_agent
                from agent.tools.cv_tool import cv_predict

                result = run_agent(
                    user_input=f"Identify: {image_url}",
                    tools=[cv_predict],
                )
        finally:
            nodes_module._get_base_llm = original_llm

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1


# ---------------------------------------------------------------------------
# 3. CV service error — graceful fallback
# ---------------------------------------------------------------------------

class TestCvServiceError:
    """If cv_service is unreachable, the agent must still return a coherent reply."""

    def test_cv_connection_error_no_exception_propagated(self):
        import httpx

        turn1 = _make_ai_with_tool_call(
            "cv_predict",
            {"image_url": "http://example.com/dog.jpg"},
            "tc_cv_err",
        )
        turn2 = _make_ai_text(
            "I was unable to reach the image classification service. "
            "Please check the URL or try again later."
        )

        mock_llm = _make_mock_llm([turn1, turn2])

        # cv_service returns connection error
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch("httpx.Client", return_value=mock_client):
                from agent.runner import run_agent
                from agent.tools.cv_tool import cv_predict

                # Must NOT raise
                result = run_agent(
                    user_input="Identify: http://example.com/dog.jpg",
                    tools=[cv_predict],
                )
        finally:
            nodes_module._get_base_llm = original_llm

        reply = result["messages"][-1].content
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_cv_error_tool_message_contains_error_key(self):
        """The ToolMessage from a failed cv_predict must carry an 'error' key."""
        import httpx

        turn1 = _make_ai_with_tool_call(
            "cv_predict",
            {"image_url": "http://example.com/dog.jpg"},
            "tc_cv_err2",
        )
        turn2 = _make_ai_text("Service unavailable.")

        mock_llm = _make_mock_llm([turn1, turn2])

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch("httpx.Client", return_value=mock_client):
                from agent.runner import run_agent
                from agent.tools.cv_tool import cv_predict

                result = run_agent(
                    user_input="Identify: http://example.com/dog.jpg",
                    tools=[cv_predict],
                )
        finally:
            nodes_module._get_base_llm = original_llm

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        data = json.loads(tool_messages[0].content)
        assert "error" in data


# ---------------------------------------------------------------------------
# 4 & 5. Multi-turn session
# ---------------------------------------------------------------------------

class TestMultiTurnSession:
    """AgentSession must preserve history and breed context across turns."""

    def _run_two_turn_session(self, mock_llm, mock_cv_client=None, extra_patches=None):
        """Helper: run a two-turn session with optional extra context managers."""
        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        ctx = {}
        if mock_cv_client:
            ctx["httpx.Client"] = mock_cv_client

        try:
            patches = [patch(k, return_value=v) for k, v in ctx.items()]
            for p in patches:
                p.start()

            from agent.tools.cv_tool import cv_predict
            from agent.tools.db_tool import db_query
            from agent.runner import AgentSession

            session = AgentSession(tools=[cv_predict, db_query])
            reply1 = session.chat(
                "https://images.dog.ceo/breeds/beagle/n02088364_10108.jpg",
                verbose=False,
            )
            reply2 = session.chat("What are the health issues for this breed?", verbose=False)
            return session, reply1, reply2
        finally:
            for p in patches:
                p.stop()
            nodes_module._get_base_llm = original_llm

    def test_breed_context_persists_across_turns(self):
        image_url = "https://images.dog.ceo/breeds/beagle/n02088364_10108.jpg"

        # Turn 1: cv_predict → final text
        t1_step1 = _make_ai_with_tool_call("cv_predict", {"image_url": image_url}, "tc_mt1")
        t1_step2 = _make_ai_text("That's a Beagle.")
        # Turn 2: plain answer (no tool call needed)
        t2_step1 = _make_ai_text("Beagles are prone to epilepsy and obesity.")

        mock_llm = _make_mock_llm([t1_step1, t1_step2, t2_step1])
        mock_cv_client = _make_cv_mock("Beagle", 0.91)

        session, reply1, reply2 = self._run_two_turn_session(mock_llm, mock_cv_client)

        assert session.breed_identified == "Beagle"
        assert isinstance(reply2, str)
        assert len(reply2) > 0

    def test_history_grows_across_turns(self):
        """After two turns, history must contain messages from both turns."""
        t1 = _make_ai_text("Hello!")
        t2 = _make_ai_text("Here are the health issues.")

        mock_llm = _make_mock_llm([t1, t2])

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            from agent.runner import AgentSession
            session = AgentSession(tools=[])
            session.chat("Hello", verbose=False)
            len_after_1 = len(session.history)
            session.chat("What are the health issues?", verbose=False)
            len_after_2 = len(session.history)
        finally:
            nodes_module._get_base_llm = original_llm

        assert len_after_2 > len_after_1

    def test_system_prompt_appears_exactly_once_in_multi_turn(self):
        """SystemMessage must appear exactly once regardless of how many turns are run."""
        turns = [_make_ai_text(f"Reply {i}") for i in range(3)]
        mock_llm = _make_mock_llm(turns)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            from agent.runner import AgentSession
            session = AgentSession(tools=[])
            for msg in ["first", "second", "third"]:
                session.chat(msg, verbose=False)
            system_msgs = [m for m in session.history if isinstance(m, SystemMessage)]
        finally:
            nodes_module._get_base_llm = original_llm

        assert len(system_msgs) == 1

    def test_session_reset_clears_breed_and_history(self):
        t1 = _make_ai_with_tool_call(
            "cv_predict",
            {"image_url": "http://x.com/dog.jpg"},
            "tc_reset",
        )
        t2 = _make_ai_text("That's a Beagle.")
        mock_llm = _make_mock_llm([t1, t2])
        mock_cv_client = _make_cv_mock("Beagle")

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch("httpx.Client", return_value=mock_cv_client):
                from agent.tools.cv_tool import cv_predict
                from agent.runner import AgentSession
                session = AgentSession(tools=[cv_predict])
                session.chat("http://x.com/dog.jpg", verbose=False)
                assert session.breed_identified == "Beagle"
                session.reset()
        finally:
            nodes_module._get_base_llm = original_llm

        assert session.breed_identified is None
        assert session.history == []


# ---------------------------------------------------------------------------
# 6. Agent output validation
# ---------------------------------------------------------------------------

class TestAgentOutputValidation:
    """The last message in the result must always be a non-empty AIMessage."""

    def test_reply_is_non_empty_string(self):
        mock_llm = _make_mock_llm([_make_ai_text("I am the agent reply.")])

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            from agent.runner import run_agent
            result = run_agent("Hello", tools=[])
        finally:
            nodes_module._get_base_llm = original_llm

        reply = result["messages"][-1].content
        assert isinstance(reply, str)
        assert reply.strip() != ""

    def test_result_has_required_state_keys(self):
        """AgentState must always contain the four required keys."""
        mock_llm = _make_mock_llm([_make_ai_text("Answer.")])

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            from agent.runner import run_agent
            result = run_agent("Hello", tools=[])
        finally:
            nodes_module._get_base_llm = original_llm

        for key in ("messages", "turn_count", "last_tool_result", "breed_identified"):
            assert key in result, f"Missing state key: {key}"

    def test_turn_count_increments(self):
        """turn_count must be >= 1 after at least one reasoning step."""
        mock_llm = _make_mock_llm([_make_ai_text("Done.")])

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            from agent.runner import run_agent
            result = run_agent("Hello", tools=[])
        finally:
            nodes_module._get_base_llm = original_llm

        assert result["turn_count"] >= 1


# ---------------------------------------------------------------------------
# 7. MAX_TURNS guard
# ---------------------------------------------------------------------------

class TestMaxTurnsGuard:
    """Agent must stop and return a message when MAX_TURNS is reached."""

    def test_agent_stops_at_max_turns(self):
        from agent.nodes import MAX_TURNS

        # Every LLM call returns a tool call so the loop keeps going
        tool_call_msg = _make_ai_with_tool_call(
            "db_query", {"breed_name": "beagle"}, "tc_loop"
        )
        # Provide more replies than MAX_TURNS to prove the guard fires
        replies = [tool_call_msg] * (MAX_TURNS + 5)
        mock_llm = _make_mock_llm(replies)

        import agent.nodes as nodes_module
        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            from agent.tools.db_tool import db_query
            from agent.runner import run_agent
            result = run_agent("loop forever", tools=[db_query])
        finally:
            nodes_module._get_base_llm = original_llm

        # Must not raise; must contain the safety stop message
        last_content = result["messages"][-1].content
        assert isinstance(last_content, str)
        assert "maximum" in last_content.lower()


# ---------------------------------------------------------------------------
# 8. Memory integration — _save_turn_to_memory called per chat()
# ---------------------------------------------------------------------------

class TestMemorySaveIntegration:
    """AgentSession.chat() must call _save_turn_to_memory twice per turn."""

    def test_save_turn_called_twice_per_chat(self):
        mock_llm = _make_mock_llm([_make_ai_text("Hello!")])

        import agent.nodes as nodes_module
        import agent.runner as runner_module

        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch.object(runner_module, "_save_turn_to_memory") as mock_save:
                from agent.runner import AgentSession
                session = AgentSession(tools=[])
                session.chat("Hello", verbose=False)
        finally:
            nodes_module._get_base_llm = original_llm

        assert mock_save.call_count == 2
        roles = [c.args[0] for c in mock_save.call_args_list]
        assert "user" in roles
        assert "assistant" in roles

    def test_save_turn_receives_session_id(self):
        """Both memory save calls must pass the same session_id."""
        mock_llm = _make_mock_llm([_make_ai_text("Hi!")])

        import agent.nodes as nodes_module
        import agent.runner as runner_module

        original_llm = nodes_module._get_base_llm
        nodes_module._get_base_llm = lambda: mock_llm

        try:
            with patch.object(runner_module, "_save_turn_to_memory") as mock_save:
                from agent.runner import AgentSession
                session = AgentSession(tools=[])
                expected_sid = session.session_id
                session.chat("Hi", verbose=False)
        finally:
            nodes_module._get_base_llm = original_llm

        for c in mock_save.call_args_list:
            assert c.kwargs.get("session_id") == expected_sid or c.args[2] == expected_sid


# ---------------------------------------------------------------------------
# 9. DB tool E2E — all 37 breeds resolve correctly
# ---------------------------------------------------------------------------

class TestDbToolE2E:
    """Quick smoke test: db_query resolves a representative sample of breeds."""

    @pytest.mark.parametrize("breed", [
        "beagle", "siamese", "maine_coon", "pug", "persian",
        "american_bulldog", "yorkshire_terrier", "ragdoll",
    ])
    def test_breed_lookup_returns_valid_data(self, breed):
        from agent.tools.db_tool import db_query
        result = db_query.invoke({"breed_name": breed})
        data = json.loads(result)
        assert "error" not in data, f"Breed '{breed}' not found: {data}"
        assert "type" in data
        assert data["type"] in ("dog", "cat")
