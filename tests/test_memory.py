"""
Tests for Phase 6 — memory and vector database.

These tests use mocks to avoid real Pinecone/OpenAI calls during CI.
Integration tests (marked @pytest.mark.integration) require real API keys
and a live Pinecone index — they are skipped by default.

Run unit tests only:
    pytest tests/test_memory.py -v

Run all (including integration):
    pytest tests/test_memory.py -v -m "not skip_ci"
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Unit tests — embedder (mocked OpenAI)
# ---------------------------------------------------------------------------

class TestEmbedder:
    """Tests for agent.memory.embedder — all mocked."""

    def test_embed_text_returns_vector(self):
        fake_vector = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_vector)]

        with patch("agent.memory.embedder._get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_client_fn.return_value = mock_client

            from agent.memory.embedder import embed_text
            result = embed_text("Beagle health issues")

        assert isinstance(result, list)
        assert len(result) == 1536

    def test_embed_text_raises_on_empty(self):
        from agent.memory.embedder import embed_text
        with pytest.raises(ValueError, match="empty"):
            embed_text("   ")

    def test_embed_batch_returns_ordered_vectors(self):
        fake_vectors = [[float(i)] * 1536 for i in range(3)]
        mock_items = [
            MagicMock(embedding=v, index=i)
            for i, v in enumerate(fake_vectors)
        ]
        mock_response = MagicMock()
        mock_response.data = mock_items

        with patch("agent.memory.embedder._get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_client_fn.return_value = mock_client

            from agent.memory.embedder import embed_batch
            results = embed_batch(["text1", "text2", "text3"])

        assert len(results) == 3

    def test_embed_batch_empty_input(self):
        from agent.memory.embedder import embed_batch
        result = embed_batch([])
        assert result == []

    def test_get_embedding_dim(self):
        from agent.memory.embedder import get_embedding_dim
        assert get_embedding_dim() == 1536


# ---------------------------------------------------------------------------
# Unit tests — store (mocked Pinecone)
# ---------------------------------------------------------------------------

class TestStore:
    """Tests for agent.memory.store — Pinecone calls fully mocked."""

    def _make_mock_index(self):
        mock_index = MagicMock()
        mock_index.upsert.return_value = None
        # Simulate query response
        mock_match = MagicMock()
        mock_match.id = "test-id-001"
        mock_match.score = 0.92
        mock_match.metadata = {"text": "Beagle is a friendly dog.", "role": "user"}
        mock_index.query.return_value = MagicMock(matches=[mock_match])
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"conversations": {"vector_count": 5}},
            "dimension": 1536,
        }
        return mock_index

    def test_upsert_vector_returns_id(self):
        mock_index = self._make_mock_index()

        with patch("agent.memory.store._get_index", return_value=mock_index):
            from agent.memory import store
            # Reset singleton to force re-init
            store._index = mock_index

            doc_id = store.upsert_vector(
                vector=[0.1] * 1536,
                metadata={"text": "test", "role": "user"},
            )

        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_query_similar_returns_results(self):
        mock_index = self._make_mock_index()

        with patch("agent.memory.store._get_index", return_value=mock_index):
            from agent.memory import store
            store._index = mock_index

            results = store.query_similar(vector=[0.1] * 1536, top_k=3)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["score"] == 0.92
        assert "metadata" in results[0]

    def test_index_stats_returns_dict(self):
        mock_index = self._make_mock_index()

        with patch("agent.memory.store._get_index", return_value=mock_index):
            from agent.memory import store
            store._index = mock_index

            stats = store.index_stats()

        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Unit tests — manager (mocked embedder + store)
# ---------------------------------------------------------------------------

class TestMemoryManager:
    """Tests for agent.memory.manager — all external calls mocked."""

    def test_save_conversation_turn_success(self):
        fake_vector = [0.1] * 1536

        with (
            patch("agent.memory.manager._safe_embed", return_value=fake_vector),
            patch("agent.memory.manager._safe_upsert", return_value="mock-id-123"),
        ):
            from agent.memory.manager import save_conversation_turn
            result = save_conversation_turn(
                role="user",
                text="What breed is this dog?",
                session_id="session-abc",
            )

        assert result == "mock-id-123"

    def test_save_conversation_turn_empty_text(self):
        from agent.memory.manager import save_conversation_turn
        result = save_conversation_turn(role="user", text="", session_id="s1")
        assert result is None

    def test_save_conversation_turn_embed_failure(self):
        with patch("agent.memory.manager._safe_embed", return_value=None):
            from agent.memory.manager import save_conversation_turn
            result = save_conversation_turn(
                role="user", text="Hello", session_id="s1"
            )
        assert result is None

    def test_retrieve_relevant_context_returns_list(self):
        fake_vector = [0.1] * 1536
        fake_results = [
            {
                "id": "id1",
                "score": 0.88,
                "metadata": {
                    "role": "user",
                    "text": "Tell me about Beagles.",
                    "session_id": "s1",
                    "breed": "beagle",
                    "timestamp": 1700000000.0,
                },
            }
        ]

        with (
            patch("agent.memory.manager._safe_embed", return_value=fake_vector),
            patch("agent.memory.manager._safe_query", return_value=fake_results),
        ):
            from agent.memory.manager import retrieve_relevant_context
            results = retrieve_relevant_context("Beagle health", top_k=5)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["role"] == "user"
        assert results[0]["score"] == 0.88

    def test_retrieve_returns_empty_on_embed_failure(self):
        with patch("agent.memory.manager._safe_embed", return_value=None):
            from agent.memory.manager import retrieve_relevant_context
            results = retrieve_relevant_context("anything")
        assert results == []

    def test_search_breed_knowledge_returns_list(self):
        fake_vector = [0.1] * 1536
        fake_results = [
            {
                "id": "breed_beagle",
                "score": 0.95,
                "metadata": {
                    "breed": "beagle",
                    "type": "dog",
                    "text": "Breed: Beagle | ...",
                },
            }
        ]

        with (
            patch("agent.memory.manager._safe_embed", return_value=fake_vector),
            patch("agent.memory.manager._safe_query", return_value=fake_results),
        ):
            from agent.memory.manager import search_breed_knowledge
            results = search_breed_knowledge("friendly small dog")

        assert len(results) == 1
        assert results[0]["breed"] == "beagle"


# ---------------------------------------------------------------------------
# Unit tests — memory tools (mocked manager)
# ---------------------------------------------------------------------------

class TestMemoryTools:
    """Tests for the LangChain tool wrappers in memory_tool.py."""

    def test_memory_search_returns_json(self):
        fake_results = [
            {"role": "user", "text": "Tell me about the Siamese cat.", "score": 0.9, "breed": "siamese"}
        ]

        with patch("agent.memory.manager.retrieve_relevant_context", return_value=fake_results):
            from agent.tools.memory_tool import memory_search
            result = memory_search.invoke({"query": "Siamese cat"})

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1

    def test_memory_search_no_results(self):
        with patch("agent.memory.manager.retrieve_relevant_context", return_value=[]):
            from agent.tools.memory_tool import memory_search
            result = memory_search.invoke({"query": "something obscure"})

        data = json.loads(result)
        assert "message" in data
        assert data["results"] == []

    def test_breed_semantic_search_returns_json(self):
        fake_results = [
            {"breed": "maine_coon", "type": "cat", "text": "...", "score": 0.93}
        ]

        with patch("agent.memory.manager.search_breed_knowledge", return_value=fake_results):
            from agent.tools.memory_tool import breed_semantic_search
            result = breed_semantic_search.invoke({"query": "large gentle cat"})

        data = json.loads(result)
        assert "results" in data
        assert data["results"][0]["breed"] == "maine_coon"

    def test_breed_semantic_search_handles_error(self):
        with patch("agent.memory.manager.search_breed_knowledge", side_effect=Exception("Pinecone down")):
            from agent.tools.memory_tool import breed_semantic_search
            result = breed_semantic_search.invoke({"query": "anything"})

        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Unit tests — runner.py memory integration
# ---------------------------------------------------------------------------

class TestRunnerMemoryIntegration:
    """Verify AgentSession saves turns to memory and has a session_id."""

    def test_session_has_session_id(self):
        from agent.runner import AgentSession
        session = AgentSession(tools=[])
        assert hasattr(session, "session_id")
        assert isinstance(session.session_id, str)
        assert len(session.session_id) == 36  # UUID4 format

    def test_reset_generates_new_session_id(self):
        from agent.runner import AgentSession
        session = AgentSession(tools=[])
        old_id = session.session_id
        session.reset()
        assert session.session_id != old_id

    def test_save_turn_called_on_chat(self):
        """AgentSession.chat should call _save_turn_to_memory twice (user + assistant)."""
        import agent.runner as runner_module

        mock_result = {
            "messages": [
                MagicMock(content="Test reply"),
            ],
            "breed_identified": None,
        }

        with (
            patch.object(runner_module, "run_agent", return_value=mock_result),
            patch.object(runner_module, "_save_turn_to_memory") as mock_save,
        ):
            from agent.runner import AgentSession
            session = AgentSession(tools=[])
            session.chat("Hello", verbose=False)

        assert mock_save.call_count == 2
        calls = [c.kwargs["role"] if c.kwargs else c.args[0] for c in mock_save.call_args_list]
        roles_called = [mock_save.call_args_list[0][0][0], mock_save.call_args_list[1][0][0]]
        assert "user" in roles_called
        assert "assistant" in roles_called
