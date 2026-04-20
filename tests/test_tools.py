"""
Unit tests for Phase 5 — agent tools.

All external calls (httpx, duckduckgo_search) are mocked.
Tests run fully offline with no API keys required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Tests: cv_predict
# ---------------------------------------------------------------------------

class TestCvPredict:
    def test_successful_prediction(self):
        """Happy path: cv_service returns a valid prediction."""
        fake_response = {
            "breed": "Siamese",
            "confidence": 0.93,
            "top5": [{"breed": "Siamese", "confidence": 0.93}],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = lambda: None

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from agent.tools.cv_tool import cv_predict
            result = cv_predict.invoke({"image_url": "http://example.com/cat.jpg"})

        data = json.loads(result)
        assert data["breed"] == "Siamese"
        assert data["confidence"] == 0.93

    def test_connection_error_returns_error_json(self):
        """cv_service unreachable → error JSON, no exception propagated."""
        import httpx

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from agent.tools.cv_tool import cv_predict
            result = cv_predict.invoke({"image_url": "http://example.com/dog.jpg"})

        data = json.loads(result)
        assert "error" in data
        assert "cv-service" in data["error"].lower() or "connect" in data["error"].lower()

    def test_http_error_returns_error_json(self):
        """cv_service returns 500 → error JSON."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        http_error = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_resp.raise_for_status.side_effect = http_error
            mock_client.__enter__ = lambda s: mock_client
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from agent.tools.cv_tool import cv_predict
            result = cv_predict.invoke({"image_url": "http://example.com/dog.jpg"})

        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: db_query
# ---------------------------------------------------------------------------

class TestDbQuery:
    def test_exact_breed_lookup(self):
        from agent.tools.db_tool import db_query
        result = db_query.invoke({"breed_name": "siamese"})
        data = json.loads(result)
        assert data["type"] == "cat"
        assert "vocal" in data["temperament"].lower()

    def test_case_insensitive_lookup(self):
        from agent.tools.db_tool import db_query
        result = db_query.invoke({"breed_name": "BEAGLE"})
        data = json.loads(result)
        assert "error" not in data
        assert data["type"] == "dog"

    def test_space_vs_underscore_normalisation(self):
        from agent.tools.db_tool import db_query
        result = db_query.invoke({"breed_name": "Maine Coon"})
        data = json.loads(result)
        assert "error" not in data
        assert data["size"] == "large"

    def test_unknown_breed_returns_error(self):
        from agent.tools.db_tool import db_query
        result = db_query.invoke({"breed_name": "Golden Retriever"})
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_partial_match_works(self):
        from agent.tools.db_tool import db_query
        result = db_query.invoke({"breed_name": "bulldog"})  # matches american_bulldog
        data = json.loads(result)
        assert "error" not in data

    def test_all_37_breeds_are_present(self):
        from agent.tools.db_tool import _BREED_DB
        assert len(_BREED_DB) == 37


# ---------------------------------------------------------------------------
# Tests: web_search
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_returns_results_list(self):
        fake_results = [
            {"title": "Siamese cat health", "href": "http://a.com", "body": "They are vocal..."},
            {"title": "Siamese breed info", "href": "http://b.com", "body": "Origin: Thailand"},
        ]

        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = iter(fake_results)
        mock_ddgs.__enter__ = lambda s: mock_ddgs
        mock_ddgs.__exit__ = MagicMock(return_value=False)

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            from agent.tools.search_tool import web_search
            result = web_search.invoke({"query": "Siamese cat health problems"})

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "Siamese cat health"

    def test_handles_ddg_exception(self):
        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = Exception("rate limited")
        mock_ddgs.__enter__ = lambda s: mock_ddgs
        mock_ddgs.__exit__ = MagicMock(return_value=False)

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            from agent.tools.search_tool import web_search
            result = web_search.invoke({"query": "anything"})

        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests: ALL_TOOLS list
# ---------------------------------------------------------------------------

class TestAllTools:
    def test_all_tools_exported(self):
        from agent.tools import ALL_TOOLS
        names = [t.name for t in ALL_TOOLS]
        assert "cv_predict" in names
        assert "web_search" in names
        assert "db_query" in names

    def test_tools_are_langchain_compatible(self):
        """Each tool must have .name, .description, and be callable via .invoke()."""
        from agent.tools import ALL_TOOLS
        for t in ALL_TOOLS:
            assert hasattr(t, "name")
            assert hasattr(t, "description")
            assert callable(getattr(t, "invoke", None))
