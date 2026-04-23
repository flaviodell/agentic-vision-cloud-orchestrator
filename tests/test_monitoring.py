"""
Prometheus monitoring metrics.

All tests are fully offline (no API keys, no Docker, no real model loads).
Tests import metric objects directly and verify counters/histograms behave correctly.
"""

import sys
import os
import pytest

# Allow importing cv_service modules from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cv_service"))


# ---------------------------------------------------------------------------
# CV Service metrics
# ---------------------------------------------------------------------------

class TestCvServiceMetrics:
    def test_all_metric_objects_exist(self):
        """All four Prometheus metrics must be importable and non-None."""
        from app.metrics import (
            INFERENCE_LATENCY,
            PREDICTIONS_TOTAL,
            CONFIDENCE_SCORE,
            MODEL_LOAD_TIME,
        )
        assert INFERENCE_LATENCY is not None
        assert PREDICTIONS_TOTAL is not None
        assert CONFIDENCE_SCORE is not None
        assert MODEL_LOAD_TIME is not None

    def test_metrics_app_is_asgi(self):
        """make_asgi_app() must return a callable ASGI app."""
        from app.metrics import metrics_app
        assert callable(metrics_app)

    def test_predictions_counter_increments(self):
        """Counter must increment by exactly 1 per .inc() call."""
        from app.metrics import PREDICTIONS_TOTAL
        label = {"breed": "_test_breed_counter", "status": "success"}
        before = PREDICTIONS_TOTAL.labels(**label)._value.get()
        PREDICTIONS_TOTAL.labels(**label).inc()
        after = PREDICTIONS_TOTAL.labels(**label)._value.get()
        assert after == before + 1.0

    def test_predictions_counter_error_label(self):
        """Error label must be tracked independently from success."""
        from app.metrics import PREDICTIONS_TOTAL
        label = {"breed": "_test_breed_err", "status": "error"}
        before = PREDICTIONS_TOTAL.labels(**label)._value.get()
        PREDICTIONS_TOTAL.labels(**label).inc()
        after = PREDICTIONS_TOTAL.labels(**label)._value.get()
        assert after == before + 1.0

    def test_inference_latency_histogram_observes(self):
        """Histogram must accept valid float observations without raising."""
        from app.metrics import INFERENCE_LATENCY
        for val in [0.05, 0.25, 1.0, 5.0]:
            INFERENCE_LATENCY.observe(val)  # must not raise

    def test_confidence_score_histogram_observes(self):
        """Confidence histogram must accept values in [0, 1]."""
        from app.metrics import CONFIDENCE_SCORE
        for val in [0.1, 0.5, 0.9, 0.99]:
            CONFIDENCE_SCORE.observe(val)

    def test_model_load_time_gauge_set(self):
        """Gauge.set() must update the stored value."""
        from app.metrics import MODEL_LOAD_TIME
        MODEL_LOAD_TIME.set(7.42)
        assert MODEL_LOAD_TIME._value.get() == pytest.approx(7.42)


# ---------------------------------------------------------------------------
# Agent monitoring metrics
# ---------------------------------------------------------------------------

class TestAgentMetrics:
    def test_all_agent_metric_objects_exist(self):
        """All agent Prometheus metrics must be importable."""
        from agent.monitoring.metrics import (
            AGENT_TURN_LATENCY,
            AGENT_TURNS_PER_SESSION,
            AGENT_TOOL_CALLS_TOTAL,
            AGENT_SESSIONS_TOTAL,
        )
        assert AGENT_TURN_LATENCY is not None
        assert AGENT_TURNS_PER_SESSION is not None
        assert AGENT_TOOL_CALLS_TOTAL is not None
        assert AGENT_SESSIONS_TOTAL is not None

    def test_record_session_increments_counter(self):
        """record_session must bump AGENT_SESSIONS_TOTAL by 1."""
        from agent.monitoring.metrics import record_session, AGENT_SESSIONS_TOTAL
        before = AGENT_SESSIONS_TOTAL._value.get()
        record_session(turn_count=4)
        after = AGENT_SESSIONS_TOTAL._value.get()
        assert after == before + 1.0

    def test_record_session_observes_histogram(self):
        """record_session must feed the turn count into AGENT_TURNS_PER_SESSION."""
        from agent.monitoring.metrics import record_session, AGENT_TURNS_PER_SESSION

        def _count(h):
            return next(s.value for s in h.collect()[0].samples if s.name.endswith("_count"))

        before_count = _count(AGENT_TURNS_PER_SESSION)
        record_session(turn_count=3)
        after_count = _count(AGENT_TURNS_PER_SESSION)
        assert after_count == before_count + 1

    def test_record_tool_call_increments_counter(self):
        """record_tool_call must increment the labelled counter by 1."""
        from agent.monitoring.metrics import record_tool_call, AGENT_TOOL_CALLS_TOTAL
        before = AGENT_TOOL_CALLS_TOTAL.labels(tool_name="_test_tool")._value.get()
        record_tool_call("_test_tool")
        after = AGENT_TOOL_CALLS_TOTAL.labels(tool_name="_test_tool")._value.get()
        assert after == before + 1.0

    def test_record_tool_call_different_tools_are_independent(self):
        """Different tool_name labels must not interfere with each other."""
        from agent.monitoring.metrics import record_tool_call, AGENT_TOOL_CALLS_TOTAL
        before_a = AGENT_TOOL_CALLS_TOTAL.labels(tool_name="_tool_a")._value.get()
        before_b = AGENT_TOOL_CALLS_TOTAL.labels(tool_name="_tool_b")._value.get()
        record_tool_call("_tool_a")
        after_a = AGENT_TOOL_CALLS_TOTAL.labels(tool_name="_tool_a")._value.get()
        after_b = AGENT_TOOL_CALLS_TOTAL.labels(tool_name="_tool_b")._value.get()
        assert after_a == before_a + 1.0
        assert after_b == before_b  # must be unchanged

    def test_agent_turn_latency_histogram_observes(self):
        """AGENT_TURN_LATENCY must accept valid latency observations."""
        from agent.monitoring.metrics import AGENT_TURN_LATENCY
        for val in [0.5, 2.5, 10.0]:
            AGENT_TURN_LATENCY.observe(val)


# ---------------------------------------------------------------------------
# Integration: runner imports monitoring without errors
# ---------------------------------------------------------------------------

class TestRunnerMonitoringIntegration:
    def test_runner_imports_cleanly(self):
        """runner.py must import without raising even without API keys set."""
        import importlib
        import agent.runner as runner_mod
        importlib.reload(runner_mod)  # re-import to catch any import-time errors

    def test_run_agent_records_latency(self):
        """
        run_agent must call AGENT_TURN_LATENCY.observe() after graph finishes.
        We mock the graph to avoid real LLM calls.
        """
        from unittest.mock import patch, MagicMock
        from langchain_core.messages import AIMessage
        from agent.monitoring.metrics import AGENT_TURN_LATENCY

        mock_state = {
            "messages": [AIMessage(content="Mocked reply")],
            "turn_count": 1,
            "last_tool_result": None,
            "breed_identified": None,
        }
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_state

        def _count(h):
            return next(s.value for s in h.collect()[0].samples if s.name.endswith("_count"))

        before = _count(AGENT_TURN_LATENCY)

        with patch("agent.runner.build_graph", return_value=mock_graph):
            from agent.runner import run_agent
            run_agent("test input", tools=[])

        after = _count(AGENT_TURN_LATENCY)
        assert after == before + 1
