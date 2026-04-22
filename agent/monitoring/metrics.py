"""
Prometheus metrics for the LangGraph agent runner.

Exposes:
  - agent_turn_latency_seconds   (Histogram) — latency per graph invocation
  - agent_turns_per_session      (Histogram) — turns used per session
  - agent_tool_calls_total       (Counter)   — tool calls by tool name
  - agent_sessions_total         (Counter)   — total sessions started
"""

from prometheus_client import Counter, Histogram

AGENT_TURN_LATENCY = Histogram(
    "agent_turn_latency_seconds",
    "Latency of a single agent graph invocation (run_agent call)",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0],
)

AGENT_TURNS_PER_SESSION = Histogram(
    "agent_turns_per_session",
    "Number of LangGraph turns used per complete session",
    buckets=[1, 2, 3, 5, 7, 10],
)

AGENT_TOOL_CALLS_TOTAL = Counter(
    "agent_tool_calls_total",
    "Total tool invocations by tool name",
    ["tool_name"],
)

AGENT_SESSIONS_TOTAL = Counter(
    "agent_sessions_total",
    "Total agent sessions started",
)


def record_session(turn_count: int) -> None:
    """Call at end of session to record aggregate metrics."""
    AGENT_TURNS_PER_SESSION.observe(turn_count)
    AGENT_SESSIONS_TOTAL.inc()


def record_tool_call(tool_name: str) -> None:
    """Call each time a tool is invoked."""
    AGENT_TOOL_CALLS_TOTAL.labels(tool_name=tool_name).inc()
