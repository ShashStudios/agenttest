"""Capture query/response for agenttest diff."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_record_data: ContextVar[dict[str, Any] | None] = ContextVar("agenttest_record", default=None)


def record(query: str, response: str, score: float | None = None) -> None:
    """
    Record query and response for this test. Used by `agenttest diff` to show
    before/after when comparing runs.

    Call this in your @eval test with the agent's input and output. Optional
    score (e.g. from judge.relevance()) improves the diff display.
    """
    current = _record_data.get(None)
    if current is None:
        current = {}
        _record_data.set(current)
    current["query"] = query
    current["response"] = response
    current["score"] = score


def _clear_record() -> None:
    """Clear recorded data (called by runner before each test)."""
    _record_data.set(None)


def _get_record() -> dict[str, Any] | None:
    """Get recorded data (called by runner after each test)."""
    return _record_data.get(None)
