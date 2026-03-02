"""agenttest — The pytest of AI agents."""

from __future__ import annotations

from .record import record
from .assertions import (
    assert_compare,
    assert_conciseness,
    assert_contains_action,
    assert_faithfulness,
    assert_no_hallucination,
    assert_no_toxicity,
    assert_relevance,
    assert_score,
    assert_tone,
)
from .judge import Judge, judge
from .reporter import load_last_report, report_results
from .runner import discover_tests, run_tests

__all__ = [
    "eval",
    "judge",
    "record",
    "EvalResult",
    "Judge",
    "assert_tone",
    "assert_contains_action",
    "assert_no_hallucination",
    "assert_relevance",
    "assert_no_toxicity",
    "assert_faithfulness",
    "assert_conciseness",
    "assert_score",
    "assert_compare",
    "discover_tests",
    "run_tests",
    "load_last_report",
    "report_results",
]


def eval(fn: object) -> object:
    """
    Decorator to mark a function as an agent eval test.
    Discovered by agenttest run.
    """
    # Passthrough - runner discovers by parsing source for @eval
    return fn


class EvalResult:
    """Result of a single eval run."""

    def __init__(
        self,
        test_name: str,
        status: str,
        duration: float = 0,
        error_message: str | None = None,
        scores: dict[str, float] | None = None,
    ):
        self.test_name = test_name
        self.status = status
        self.duration = duration
        self.error_message = error_message
        self.scores = scores or {}
