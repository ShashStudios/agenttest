"""Assertion helpers for agent evaluations."""

from __future__ import annotations

from typing import Any

from .judge import Judge, judge

__all__ = [
    "assert_tone",
    "assert_contains_action",
    "assert_no_hallucination",
    "assert_relevance",
    "assert_no_toxicity",
    "assert_faithfulness",
    "assert_conciseness",
    "assert_score",
    "assert_compare",
]


def _format_explanation(data: dict[str, Any]) -> str:
    """Extract explanation from judge JSON response."""
    for key in ("explanation", "reason"):
        if key in data and data[key]:
            return str(data[key])
    return ""


def assert_tone(
    response: str,
    expected: str,
    judge_instance: Judge | None = None,
) -> None:
    """
    Assert the response has the expected tone (empathetic, professional, aggressive, neutral).
    """
    j = judge_instance or judge
    actual_tone = j.tone(response)
    if actual_tone != expected:
        raise AssertionError(
            f"Expected tone '{expected}', got '{actual_tone}'"
        )


def assert_contains_action(
    response: str,
    action: str,
    judge_instance: Judge | None = None,
) -> None:
    """Assert the response contains or implies the given action."""
    j = judge_instance or judge
    if not j.contains_action(response, action):
        raise AssertionError(
            f"Expected response to contain action '{action}', but it did not"
        )


def assert_no_hallucination(
    response: str,
    context: str | None = None,
    judge_instance: Judge | None = None,
) -> None:
    """Assert the response contains no hallucinations."""
    j = judge_instance or judge
    if not j.no_hallucination(response, context):
        raise AssertionError(
            "Response appears to contain hallucinations (claims not supported by context)"
        )


def assert_relevance(
    response: str,
    query: str,
    min_score: float = 0.7,
    judge_instance: Judge | None = None,
) -> None:
    """Assert relevance score of response to query is at least min_score."""
    j = judge_instance or judge
    score = j.relevance(response, query)
    if score < min_score:
        raise AssertionError(
            f"Relevance score {score:.2f} below threshold {min_score}"
        )


def assert_no_toxicity(
    response: str,
    judge_instance: Judge | None = None,
) -> None:
    """Assert the response is not toxic."""
    j = judge_instance or judge
    if j.toxicity(response):
        raise AssertionError("Response contains toxic content")


def assert_faithfulness(
    response: str,
    source: str,
    min_score: float = 0.8,
    judge_instance: Judge | None = None,
) -> None:
    """Assert faithfulness score of response to source is at least min_score."""
    j = judge_instance or judge
    score = j.faithfulness(response, source)
    if score < min_score:
        raise AssertionError(
            f"Faithfulness score {score:.2f} below threshold {min_score}"
        )


def assert_conciseness(
    response: str,
    expected: str = "good",
    judge_instance: Judge | None = None,
) -> None:
    """Assert conciseness is expected (too_short, good, too_long)."""
    j = judge_instance or judge
    actual = j.conciseness(response)
    if actual != expected:
        raise AssertionError(
            f"Expected conciseness '{expected}', got '{actual}'"
        )


def assert_score(
    response: str,
    criteria: str,
    min_score: float = 0.7,
    judge_instance: Judge | None = None,
) -> None:
    """Assert custom score meets minimum."""
    j = judge_instance or judge
    score = j.score(response, criteria)
    if score < min_score:
        raise AssertionError(
            f"Score {score:.2f} below threshold {min_score} for criteria: {criteria}"
        )


def assert_compare(
    response_a: str,
    response_b: str,
    criteria: str,
    expected_winner: str,
    judge_instance: Judge | None = None,
) -> None:
    """Assert expected_winner ('a', 'b', or 'tie') wins the comparison."""
    j = judge_instance or judge
    winner = j.compare(response_a, response_b, criteria)
    if winner != expected_winner:
        raise AssertionError(
            f"Expected winner '{expected_winner}', got '{winner}'"
        )
