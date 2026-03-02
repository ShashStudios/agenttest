"""Example agent eval tests demonstrating all judge functions."""

from agenttest import eval, judge, record


def my_agent(query: str) -> str:
    """Mock agent - replace with your actual agent call."""
    if "refund" in query.lower():
        return (
            "I'm sorry to hear you're unhappy. Our refund policy allows "
            "returns within 30 days. Would you like me to start the refund process?"
        )
    if "hours" in query.lower():
        return "We're open Monday-Friday 9am-5pm EST."
    return "How can I help you today?"


@eval
def test_tone_empathetic():
    query = "I want a refund"
    response = my_agent(query)
    record(query, response)
    assert judge.tone(response) == "empathetic"
    assert judge.contains_action(response, "refund_policy")
    assert judge.no_hallucination(response)


@eval
def test_relevance_and_toxicity():
    query = "What are your hours?"
    response = my_agent(query)
    record(query, response)
    assert judge.relevance(response, "business hours") >= 0.3
    assert not judge.toxicity(response)


@eval
def test_conciseness():
    query = "Hi"
    response = my_agent(query)
    record(query, response)
    assert judge.conciseness(response) in ("good", "too_short", "too_long")


@eval
def test_custom_score():
    query = "I want a refund"
    response = my_agent(query)
    score = judge.score(
        response,
        criteria="Does the response show empathy and offer a clear next step?",
    )
    record(query, response, score)
    assert score >= 0.5


@eval
def test_compare_versions():
    query = "I want a refund"
    response_a = my_agent(query)
    record(query, response_a)
    response_b = "NO REFUNDS. EVER."  # Worse response
    winner = judge.compare(
        response_a,
        response_b,
        criteria="Which response is more helpful and empathetic?",
    )
    assert winner in ("a", "b", "tie")


@eval
def test_faithfulness_with_source():
    source = "Refunds: 30 days, unopened items only."
    response = "You can get a refund within 30 days if the item is unopened."
    record("Refund policy?", response, judge.faithfulness(response, source))
    assert judge.faithfulness(response, source) >= 0.7


@eval
def test_intentional_fail():
    """This test intentionally fails to demonstrate failure output."""
    query = "Hello"
    response = my_agent(query)
    record(query, response)
    assert judge.tone(response) == "aggressive"  # Will fail - tone is likely neutral
