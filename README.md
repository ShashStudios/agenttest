# agenttest

**The pytest of AI agents.** Catch regressions before they reach prod.

[![PyPI version](https://img.shields.io/pypi/v/agenttest.svg)](https://pypi.org/project/agenttest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
<!-- [![CI](https://github.com/YOUR_ORG/agenttest/actions/workflows/agenttest.yml/badge.svg)](https://github.com/YOUR_ORG/agenttest/actions) -->

---

## You ship an agent. You change a prompt. Did it get better or worse?

**You have no idea.** No test suite. No CI. No diff. Just deploy and hope.

Every team shipping AI agents hits the same wall: your "eval" is manually pasting examples into a playground. One prompt tweak could break everything—or fix everything—and you won't know until a user complains.

---

## 30-Second Quickstart

```bash
pip install agenttest
export ANTHROPIC_API_KEY=your_key
agenttest init
agenttest run
```

**Or from scratch:**

```python
# agent_test_example.py
from agenttest import eval, judge

def my_agent(query: str) -> str:
    return "Your agent's response"  # Replace with real agent

@eval
def test_customer_support():
    response = my_agent("I want a refund")
    assert judge.tone(response) == "empathetic"
    assert judge.no_hallucination(response)
```

```bash
agenttest run
```

---

## Features

- **Code-first** — Tests are just Python. No YAML. No config hell.
- **LLM-as-judge** — 9 built-in scorers: tone, hallucination, relevance, toxicity, faithfulness, conciseness, custom criteria, A/B compare.
- **Local & CI** — Runs anywhere. Add 4 lines to GitHub Actions. No account. No dashboard.
- **`agenttest diff`** — Side-by-side view of how your agent's responses changed between two runs. The git diff for agent behavior.
- **Caching** — Judge results cached in `.agenttest_cache/` to avoid redundant API calls.
- **Parallel** — `--workers 4` for faster runs.

---

## agenttest diff — The Git Diff for Agent Behavior

See exactly how your agent's responses changed between two runs:

```bash
agenttest run --tag v1    # Before your prompt change
agenttest run --tag v2    # After your prompt change
agenttest diff v1 v2
```

```
test_customer_support_refund:
  BEFORE: "I cannot help with refunds"           pass
  AFTER:  "I'd be happy to process that for you" pass
  DELTA:  ✓ improved

test_helpful_tone:
  BEFORE: "Our policy states no returns"         fail
  AFTER:  "I'm sorry to hear that. Let me help"   pass
  DELTA:  +1 ↑
```

---

## Judge Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `judge.tone(response)` | `str` | empathetic, professional, aggressive, neutral |
| `judge.no_hallucination(response, context?)` | `bool` | True if no hallucination |
| `judge.contains_action(response, action)` | `bool` | Response mentions/implies the action |
| `judge.relevance(response, query)` | `float` | 0.0–1.0 relevance |
| `judge.toxicity(response)` | `bool` | True if toxic |
| `judge.faithfulness(response, source)` | `float` | 0.0–1.0 faithfulness |
| `judge.conciseness(response)` | `str` | too_short, good, too_long |
| `judge.score(response, criteria)` | `float` | Custom 0.0–1.0 score |
| `judge.compare(a, b, criteria)` | `str` | "a", "b", or "tie" |

---

## CI in 4 Lines

```yaml
# .github/workflows/agenttest.yml
- run: pip install agenttest
- run: agenttest run
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Every PR shows whether your agent got better or worse.

---

## agenttest vs Braintrust / DeepEval / Promptfoo

| | agenttest | Braintrust | DeepEval | Promptfoo |
|---|:---:|:---:|:---:|:---:|
| **No account required** | ✅ | ❌ | ❌ | ❌ |
| **No vendor lock-in** | ✅ | ❌ | ❌ | ❌ |
| **Lives in your codebase** | ✅ | ❌ | ❌ | ❌ |
| **Behavior diff (before/after)** | ✅ | ❌ | ❌ | ❌ |
| **Runs locally** | ✅ | ✅ | ✅ | ✅ |
| **MIT License** | ✅ | ❌ | ❌ | ✅ |
| **Code-first API** | ✅ | ⚠️ | ⚠️ | ⚠️ |

**agenttest** = pytest for agents. No dashboards. No SaaS. Your tests, your repo, your CI.

---

## Config

```toml
# agenttest.toml
[agenttest]
model = "claude-3-5-haiku-latest"
timeout_seconds = 30
workers = 4
fail_threshold = 0.8
cache = true

[agenttest.env]
ANTHROPIC_API_KEY = "$ANTHROPIC_API_KEY"
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
