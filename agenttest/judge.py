"""LLM-as-judge scoring functions using Anthropic API."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from .config import get_api_key, load_config

CACHE_DIR = Path(".agenttest_cache")


def _cache_key(func_name: str, *args: Any, **kwargs: Any) -> str:
    """Generate deterministic cache key from function name and arguments."""
    payload = [func_name] + [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in sorted(kwargs.items())]
    raw = "|".join(payload)
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached(cache_key: str, config: dict[str, Any]) -> tuple[bool, Any | None]:
    """Return (hit, value). Value is None if miss."""
    if not config.get("cache", True):
        return False, None
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if not cache_path.is_file():
        return False, None
    try:
        data = json.loads(cache_path.read_text())
        return True, data.get("result")
    except Exception:
        return False, None


def _set_cached(cache_key: str, result: Any, config: dict[str, Any]) -> None:
    """Store result in cache."""
    if not config.get("cache", True):
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    cache_path.write_text(json.dumps({"result": result}))


def _call_judge(
    system_prompt: str,
    user_prompt: str,
    response_format: str,
    config: dict[str, Any],
    cache_key: str | None = None,
) -> str:
    """Call Anthropic API and return raw response text."""
    if cache_key:
        hit, cached = _get_cached(cache_key, config)
        if hit:
            return cached

    try:
        client = Anthropic(api_key=get_api_key(config))
        model = config.get("model", "claude-3-5-haiku-latest")

        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = msg.content[0].text if msg.content else ""

        if cache_key:
            _set_cached(cache_key, text, config)
        return text
    except Exception as e:
        err_name = type(e).__name__
        if "Authentication" in err_name or "401" in str(e):
            raise ValueError(
                "Invalid ANTHROPIC_API_KEY. Check your key at https://console.anthropic.com/"
            ) from e
        if "Rate" in err_name or "429" in str(e):
            raise ValueError(
                "Anthropic API rate limit exceeded. Try again later or reduce --workers."
            ) from e
        raise


class Judge:
    """
    LLM-as-judge scoring functions. Each returns typed, deterministic results.
    Results are cached in .agenttest_cache/ when config cache=true.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or load_config()

    def no_hallucination(self, response: str, context: str | None = None, explain: bool = False) -> bool:
        """
        Check if the response contains hallucinations (claims not supported by context).

        Returns True if no hallucination detected.
        """
        cache_key = _cache_key("no_hallucination", response, context, explain) if self._config.get("cache") else None
        sys = """You are a strict fact-checker. Determine if the response contains any claims, facts, or details that are NOT supported by the given context. If no context is provided, check if the response makes unsupported factual claims that could be hallucinations.

Output ONLY a JSON object with this exact structure:
{"hallucination": false} if the response has NO hallucinations
{"hallucination": true, "reason": "brief explanation"} if there ARE hallucinations
{"explanation": "optional extra detail"} only when explain is requested"""
        user = f"Response:\n{response}\n\n"
        if context:
            user += f"Context:\n{context}\n\n"
        user += "Does the response contain hallucinations? Output JSON only."
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        try:
            data = _extract_json(raw)
            return not data.get("hallucination", True)
        except (json.JSONDecodeError, KeyError):
            return True

    def tone(self, response: str, explain: bool = False) -> str:
        """
        Classify the tone of the response. Returns one of:
        empathetic, professional, aggressive, neutral.
        """
        cache_key = _cache_key("tone", response, explain) if self._config.get("cache") else None
        sys = """You classify the tone of text. Output ONLY a JSON object:
{"tone": "empathetic"|"professional"|"aggressive"|"neutral"}
Optionally: {"explanation": "brief reason"} if asked to explain."""
        user = f"Classify the tone of this response:\n\n{response}"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        return str(data.get("tone", "neutral"))

    def contains_action(self, response: str, action: str, explain: bool = False) -> bool:
        """Check if the response contains or implies the given action (e.g. refund_policy)."""
        cache_key = _cache_key("contains_action", response, action, explain) if self._config.get("cache") else None
        sys = """You check if a response contains or clearly implies a specific action/topic. Output ONLY:
{"contains": true|false}
Optional: {"explanation": "..."}"""
        user = f"Response:\n{response}\n\nAction/topic to look for: {action}\n\nDoes it contain or imply this action?"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        return bool(data.get("contains", False))

    def relevance(self, response: str, query: str, explain: bool = False) -> float:
        """Score how relevant the response is to the query. Returns 0.0 to 1.0."""
        cache_key = _cache_key("relevance", response, query, explain) if self._config.get("cache") else None
        sys = """You score relevance of a response to a query. Output ONLY a JSON object:
{"score": 0.0 to 1.0}
Optional: {"explanation": "..."}"""
        user = f"Query: {query}\n\nResponse: {response}\n\nRelevance score (0.0-1.0)?"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        return float(data.get("score", 0.0))

    def toxicity(self, response: str, explain: bool = False) -> bool:
        """Return True if the response contains toxic/harmful content."""
        cache_key = _cache_key("toxicity", response, explain) if self._config.get("cache") else None
        sys = """You detect toxic/harmful content (hate, harassment, violence, etc). Output ONLY:
{"toxic": true|false}
Optional: {"explanation": "..."}"""
        user = f"Is this response toxic?\n\n{response}"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        return bool(data.get("toxic", False))

    def faithfulness(self, response: str, source: str, explain: bool = False) -> float:
        """Score how faithful the response is to the source. Returns 0.0 to 1.0."""
        cache_key = _cache_key("faithfulness", response, source, explain) if self._config.get("cache") else None
        sys = """You score how faithful a response is to a source document. No unsupported claims. Output ONLY:
{"score": 0.0 to 1.0}
Optional: {"explanation": "..."}"""
        user = f"Source:\n{source}\n\nResponse:\n{response}\n\nFaithfulness score?"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        return float(data.get("score", 0.0))

    def conciseness(self, response: str, explain: bool = False) -> str:
        """Classify length: too_short, good, too_long."""
        cache_key = _cache_key("conciseness", response, explain) if self._config.get("cache") else None
        sys = """You classify response length. Output ONLY:
{"conciseness": "too_short"|"good"|"too_long"}
Optional: {"explanation": "..."}"""
        user = f"Classify the length of this response:\n\n{response}"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        return str(data.get("conciseness", "good"))

    def score(self, response: str, criteria: str, explain: bool = False) -> float:
        """Custom score 0.0-1.0 based on criteria."""
        cache_key = _cache_key("score", response, criteria, explain) if self._config.get("cache") else None
        sys = """You score a response 0.0 to 1.0 based on given criteria. Output ONLY:
{"score": 0.0 to 1.0}
Optional: {"explanation": "..."}"""
        user = f"Criteria: {criteria}\n\nResponse:\n{response}\n\nScore (0.0-1.0)?"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        return float(data.get("score", 0.0))

    def compare(
        self, response_a: str, response_b: str, criteria: str, explain: bool = False
    ) -> str:
        """Compare two responses. Returns 'a', 'b', or 'tie'."""
        cache_key = _cache_key("compare", response_a, response_b, criteria, explain) if self._config.get("cache") else None
        sys = """You compare two responses A and B. Output ONLY:
{"winner": "a"|"b"|"tie"}
Optional: {"explanation": "..."}"""
        user = f"Criteria: {criteria}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\nWhich is better? a, b, or tie?"
        raw = _call_judge(sys, user, "json", self._config, cache_key)
        data = _extract_json(raw)
        w = str(data.get("winner", "tie")).lower()
        return w if w in ("a", "b", "tie") else "tie"


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from model output (handle markdown code blocks)."""
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        if start >= 0:
            rest = text[start:]
            if rest.startswith("```json"):
                rest = rest[7:]
            elif rest.startswith("```"):
                rest = rest[3:]
            end = rest.find("```")
            if end >= 0:
                rest = rest[:end]
            text = rest
    return json.loads(text)


# Global singleton for convenience
judge = Judge()
