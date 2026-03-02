"""LLM provider backends for the judge. Supports API keys, CLI tools, and Ollama."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

CACHE_DIR = __import__("pathlib", fromlist=["Path"]).Path(".agenttest_cache")


def _check_cmd(cmd: list[str]) -> bool:
    """Return True if command exists and runs successfully."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _run_cmd(cmd: list[str], input_text: str, timeout: int = 60) -> str:
    """Run command with input, return stdout."""
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0 and result.stderr:
            raise RuntimeError(f"Command failed: {result.stderr[:500]}")
        return result.stdout or ""
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s")


def detect_provider(config: dict[str, Any] | None = None) -> str:
    """
    Auto-detect available judge provider. Priority:
    1. ANTHROPIC_API_KEY → anthropic
    2. OPENAI_API_KEY → openai
    3. claude CLI → claude
    4. codex CLI → codex
    5. ollama running → ollama
    6. Raise ValueError with all options
    """
    env = (config or {}).get("env", {})
    if env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if _check_cmd(["claude", "--version"]):
        return "claude"
    if _check_cmd(["codex", "--version"]):
        return "codex"
    if _check_ollama():
        return "ollama"

    raise ValueError(
        "No judge provider found. Set one of:\n"
        "  - ANTHROPIC_API_KEY (anthropic API)\n"
        "  - OPENAI_API_KEY (openai API)\n"
        "  - Install Claude Code CLI: claude -p 'prompt'\n"
        "  - Install Codex CLI: codex\n"
        "  - Run Ollama: ollama serve\n"
        "Or use: agenttest run --judge claude|codex|ollama|anthropic|openai"
    )


def _check_ollama() -> bool:
    """Check if Ollama is running (localhost:11434)."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def call_llm(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
    provider: str | None = None,
) -> str:
    """
    Call LLM with the given prompts. Dispatches to the configured provider.
    """
    prov = provider or config.get("judge") or config.get("provider") or detect_provider(config)
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    if prov == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, config)
    if prov == "openai":
        return _call_openai(system_prompt, user_prompt, config)
    if prov == "claude":
        return _call_claude_cli(full_prompt, config)
    if prov == "codex":
        return _call_codex_cli(full_prompt, config)
    if prov == "ollama":
        return _call_ollama(full_prompt, config)

    raise ValueError(f"Unknown provider: {prov}. Use: anthropic, openai, claude, codex, ollama")


def _call_anthropic(system_prompt: str, user_prompt: str, config: dict[str, Any]) -> str:
    """Call Anthropic API."""
    from anthropic import Anthropic
    from .config import get_api_key

    key = get_api_key(config)
    client = Anthropic(api_key=key)
    model = config.get("model", "claude-3-5-haiku-latest")
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return msg.content[0].text if msg.content else ""


def _call_openai(system_prompt: str, user_prompt: str, config: dict[str, Any]) -> str:
    """Call OpenAI API."""
    from openai import OpenAI

    env = config.get("env", {})
    key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Set it in your environment or agenttest.toml")
    client = OpenAI(api_key=key)
    model = config.get("openai_model", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_claude_cli(prompt: str, config: dict[str, Any]) -> str:
    """Call Claude Code CLI: claude -p 'prompt'."""
    timeout = config.get("timeout_seconds", 30) * 2  # Judge can take longer
    return _run_cmd(["claude", "-p", prompt], "", timeout=timeout)


def _call_codex_cli(prompt: str, config: dict[str, Any]) -> str:
    """Call Codex CLI. Pass prompt as first argument."""
    timeout = config.get("timeout_seconds", 30) * 2
    # codex <prompt> or prompt via stdin - try arg first
    return _run_cmd(["codex", prompt], "", timeout=timeout)


def _call_ollama(prompt: str, config: dict[str, Any]) -> str:
    """Call Ollama local API."""
    import urllib.request

    model = config.get("ollama_model", "llama3.2")
    url = "http://localhost:11434/api/generate"
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    return data.get("response", "")
