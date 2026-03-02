"""Configuration loader for agenttest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import toml
except ImportError:
    toml = None  # type: ignore[assignment]

DEFAULT_CONFIG = {
    "model": "claude-3-5-haiku-latest",
    "timeout_seconds": 30,
    "workers": 4,
    "fail_threshold": 0.8,
    "cache": True,
}

CONFIG_FILENAMES = ("agenttest.toml", "pyproject.toml")


def _find_config_path(start: Path) -> Path | None:
    """Find agenttest.toml or [tool.agenttest] in pyproject.toml."""
    current = start.resolve()
    for _ in range(10):  # Max 10 levels up
        for name in CONFIG_FILENAMES:
            path = current / name
            if path.is_file():
                if name == "agenttest.toml":
                    return path
                if name == "pyproject.toml":
                    try:
                        data = toml.load(path)
                        if "tool" in data and "agenttest" in data["tool"]:
                            return path
                    except Exception:
                        pass
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML file."""
    if toml is None:
        raise ImportError(
            "toml package is required for config loading. Install with: pip install toml"
        )
    return toml.load(path)


def _resolve_env(value: str) -> str:
    """Resolve $VAR or ${VAR} in string. Missing vars become empty string."""
    if not isinstance(value, str):
        return value
    if value.startswith("$") and len(value) > 1:
        var = value[1:]
        if var.startswith("{"):
            var = var[1:-1]
        return os.environ.get(var, "")
    return value


def load_config(start_dir: Path | str | None = None) -> dict[str, Any]:
    """
    Load agenttest configuration from agenttest.toml or pyproject.toml.

    Searches upward from start_dir (default: cwd) for config files.
    Environment variables in config values (e.g. $ANTHROPIC_API_KEY) are resolved.

    Returns:
        Merged config dict with defaults.
    """
    start = Path(start_dir or os.getcwd())
    config: dict[str, Any] = {**DEFAULT_CONFIG}

    path = _find_config_path(start)
    if path:
        raw = _load_toml(path)
        if path.name == "pyproject.toml":
            agent = raw.get("tool", {}).get("agenttest", {})
        else:
            agent = raw.get("agenttest", {})

        # Merge [agenttest] section
        for key, value in agent.items():
            if key == "env":
                continue
            if value is not None:
                config[key] = value

        # Resolve [agenttest.env]
        env_section = agent.get("env", {})
        for key, val in env_section.items():
            if isinstance(val, str):
                config.setdefault("env", {})[key] = _resolve_env(val)
            else:
                config.setdefault("env", {})[key] = val

    return config


def get_api_key(config: dict[str, Any] | None = None) -> str:
    """
    Get Anthropic API key from config env or ANTHROPIC_API_KEY.

    Raises:
        ValueError: If no API key is found.
    """
    if config is None:
        config = load_config()
    env = config.get("env", {})
    key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in your environment or in agenttest.toml under agenttest.env."
        )
    return str(key)
