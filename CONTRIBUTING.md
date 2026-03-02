# Contributing to agenttest

Thank you for considering contributing to agenttest! This document provides guidelines for contributing.

## Development Setup

```bash
git clone https://github.com/your-org/agenttest.git
cd agenttest
pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests (pytest)
pytest tests/

# Agent evals (requires ANTHROPIC_API_KEY)
agenttest run
```

## Code Style

- Use type hints on all public functions
- Add docstrings to public APIs
- Follow existing patterns in the codebase

## Pull Request Process

1. Fork the repo and create a branch
2. Make your changes with tests
3. Ensure `pytest tests/` and `agenttest run` pass
4. Submit a PR with a clear description

## Questions?

Open an issue for discussion.
