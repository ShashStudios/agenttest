"""Tests for the agenttest runner (unit tests)."""

from pathlib import Path

import pytest

from agenttest.runner import discover_tests, _has_eval_decorator
import ast


def test_has_eval_decorator():
    """Test decorator detection."""
    code = """
@eval
def foo():
    pass
"""
    tree = ast.parse(code)
    func = tree.body[0]
    assert _has_eval_decorator(func) is True

    code2 = """
def bar():
    pass
"""
    tree2 = ast.parse(code2)
    func2 = tree2.body[0]
    assert _has_eval_decorator(func2) is False


def test_discover_tests():
    """Test test discovery finds our example tests."""
    tests_dir = Path(__file__).parent
    tests = discover_tests(tests_dir)
    assert len(tests) >= 7  # We have 7 @eval tests in agent_test_example.py
    names = [t[1] for t in tests]
    assert any("test_tone_empathetic" in n for n in names)
    assert any("test_intentional_fail" in n for n in names)


def test_discover_tests_with_filter():
    """Test filter reduces discovered tests."""
    tests_dir = Path(__file__).parent
    tests = discover_tests(tests_dir, filter_pattern="tone")
    assert len(tests) >= 1
    assert all("tone" in t[1].lower() for t in tests)
