"""Test discovery and execution engine."""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
import traceback
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Generator

from .config import load_config
from .record import _clear_record, _get_record

EVAL_DECORATOR = "eval"


def discover_tests(
    path: Path | str,
    filter_pattern: str | None = None,
) -> list[tuple[str, str, Callable[..., Any]]]:
    """
    Discover all @eval-decorated functions in agent_test_*.py or *_agent_test.py files.

    Returns list of (file_path, test_name, test_func).
    """
    path = Path(path).resolve()
    if not path.exists():
        return []

    collected: list[tuple[str, str, Callable[..., Any]]] = []
    for p in path.rglob("*.py"):
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith("agent_test_") and name.endswith(".py")) and not (
            "_agent_test.py" in name
        ):
            continue
        try:
            for fn_name, fn in _collect_evals_from_file(p):
                try:
                    rel = p.relative_to(path)
                except ValueError:
                    rel = p
                full_name = f"{rel!s}::{fn_name}"
                if filter_pattern and filter_pattern.lower() not in full_name.lower():
                    continue
                collected.append((str(p), full_name, fn))
        except Exception as e:
            # Skip broken files but could log
            raise RuntimeError(f"Failed to load {p}: {e}") from e
    return collected


def _collect_evals_from_file(file_path: Path) -> Generator[tuple[str, Callable[..., Any]], None, None]:
    """Parse file and yield (name, func) for each @eval-decorated function."""
    try:
        source = file_path.read_text()
    except OSError:
        return
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if _has_eval_decorator(node):
                mod = _load_module(file_path)
                if mod is not None and hasattr(mod, node.name):
                    fn = getattr(mod, node.name)
                    if callable(fn):
                        yield node.name, fn


def _has_eval_decorator(node: ast.FunctionDef) -> bool:
    """Check if function has @eval decorator."""
    for d in node.decorator_list:
        if isinstance(d, ast.Name) and d.id == EVAL_DECORATOR:
            return True
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
            if d.func.id == EVAL_DECORATOR:
                return True
    return False


def _load_module(file_path: Path) -> Any:
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(
        f"agenttest_module_{file_path.stem}", file_path
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    agenttest_parent = Path(__file__).resolve().parent.parent
    if str(agenttest_parent) not in sys.path:
        sys.path.insert(0, str(agenttest_parent))
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def run_test(
    test_path: str,
    test_name: str,
    test_func: Callable[..., Any],
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """
    Run a single test and return result dict.
    """
    start = time.time()
    _clear_record()
    try:
        test_func()
        status = "pass"
        error_message = None
        assertion_line = None
    except AssertionError as e:
        status = "fail"
        error_message = str(e)
        assertion_line = traceback.format_exc()
    except Exception as e:
        status = "error"
        error_message = str(e)
        assertion_line = traceback.format_exc()
    duration = time.time() - start

    recorded = _get_record()
    if recorded is None:
        recorded = {}

    return {
        "test_path": test_path,
        "test_name": test_name,
        "status": status,
        "duration": duration,
        "error_message": error_message,
        "assertion_traceback": assertion_line,
        "scores": {},
        "recorded": recorded,
    }


def run_tests(
    path: Path | str,
    filter_pattern: str | None = None,
    workers: int = 1,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    """
    Discover and run all matching tests. Returns list of result dicts.
    """
    tests = discover_tests(path, filter_pattern)
    if not tests:
        return []

    config = load_config(Path(path) if isinstance(path, (str, Path)) else path)
    workers = workers or config.get("workers", 1)
    timeout_seconds = timeout_seconds or config.get("timeout_seconds", 30)

    results: list[dict[str, Any]] = []

    if workers <= 1:
        for test_path, test_name, test_func in tests:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(run_test, test_path, test_name, test_func, timeout_seconds)
                try:
                    res = fut.result(timeout=timeout_seconds + 5)  # Buffer for run_test overhead
                    results.append(res)
                except FuturesTimeoutError:
                    results.append(
                        {
                            "test_path": test_path,
                            "test_name": test_name,
                            "status": "error",
                            "duration": timeout_seconds,
                            "error_message": f"Test timed out after {timeout_seconds}s",
                            "assertion_traceback": None,
                            "scores": {},
                            "recorded": {},
                        }
                    )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(run_test, test_path, test_name, test_func, timeout_seconds): (
                    test_path,
                    test_name,
                )
                for test_path, test_name, test_func in tests
            }
            for future in as_completed(futures, timeout=timeout_seconds * len(tests) + 60):
                try:
                    res = future.result(timeout=timeout_seconds)
                    results.append(res)
                except FuturesTimeoutError:
                    test_path, test_name = futures[future]
                    results.append(
                        {
                            "test_path": test_path,
                            "test_name": test_name,
                            "status": "error",
                            "duration": timeout_seconds,
                            "error_message": f"Test timed out after {timeout_seconds}s",
                            "assertion_traceback": None,
                            "scores": {},
                            "recorded": {},
                        }
                    )
                except Exception as e:
                    test_path, test_name = futures[future]
                    results.append(
                        {
                            "test_path": test_path,
                            "test_name": test_name,
                            "status": "error",
                            "duration": 0,
                            "error_message": str(e),
                            "assertion_traceback": traceback.format_exc(),
                            "scores": {},
                            "recorded": {},
                        }
                    )

    return results
