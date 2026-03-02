"""CLI for agenttest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .config import load_config
from .reporter import RESULTS_FILE, load_last_report, report_compare, report_diff, report_results
from .runner import discover_tests, run_tests

console = Console()


def _results_path(tag: str | None = None) -> Path:
    """Path for results file, optionally tagged."""
    if tag:
        return Path(f".agenttest_results_{tag}.json")
    return RESULTS_FILE


@click.group()
@click.version_option(version="0.1.0")
def app() -> None:
    """The pytest of AI agents. Eval-driven testing for LLM applications."""
    pass


@app.command()
@click.option("--path", "-p", default=".", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--filter", "-k", "filter_pattern", default=None)
@click.option("--workers", "-w", default=4, type=int)
@click.option("--tag", "-t", default=None)
def run(path: str, filter_pattern: Optional[str], workers: int, tag: Optional[str]) -> None:
    """Discover and run all agent eval tests."""
    from .config import get_api_key

    path_obj = Path(path)
    config = load_config(path_obj)
    try:
        get_api_key(config)
    except ValueError as e:
        console.print(str(e), style="red")
        sys.exit(1)

    timeout = config.get("timeout_seconds", 30)
    fail_threshold = config.get("fail_threshold", 0.8)

    tests = discover_tests(path_obj, filter_pattern)
    if not tests:
        console.print(Panel(
            "No tests found. Create files matching agent_test_*.py or *_agent_test.py with @eval decorated functions.",
            title="[yellow]No Tests[/yellow]",
            border_style="yellow",
        ))
        sys.exit(1)

    console.print(f"[cyan]Running {len(tests)} test(s)...[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing evals...", total=len(tests))
        results = run_tests(path_obj, filter_pattern, workers=workers, timeout_seconds=timeout)
        progress.update(task, completed=len(tests))

    report = report_results(results, fail_threshold=fail_threshold)

    if tag:
        out = _results_path(tag)
        out.write_text(json.dumps(report, indent=2))
        console.print(f"\n[dim]Results saved to {out}[/dim]")

    total = report.get("total", 0)
    passed = report.get("passed", 0)
    if total > 0 and (passed / total) < fail_threshold:
        sys.exit(1)


@app.command()
@click.option("--path", "-p", default=".", type=click.Path())
def init(path: str) -> None:
    """Scaffold agent_test_example.py, agenttest.toml, and GitHub Actions workflow."""
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)

    example_content = '''"""Example agent eval tests. Replace my_agent with your agent."""

from agenttest import eval, judge, record


def my_agent(query: str) -> str:
    """Mock agent - replace with your actual agent call."""
    if "refund" in query.lower():
        return (
            "I'm sorry to hear you're unhappy. Our refund policy allows "
            "returns within 30 days. Would you like me to start the refund process?"
        )
    return "How can I help you today?"


@eval
def test_customer_support_refund():
    query = "I want a refund"
    response = my_agent(query)
    record(query, response)
    assert judge.tone(response) == "empathetic"
    assert judge.contains_action(response, "refund_policy")
    assert judge.no_hallucination(response)


@eval
def test_customer_support_generic():
    query = "What are your hours?"
    response = my_agent(query)
    record(query, response)
    assert judge.relevance(response, "business hours") >= 0.3  # May be generic
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
'''

    toml_content = '''[agenttest]
model = "claude-3-5-haiku-latest"
timeout_seconds = 30
workers = 4
fail_threshold = 0.8
cache = true

[agenttest.env]
ANTHROPIC_API_KEY = "$ANTHROPIC_API_KEY"
'''

    workflow_content = '''name: Agent Evals
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install agenttest-py
      - run: agenttest run
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
'''

    (path_obj / "agent_test_example.py").write_text(example_content)
    (path_obj / "agenttest.toml").write_text(toml_content)
    workflows_dir = path_obj / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    (workflows_dir / "agenttest.yml").write_text(workflow_content)

    console.print("[green]✓[/green] Created agent_test_example.py")
    console.print("[green]✓[/green] Created agenttest.toml")
    console.print("[green]✓[/green] Created .github/workflows/agenttest.yml")
    console.print("\n[dim]Run: agenttest run[/dim]")
    console.print("[dim]Set ANTHROPIC_API_KEY in your environment or agenttest.toml[/dim]")


@app.command()
def report() -> None:
    """Show last run results from .agenttest_results.json."""
    data = load_last_report()
    if not data:
        console.print("[yellow]No previous run found. Run 'agenttest run' first.[/yellow]")
        sys.exit(1)

    report_results(data.get("results", []), live=False)


@app.command()
@click.argument("v1")
@click.argument("v2")
def compare(v1: str, v2: str) -> None:
    """Compare pass/fail metrics between two tagged runs."""
    p1 = _results_path(v1)
    p2 = _results_path(v2)
    if not p1.is_file():
        console.print(f"[red]No results found for tag '{v1}'. Run: agenttest run --tag {v1}[/red]")
        sys.exit(1)
    if not p2.is_file():
        console.print(f"[red]No results found for tag '{v2}'. Run: agenttest run --tag {v2}[/red]")
        sys.exit(1)

    r1 = json.loads(p1.read_text())
    r2 = json.loads(p2.read_text())
    report_compare(r1, r2, tag_a=v1, tag_b=v2)


@app.command()
@click.argument("v1")
@click.argument("v2")
def diff(v1: str, v2: str) -> None:
    """Show side-by-side diff of agent responses between two runs. The git diff for agent behavior."""
    p1 = _results_path(v1)
    p2 = _results_path(v2)
    if not p1.is_file():
        console.print(f"[red]No results found for tag '{v1}'. Run: agenttest run --tag {v1}[/red]")
        sys.exit(1)
    if not p2.is_file():
        console.print(f"[red]No results found for tag '{v2}'. Run: agenttest run --tag {v2}[/red]")
        sys.exit(1)

    r1 = json.loads(p1.read_text())
    r2 = json.loads(p2.read_text())
    report_diff(r1, r2, tag_a=v1, tag_b=v2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
