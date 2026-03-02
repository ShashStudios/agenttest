"""Terminal output (Rich) and JSON report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

RESULTS_FILE = Path(".agenttest_results.json")


def report_results(
    results: list[dict[str, Any]],
    *,
    live: bool = True,
    fail_threshold: float = 0.8,
) -> dict[str, Any]:
    """
    Display results in terminal and return summary. Writes .agenttest_results.json.
    """
    console = Console()
    passed = sum(1 for r in results if r.get("status") == "pass")
    failed = sum(1 for r in results if r.get("status") == "fail")
    errors = sum(1 for r in results if r.get("status") == "error")
    total = len(results)

    # Summary table
    table = Table(
        title="Agent Eval Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Test", style="dim")
    table.add_column("Status", min_width=8)
    table.add_column("Duration", justify="right")
    table.add_column("Details", overflow="fold")

    for r in results:
        name = r.get("test_name", "?")
        status = r.get("status", "unknown")
        duration = r.get("duration", 0)
        details = r.get("error_message", "") or ""

        if status == "pass":
            status_text = Text("✅ pass", style="bold green")
        elif status == "fail":
            status_text = Text("❌ fail", style="bold red")
        else:
            status_text = Text("⚠️ error", style="bold yellow")

        table.add_row(
            name,
            status_text,
            f"{duration:.2f}s",
            details[:80] + "..." if len(details) > 80 else details,
        )

    console.print()
    console.print(table)
    console.print()

    # Overall score
    score_str = f"{passed}/{total} passed"
    if total > 0:
        pct = passed / total * 100
        if pct >= 100:
            style = "bold green"
        elif pct >= fail_threshold * 100:
            style = "bold yellow"
        else:
            style = "bold red"
        console.print(Panel(score_str, title="Overall", border_style=style))
    else:
        console.print(Panel("No tests found", title="Overall", border_style="yellow"))

    # Show failure details
    for r in results:
        if r.get("status") in ("fail", "error"):
            console.print()
            console.print(Panel(
                r.get("assertion_traceback") or r.get("error_message", "Unknown error"),
                title=f"[red]Failure: {r.get('test_name', '?')}[/red]",
                border_style="red",
            ))

    # Build and save JSON report
    report = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "results": results,
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=2))

    return report


def load_last_report() -> dict[str, Any] | None:
    """Load last run results from .agenttest_results.json."""
    if not RESULTS_FILE.is_file():
        return None
    try:
        return json.loads(RESULTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def report_compare(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    tag_a: str = "v1",
    tag_b: str = "v2",
) -> None:
    """Display comparison between two runs."""
    console = Console()
    table = Table(title=f"Compare {tag_a} vs {tag_b}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column(tag_a, justify="right")
    table.add_column(tag_b, justify="right")
    table.add_column("Δ", justify="right")

    total_a = report_a.get("total", 0)
    total_b = report_b.get("total", 0)
    passed_a = report_a.get("passed", 0)
    passed_b = report_b.get("passed", 0)

    def _delta(a: int | float, b: int | float) -> str:
        d = (b or 0) - (a or 0)
        if d > 0:
            return f"[green]+{d}[/green]"
        if d < 0:
            return f"[red]{d}[/red]"
        return "0"

    table.add_row("Total", str(total_a), str(total_b), _delta(total_a, total_b))
    table.add_row("Passed", str(passed_a), str(passed_b), _delta(passed_a, passed_b))
    pct_a = (passed_a / total_a * 100) if total_a else 0
    pct_b = (passed_b / total_b * 100) if total_b else 0
    table.add_row("Pass %", f"{pct_a:.1f}%", f"{pct_b:.1f}%", _delta(pct_a, pct_b))

    console.print()
    console.print(table)


def report_diff(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    tag_a: str = "v1",
    tag_b: str = "v2",
) -> None:
    """
    Display side-by-side diff of agent responses between two runs.
    The git diff for agent behavior.
    """
    console = Console()
    results_a = {r["test_name"]: r for r in report_a.get("results", [])}
    results_b = {r["test_name"]: r for r in report_b.get("results", [])}

    all_tests = sorted(set(results_a) | set(results_b))
    if not all_tests:
        console.print("[yellow]No tests found in either run.[/yellow]")
        return

    def _short_name(name: str) -> str:
        return name.split("::")[-1] if "::" in name else name

    def _score_str(r: dict[str, Any]) -> str:
        rec = r.get("recorded") or {}
        if rec.get("score") is not None:
            return f"{float(rec['score']):.2f}"
        return "pass" if r.get("status") == "pass" else "fail"

    def _trunc(s: str, n: int = 50) -> str:
        if not s:
            return "(no response)"
        s = s.replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s

    def _num_score(r: dict[str, Any]) -> float:
        rec = r.get("recorded") or {}
        if isinstance(rec.get("score"), (int, float)):
            return float(rec["score"])
        return 1.0 if r.get("status") == "pass" else 0.0

    console.print()
    console.print(Panel(f"[bold]Behavior diff: {tag_a} → {tag_b}[/bold]", border_style="cyan"))
    console.print()

    for test_name in all_tests:
        ra = results_a.get(test_name, {})
        rb = results_b.get(test_name, {})
        short = _short_name(test_name)

        resp_a = (ra.get("recorded") or {}).get("response", "")
        resp_b = (rb.get("recorded") or {}).get("response", "")
        score_a = _score_str(ra) if ra else "—"
        score_b = _score_str(rb) if rb else "—"

        num_a = _num_score(ra) if ra else 0
        num_b = _num_score(rb) if rb else 0
        d = num_b - num_a
        if d > 0:
            delta = f"[green]+{d:.2f} ↑[/green]" if d != 1.0 else "[green]✓ improved[/green]"
        elif d < 0:
            delta = f"[red]{d:.2f} ↓[/red]" if d != -1.0 else "[red]✗ regressed[/red]"
        else:
            delta = "—"

        console.print(f"[bold cyan]{short}:[/bold cyan]")
        console.print(f"  [dim]BEFORE:[/dim] {_trunc(resp_a) or '(no data)'}  [dim]score:[/dim] {score_a}")
        console.print(f"  [dim]AFTER:[/dim]  {_trunc(resp_b) or '(no data)'}  [dim]score:[/dim] {score_b}")
        console.print(f"  [dim]DELTA:[/dim]  {delta}")
        console.print()
