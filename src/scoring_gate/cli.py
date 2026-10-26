"""Command-line entry point: run the gate on a gold set and exit non-zero on failure."""
import json
from pathlib import Path

import pandas as pd
import typer

from scoring_gate.gate import format_report, load_config, run_gate

app = typer.Typer(add_completion=False, help="Release gate for automated scoring models.")


@app.command()
def main(
    data: Path = typer.Option(..., help="CSV of gold-set scores (human and model)."),
    config: Path = typer.Option(Path("gate.yaml"), help="YAML file with gate thresholds."),
    report: Path = typer.Option(None, help="Optional path to save the markdown report."),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
):
    """Score DATA against the thresholds in CONFIG. Exit code 0 = pass, 1 = fail."""
    result = run_gate(pd.read_csv(data), load_config(config))
    text = format_report(result)

    typer.echo(json.dumps(result, indent=2) if as_json else text)
    if report:
        report.write_text(text)

    raise typer.Exit(code=0 if result["passed"] else 1)