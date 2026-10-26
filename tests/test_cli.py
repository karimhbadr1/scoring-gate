from pathlib import Path

from typer.testing import CliRunner

from scoring_gate.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def run(*args):
    return runner.invoke(app, [*args, "--config", str(ROOT / "gate.yaml")])


def test_pass_exits_zero():
    result = run("--data", str(ROOT / "tests" / "fixtures" / "fair.csv"))
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_fail_exits_one():
    result = run("--data", str(ROOT / "tests" / "fixtures" / "biased.csv"))
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_report_file_written(tmp_path):
    out = tmp_path / "report.md"
    run("--data", str(ROOT / "tests" / "fixtures" / "fair.csv"), "--report", str(out))
    assert out.read_text().startswith("## Scoring gate")