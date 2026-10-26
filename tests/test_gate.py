from pathlib import Path

import pandas as pd

from scoring_gate.gate import format_report, load_config, run_gate

ROOT = Path(__file__).resolve().parents[1]          # the top-level scoring-gate folder
CONFIG = load_config(ROOT / "gate.yaml")
FIX = ROOT / "tests" / "fixtures"

def test_fair_model_passes():
    report = run_gate(pd.read_csv(FIX /"fair.csv"), CONFIG)
    assert report["passed"]


def test_biased_model_fails_on_fairness_not_agreement():
    report = run_gate(pd.read_csv(FIX / "biased.csv"), CONFIG)
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert not report["passed"]
    assert "SMD (Female)" in failed
    assert "QWK lower 95% CI" not in failed            # overall agreement still looks fine


def test_report_renders_markdown():
    report = run_gate(pd.read_csv(FIX/"fair.csv"), CONFIG)
    assert format_report(report).startswith("## Scoring gate: PASS")