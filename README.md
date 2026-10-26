# scoring-gate

A release gate for automated scoring models. Before a new or updated scoring model
replaces the live one, it must agree with expert raters, treat groups equally, and
make similar pass/fail errors across groups, on a fixed human-scored gold set.

## What it checks

| Check | Why |
|---|---|
| QWK with humans, judged on the lower 95% bootstrap bound | Agreement on an ordinal scale, with its uncertainty |
| Gap to human–human QWK | Human raters don't agree perfectly; that's the realistic ceiling |
| Standardised mean difference by group | Is the model scoring any group higher or lower than humans do? |
| Type II (false-fail) rate gap between groups at the pass mark | Does the model wrongly fail one group more often? |

Thresholds live in `gate.yaml`, agreed before results are seen, so changing one is a reviewable diff.

## Usage

```bash
pip install -e ".[dev]"
scoring-gate --data data/gold.csv --config gate.yaml --report report.md
```

Exit code 0 = pass, 1 = fail. Add `--json` for machine-readable output.

## CI

GitHub Actions runs the unit tests, then the gate, on every pull request. A failing
gate blocks the merge and posts the report to the run summary.
Example: a candidate model that under-scores female speakers passes on overall
agreement (QWK .79, CI lower bound .76) but fails on SMD (−.38) and the false-fail gap (.21).

## Layout

```
src/scoring_gate/metrics.py   QWK, bootstrap CI, SMD, decision errors, Cohen's d
src/scoring_gate/gate.py      thresholds → pass/fail report
src/scoring_gate/cli.py       command-line entry point (exit codes)
tests/                        unit tests on fixed fixtures, QWK checked against scikit-learn
scripts/make_gold_set.py      synthetic gold sets (fair and biased model)
```

## Limitations and next steps

- Data is synthetic. In production, a CI step would first re-score the gold set with
  the candidate model, then run the gate on those scores.
- Human score 1 is the reference; an adjudicated or averaged score would be better.
- Next: re-score consistency (LLM non-determinism), an equivalence test against the
  current model, per-item checks, and a production re-scoring sample for drift.