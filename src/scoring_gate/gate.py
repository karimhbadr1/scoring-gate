"""Turn metrics into a pass/fail release decision, using thresholds from a config file."""
import yaml

from scoring_gate.metrics import bootstrap_ci, decision_errors, qwk, smd_by_group


def load_config(path):
    """Read the gate thresholds from a YAML file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def check(name, value, threshold, passed):
    """One row of the report."""
    return {"name": name, "value": round(float(value), 3),
            "threshold": threshold, "passed": bool(passed)}


def run_gate(df, config):
    """Run every check and return a report: overall pass flag plus one row per check."""
    cols = config["columns"]
    ref, second, model, group = cols["reference"], cols["second_rater"], cols["model"], cols["group"]
    agr, fair, dec = config["agreement"], config["fairness"], config["decision"]
    checks = []

    # 1. Agreement with humans, judged on the lower end of its confidence interval
    q = qwk(df[ref], df[model])
    lo, hi = bootstrap_ci(qwk, df[ref], df[model], n_boot=agr["n_boot"])
    checks.append(check("QWK lower 95% CI", lo, f">= {agr['min_qwk_lower_ci']}",
                        lo >= agr["min_qwk_lower_ci"]))

    # 2. Gap to the human-human ceiling
    ceiling = qwk(df[ref], df[second])
    gap = ceiling - q
    checks.append(check("Gap to human-human QWK", gap, f"<= {agr['max_gap_to_human']}",
                        gap <= agr["max_gap_to_human"]))

    # 3. Fairness: SMD in every group
    for g, value in smd_by_group(df, group, ref, model).items():
        checks.append(check(f"SMD ({g})", value, f"abs <= {fair['max_abs_smd']}",
                            abs(value) <= fair["max_abs_smd"]))

    # 4. Decision errors at the cut: false-fail rates shouldn't differ much by group
    type_ii = {g: decision_errors(d[ref], d[model], dec["cut"])["type_ii_rate"]
               for g, d in df.groupby(group)}
    spread = max(type_ii.values()) - min(type_ii.values())
    checks.append(check("Type II rate gap between groups", spread,
                        f"<= {dec['max_type_ii_gap']}", spread <= dec["max_type_ii_gap"]))

    return {
        "passed": all(c["passed"] for c in checks),
        "summary": {"qwk": round(float(q), 3), "qwk_ci": [round(lo, 3), round(hi, 3)],
                    "human_human_qwk": round(float(ceiling), 3),
                    "type_ii_by_group": {g: round(float(v), 3) for g, v in type_ii.items()}},
        "checks": checks,
    }


def format_report(report):
    """Render the report as a markdown table (readable in a terminal or a PR comment)."""
    lines = [f"## Scoring gate: {'PASS' if report['passed'] else 'FAIL'}", "",
             "| Check | Value | Threshold | Result |", "|---|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['name']} | {c['value']} | {c['threshold']} | "
                     f"{'pass' if c['passed'] else 'FAIL'} |")
    return "\n".join(lines)