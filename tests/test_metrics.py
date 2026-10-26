import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scoring_gate.metrics import (qwk, exact_agreement, bootstrap_ci, smd,
                                  smd_by_group, decision_errors, cohens_d)

rng = np.random.default_rng(1)
A = rng.integers(0, 5, 300)
B = np.clip(A + rng.integers(-1, 2, 300), 0, 4)


def test_qwk_perfect_agreement_is_one():
    assert qwk(A, A) == 1.0


def test_qwk_matches_sklearn():
    expected = cohen_kappa_score(A, B, weights="quadratic", labels=range(5))
    assert np.isclose(qwk(A, B), expected)


def test_exact_agreement_bounds():
    assert exact_agreement(A, A) == 1.0
    assert 0 <= exact_agreement(A, B) <= 1


def test_bootstrap_ci_contains_estimate():
    lo, hi = bootstrap_ci(qwk, A, B, n_boot=300)
    assert lo <= qwk(A, B) <= hi


def test_smd_sign():
    human = np.array([2, 3, 2, 3])
    assert smd(human, human - 1) < 0      # model lower than humans
    assert smd(human, human) == 0


def test_smd_by_group_detects_biased_group():
    df = pd.DataFrame({"group": ["F"] * 4 + ["M"] * 4,
                       "human_1": [2, 3, 2, 3] * 2,
                       "model_score": [1, 2, 1, 2, 2, 3, 2, 3]})
    out = smd_by_group(df)
    assert out["F"] < -0.5 and out["M"] == 0


def test_decision_errors_known_case():
    human = [1, 1, 3, 3]          # cut = 2: two below, two above
    model = [3, 1, 1, 3]          # one false pass, one false fail
    e = decision_errors(human, model, cut=2)
    assert e["type_i_rate"] == 0.5 and e["type_ii_rate"] == 0.5


def test_cohens_d_zero_when_groups_equal():
    df = pd.DataFrame({"group": ["Female", "Male"] * 4, "s": [1, 1, 2, 2, 3, 3, 4, 4]})
    assert abs(cohens_d(df, "s")) < 1e-12