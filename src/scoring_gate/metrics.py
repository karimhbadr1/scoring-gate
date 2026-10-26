"""Agreement, fairness and decision metrics for automated scoring."""
import numpy as np
import pandas as pd


def qwk(a, b, k=5):
    """Quadratic weighted kappa between two raters on a 0..k-1 integer scale.

    1 = perfect agreement, 0 = chance level. Disagreements are penalised by
    the squared distance between scores, so 0 vs 4 costs far more than 2 vs 3.
    """
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    observed = np.zeros((k, k))
    np.add.at(observed, (a, b), 1)                       # confusion matrix of counts
    expected = np.outer(np.bincount(a, minlength=k),
                        np.bincount(b, minlength=k)) / n  # counts expected by chance
    i, j = np.indices((k, k))
    weights = (i - j) ** 2 / (k - 1) ** 2                 # 0 on the diagonal, 1 at the corners
    return 1 - (weights * observed).sum() / (weights * expected).sum()


def exact_agreement(a, b):
    """Share of responses where the two raters give the identical score."""
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def bootstrap_ci(metric, a, b, n_boot=1000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval for a two-rater metric.

    Resamples responses with replacement, recomputes the metric each time,
    and takes the middle 95% of the results.
    """
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                       # resample row positions
        stats.append(metric(a[idx], b[idx]))
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def smd(human, model):
    """Standardised mean difference: (model mean - human mean) / human SD.

    Negative means the model scores lower than humans on the same responses.
    """
    human, model = np.asarray(human, float), np.asarray(model, float)
    return float((model.mean() - human.mean()) / human.std(ddof=1))


def smd_by_group(df, group_col="group", human_col="human_1", model_col="model_score"):
    """SMD for each group, e.g. {'Female': -0.52, 'Male': 0.01}."""
    return {g: smd(d[human_col], d[model_col]) for g, d in df.groupby(group_col)}


def decision_errors(human, model, cut):
    """Classification errors at a pass mark, treating human scores as the truth.

    Type I (false pass): model passes someone humans place below the cut.
    Type II (false fail): model fails someone humans place at or above it.
    """
    h = np.asarray(human) >= cut
    m = np.asarray(model) >= cut
    tp, fp = np.sum(h & m), np.sum(~h & m)
    fn, tn = np.sum(h & ~m), np.sum(~h & ~m)
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if tp else 0.0
    return {
        "type_i_rate": 1 - specificity,
        "type_ii_rate": 1 - sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def cohens_d(df, score_col, group_col="group", focal="Female", reference="Male"):
    """Standardised score gap between two groups, using the pooled SD."""
    f = df.loc[df[group_col] == focal, score_col].astype(float)
    r = df.loc[df[group_col] == reference, score_col].astype(float)
    pooled = np.sqrt(((len(f) - 1) * f.var() + (len(r) - 1) * r.var())
                     / (len(f) + len(r) - 2))
    return float((f.mean() - r.mean()) / pooled)