"""Generate synthetic gold sets for testing the scoring gate.

Each row is one candidate response, scored 0-4 by two human raters and by
the model. Both files share the same people and human scores (same seed),
so the only difference between them is the model's bias.
"""
import numpy as np
import pandas as pd


def to_score(x):
    """Round to the nearest whole score and keep it on the 0-4 scale."""
    return np.clip(np.rint(x), 0, 4).astype(int)


def make_gold_set(n=600, female_bias=0.0, seed=42):
    rng = np.random.default_rng(seed)          # reproducible random numbers

    # Half male, half female, in random order
    group = rng.permutation(np.repeat(["Male", "Female"], n // 2))

    # Each person's true ability on the 0-4 scale (mean 2)
    true = 2 + 0.95 * rng.standard_normal(n)

    # Two independent human ratings: truth plus rater noise
    human_1 = to_score(true + 0.32 * rng.standard_normal(n))
    human_2 = to_score(true + 0.32 * rng.standard_normal(n))

    # Model rating: truth plus model noise, minus a bias for female speakers
    model_raw = true + 0.30 * rng.standard_normal(n)
    model_raw = np.where(group == "Female", model_raw - female_bias, model_raw)
    model_score = to_score(model_raw)

    return pd.DataFrame({
        "response_id": np.arange(1, n + 1),
        "group": group,
        "human_1": human_1,
        "human_2": human_2,
        "model_score": model_score,
    })


if __name__ == "__main__":
    make_gold_set().to_csv("data/gold.csv", index=False)
    make_gold_set(female_bias=0.5).to_csv("data/gold_biased.csv", index=False)
    print("Wrote data/gold.csv and data/gold_biased.csv")