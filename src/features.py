"""Steps 2-5 of plan section 3: log-returns, pair alignment, rolling Pearson label, rho stats."""
from itertools import combinations

import numpy as np
import pandas as pd

from src.data import GROUPS, START, load_raw, traded

K = 20


def returns_table(codes):
    """Wide table of log-returns over each stock's own traded days; NaN on days it did not trade."""
    return pd.DataFrame({c: np.log(traded(load_raw(c)).close).diff() for c in codes})


def pairs():
    return [(g, a, b) for g, codes in GROUPS.items() for a, b in combinations(codes, 2)]


def pair_frame(returns, a, b, k=K, start=START):
    """Inner join on dates both stocks traded (no filling), rho(t) = Pearson over t-k+1..t."""
    df = returns[[a, b]].dropna().set_axis(["r_A", "r_B"], axis=1)
    df["rho"] = df.r_A.rolling(k).corr(df.r_B)
    return df.loc[start:, ["rho", "r_A", "r_B"]]


def rho_stats(rho):
    return {
        "n_days": len(rho), "mean": rho.mean(), "std": rho.std(), "min": rho.min(), "max": rho.max(),
        "acf1_rho": rho.autocorr(1), "acf1_drho": rho.diff().autocorr(1),
        "n_abs_rho_lt_0.05": int((rho.abs() < 0.05).sum()),
    }
