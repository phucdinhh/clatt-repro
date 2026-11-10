"""Steps 2-5 of plan section 3: log-returns, pair alignment, rolling Pearson label, rho stats."""
from itertools import combinations

import numpy as np
import pandas as pd

from src.data import GROUPS, RAW, START, load_raw, traded

K = 20
# extension 3: market features from data/raw/index/ (python -m scripts.download_baostock --index)
INDICES = {"cyb": "sz.399006", "hs300": "sh.000300"}  # ChiNext Index, CSI 300
MARKET_FEATURES = [f"m_{x}_{name}" for name in INDICES for x in ("ret", "vol")]


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


LEAVE_FEATURE = "x_leave"


def add_leave(df, k=K):
    """Extension 3b: x_leave(t) = z_A * z_B of day t-k+1, the day that drops out of the rho window at t+1,
    standardized with the mean/std of the current window t-k+1..t. Uses returns up to day t only."""
    out = df.copy()
    z = {}
    for c in ("r_A", "r_B"):
        z[c] = (df[c].shift(k - 1) - df[c].rolling(k).mean()) / df[c].rolling(k).std()
    out[LEAVE_FEATURE] = z["r_A"] * z["r_B"]
    return out


def market_table(k=K):
    """Per index, on its own trading calendar: m_ret = log-return of day t, m_vol = std of the last k log-returns
    (t-k+1..t). Every value uses data up to day t only."""
    cols = {}
    for name, code in INDICES.items():
        close = pd.read_csv(RAW / "index" / f"{code}.csv", parse_dates=["date"]).set_index("date").close
        r = np.log(close).diff()
        cols[f"m_ret_{name}"], cols[f"m_vol_{name}"] = r, r.rolling(k).std()
    return pd.DataFrame(cols)


def add_market(df, k=K):
    """Attach the market features to a pair frame by date (the pair's dates are a subset of market days)."""
    return df.join(market_table(k), how="left")  # NaN only in the warm-up rows; checked after trimming


def rho_stats(rho):
    return {
        "n_days": len(rho), "mean": rho.mean(), "std": rho.std(), "min": rho.min(), "max": rho.max(),
        "acf1_rho": rho.autocorr(1), "acf1_drho": rho.diff().autocorr(1),
        "n_abs_rho_lt_0.05": int((rho.abs() < 0.05).sum()),
    }
