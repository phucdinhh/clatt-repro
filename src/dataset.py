"""Steps 6-8 of plan section 3: sliding windows, purged time split, MinMax scaling.

`python -m src.dataset` runs the whole pipeline and writes the stage-1 tables and figures.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

from src.data import CODES, END, PAPER_DAYS, PROCESSED, RESULTS, START, load_raw, traded
from src.features import K, pair_frame, pairs, returns_table, rho_stats

T = 68
FEATURES = ["rho", "r_A", "r_B"]
FRACS = (0.7, 0.1, 0.2)


def split_bounds(n, gap, fracs=FRACS):
    """Sample-index slices for train/val/test; `gap` samples are purged at each boundary first."""
    usable = n - 2 * gap
    n_tr, n_va = round(usable * fracs[0]), round(usable * fracs[1])
    return {"train": slice(0, n_tr), "val": slice(n_tr + gap, n_tr + gap + n_va),
            "test": slice(n_tr + n_va + 2 * gap, n)}


def make_dataset(df, T=T, k=K, features=FEATURES, target="next", fixed_split_T=None):
    """Sample i: X = rows i..i+T-1 of `features`, y = rho at row i+T-1+h. Scaler fit on train rows only.

    target="next": h=1, y = rho(t+1) (overlaps X by k-1 returns).
    target="future_window": h=k, y = Pearson over returns t+1..t+k, no return shared with X.
    fixed_split_T: split as if T were this (>= T), so every split keeps the same label dates for any T.
    """
    h = {"next": 1, "future_window": k}[target]
    n = len(df) - T - h + 1
    Ts = fixed_split_T or T
    if Ts < T:
        raise ValueError(f"fixed_split_T={Ts} < T={T}")
    # sample j under T has the label date of sample j-(Ts-T) under Ts; the purge of Ts covers T
    splits = {name: slice(s.start + Ts - T, s.stop + Ts - T)
              for name, s in split_bounds(n - (Ts - T), gap=k + Ts + h - 1).items()}
    s_tr = splits["train"]
    train_rows = df.iloc[s_tr.start: s_tr.stop + T - 1 + h]  # every row a train input or label touches
    scaler = MinMaxScaler().fit(train_rows[features].to_numpy())
    v = scaler.transform(df[features].to_numpy()).astype(np.float32)
    X = sliding_window_view(v, T, axis=0)[:n].transpose(0, 2, 1)  # (n, T, F)
    y = v[T - 1 + h:, features.index("rho")] if "rho" in features else None
    rho = df.rho.to_numpy()
    return {"X": X, "y": y, "x_end": df.index[T - 1: T - 1 + n], "y_date": df.index[T - 1 + h:],
            "y_raw": rho[T - 1 + h:], "rho_prev": rho[T - 1: T - 1 + n],  # original scale, for metrics
            "splits": splits, "scaler": scaler, "train_rows": train_rows[features], "h": h}


def get_pair_data(pair, T=T, k=K, features=FEATURES, fixed_split_T=None):
    """`pair` is 'sz.300015_sz.300274'; rebuilt from raw CSVs so k and T can vary."""
    a, b = pair.split("_")
    return make_dataset(pair_frame(returns_table([a, b]), a, b, k), T, k, features, fixed_split_T=fixed_split_T)


def halt_runs(halted, market_days):
    """Compress halt dates into 'first..last' runs of consecutive market days."""
    pos = market_days.get_indexer(halted)
    runs, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[i - 1] + 1:
            a, b = halted[start].date(), halted[i - 1].date()
            runs.append(f"{a}" if a == b else f"{a}..{b}")
            start = i
    return "; ".join(runs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables, figures = RESULTS / "tables", RESULTS / "figures"
    for d in (tables, figures, PROCESSED / "pairs"):
        d.mkdir(parents=True, exist_ok=True)

    summary = []
    for c in CODES:
        raw = load_raw(c).loc[START:END]
        ok = traded(raw)
        halted = raw.index.difference(ok.index)
        summary.append({"code": c, "market_days": len(raw), "traded_days": len(ok),
                        "halt_days": len(halted), "paper_days": PAPER_DAYS,
                        "diff_vs_paper": len(ok) - PAPER_DAYS, "halt_dates": halt_runs(halted, raw.index)})
    pd.DataFrame(summary).to_csv(tables / "data_summary.csv", index=False)

    returns = returns_table(CODES)
    returns.to_parquet(PROCESSED / "returns.parquet")

    stats, split_rows = [], []
    for g, a, b in pairs():
        name = f"{a}_{b}"
        df = pair_frame(returns, a, b)
        df.to_parquet(PROCESSED / "pairs" / f"{name}.parquet")
        ds = make_dataset(df)
        stats.append({"group": g, "pair": name, **rho_stats(df.rho)})

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df.index, df.rho, lw=0.8, color="black")
        for (split, s), color in zip(ds["splits"].items(), ["tab:blue", "tab:orange", "tab:green"]):
            dates = ds["y_date"][s]
            split_rows.append({"group": g, "pair": name, "split": split, "first_label": dates[0].date(),
                               "last_label": dates[-1].date(), "first_input": df.index[s.start].date(), "n_samples": len(dates)})
            ax.axvspan(dates[0], dates[-1], color=color, alpha=0.15, label=split)
        ax.set(title=f"rho (k={K}) {a} vs {b}", ylabel="rho", ylim=(-1, 1))
        ax.legend(loc="lower left", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"rho_series_{name}.png", dpi=120)
        plt.close(fig)

    pd.DataFrame(stats).round(4).to_csv(tables / "rho_stats.csv", index=False)
    pd.DataFrame(split_rows).to_csv(tables / "split_dates.csv", index=False)


if __name__ == "__main__":
    main()
