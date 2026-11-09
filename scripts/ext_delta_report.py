"""Extension 1 (after stage 7): CLATT forecasting the change rho(t+1) - rho(t) ("CLATT-Δρ").

python -m scripts.ext_delta_report            tables + figures in results/ext/delta/ (git-ignored)
python -m scripts.ext_delta_report --split val  same on validation only (allowed before the test run)

Runs come from:  python -m src.run --exp ext_delta --models CLATT --group all --seeds 0 1 2 3 4
                 --config-override target=delta [--final]
Design fixed before the test read: results/ext/delta/DESIGN.md.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import stats  # noqa: E402
from src.aggregate import collect  # noqa: E402
from src.data import RESULTS  # noqa: E402
from src.evaluate import by_level, by_pair, missing_runs  # noqa: E402

EXP = "ext_delta"
OUT = RESULTS / "ext" / "delta"
METRICS = ["mse", "mae", "r2", "skill", "r2_delta", "dir_acc"]
# (label, exp_b, model_b, pre-registered?)
VERSUS = [("CLATT-Δρ vs Persistence", "main_A", "Persistence", True),
          ("CLATT-Δρ vs AR(1)", "main_A", "AR(1)", True),
          ("CLATT-Δρ vs CLATT (Table 1)", "main_A", "CLATT", True),
          ("CLATT-Δρ vs CLATT (tuned)", "main_B", "CLATT", False),
          ("CLATT-Δρ vs LSTM (tuned)", "main_B", "LSTM", False)]
SHOW = [("main_A", "CLATT", "CLATT (Table 1)"), ("main_B", "CLATT", "CLATT (tuned)"),
        ("main_B", "LSTM", "LSTM (tuned)"), (EXP, "CLATT", "CLATT-Δρ"),
        ("main_A", "Persistence", "Persistence"), ("main_A", "AR(1)", "AR(1)")]


def overview(split):
    """Mean ± std over seeds of the pair mean, per group and overall, for CLATT-Δρ and the reference models."""
    frames = []
    for exp, model, name in SHOW:
        long = collect(exp)
        long = long[(long.model == model) & (long.split == split) & long.metric.isin(METRICS)].copy()
        long["value"] = long.value.astype(float)
        if long.empty:
            continue
        for keys in (["group"], []):
            t = by_level(long, keys).assign(model=name)
            frames.append(t if keys else t.assign(group="all"))
    t = pd.concat(frames)
    t["cell"] = [f"{m:.4f}" if pd.isna(s) else f"{m:.4f} ± {s:.4f}" for m, s in zip(t["mean"], t["std"])]
    wide = t.pivot_table(index=["group", "model"], columns="metric", values="cell", aggfunc="first")
    order = {n: i for i, (_, _, n) in enumerate(SHOW)}
    wide = wide.reset_index()
    wide["group"] = wide.group.astype(str)
    wide = wide.sort_values(["group", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
    return wide[["group", "model"] + [m for m in METRICS if m in wide]]


def tests():
    per, summ = [], []
    for label, exp_b, model_b, pre in VERSUS:
        p, s = stats.compare(EXP, "CLATT", exp_b, model_b, label)
        per.append(p)
        summ.append(s.assign(pre_registered=pre))
    return pd.concat(per, ignore_index=True), pd.concat(summ, ignore_index=True)


def fig_pred(pairs=("sz.300015_sz.300274", "sz.300014_sz.300122"), days=100):
    fig, axes = plt.subplots(len(pairs), 1, figsize=(11, 4 * len(pairs)))
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        y = None
        for exp, model, name in [(EXP, "CLATT", "CLATT-Δρ"), ("main_A", "CLATT", "CLATT (Table 1)"),
                                 ("main_A", "Persistence", "Persistence")]:
            y, pred = stats.mean_forecast(stats.load_preds(exp, model, pair))
            dates = pd.to_datetime(next(iter(stats.load_preds(exp, model, pair).values())).date)
            ax.plot(dates[:days], pred[:days], lw=1, label=name)
        ax.plot(dates[:days], y[:days], color="black", lw=1.6, label="true ρ(t+1)")
        ax.set(title=f"{pair}: first {days} test days", ylabel="ρ")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pred_vs_true.png", dpi=130)
    plt.close(fig)


def fig_skill(per):
    d = per[per.comparison == "CLATT-Δρ vs Persistence"]
    base = per[per.comparison == "CLATT-Δρ vs CLATT (Table 1)"]
    clatt_skill = 1 - base.mse_b.to_numpy() / d.mse_b.to_numpy()  # CLATT (Table 1) vs Persistence
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, 100 * clatt_skill, 0.4, label="CLATT (Table 1)", color="tab:gray")
    ax.bar(x + 0.2, 100 * d.rel_skill, 0.4, label="CLATT-Δρ", color="tab:blue")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [p.replace("sz.", "").replace("sh.", "").replace("_", "\n") for p in d.pair], fontsize=8)
    ax.set(ylabel="skill vs Persistence (%)", title="Test skill = 1 - MSE/MSE(Persistence), mean over seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "skill_vs_persistence.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="test")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    long = collect(EXP)
    long = long[long.split == args.split]
    gaps = missing_runs(long.assign(value=0.0), [0, 1, 2, 3, 4])
    if gaps:
        print(f"WARNING: {len(gaps)} runs missing, e.g. {gaps[:3]}")
    ov = overview(args.split)
    ov.to_csv(OUT / f"overview_{args.split}.csv", index=False)
    print(ov.to_string(index=False))
    if args.split == "test":
        long["value"] = long.value.astype(float)
        by_pair(long[long.metric.isin(METRICS)]).to_csv(OUT / "by_pair_test.csv", index=False)
        per, summ = tests()
        per.to_csv(OUT / "dm_by_pair.csv", index=False)
        summ.to_csv(OUT / "tests.csv", index=False)
        cols = ["comparison", "group", "mse_a", "mse_b", "mean_rel_skill", "n_dm_a_better", "n_dm_b_better",
                "wilcoxon_p", "verdict_a_better", "pre_registered"]
        print(summ[cols].round(4).to_string(index=False))
        fig_pred()
        fig_skill(per)


if __name__ == "__main__":
    main()
