"""Extension 2 (after stage 7): one CLATT per group, pooled over the 6 pairs ("CLATT-group").

python -m scripts.ext_group_report               tables + figures in results/ext/group/ (git-ignored)
python -m scripts.ext_group_report --split val   validation only

Runs: python -m src.run --exp ext_group[_delta] --models CLATT --group 1|2 --seeds 0 1 2 3 4 --pooled
      [--config-override target=delta] --final.   Design fixed before the test read: results/ext/group/DESIGN.md.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.ext_delta_report import METRICS  # noqa: E402
from src import stats  # noqa: E402
from src.aggregate import collect  # noqa: E402
from src.data import RESULTS  # noqa: E402
from src.evaluate import by_level, missing_runs  # noqa: E402

OUT = RESULTS / "ext" / "group"
# (exp, model, display name)
SHOW = [("main_A", "CLATT", "CLATT (Table 1)"), ("ext_group", "CLATT", "CLATT-group"),
        ("ext_delta", "CLATT", "CLATT-Δρ"), ("ext_group_delta", "CLATT", "CLATT-group-Δρ"),
        ("main_B", "LSTM", "LSTM (tuned)"), ("main_A", "Persistence", "Persistence"), ("main_A", "AR(1)", "AR(1)")]
# (label, exp_a, exp_b, model_b, pre-registered?)
VERSUS = [("CLATT-group vs CLATT (Table 1)", "ext_group", "main_A", "CLATT", True),
          ("CLATT-group vs Persistence", "ext_group", "main_A", "Persistence", True),
          ("CLATT-group vs AR(1)", "ext_group", "main_A", "AR(1)", True),
          ("CLATT-group-Δρ vs CLATT-Δρ", "ext_group_delta", "ext_delta", "CLATT", True),
          ("CLATT-group-Δρ vs Persistence", "ext_group_delta", "main_A", "Persistence", True),
          ("CLATT-group-Δρ vs AR(1)", "ext_group_delta", "main_A", "AR(1)", True),
          ("CLATT-group vs LSTM (tuned)", "ext_group", "main_B", "LSTM", False)]


def overview(split):
    frames = []
    for exp, model, name in SHOW:
        long = collect(exp)
        long = long[(long.model == model) & (long.split == split) & long.metric.isin(METRICS)].copy()
        if long.empty:
            continue
        long["value"] = long.value.astype(float)
        for keys in (["group"], []):
            t = by_level(long, keys).assign(model=name)
            frames.append(t if keys else t.assign(group="all"))
    t = pd.concat(frames)
    t["cell"] = [f"{m:.4f}" if pd.isna(s) else f"{m:.4f} ± {s:.4f}" for m, s in zip(t["mean"], t["std"])]
    wide = t.pivot_table(index=["group", "model"], columns="metric", values="cell", aggfunc="first").reset_index()
    order = {n: i for i, (_, _, n) in enumerate(SHOW)}
    wide["group"] = wide.group.astype(str)
    wide = wide.sort_values(["group", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
    return wide[["group", "model"] + [m for m in METRICS if m in wide]]


def tests():
    per, summ = [], []
    for label, exp_a, exp_b, model_b, pre in VERSUS:
        p, s = stats.compare(exp_a, "CLATT", exp_b, model_b, label)
        per.append(p)
        summ.append(s.assign(pre_registered=pre))
    return pd.concat(per, ignore_index=True), pd.concat(summ, ignore_index=True)


def fig_mse_by_pair():
    """Test MSE per pair (mean over seeds) for the four CLATT variants and Persistence."""
    rows = []
    for exp, model, name in SHOW[:4] + [SHOW[5]]:
        long = collect(exp)
        long = long[(long.model == model) & (long.split == "test") & (long.metric == "mse")]
        rows.append(long.assign(value=long.value.astype(float)).groupby("pair").value.mean().rename(name))
    t = pd.concat(rows, axis=1)
    x = np.arange(len(t))
    w = 0.8 / t.shape[1]
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, c in enumerate(t.columns):
        ax.bar(x + (i - (t.shape[1] - 1) / 2) * w, t[c], w, label=c, color="black" if c == "Persistence" else None)
    ax.set_xticks(x, [p.replace("sz.", "").replace("sh.", "").replace("_", "\n") for p in t.index], fontsize=8)
    ax.set(ylabel="test MSE", title="Test MSE per pair (mean over 5 seeds)")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(OUT / "mse_by_pair.png", dpi=130)
    plt.close(fig)


def fig_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=False)
    for ax, g in zip(axes, (1, 2)):
        for s in range(5):
            f = RESULTS / "runs" / "ext_group" / "CLATT" / f"_group{g}" / f"seed{s}" / "history.csv"
            if f.exists():
                h = pd.read_csv(f)
                ax.plot(h.epoch, h.val_loss, lw=1, label=f"seed {s}")
        ax.set(title=f"CLATT-group, group {g}: pooled val loss (scaled)", xlabel="epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="test")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for exp in ("ext_group", "ext_group_delta"):
        long = collect(exp)
        gaps = missing_runs(long[long.split == args.split], [0, 1, 2, 3, 4])
        if gaps:
            print(f"WARNING {exp}: {len(gaps)} runs missing, e.g. {gaps[:3]}")
    ov = overview(args.split)
    ov.to_csv(OUT / f"overview_{args.split}.csv", index=False)
    print(ov.to_string(index=False))
    if args.split == "test":
        per, summ = tests()
        per.to_csv(OUT / "dm_by_pair.csv", index=False)
        summ.to_csv(OUT / "tests.csv", index=False)
        cols = ["comparison", "group", "mse_a", "mse_b", "mean_rel_skill", "n_dm_a_better", "n_dm_b_better",
                "wilcoxon_p", "verdict_a_better", "pre_registered"]
        print(summ[cols].round(4).to_string(index=False))
        fig_mse_by_pair()
        fig_loss_curves()


if __name__ == "__main__":
    main()
