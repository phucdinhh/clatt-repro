"""python -m src.plots <exp>: train/val loss per epoch for every deep-model run of an exp.
python -m src.plots --final main_A main_B: true vs predicted (illustration pairs) and skill per pair (stage 7.6).
python -m src.plots --ablation: bar chart of results/tables/ablation.csv (stage 8.2)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import RESULTS  # noqa: E402
from src.train import RUNS  # noqa: E402


def loss_curves(exp):
    # shortcut: one panel per run, unreadable beyond a few runs; fine for smoke checks, aggregate before stage 5
    runs = sorted((RUNS / exp).glob("*/*/seed*/history.csv"))
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.5), squeeze=False)
    for ax, h in zip(axes[0], runs):
        d = pd.read_csv(h)
        ax.plot(d.epoch, d.train_loss, label="train")
        ax.plot(d.epoch, d.val_loss, label="val")
        ax.set(title=f"{h.parts[-4]} {h.parts[-3]} {h.parts[-2]}", xlabel="epoch", ylabel="MSE (scaled)", yscale="log")
        ax.legend()
    fig.tight_layout()
    out = RESULTS / "figures" / f"{exp}_loss_curves.png"
    fig.savefig(out, dpi=120)
    return out


def pred_vs_true(exp, models=("CLATT", "CNN-BiLSTM", "Persistence"), baselines=None, days=100):
    """First `days` test labels of each illustration pair (the paper shows the first 100), seed-mean forecasts."""
    import yaml

    from src.baselines import BASELINES
    from src.data import ROOT
    from src.stats import load_preds, mean_forecast
    pick = yaml.safe_load((ROOT / "configs" / "stage7_decisions.yaml").read_text())["illustration_pairs"]
    fig, axes = plt.subplots(len(pick), 1, figsize=(11, 3.4 * len(pick)), squeeze=False)
    for ax, (g, pair) in zip(axes[:, 0], pick.items()):
        for i, m in enumerate(models):
            src_exp = baselines if (baselines and m in BASELINES) else exp
            preds = load_preds(src_exp, m, pair)
            y, fc = mean_forecast(preds)
            dates = pd.to_datetime(next(iter(preds.values())).date)[:days]
            if i == 0:
                ax.plot(dates, y[:days], color="black", lw=1.6, label="true rho(t+1)")
            ax.plot(dates, fc[:days], lw=1, label=f"{m} ({len(preds)} seed{'s' if len(preds) > 1 else ''})")
        ax.set(title=f"{exp}, group {g}, {pair}: first {days} test days", ylabel="rho")
        ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    out = RESULTS / "figures" / f"{exp}_pred_vs_true.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def skill_by_pair(models=("CLATT", "CNN-BiLSTM", "CNN-LSTM", "LSTM", "CNN", "AR(1)", "Ridge", "MA5")):
    """results/tables/skill_by_pair.csv (from src.evaluate --compare): skill vs persistence per pair."""
    d = pd.read_csv(RESULTS / "tables" / "skill_by_pair.csv")
    exps = list(d.exp.unique())
    fig, axes = plt.subplots(len(exps), 1, figsize=(12, 3.8 * len(exps)), squeeze=False, sharey=True)
    for ax, exp in zip(axes[:, 0], exps):
        e = d[d.exp == exp]
        order = list(dict.fromkeys(e.sort_values(["group", "pair"]).pair))
        ms = [m for m in models if m in set(e.model)]
        w = 0.8 / len(ms)
        for i, m in enumerate(ms):
            v = e[e.model == m].set_index("pair").reindex(order)
            ax.bar([j + i * w for j in range(len(order))], v.skill, w, yerr=np.nan_to_num(v.skill_std), label=m,
                   error_kw={"lw": 0.6})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks([j + 0.4 - w / 2 for j in range(len(order))])
        ax.set_xticklabels([p.replace("sz.", "").replace("sh.", "").replace("_", "\n") for p in order], fontsize=7)
        ax.set(title=f"{exp}: skill = 1 - MSE/MSE(persistence) on test (mean ± std over seeds)", ylabel="skill")
        ax.legend(fontsize=7, ncol=len(ms))
    fig.tight_layout()
    out = RESULTS / "figures" / "skill_by_pair.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def ablation_bar():
    d = pd.read_csv(RESULTS / "tables" / "ablation.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = ["all", "1", "2"]
    w = 0.25
    for i, g in enumerate(groups):
        v = d[d.group.astype(str) == g]
        ax.bar([j + i * w for j in range(len(v))], v.mse_change_pct, w, label=f"group {g}")
    ax.axhline(0, color="black", lw=0.8)
    v = d[d.group.astype(str) == "all"]
    ax.set_xticks([j + w for j in range(len(v))])
    ax.set_xticklabels([f"{r.variant}\np={r.wilcoxon_p:.3f}" for r in v.itertuples()], fontsize=8)
    ax.set(ylabel="test MSE change vs full CLATT (%)",
           title="Ablations (protocol A; >0: removing the part hurts; p = Wilcoxon, 12 pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "figures" / "ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--final"]:
        exps = args[1:]
        for exp in exps:
            print(pred_vs_true(exp, baselines=exps[0] if exp != exps[0] else None))
        print(skill_by_pair())
    elif args == ["--ablation"]:
        print(ablation_bar())
    else:
        print(loss_curves(args[0]))
