"""Extension 3 (after stage 7): extra inputs for CLATT-Δρ.

  3a  market features (ChiNext and CSI 300: daily log-return and 20-day volatility)      exp ext_market
  3b  departing-day feature x_leave (z_A*z_B of the day leaving the rho window at t+1)   exp ext_leave
  3c  both                                                                               exp ext_leave_market

Two deterministic baselines that use the same information as 3b, so a gain is not credited to the network:
  Persist-19    rho over the last k-1 = 19 days (the part of the next window already known)
  Linear-leave  OLS of Δρ on [rho(t), x_leave(t)], fit on each pair's train samples; forecast rho(t) + Δρ̂

python -m scripts.ext_feat_report --baselines    write the two baselines as runs in results/runs/ext_feat_base/
python -m scripts.ext_feat_report [--split val]  tables + figures in results/ext/features/ (git-ignored)
Design fixed before the test read: results/ext/features/DESIGN.md.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from scripts.ext_delta_report import METRICS  # noqa: E402
from src import stats  # noqa: E402
from src.aggregate import collect  # noqa: E402
from src.data import RESULTS  # noqa: E402
from src.dataset import get_pair_data  # noqa: E402
from src.evaluate import by_level, missing_runs  # noqa: E402
from src.features import K, LEAVE_FEATURE, pair_frame, pairs, returns_table  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.train import RUNS, load_config  # noqa: E402

OUT = RESULTS / "ext" / "features"
BASE = "ext_feat_base"
SHOW = [("ext_delta", "CLATT", "CLATT-Δρ"), ("ext_market", "CLATT", "CLATT-Δρ + market"),
        ("ext_leave", "CLATT", "CLATT-Δρ + leave"), ("ext_leave_market", "CLATT", "CLATT-Δρ + leave + market"),
        ("main_A", "Persistence", "Persistence"), ("main_A", "AR(1)", "AR(1)"),
        (BASE, "Persist-19", "Persist-19"), (BASE, "Linear-leave", "Linear-leave")]
# (label, exp_a, model_a, exp_b, model_b, pre-registered?)
VERSUS = [
    ("market vs CLATT-Δρ", "ext_market", "CLATT", "ext_delta", "CLATT", True),
    ("market vs Persistence", "ext_market", "CLATT", "main_A", "Persistence", True),
    ("market vs AR(1)", "ext_market", "CLATT", "main_A", "AR(1)", True),
    ("leave vs CLATT-Δρ", "ext_leave", "CLATT", "ext_delta", "CLATT", True),
    ("leave vs Persistence", "ext_leave", "CLATT", "main_A", "Persistence", True),
    ("leave vs AR(1)", "ext_leave", "CLATT", "main_A", "AR(1)", True),
    ("leave vs Persist-19", "ext_leave", "CLATT", BASE, "Persist-19", True),
    ("leave vs Linear-leave", "ext_leave", "CLATT", BASE, "Linear-leave", True),
    ("leave+market vs leave", "ext_leave_market", "CLATT", "ext_leave", "CLATT", True),
    ("leave+market vs Persistence", "ext_leave_market", "CLATT", "main_A", "Persistence", True),
    ("Linear-leave vs Persistence", BASE, "Linear-leave", "main_A", "Persistence", True),
    ("Persist-19 vs Persistence", BASE, "Persist-19", "main_A", "Persistence", True),
]


def write_baselines(final=True):
    """Deterministic baselines for 3b, saved like seed-0 runs so src.stats can compare them."""
    cfg = load_config("CLATT")
    for _, a, b in pairs():
        pair = f"{a}_{b}"
        d = get_pair_data(pair, cfg["T"], cfg["k"], features=["rho", "r_A", "r_B", LEAVE_FEATURE], target="delta")
        full = pair_frame(returns_table([a, b]), a, b, K, start=None)
        rho19 = full.r_A.rolling(K - 1).corr(full.r_B).reindex(d["x_end"]).to_numpy()
        leave = d["X"][:, -1, 3]  # scaled x_leave at day t (scaling does not change an OLS fit)
        Z = np.column_stack([d["rho_prev"], leave])
        dy = d["y_raw"] - d["rho_prev"]
        tr = d["splits"]["train"]
        ols = LinearRegression().fit(Z[tr], dy[tr])
        preds = {"Persist-19": rho19, "Linear-leave": d["rho_prev"] + ols.predict(Z)}
        for name, pred in preds.items():
            out = RUNS / BASE / name / pair / "seed0"
            out.mkdir(parents=True, exist_ok=True)
            metrics = {}
            for split in ("val", "test") if final else ("val",):
                sl = d["splits"][split]
                pd.DataFrame({"date": d["y_date"][sl].date, "y_true": d["y_raw"][sl], "y_pred": pred[sl],
                              "rho_prev": d["rho_prev"][sl]}).to_csv(out / f"preds_{split}.csv", index=False)
                metrics[split] = compute_all(d["y_raw"][sl], pred[sl], d["rho_prev"][sl])
            info = {"n_params": 3 if name == "Linear-leave" else 0, "train_seconds": 0.0}
            if name == "Linear-leave":
                info["coef_rho"], info["coef_leave"] = map(float, ols.coef_)
            (out / "metrics.json").write_text(json.dumps({**metrics, **info}, indent=2))


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
    for label, ea, ma, eb, mb, pre in VERSUS:
        p, s = stats.compare(ea, ma, eb, mb, label)
        per.append(p)
        summ.append(s.assign(pre_registered=pre))
    return pd.concat(per, ignore_index=True), pd.concat(summ, ignore_index=True)


def fig_skill():
    rows = {}
    for exp, model, name in SHOW:
        if model == "Persistence":
            continue
        long = collect(exp)
        long = long[(long.model == model) & (long.split == "test") & (long.metric == "skill")]
        rows[name] = long.assign(value=long.value.astype(float)).groupby("pair").value.mean()
    t = pd.DataFrame(rows)
    x = np.arange(len(t))
    w = 0.85 / t.shape[1]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for i, c in enumerate(t.columns):
        ax.bar(x + (i - (t.shape[1] - 1) / 2) * w, 100 * t[c], w, label=c)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [p.replace("sz.", "").replace("sh.", "").replace("_", "\n") for p in t.index], fontsize=8)
    ax.set(ylabel="skill vs Persistence (%)", title="Test skill = 1 - MSE/MSE(Persistence), mean over seeds")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "skill_by_pair.png", dpi=130)
    plt.close(fig)


def fig_pred(pair="sz.300015_sz.300274", days=100):
    fig, ax = plt.subplots(figsize=(11, 4))
    y = dates = None
    for exp, model, name in [("ext_leave", "CLATT", "CLATT-Δρ + leave"), (BASE, "Linear-leave", "Linear-leave"),
                             ("main_A", "Persistence", "Persistence")]:
        preds = stats.load_preds(exp, model, pair)
        y, pred = stats.mean_forecast(preds)
        dates = pd.to_datetime(next(iter(preds.values())).date)
        ax.plot(dates[:days], pred[:days], lw=1, label=name)
    ax.plot(dates[:days], y[:days], color="black", lw=1.6, label="true ρ(t+1)")
    ax.set(title=f"{pair}: first {days} test days", ylabel="ρ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pred_vs_true.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--baselines", action="store_true", help="write Persist-19 and Linear-leave runs first")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    if args.baselines:
        write_baselines(final=args.split == "test")
    for exp in ("ext_market", "ext_leave", "ext_leave_market"):
        long = collect(exp)
        if not long.empty and (gaps := missing_runs(long[long.split == args.split], [0, 1, 2, 3, 4])):
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
        print(summ[summ.group == "all"][cols].round(4).to_string(index=False))
        fig_skill()
        fig_pred()


if __name__ == "__main__":
    main()
