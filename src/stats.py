"""Stage 7.5 / 8.2: significance tests on the test split.

python -m src.stats main_A                       CLATT vs Persistence, AR(1), CNN-BiLSTM; Spearman vs Table 5
python -m src.stats main_B --baselines main_A    same for protocol B (baselines reused from protocol A)
python -m src.stats ablation                     abl_A1..A4 CLATT vs main_A CLATT -> results/tables/ablation.csv

Decisions fixed before the test split was read (configs/stage7_decisions.yaml):
- DM: squared errors, horizon 1, Harvey-Leybourne-Newbold small-sample correction, Student t with n-1 df,
  two-sided. Primary: the seed-mean forecast of each deep model (one DM per pair). Also reported: the share
  of seeds whose own DM is significant in the same direction. Baselines are deterministic (seed 0).
- Wilcoxon signed-rank, two-sided, over the 12 pairs on seed-mean MSE.
- Verdict "A better than B": mean over pairs of 1 - MSE_A/MSE_B > 0 AND primary DM p < 0.05 with A's loss lower
  on at least 7 of 12 pairs.
- Spearman: rank of the 5 deep models by test MSE (mean over pairs and seeds), per group, against the paper's
  Table 5 MSE (dataset 1 = group 1, dataset 2 = group 2 [INFERENCE]). MAE is reported too because Table 5's
  CNN-BiLSTM MSE in dataset 2 (0.0592) looks like a typo for 0.0059.
"""
import argparse

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.baselines import BASELINES
from src.data import RESULTS, ROOT
from src.features import pairs
from src.train import RUNS

TABLES = RESULTS / "tables"
ALPHA, MIN_PAIRS = 0.05, 7
COMPARE_TO = ["Persistence", "AR(1)", "CNN-BiLSTM"]
TABLE5 = {  # [PAPER] Table 5
    1: {"mse": {"CNN": 0.0077, "LSTM": 0.0076, "CNN-LSTM": 0.0064, "CNN-BiLSTM": 0.0059, "CLATT": 0.0032},
        "mae": {"CNN": 0.0681, "LSTM": 0.0656, "CNN-LSTM": 0.0594, "CNN-BiLSTM": 0.0572, "CLATT": 0.0414}},
    2: {"mse": {"CNN": 0.0060, "LSTM": 0.0052, "CNN-LSTM": 0.0056, "CNN-BiLSTM": 0.0592, "CLATT": 0.0024},
        "mae": {"CNN": 0.0561, "LSTM": 0.0620, "CNN-LSTM": 0.0605, "CNN-BiLSTM": 0.0557, "CLATT": 0.0374}},
}


def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared errors with the Harvey et al. (1997) correction.
    Negative stat: forecast 1 has the lower loss. Returns (stat, two-sided p)."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    dc = d - d.mean()
    var = (dc @ dc + 2 * sum(dc[k:] @ dc[:-k] for k in range(1, h))) / n  # long-run variance, h-1 lags
    if var <= 0:
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def load_preds(exp, model, pair, split="test"):
    """Seed -> DataFrame(date, y_true, y_pred)."""
    files = sorted((RUNS / exp / model / pair).glob(f"seed*/preds_{split}.csv"))
    if not files:
        raise FileNotFoundError(f"no preds_{split}.csv for {exp}/{model}/{pair}")
    return {int(f.parent.name.removeprefix("seed")): pd.read_csv(f) for f in files}


def mean_forecast(preds):
    frames = list(preds.values())
    for f in frames[1:]:
        if not (f.date.equals(frames[0].date) and np.allclose(f.y_true, frames[0].y_true)):
            raise ValueError("seeds disagree on test dates or labels")
    return frames[0].y_true.to_numpy(), np.mean([f.y_pred for f in frames], axis=0)


def compare_pair(a, b):
    """a, b: seed -> preds for one pair. Primary DM on seed-mean forecasts; per-seed DM share."""
    y, pa = mean_forecast(a)
    y_b, pb = mean_forecast(b)
    if not np.allclose(y, y_b):
        raise ValueError("models disagree on test labels")
    dm, p = dm_test(pa - y, pb - y)
    mse_a = np.mean([np.mean((f.y_pred - f.y_true) ** 2) for f in a.values()])
    mse_b = np.mean([np.mean((f.y_pred - f.y_true) ** 2) for f in b.values()])
    per_seed = []
    for s, fa in a.items():
        fb = b.get(s, next(iter(b.values())))  # deterministic baselines: their single seed
        per_seed.append(dm_test(fa.y_pred - fa.y_true, fb.y_pred - fb.y_true))
    return {"n": len(y), "mse_a": mse_a, "mse_b": mse_b, "rel_skill": 1 - mse_a / mse_b,
            "mse_a_meanfc": float(np.mean((pa - y) ** 2)), "dm": dm, "p": p,
            "a_better_sig": dm < 0 and p < ALPHA, "b_better_sig": dm > 0 and p < ALPHA,
            "seeds_a_better_sig": float(np.mean([d < 0 and q < ALPHA for d, q in per_seed])),
            "seeds_b_better_sig": float(np.mean([d > 0 and q < ALPHA for d, q in per_seed])),
            "n_seeds_a": len(a), "n_seeds_b": len(b)}


def compare(exp_a, model_a, exp_b, model_b, label=None):
    rows = []
    for g, x, y in pairs():
        pair = f"{x}_{y}"
        r = compare_pair(load_preds(exp_a, model_a, pair), load_preds(exp_b, model_b, pair))
        rows.append({"comparison": label or f"{model_a} vs {model_b}", "group": g, "pair": pair, **r})
    per_pair = pd.DataFrame(rows)
    return per_pair, summarize(per_pair)


def summarize(per_pair):
    out = []
    for (label, group), d in [((c, "all"), d) for c, d in per_pair.groupby("comparison", sort=False)] + \
            [((c, g), d) for (c, g), d in per_pair.groupby(["comparison", "group"], sort=False)]:
        w = stats.wilcoxon(d.mse_a, d.mse_b) if len(d) > 1 and (d.mse_a != d.mse_b).any() else None
        n_better = int(d.a_better_sig.sum())
        row = {"comparison": label, "group": group, "n_pairs": len(d),
               "mse_a": d.mse_a.mean(), "mse_b": d.mse_b.mean(), "mean_rel_skill": d.rel_skill.mean(),
               "n_pairs_rel_skill_pos": int((d.rel_skill > 0).sum()),
               "n_dm_a_better": n_better, "n_dm_b_better": int(d.b_better_sig.sum()),
               "wilcoxon_stat": w.statistic if w else np.nan, "wilcoxon_p": w.pvalue if w else np.nan}
        if group == "all":
            row["verdict_a_better"] = bool(row["mean_rel_skill"] > 0 and n_better >= MIN_PAIRS)
        out.append(row)
    return pd.DataFrame(out)


def spearman(exp):
    o = pd.read_csv(TABLES / f"{exp}_test_by_group.csv")
    rows = []
    for g, paper in TABLE5.items():
        for metric, ref in paper.items():
            ours = o[(o.group == g) & (o.metric == metric)].set_index("model")["mean"]
            models = [m for m in ref if m in ours.index]
            rho, p = stats.spearmanr([ours[m] for m in models], [ref[m] for m in models])
            rank = ours[models].rank().astype(int).to_dict()
            rows.append({"group": g, "metric": metric, "spearman": rho, "p": p, "n_models": len(models),
                         "our_rank": " < ".join(sorted(models, key=ours.get)),
                         "paper_rank": " < ".join(sorted(models, key=ref.get)),
                         **{f"rank_{m}": rank[m] for m in models}})
    return pd.DataFrame(rows)


def main_tests(exp, baselines=None):
    parts = [compare(exp, "CLATT", baselines if (b in BASELINES and baselines) else exp, b,
                     f"CLATT vs {b}") for b in COMPARE_TO]
    per_pair = pd.concat([p for p, _ in parts], ignore_index=True)
    summary = pd.concat([s for _, s in parts], ignore_index=True)
    per_pair.to_csv(TABLES / f"{exp}_dm_by_pair.csv", index=False)
    summary.to_csv(TABLES / f"{exp}_tests.csv", index=False)
    sp = spearman(exp)
    sp.to_csv(TABLES / f"{exp}_spearman_table5.csv", index=False)
    return per_pair, summary, sp


def ablation(variants=("A1", "A2", "A3", "A4"), full="main_A"):
    desc = yaml.safe_load((ROOT / "configs" / "ablations.yaml").read_text())
    per, summ = [], []
    for v in variants:
        p, s = compare(f"abl_{v}", "CLATT", full, "CLATT", f"{v} vs CLATT")
        per.append(p.assign(variant=v))
        s = s.assign(variant=v, desc=desc[v]["desc"])
        summ.append(s)
    per, summ = pd.concat(per, ignore_index=True), pd.concat(summ, ignore_index=True)
    # mse_a = variant, mse_b = full CLATT; positive change = removing the part hurts
    summ["mse_change_pct"] = 100 * (summ.mse_a / summ.mse_b - 1)
    pct = per.assign(pct=100 * (per.mse_a / per.mse_b - 1))
    summ["mse_change_pct_pair_mean"] = [pct[(pct.variant == v) & ((pct.group == g) if g != "all" else True)].pct.mean()
                                        for v, g in zip(summ.variant, summ.group)]
    summ = summ.rename(columns={"mse_a": "mse_variant", "mse_b": "mse_full",
                                "n_dm_a_better": "n_dm_variant_better", "n_dm_b_better": "n_dm_full_better",
                                "verdict_a_better": "verdict_variant_better"})
    cols = ["variant", "desc", "group", "n_pairs", "mse_full", "mse_variant", "mse_change_pct",
            "mse_change_pct_pair_mean", "n_dm_full_better", "n_dm_variant_better", "wilcoxon_p",
            "verdict_variant_better"]
    summ[cols].to_csv(TABLES / "ablation.csv", index=False)
    per.to_csv(TABLES / "ablation_dm_by_pair.csv", index=False)
    return summ[cols]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("exp", help="an exp name, or 'ablation'")
    p.add_argument("--baselines")
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)
    if args.exp == "ablation":
        print(ablation().round(4).to_string(index=False))
        return
    _, summary, sp = main_tests(args.exp, args.baselines)
    print(summary.round(4).to_string(index=False))
    print(sp[["group", "metric", "spearman", "p", "our_rank", "paper_rank"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
