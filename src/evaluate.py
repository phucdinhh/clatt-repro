"""Stage 7.4: test-split tables for one exp.

python -m src.evaluate main_A                      deep models + baselines of main_A
python -m src.evaluate main_B --baselines main_A   protocol B reuses the protocol A baselines (no hyperparameters)
python -m src.evaluate --compare main_A main_B     side-by-side table and skill per pair for the report

Writes results/tables/<exp>_test_by_pair.csv, _by_group.csv, _overall.csv (long format: model, [group,] [pair,]
metric, mean, std, n_seeds, n_pairs, n_dropped). by_pair: mean/std over seeds. by_group/overall: for each seed
the mean over the pairs, then mean/std over seeds (baselines are deterministic, 1 seed, so std is NaN).
Non-finite values (mape when y_true == 0) are dropped and counted in n_dropped, never averaged.
"""
import argparse

import numpy as np
import pandas as pd

from src.aggregate import collect
from src.baselines import BASELINES
from src.data import RESULTS
from src.features import pairs
from src.models import MODELS

TABLES = RESULTS / "tables"
METRICS = ["mse", "rmse", "mae", "r2", "mape", "mape_excl", "smape", "skill", "r2_delta", "dir_acc"]
ORDER = MODELS[::-1] + list(BASELINES)  # CLATT first


def load_long(exp, baselines=None, split="test"):
    long = collect(exp)
    if baselines:
        long = pd.concat([long[~long.model.isin(BASELINES)],
                          collect(baselines).query("model in @BASELINES")], ignore_index=True)
    long = long[(long.split == split) & long.metric.isin(METRICS)].copy()
    if long.empty:
        raise SystemExit(f"no {split} metrics in {exp}" + (f" + {baselines}" if baselines else ""))
    long["value"] = long.value.astype(float)
    return long


def missing_runs(long, seeds):
    """(model, pair, seed) triples the grid expects but `long` lacks; baselines are expected at seed 0 only."""
    all_pairs = [f"{a}_{b}" for _, a, b in pairs()]
    have = set(zip(long.model, long.pair, long.seed))
    return [(m, p, s) for m in long.model.unique() for p in all_pairs
            for s in ([0] if m in BASELINES else seeds) if (m, p, s) not in have]


def _finite(long):
    ok = np.isfinite(long.value)
    dropped = long[~ok].groupby(["model", "group", "pair", "metric"]).size().rename("n_dropped")
    return long[ok], dropped


def by_pair(long):
    fin, dropped = _finite(long)
    out = (fin.groupby(["model", "group", "pair", "metric"]).value
           .agg(mean="mean", std="std", n_seeds="count").reset_index())
    out = out.merge(dropped.reset_index(), how="left", on=["model", "group", "pair", "metric"])
    return _sort(_count(out))


def by_level(long, keys):
    """keys = ["group"] or []: pair mean per seed, then mean/std over seeds."""
    fin, dropped = _finite(long)
    per_seed = (fin.groupby(["model", *keys, "metric", "seed"])
                .agg(value=("value", "mean"), n_pairs=("pair", "nunique")).reset_index())
    out = (per_seed.groupby(["model", *keys, "metric"])
           .agg(mean=("value", "mean"), std=("value", "std"), n_seeds=("seed", "nunique"),
                n_pairs=("n_pairs", "min")).reset_index())
    d = dropped.groupby(["model", *([k for k in keys]), "metric"]).sum().rename("n_dropped").reset_index()
    out = out.merge(d, how="left", on=["model", *keys, "metric"])
    return _sort(_count(out))


def _count(out):  # pairs with nothing dropped have no n_dropped row after the merge
    return out.assign(n_dropped=out.n_dropped.where(out.n_dropped.notna(), 0).astype(int))


def _sort(df):
    rank = df.model.map({m: i for i, m in enumerate(ORDER)})
    df = df.assign(_m=rank.where(rank.notna(), 99),
                   _k=df.metric.map({m: i for i, m in enumerate(METRICS)}))
    keys = ["_m"] + [c for c in ("group", "pair") if c in df] + ["_k"]
    return df.sort_values(keys).drop(columns=["_m", "_k"]).reset_index(drop=True)


def evaluate(exp, baselines=None, seeds=None):
    long = load_long(exp, baselines)
    seeds = seeds or sorted(long[~long.model.isin(BASELINES)].seed.unique())
    gaps = missing_runs(long, seeds)
    if gaps:
        print(f"WARNING {exp}: {len(gaps)} runs missing, e.g. {gaps[:5]}")
    tables = {"by_pair": by_pair(long), "by_group": by_level(long, ["group"]), "overall": by_level(long, [])}
    for name, df in tables.items():
        df.to_csv(TABLES / f"{exp}_test_{name}.csv", index=False)
    return tables, gaps


def fmt(mean, std, digits=4):
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def compare(exps, metrics=("mse", "mae", "r2", "mape", "skill", "dir_acc")):
    """main_compare.csv (model x protocol x group, 'mean ± std over seeds') and skill_by_pair.csv."""
    rows, skill = [], []
    for exp in exps:
        g = pd.read_csv(TABLES / f"{exp}_test_by_group.csv")
        o = pd.read_csv(TABLES / f"{exp}_test_overall.csv").assign(group="all")
        for (model, group), d in pd.concat([g, o]).groupby(["model", "group"], sort=False):
            d = d.set_index("metric")
            rows.append({"exp": exp, "model": model, "group": group, "n_seeds": int(d.n_seeds.max()),
                         **{m: fmt(d.loc[m, "mean"], d.loc[m, "std"]) for m in metrics if m in d.index}})
        p = pd.read_csv(TABLES / f"{exp}_test_by_pair.csv").query("metric == 'skill'")
        skill.append(p.assign(exp=exp)[["exp", "model", "group", "pair", "mean", "std", "n_seeds"]])
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "main_compare.csv", index=False)
    pd.concat(skill).rename(columns={"mean": "skill", "std": "skill_std"}).to_csv(
        TABLES / "skill_by_pair.csv", index=False)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("exp", nargs="*")
    p.add_argument("--baselines", help="exp whose baseline runs are reused")
    p.add_argument("--compare", nargs="+", metavar="EXP")
    args = p.parse_args(argv)
    if args.compare:
        print(compare(args.compare).to_string(index=False))
    for exp in args.exp:
        tables, _ = evaluate(exp, args.baselines)
        print(tables["by_group"].query("metric in ['mse', 'r2', 'skill']").round(5).to_string(index=False))


if __name__ == "__main__":
    main()
