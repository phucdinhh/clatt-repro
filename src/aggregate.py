"""python -m src.aggregate <exp>: all metrics.json of an exp -> long CSV + mean/std summary per (model, group)."""
import json
import sys

import pandas as pd

from src.data import RESULTS
from src.features import pairs
from src.train import RUNS

COLUMNS = ["exp", "model", "pair", "group", "seed", "split", "metric", "value"]


def collect(exp):
    group = {f"{a}_{b}": g for g, a, b in pairs()}
    rows = []
    for f in sorted((RUNS / exp).glob("*/*/seed*/metrics.json")):
        model, pair, seed = f.parts[-4], f.parts[-3], int(f.parts[-2].removeprefix("seed"))
        for key, value in json.loads(f.read_text()).items():
            items = value.items() if isinstance(value, dict) else [(key, value)]
            split = key if isinstance(value, dict) else "run"
            rows += [(exp, model, pair, group[pair], seed, split, m, v) for m, v in items]
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(long):
    return (long.groupby(["model", "group", "split", "metric"]).value
            .agg(mean="mean", std="std", n="count").reset_index())


def main(exp):
    long = collect(exp)
    tables = RESULTS / "tables"
    long.to_csv(tables / f"{exp}_metrics_long.csv", index=False)
    summarize(long).to_csv(tables / f"{exp}_summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
