"""Extension 2 (after stage 7): one deep model per group, trained on the 6 pairs of the group together.

Each pair keeps its own dataset (own MinMax scaler fit on its own train rows, own purged split); the train samples
of the 6 pairs are pooled for training and the val samples are pooled for early stopping. The model is evaluated
on each pair separately, and the outputs are written per pair in the usual layout
results/runs/<exp>/<model>/<pair>/seed<k>/ so src.aggregate / src.stats work unchanged. The shared model and its
loss history are saved once in results/runs/<exp>/<model>/_group<g>/seed<k>/.

Leakage across pairs: every pair's train label dates must end before any pair's val input dates start, and val
before test (asserted in `check_cross_pair`).
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import FEATURES, get_pair_data
from src.features import pairs
from src.metrics import compute_all
from src.models import build, n_params
from src.train import RUNS, fit, log, predict, set_seed, to_rho, write_env


def group_pairs(group):
    return [f"{a}_{b}" for g, a, b in pairs() if g == group]


def check_cross_pair(datas, T):
    """Pooling must not let one pair's train (or val) samples see dates another pair uses for val (or test).
    A split touches dates from T-1 rows before its first input-end date up to its last label date."""
    span = {}
    for pair, d in datas.items():
        for name, sl in d["splits"].items():
            first = d["all_dates"][d["all_dates"].get_loc(d["x_end"][sl.start]) - (T - 1)]
            span[(pair, name)] = (first, d["y_date"][sl.stop - 1])
    for earlier, later in (("train", "val"), ("val", "test")):
        last = max(v[1] for (_, n), v in span.items() if n == earlier)
        first = min(v[0] for (_, n), v in span.items() if n == later)
        if not last < first:
            raise ValueError(f"pooled {earlier} reaches {last.date()}, but some pair's {later} inputs start "
                             f"{first.date()}")
    return span


def load_group(group, cfg):
    datas = {}
    for pair in group_pairs(group):
        d = get_pair_data(pair, cfg["T"], cfg["k"], features=cfg.get("features", FEATURES),
                          fixed_split_T=cfg["fixed_split_T"], target=cfg.get("target", "next"))
        d["all_dates"] = _pair_dates(pair, cfg["k"])
        datas[pair] = d
    check_cross_pair(datas, cfg["T"])
    return datas


def _pair_dates(pair, k):
    from src.features import pair_frame, returns_table
    a, b = pair.split("_")
    return pair_frame(returns_table([a, b]), a, b, k).index


def train_group(model_name, group, seed, cfg, exp, final=False, force=False):
    """Train one model on the pooled train samples of `group`; write per-pair outputs. Returns {pair: metrics}."""
    split_eval = ("val", "test") if final else ("val",)
    pairs_ = group_pairs(group)
    outs = {p: RUNS / exp / model_name / p / f"seed{seed}" for p in pairs_}
    shared = RUNS / exp / model_name / f"_group{group}" / f"seed{seed}"
    new_cfg = {"exp": exp, "model": model_name, "group": group, "pooled_pairs": pairs_, "seed": seed,
               "split_eval": list(split_eval), **cfg}
    if final and force and any((o / "preds_test.csv").exists() for o in outs.values()):
        raise ValueError(f"{exp} group {group} seed {seed} has already read the test split")
    if all((o / "metrics.json").exists() for o in outs.values()) and not force:
        saved = yaml.safe_load((shared / "config.yaml").read_text())
        new = yaml.safe_load(yaml.safe_dump(new_cfg))
        diff = sorted(k for k in saved.keys() | new.keys() if saved.get(k, "<missing>") != new.get(k, "<missing>"))
        if diff:
            raise ValueError(f"existing pooled run differs in: {', '.join(diff)}; use a new --exp or --force")
        log.info("skip %s group %d seed %d (pooled run exists)", exp, group, seed)
        return {p: json.loads((o / "metrics.json").read_text()) for p, o in outs.items()}

    set_seed(seed)
    torch.set_num_threads(cfg["num_threads"])
    datas = load_group(group, cfg)
    pool = {s: (np.concatenate([d["X"][d["splits"][s]] for d in datas.values()]),
                np.concatenate([d["y"][d["splits"][s]] for d in datas.values()])) for s in ("train", "val")}
    n_features = next(iter(datas.values()))["X"].shape[2]
    hp = {k: cfg[k] for k in ("hidden", "filters", "dropout", "layers", "head") if k in cfg}
    model = build(model_name, n_features, cfg["T"], **hp)
    start = time.perf_counter()
    history, best_epoch = fit(model, *pool["train"], *pool["val"], cfg, seed)
    train_seconds = time.perf_counter() - start
    shared.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(shared / "history.csv", index=False)
    torch.save(model.state_dict(), shared / "model.pt")
    info = {"best_epoch": best_epoch, "n_params": n_params(model), "train_seconds": train_seconds,
            "seconds_per_epoch": float(np.mean([h["seconds"] for h in history])),
            "n_train_pooled": len(pool["train"][0]), "pooled_group": group}

    results = {}
    for pair, d in datas.items():
        out = outs[pair]
        out.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for split in split_eval:
            sl = d["splits"][split]
            y_true, rho_prev = d["y_raw"][sl], d["rho_prev"][sl]
            y_pred = to_rho(d, sl, predict(model, d["X"][sl]))
            pd.DataFrame({"date": d["y_date"][sl].date, "y_true": y_true, "y_pred": y_pred,
                          "rho_prev": rho_prev}).to_csv(out / f"preds_{split}.csv", index=False)
            metrics[split] = compute_all(y_true, y_pred, rho_prev)
        dates = {k: [str(d["y_date"][v][0].date()), str(d["y_date"][v][-1].date())] for k, v in d["splits"].items()}
        (out / "config.yaml").write_text(yaml.safe_dump({**new_cfg, "pair": pair, "split_label_dates": dates},
                                                        sort_keys=False))
        result = {**metrics, **info}
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        results[pair] = result
    (shared / "config.yaml").write_text(yaml.safe_dump(new_cfg, sort_keys=False))
    write_env(shared / "env.txt")
    return results
