"""Train or fit one model on one pair with one seed and save everything plan section 5.3 asks for."""
import copy
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import BASELINES
from src.data import RESULTS, ROOT
from src.dataset import FEATURES, get_pair_data
from src.metrics import compute_all
from src.models import TABLE1, build, n_params

RUNS = RESULTS / "runs"
TUNED = ROOT / "configs" / "tuned"
ABLATION_KEYS = {"features", "head"}  # stage 8 switches; absent from base.yaml so protocol A configs are unchanged
log = logging.getLogger("clatt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Required by torch.use_deterministic_algorithms on CUDA to prevent LSTM RuntimeError
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def tuned_path(model_name, group):
    return TUNED / f"{model_name.lower().replace('-', '_')}_g{group}.yaml"


def load_config(model_name, overrides=(), tuned_group=None):
    """configs/base.yaml + configs/models/<model>.yaml [+ configs/tuned/<model>_g<group>.yaml if `tuned_group`]
    + `key=value` overrides (values parsed as YAML).

    Protocol A: tuned_group=None, identical to before stage 7. Protocol B: the stage 6 best (hidden, filters,
    dropout, batch_size, lr, max_epochs) replaces Table 1, and `tuned_from` records the file. Baselines have
    no tuned file and are unchanged."""
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    known = set(base) | {k for p in TABLE1.values() for k in p} | ABLATION_KEYS
    cfg = {**base, **TABLE1.get(model_name, {})}
    if tuned_group is not None and model_name in TABLE1:
        path = tuned_path(model_name, tuned_group)
        tuned = yaml.safe_load(path.read_text())
        if tuned.pop("name") != model_name:
            raise ValueError(f"{path} is not for {model_name}")
        cfg.update(tuned)
        cfg["tuned_from"] = str(path.relative_to(ROOT))
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in known:
            raise KeyError(f"unknown config key: {key}")
        value = yaml.safe_load(value)
        if isinstance(cfg.get(key), float) and isinstance(value, str):
            value = float(value)  # PyYAML reads "1e-3" (no dot) as a string
        cfg[key] = value
    return cfg


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).squeeze(1).numpy()


def fit(model, X_tr, y_tr, X_va, y_va, cfg, seed, on_epoch=None):
    """Adam + MSE on scaled y, grad clipping, early stopping on val MSE; restores the best weights.
    `on_epoch(epoch, val_loss)` is called after each epoch (Optuna pruning; it may raise)."""
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=cfg["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = {"mse": torch.nn.functional.mse_loss, "mae": torch.nn.functional.l1_loss}[cfg["loss"]]
    best, best_state, best_epoch, wait, history = np.inf, None, 0, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        start = time.perf_counter()
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
            opt.step()
            total += loss.item() * len(xb)
        val_loss = float(np.mean((predict(model, X_va) - y_va) ** 2))
        history.append({"epoch": epoch, "train_loss": total / len(X_tr), "val_loss": val_loss,
                        "seconds": time.perf_counter() - start})
        log.info("  epoch %3d  train %.5f  val %.5f  %.2fs", epoch, total / len(X_tr), val_loss, history[-1]["seconds"])
        if on_epoch:
            on_epoch(epoch, val_loss)
        if val_loss < best:
            best, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history, best_epoch


def write_env(path):
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    path.write_text(f"python {sys.version}\nplatform {platform.platform()}\ncpu {platform.processor()} "
                    f"x{os.cpu_count()}, torch threads {torch.get_num_threads()}\n\n{freeze}")


def train_one(model_name, pair, seed, cfg, split_eval=("val",), exp="smoke", final=False, force=False):
    if "test" in split_eval and not final:
        raise ValueError("evaluating on the test split requires --final")
    out = RUNS / exp / model_name / pair / f"seed{seed}"
    if final and force and (out / "preds_test.csv").exists():
        raise ValueError(f"{out} has already read the test split; --force would read it again (stage 7 rule)")
    new_cfg = {"exp": exp, "model": model_name, "pair": pair, "seed": seed, "split_eval": list(split_eval), **cfg}
    if (out / "metrics.json").exists() and not force:
        cfg_path = out / "config.yaml"
        if not cfg_path.exists():
            raise ValueError(f"existing run at {out} is missing config.yaml; use a new --exp or --force")
        saved = yaml.safe_load(cfg_path.read_text())
        new = yaml.safe_load(yaml.safe_dump(new_cfg))  # same types as the saved file
        diff_keys = sorted(k for k in saved.keys() | new.keys()
                           if k != "split_label_dates" and saved.get(k, "<missing>") != new.get(k, "<missing>"))
        if diff_keys:
            raise ValueError(f"existing run differs in: {', '.join(diff_keys)}; use a new --exp or --force")
        rel_path = out.relative_to(RESULTS) if out.is_relative_to(RESULTS) else out
        log.info("skip %s (metrics.json exists)", rel_path)
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.set_num_threads(cfg["num_threads"])

    data = get_pair_data(pair, cfg["T"], cfg["k"], features=cfg.get("features", FEATURES),
                         fixed_split_T=cfg["fixed_split_T"])
    X, y, s = data["X"], data["y"], data["splits"]
    start = time.perf_counter()
    if model_name in BASELINES:
        model = BASELINES[model_name]().fit(X[s["train"]], y[s["train"]], X[s["val"]], y[s["val"]])
        info = {"n_params": model.n_params}
        run_predict = model.predict
    else:
        hp = {k: cfg[k] for k in ("hidden", "filters", "dropout", "layers", "head") if k in cfg}
        model = build(model_name, X.shape[2], cfg["T"], **hp)
        history, best_epoch = fit(model, X[s["train"]], y[s["train"]], X[s["val"]], y[s["val"]], cfg, seed)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        torch.save(model.state_dict(), out / "model.pt")
        info = {"best_epoch": best_epoch, "n_params": n_params(model),
                "seconds_per_epoch": float(np.mean([h["seconds"] for h in history]))}
        run_predict = lambda x: predict(model, x)  # noqa: E731
    info["train_seconds"] = time.perf_counter() - start

    lo, span = data["scaler"].data_min_[0], data["scaler"].data_range_[0]  # rho is column 0
    metrics = {}
    for split in split_eval:
        sl = s[split]
        y_true, rho_prev = data["y_raw"][sl], data["rho_prev"][sl]
        if model_name == "Persistence":
            # Persistence is exactly rho(t); avoids float32 scale round trip
            y_pred = rho_prev
        else:
            y_pred = run_predict(X[sl]).astype(np.float64) * span + lo
        pd.DataFrame({"date": data["y_date"][sl].date, "y_true": y_true, "y_pred": y_pred,
                      "rho_prev": rho_prev}).to_csv(out / f"preds_{split}.csv", index=False)
        metrics[split] = compute_all(y_true, y_pred, rho_prev)
        if model_name == "CLATT" and model.attn is not None:  # ablations A2 and A4 have no attention weights
            model.eval()
            with torch.no_grad():
                np.save(out / f"attention_{split}.npy", model(torch.tensor(X[sl]), return_attention=True)[1].numpy())

    split_dates = {k: [str(data["y_date"][v][0].date()), str(data["y_date"][v][-1].date())] for k, v in s.items()}
    (out / "config.yaml").write_text(yaml.safe_dump({**new_cfg, "split_label_dates": split_dates}, sort_keys=False))
    write_env(out / "env.txt")
    result = {**metrics, **info}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))  # written last: marks the run complete
    return result
