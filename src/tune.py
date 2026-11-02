"""Stage 6, protocol B: the same Optuna budget for every deep model (val only).

python -m src.tune --model CLATT --group 1   one study, 20 trials, resumable (results/optuna/study.db)
python -m src.tune --report                  configs/tuned/*.yaml, tuning_summary.csv, tuning_history.png
"""
import argparse
import logging

import numpy as np
import optuna
import pandas as pd
import yaml

from src.data import RESULTS, ROOT
from src.dataset import get_pair_data
from src.features import pairs
from src.models import MODELS, TABLE1, build, n_params
from src.train import fit, load_config, log, predict, set_seed

N_TRIALS = 20
SEED = 0
MAX_EPOCHS = 80
STORAGE = RESULTS / "optuna" / "study.db"
TUNED = ROOT / "configs" / "tuned"
DONE = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def representative_pairs(group, cfg):
    """Largest and smallest std of rho over the train rows (not the full series, which includes later splits)."""
    std = {}
    for g, a, b in pairs():
        if g == group:
            ds = get_pair_data(f"{a}_{b}", cfg["T"], cfg["k"], fixed_split_T=cfg["fixed_split_T"])
            std[f"{a}_{b}"] = ds["train_rows"].rho.std()
    return [max(std, key=std.get), min(std, key=std.get)]


def sample(trial, model):
    """Search space of the stage 6 spec; sizes only, the architecture is unchanged."""
    hp = {"hidden": trial.suggest_int("hidden", 32, 256, step=8),
          "dropout": trial.suggest_float("dropout", 0.1, 0.5),
          "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
          "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True)}
    if model != "LSTM":
        hp["filters"] = trial.suggest_int("filters", 16, 128, step=8)
    return hp


def table1(model, cfg):
    # Table 1 batch sizes (23, 62) are not in {16, 32, 64}, so trial 0 is not sampled and TPE does not model it
    return {k: v for k, v in {**TABLE1[model], "lr": cfg["lr"]}.items()
            if k in ("hidden", "filters", "dropout", "batch_size", "lr") and v is not None}


def val_mse(model_name, data, cfg, on_epoch):
    """Train with seed 0 on one pair; val MSE on the original rho scale."""
    set_seed(SEED)
    X, y, s = data["X"], data["y"], data["splits"]
    model = build(model_name, X.shape[2], cfg["T"], **{k: cfg[k] for k in ("hidden", "filters", "dropout", "layers")})
    fit(model, X[s["train"]], y[s["train"]], X[s["val"]], y[s["val"]], cfg, SEED, on_epoch)
    lo, span = data["scaler"].data_min_[0], data["scaler"].data_range_[0]
    y_pred = predict(model, X[s["val"]]).astype(np.float64) * span + lo
    return float(np.mean((y_pred - data["y_raw"][s["val"]]) ** 2))


def objective(trial, model_name, rep_pairs, cfg):
    hp = table1(model_name, cfg) if trial.user_attrs.get("table1") else sample(trial, model_name)
    run_cfg = {**cfg, **hp, "max_epochs": MAX_EPOCHS}
    trial.set_user_attr("hp", hp)
    scores = []
    for i, pair in enumerate(rep_pairs):
        data = get_pair_data(pair, cfg["T"], cfg["k"], fixed_split_T=cfg["fixed_split_T"])

        def report(epoch, val_loss, offset=i * MAX_EPOCHS):  # pair i uses steps offset+1.., so trials compare like with like
            if epoch <= cfg["patience"]:  # no pruning in each pair's first epochs: with no warmup, 15/20 CLATT trials died at epoch 1-2
                return
            trial.report(val_loss, offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        scores.append(val_mse(model_name, data, run_cfg, report))
    return float(np.mean(scores))


def study_for(model_name, group):
    STORAGE.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name=f"{model_name}_g{group}", storage=f"sqlite:///{STORAGE}",
                               load_if_exists=True, direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner())


def tune(model_name, group):
    cfg = load_config(model_name)
    study = study_for(model_name, group)
    stale = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]  # left by an interrupted run
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
        study.enqueue_trial(t.params, user_attrs=t.user_attrs)  # rerun the same config
    if not study.trials:
        study.enqueue_trial({}, user_attrs={"table1": True})
    left = N_TRIALS - sum(t.state in DONE for t in study.trials)
    rep_pairs = representative_pairs(group, cfg)
    log.info("%s group %d: pairs %s, %d trials left", model_name, group, rep_pairs, left)
    if left > 0:
        study.optimize(lambda t: objective(t, model_name, rep_pairs, cfg), n_trials=left)


def report():
    TUNED.mkdir(parents=True, exist_ok=True)
    rows, history = [], []
    for model_name in MODELS:
        cfg = load_config(model_name)
        for group in (1, 2):
            try:
                study = optuna.load_study(study_name=f"{model_name}_g{group}", storage=f"sqlite:///{STORAGE}")
            except KeyError:
                print(f"skip {model_name} g{group}: no study")
                continue
            done = [t for t in study.trials if t.state in DONE]
            complete = [t for t in done if t.state == optuna.trial.TrialState.COMPLETE]
            if len(done) < N_TRIALS:
                print(f"note {model_name} g{group}: only {len(done)} trials done")
            best = min(complete, key=lambda t: t.value)
            t0 = next((t.value for t in complete if t.user_attrs.get("table1")), np.nan)
            hp = best.user_attrs["hp"]
            name = model_name.lower().replace("-", "_")
            (TUNED / f"{name}_g{group}.yaml").write_text(
                f"# stage 6 protocol B: best of {len(done)} trials, group {group}, val MSE {best.value:.6g}\n"
                + yaml.safe_dump({"name": model_name, **hp, "max_epochs": MAX_EPOCHS}, sort_keys=False))
            net = build(model_name, 3, cfg["T"], **hp, layers=cfg["layers"])  # extra keys (lr, batch_size) are ignored
            rows.append({"model": model_name, "group": group, "val_mse_table1": t0, "val_mse_best": best.value,
                         "improvement_pct": 100 * (t0 - best.value) / t0, "best_trial": best.number,
                         **hp, "n_params": n_params(net), "n_trials": len(done),
                         "n_pruned": len(done) - len(complete)})
            best_so_far = np.fmin.accumulate([t.value if t.state == optuna.trial.TrialState.COMPLETE else np.nan
                                              for t in sorted(done, key=lambda t: t.number)])
            history += [{"model": model_name, "group": group, "trial": i, "best": v} for i, v in enumerate(best_so_far)]
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "tables" / "tuning_summary.csv", index=False)
    figure(pd.DataFrame(history))
    for group, d in summary.groupby("group"):  # gap = baseline MSE / CLATT MSE - 1 (positive: CLATT better)
        d = d.set_index("model")
        if "CLATT" in d.index:
            for col in ("val_mse_table1", "val_mse_best"):
                print(f"group {group} {col}: gap vs CLATT", (d[col] / d.loc["CLATT", col] - 1).round(3).to_dict())
    print(summary.round(5).to_string(index=False))


def figure(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (group, d) in zip(axes, history.groupby("group")):
        for model_name, m in d.groupby("model", sort=False):
            ax.plot(m.trial, m.best, marker="o", ms=3, label=model_name)
        ax.set(title=f"group {group}", xlabel="trial", ylabel="best val MSE so far (original scale)", yscale="log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "tuning_history.png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--group", type=int, choices=[1, 2])
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.report:
        return report()
    if not (args.model and args.group):
        p.error("--model and --group are required unless --report")
    tune(args.model, args.group)


if __name__ == "__main__":
    main()
