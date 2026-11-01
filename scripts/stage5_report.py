"""Stage 5 tables, figure and proposed config from the sweep runs (val only).

python -m scripts.stage5_report  -> results/tables/{timestep,lstm_layers,loss_compare}.csv,
results/figures/timestep_curve.png, configs/proposed_stage5.yaml. Missing sweeps are skipped.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.aggregate import collect  # noqa: E402
from src.data import RESULTS, ROOT  # noqa: E402
from src.dataset import get_pair_data  # noqa: E402
from src.features import pairs  # noqa: E402
from src.metrics import r2  # noqa: E402
from src.train import RUNS  # noqa: E402

GRID_T = [20, 40, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 90]
T_MAX = 90
METRICS = ["r2", "mse", "skill", "r2_delta"]
PAPER_L = {1: (.9467, .8726), 2: (.9655, .9360), 3: (.9561, .9228), 4: (.9397, .8588)}  # Table 3 R², groups 1, 2
PAPER_LOSS = {"mse": (.9825, .9661), "mae": (.9820, .9644)}  # Table 4 R², groups 1, 2
N_RUNS = 12 * 3


def seed_means(exp):
    """Val metrics averaged over the 6 pairs of each group: one row per (group, seed). None unless all 36 runs exist."""
    long = collect(exp)
    if long.empty or long[["pair", "seed"]].drop_duplicates().shape[0] < N_RUNS:
        return None
    v = long[(long.split == "val") & long.metric.isin(METRICS)]
    return v.pivot_table(index=["group", "seed"], columns="metric", values="value", aggfunc="mean")


def sweep(exps):
    """{key: exp} -> {key: seed_means} for complete exps; prints the incomplete ones."""
    out = {}
    for key, exp in exps.items():
        m = seed_means(exp)
        if m is None:
            print(f"skip {exp}: incomplete")
        else:
            out[key] = m
    return out


def table(runs, name):
    rows = []
    for key, m in runs.items():
        for g, d in m.groupby(level="group"):
            rows.append({name: key, "group": g, **{f"{c}_{s}": getattr(d[c], s)() for c in METRICS for s in ("mean", "std")}})
    return pd.DataFrame(rows)


def choose(runs, prefer):
    """Lowest val MSE averaged over both groups; if `prefer` is within 1 std (over seeds) of the best, take it."""
    both = pd.DataFrame({k: m.mse.groupby(level="seed").mean() for k, m in runs.items()})  # seeds x keys
    mean, std = both.mean(), both.std()
    best = mean.idxmin()
    within = mean[mean <= mean[best] + std[best]].index.tolist()
    return (prefer if prefer in within else best), best, within


def persistence_r2():
    """Persistence R² on val per group; with fixed_split_T the val labels are the same for every T."""
    rows = []
    for g, a, b in pairs():
        ds = get_pair_data(f"{a}_{b}", T=T_MAX)
        va = ds["splits"]["val"]
        rows.append({"group": g, "r2": r2(ds["y_raw"][va], ds["rho_prev"][va])})
    return pd.DataFrame(rows).groupby("group").r2.mean()


def figure(tab, persist):
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, color in [(1, "tab:blue"), (2, "tab:red")]:
        d = tab[tab.group == g].sort_values("T")
        ax.errorbar(d["T"], d.r2_mean, yerr=d.r2_std, marker="o", ms=4, capsize=3, color=color, label=f"group {g}")
        ax.axhline(persist[g], ls="--", lw=1, color=color, label=f"persistence, group {g}")
    ax.set(xlabel="time step T", ylabel="R² (val)", title="CLATT val R² vs time step (mean ± std over 3 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "timestep_curve.png", dpi=150)


def main():
    tables = RESULTS / "tables"
    proposed = {}
    runs_T = sweep({t: f"sweep_T/T{t}" for t in GRID_T if (RUNS / "sweep_T" / f"T{t}").exists()})
    if not runs_T:
        return print("no complete sweep_T runs")
    tab = table(runs_T, "T").sort_values(["T", "group"])
    tab.to_csv(tables / "timestep.csv", index=False)
    persist = persistence_r2()
    figure(tab, persist)
    print(tab.round(4).to_string(index=False), "\npersistence R² val by group:", persist.round(4).to_dict())
    T_star, best, within = choose(runs_T, prefer=68)
    print(f"T*={T_star} (lowest MSE at T={best}; within 1 std: {within})")
    proposed["T"] = int(T_star)

    runs_L = sweep({2: f"sweep_T/T{T_star}", **{l: f"sweep_L/L{l}" for l in (1, 3, 4)}})
    if len(runs_L) == 4:
        tab = table(runs_L, "L").sort_values(["L", "group"])
        tab["paper_r2"] = [PAPER_L[l][g - 1] for l, g in zip(tab.L, tab.group)]
        tab.to_csv(tables / "lstm_layers.csv", index=False)
        L_star, best, within = choose(runs_L, prefer=2)
        print(tab.round(4).to_string(index=False), f"\nL*={L_star} (lowest MSE at L={best}; within 1 std: {within})")
        proposed["layers"] = int(L_star)

        mse_exp = f"sweep_T/T{T_star}" if L_star == 2 else f"sweep_L/L{L_star}"
        runs_loss = sweep({"mse": mse_exp, "mae": "sweep_loss/mae"})
        if len(runs_loss) == 2:
            tab = table(runs_loss, "loss")
            tab["paper_r2"] = [PAPER_LOSS[f][g - 1] for f, g in zip(tab.loss, tab.group)]
            tab.to_csv(tables / "loss_compare.csv", index=False)
            loss_star, best, within = choose(runs_loss, prefer="mse")
            print(tab.round(4).to_string(index=False), f"\nloss*={loss_star} (lowest MSE: {best}; within 1 std: {within})")
            proposed["loss"] = loss_star

    (ROOT / "configs" / "proposed_stage5.yaml").write_text(
        "# Stage 5 proposal (val only). Copy into configs/base.yaml only after the user confirms.\n"
        + yaml.safe_dump(proposed, sort_keys=False))


if __name__ == "__main__":
    main()
