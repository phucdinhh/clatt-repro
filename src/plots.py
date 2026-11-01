"""python -m src.plots <exp>: train/val loss per epoch for every deep-model run of an exp."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


if __name__ == "__main__":
    print(loss_curves(sys.argv[1]))
