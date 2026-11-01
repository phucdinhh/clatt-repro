"""`python -m src.models` writes results/tables/model_params.csv."""
import pandas as pd

from src.baselines import BASELINES
from src.data import RESULTS
from src.dataset import FEATURES, T
from src.models import TABLE1, build, n_params

STRUCTURE = {
    "CNN": "Conv1D -> ReLU -> Dropout -> Flatten -> Dense(hidden, ReLU) -> Dense(1)",
    "LSTM": "LSTM x2 -> last state -> Dropout -> Dense(1)",
    "CNN-LSTM": "Conv1D -> ReLU -> Dropout -> LSTM x2 -> last state -> Dropout -> Dense(1)",
    "CNN-BiLSTM": "Conv1D -> ReLU -> Dropout -> BiLSTM x2 -> last state (both directions) -> Dropout -> Dense(1)",
    "CLATT": "Conv1D -> ReLU -> Dropout -> BiLSTM x2 (full sequence) -> Dropout -> Attention -> Dense(1)",
}
SOURCE = {
    "CNN": "hidden/filters/dropout/batch/epochs [PAPER] Table 1; structure [INFERENCE]; kernel 3, padding same: nhóm tự chọn",
    "LSTM": "hidden/dropout/batch/epochs [PAPER] Table 1; 2 layers [PAPER] Table 3; last state [PAPER] (return_sequences=False); dropout placement [INFERENCE]",
    "CNN-LSTM": "hidden/filters/dropout/batch/epochs [PAPER] Table 1; 2 layers [PAPER] Table 3; last state [PAPER]; dropout placement [INFERENCE]; kernel 3, padding same: nhóm tự chọn",
    "CNN-BiLSTM": "hidden 126 [PAPER] (units per direction [INFERENCE]); filters/dropout/batch/epochs [PAPER] Table 1; 2 layers [PAPER] Table 3; last state [PAPER]; dropout placement [INFERENCE]; kernel 3, padding same: nhóm tự chọn",
    "CLATT": "filters, dropout after Conv1D, BiLSTM 2 layers [PAPER]; BiLSTM 168 [PAPER] (units per direction [INFERENCE]); full sequence, attention Eq. 8-9 [PAPER]; dropout after BiLSTM [INFERENCE] Fig. 4; W d x d no bias [INFERENCE]; Dense(1) linear [INFERENCE]; kernel 3, padding same: nhóm tự chọn",
}


def main():
    rows = []
    for name, p in TABLE1.items():
        rows.append({"model": name, "n_features": len(FEATURES), "n_params": n_params(build(name)), "kernel": 3,
                     "layers": None if name == "CNN" else 2, **{k: v for k, v in p.items() if k != "name"}, "structure": STRUCTURE[name], "source": SOURCE[name]})
    rows.append({**rows[-1], "model": "CLATT- (rho only)", "n_features": 1, "n_params": n_params(build("CLATT", 1)),
                 "source": SOURCE["CLATT"] + "; rho-only input as Table 6 of the paper [PAPER]"})
    for name in BASELINES:
        rows.append({"model": name, "n_features": len(FEATURES),
                     "n_params": T * len(FEATURES) + 1 if name == "Ridge" else BASELINES[name].n_params,
                     "structure": {"Persistence": "rho(t)", "MA5": "mean rho(t-4..t)", "AR(1)": "a + b rho(t), fit on train",
                                   "Ridge": "Ridge on flattened 68x3 window, alpha in logspace(-3,3,13) chosen on val"}[name],
                     "source": "plan section 4.3 (not in paper); alpha grid: nhóm tự chọn"})
    out = RESULTS / "tables" / "model_params.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows)[["model", "n_features", "n_params"]].to_string(index=False))


if __name__ == "__main__":
    main()
