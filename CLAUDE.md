# clatt-repro

Reproduction of the CLATT paper (short-term stock correlation forecasting with CNN-BiLSTM + attention) in PyTorch.
Source documents are in `docs/`: the experiment plan (PDF, sections 3 and 4.1-4.4) and the paper (PDF).
Current state: data, models and baselines exist. **Nothing is trained yet** — do not run training unless asked.

## Code map

| File | Contents |
|---|---|
| `src/data.py` | Constants (CODES, GROUPS, START, END, RESULTS, PROCESSED), `load_raw`, `traded` |
| `src/features.py` | Log-returns, stock pairing, rho label = rolling Pearson (k=20), `rho_stats` |
| `src/dataset.py` | T=68, FEATURES=[rho, r_A, r_B], sliding windows, purged train/val/test split (7/1/2), MinMax fit on train only. `python -m src.dataset` writes the stage-1 tables and figures |
| `src/models/__init__.py` | `TABLE1` ([PAPER] hyperparameters), `MODELS`, `build(name, n_features=3, T=68)`, `n_params` |
| `src/models/cnn.py` | `CNN`: Conv1D -> ReLU -> Dropout -> Flatten -> Dense(hidden, ReLU) -> Dense(1) |
| `src/models/recurrent.py` | `Attention` (Eq. 8-9) and `ConvRNN`, used for LSTM, CNN-LSTM, CNN-BiLSTM and CLATT |
| `src/models/__main__.py` | `python -m src.models` writes `results/tables/model_params.csv` |
| `src/baselines.py` | Persistence, MovingAverage (MA5), AR1, RidgeWindow; `BASELINES` dict |
| `src/train.py` | Only `set_seed` so far (training loop not written) |
| `src/evaluate.py`, `plots.py`, `stats.py`, `tune.py` | Empty files, not started |
| `tests/test_pipeline.py`, `tests/test_models.py` | Data tests; model tests (plan 4.4) and baseline sanity check |
| `configs/`, `scripts/` | `base.yaml`, `experiments/`, `models/`; `download_baostock.py`, `run_all.sh` |

## Commands

- Tests: `python -m pytest -q` (about 84 s, 25 tests; the fixture needs the real data in `data/`)
- Parameter table: `python -m src.models`
- Data pipeline: `python -m src.dataset`

## Conventions and decisions

- Model input `X` has shape `(B, T, F)` with rho in column 0. Label `y` is the scaled rho at day `t+1`. Models take `(B, T, F)` and transpose for Conv1d themselves.
- `hidden` in `TABLE1` means **units per direction** for BiLSTM. This is an interpretation, not confirmed by the paper (must be labelled `[INFERENCE]`).
- Conv1D: kernel=3, padding="same" (group's own choice). LSTM has 2 layers. Models without attention use the last state of the top layer (both directions for BiLSTM). CLATT uses the full sequence plus attention.
- "CLATT- (rho only)" is CLATT with `n_features=1`.
- Every baseline is an affine function of rho, so it inverse-transforms the same way as the deep models' output. Ridge picks alpha on val (`logspace(-3, 3, 13)`) and is fit on train only.
- The `source` column in `model_params.csv` must say which group each setting belongs to: `[PAPER]`, `[INFERENCE]` or `nhóm tự chọn` (the group's own choice). Never tag something `[PAPER]` if the paper does not state it clearly.
- `set_seed` turns on `use_deterministic_algorithms(True)`; on GPU this needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Intentional shortcuts get a short comment: `shortcut: <the limit>, <when to upgrade>`.

## Pending work (from the latest review)

1. Label "units per direction" as `[INFERENCE]` for CNN-BiLSTM and CLATT (after re-reading Table 1 / Fig. 4).
2. Add `CUBLAS_WORKSPACE_CONFIG` to `set_seed`.
3. Move each baseline's `structure` and `n_params` into its class so `__main__.py` has less hardcoding.
4. Tighten the one-batch overfit test (compare with `var(y)`) and add a test for the BiLSTM last state.
5. Create `results/tables` in `__main__.py`; write the CSV with `convert_dtypes()`.

## Working notes

- `changes.diff` in the repo root is a temporary file. Do not commit it.
- When staging, include the deleted files (`clatt.py`, `lstm.py`, `cnn_lstm.py`, `cnn_bilstm.py`) together with the other changes.
