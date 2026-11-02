# clatt-repro

A PyTorch reproduction of **CLATT**, CNN-BiLSTM with attention, from *Short-Term Stock Correlation Forecasting Based on CNN-BiLSTM Enhanced by Attention Mechanism*. The model predicts the next-day 20-day rolling Pearson correlation of a stock pair. CLATT is compared with CNN, LSTM, CNN-LSTM, CNN-BiLSTM and simple baselines (persistence, moving average, AR(1), ridge).

The paper and the experiment plan (in Vietnamese) are in `docs/`.

## Status

| Stage | Status |
|---|---|
| 0-3. Data, models, training loop | Done |
| 4. Leakage checks | Done, all gates pass |
| 5. Choose T, LSTM layers, loss (validation) | Done: T = 90 (paper 68), 2 layers, MSE |
| 6. Equal-budget tuning (Optuna, protocol B) | Done |
| 7. Main test run | Not started |
| 8. Ablations | Not started |

All results so far are on the **validation split**; the test split has not been read.

The main findings so far:
- **CLATT does not beat persistence.** It has a lower R² on validation than predicting tomorrow's correlation as today's (about 0.82 vs 0.84 in group 1).
- **CLATT's lead depends on tuning.** With the paper's Table 1 settings, CLATT beats every baseline except CNN-LSTM in group 2. After 20 Optuna trials per model, tuned LSTM, CNN-LSTM and CNN-BiLSTM all match or pass tuned CLATT.

This is seed 0 on 2 pairs per group; stage 7 will confirm or reject it. Details, numbers and deviations from the paper are in [PROGRESS.md](PROGRESS.md).

## Setup

Python 3.13 (tested), CPU only.

```bash
pip install -r requirements.txt
python -m pytest -q            # 44 tests, about 100 s
```

The price data for the 8 ChiNext / SSE stocks (2016-09 to 2022-01) is already in `data/raw/`. To download it again from baostock, run:

```bash
python -m scripts.download_baostock
```

## Usage

```bash
# Data pipeline: stage 1 tables and figures
python -m src.dataset

# Train and evaluate on validation (one pair, one seed)
python -m src.run --exp smoke --models CLATT CNN-BiLSTM Persistence --pairs 300015_300274 --seeds 0
python -m src.aggregate smoke     # results/tables/smoke_summary.csv
python -m src.plots smoke         # results/figures/smoke_loss_curves.png

# Leakage checks (about 10 min)
python -m scripts.leakage_checks

# Stage 5 sweeps and report
scripts/stage5.sh T 20 40 60 68 72 90
python -m scripts.stage5_report

# Stage 6: tune one model and group, then write the report
python -m src.tune --model CLATT --group 1
python -m src.tune --report
```

Runs are saved under `results/runs/<exp>/<model>/<pair>/seed<k>/` (git-ignored); a run with an identical config is skipped. `--final` evaluates on the test split and is reserved for stage 7.

## Layout

| Path | Contents |
|---|---|
| `src/` | Data, features, dataset, models, baselines, metrics, training, tuning |
| `scripts/` | Data download, leakage checks, stage 5 and 6 drivers |
| `configs/` | `base.yaml` (shared settings), `models/` (paper Table 1), `tuned/` (stage 6 best) |
| `results/` | `tables/`, `figures/`, `logs/` (committed); `runs/`, `optuna/` (ignored) |
| `tests/` | Pipeline, leakage, models, metrics and tuner tests |

How the code works and the conventions it follows are in [CLAUDE.md](CLAUDE.md).
