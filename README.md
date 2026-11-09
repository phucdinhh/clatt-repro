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
| 7. Main test run (protocols A and B) | Done 2026-11-08 |
| 8. Ablations A1-A4 | Done 2026-11-08 (A5-A8 not run) |

### Main results (test split, 12 pairs, mean over seeds)

| Model | MSE, protocol A (paper settings, 5 seeds) | MSE, protocol B (tuned, 3 seeds) |
|---|---|---|
| CLATT | 0.0082 | 0.0068 |
| CNN-BiLSTM | 0.0105 | 0.0071 |
| CNN-LSTM | 0.0082 | 0.0070 |
| LSTM | 0.0155 | **0.0057** |
| CNN | 0.0163 | 0.0146 |
| Persistence (ρ̂(t+1) = ρ(t)) | **0.0048** | 0.0048 |
| AR(1) | **0.0047** | 0.0047 |

- **CLATT does not beat simple baselines.** Its MSE is 70% (A) and 43% (B) above persistence, and the Diebold-Mariano test favours persistence on 11/12 (A) and 8/12 (B) pairs. No deep model beats persistence or AR(1).
- **CLATT's lead over other deep models depends on tuning.** With the paper's settings CLATT beats CNN-BiLSTM by 22% (significant on 10/12 pairs), ties CNN-LSTM, and the model ranking matches the paper in group 1 (Spearman 0.90). With equal tuning, the gap to CNN-BiLSTM is 4% and not significant, and plain LSTM is best.
- **Ablations:** removing the return inputs (-17% MSE) or the Conv1D layer (-27%) helps. Replacing attention with the last BiLSTM state changes nothing. Mean pooling is much worse.

The decision rule was fixed before the test split was read (`configs/stage7_decisions.yaml`). Full tables, tests and caveats are in [PROGRESS.md](PROGRESS.md) and `results/tables/`. Figures are in `results/figures/` (`main_A_pred_vs_true.png`, `skill_by_pair.png`, `ablation.png`).

## Setup

Python 3.13 (tested: 3.13, torch 2.15.0, numpy 2.5.4, pandas 3.1.1), CPU only. Results were produced from commit `667b2ff` plus uncommitted stage 7-8 changes; configs are pinned by `configs/FROZEN.sha256` (`python -m src.freeze check`).

```bash
pip install -r requirements.txt          # or the exact versions: requirements-lock.txt
python -m pytest -q            # 80 tests, about 1 min
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

# Stage 7-8: test runs (reads the test split once; configs must match the freeze), then tables, tests, figures
bash scripts/stage7_lane.sh 1      # main_A, then ablations A1, A2   (lanes in 3 terminals, about 5 h)
bash scripts/stage7_lane.sh 2      # main_B group 1, then A3, A4
bash scripts/stage7_lane.sh 3      # main_B group 2
bash scripts/stage7_8_report.sh
# or everything, from tests to report:  bash scripts/run_all.sh   (SEEDS_A=0 SEEDS_B=0 for a quick trial)
```

Runs are saved under `results/runs/<exp>/<model>/<pair>/seed<k>/` (git-ignored); a run with an identical config is skipped. `--final` evaluates on the test split and is reserved for stage 7.

Timing (Apple M-series, 4 threads per process): one CLATT run takes 1.5-2.5 min; stage 7-8 (828 test runs) took 5 h 04 min in 3 parallel processes.

## Layout

| Path | Contents |
|---|---|
| `src/` | Data, features, dataset, models, baselines, metrics, training, tuning |
| `scripts/` | Data download, leakage checks, stage 5-8 drivers, `run_all.sh` |
| `configs/` | `base.yaml` (shared settings), `models/` (paper Table 1), `tuned/` (stage 6 best), `ablations.yaml`, `stage7_decisions.yaml`, `FROZEN.sha256` |
| `results/` | `tables/`, `figures/`, `logs/` (committed); `runs/`, `optuna/` (ignored) |
| `tests/` | Pipeline, leakage, models, metrics and tuner tests |

How the code works and the conventions it follows are in [CLAUDE.md](CLAUDE.md).
