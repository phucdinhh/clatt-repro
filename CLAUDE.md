# clatt-repro

Reproduction of the CLATT paper (short-term stock correlation forecasting with CNN-BiLSTM + attention) in PyTorch.
Source documents are in `docs/`: the experiment plan (PDF, sections 3, 4.1-4.4, 5, 6) and the paper (PDF).
Current state: stages 0-4 done (stage 4 not committed). Stage 5 step 1 (T) done: T* = 90 (user-confirmed; paper 68 kept as a check). Step 2: L* = 2, step 3: loss* = MSE (both as the paper); stage 5 done and written to `configs/base.yaml` (T 90, layers 2, mse). Stage 6 done 2026-11-02 23:14 (`configs/tuned/`, `tuning_summary.csv`; tuned LSTM beats tuned CLATT in both groups on val; check 5 PASS). Stage 7-8 (guide of 2026-11-08): code done (`--tuned`, ablation switches, `evaluate.py`, `stats.py`, figures, lane/report/run_all scripts); configs frozen 2026-11-08 16:11 in `configs/FROZEN.sha256` (79 tests, check 5 PASS, no `preds_test.csv` existed). Stage 7 and A1-A4 done 2026-11-08 21:16 (828 test runs, report in `results/tables/main_compare.csv`, `*_tests.csv`, `ablation.csv`; conclusion in PROGRESS.md): CLATT loses to Persistence/AR(1) in both protocols, beats CNN-BiLSTM only with Table 1 settings; tuned LSTM is the best deep model; A1/A3 removals help, A2 ≈ full, A4 much worse. The test split has now been read: no further tuning or config change may use it. User decisions are in `configs/stage7_decisions.yaml`. Update CLAUDE.md and PROGRESS.md after each completed step. **The user said not to commit for now** (2026-11-08); the freeze manifest replaces the "configs frozen" commit.
**Never edit anything under `configs/` now** (`--final` refuses to run if `src.freeze check` fails), never use `--force` with `--final` (`train_one` refuses to reread test), and never tune, pick seeds or drop pairs from test numbers.

## Code map

| File | Contents |
|---|---|
| `src/data.py` | Constants (ROOT, CODES, GROUPS, START, END, RESULTS, PROCESSED), `load_raw`, `traded` |
| `src/features.py` | Log-returns, stock pairing (`pairs()`), rho label = rolling Pearson (k=20), `rho_stats` |
| `src/dataset.py` | T=68, FEATURES=[rho, r_A, r_B], sliding windows, purged train/val/test split (7/1/2), MinMax fit on train only. `make_dataset` also returns `y_raw` and `rho_prev` (original scale). `get_pair_data(pair)` rebuilds a pair from raw CSVs. `make_dataset(..., target="future_window")` makes y = Pearson over returns t+1..t+k (no overlap with X; purge grows by k-1; for ablation A5, not yet trained). `python -m src.dataset` writes the stage-1 tables and figures. `fixed_split_T` (>= T) splits as if T were that value, so every split keeps the same label dates for any T (stage 5 uses 90) |
| `src/models/__init__.py` | Loads `TABLE1` from `configs/models/*.yaml`; `MODELS`, `build(name, n_features, T, **hp)`, `n_params` |
| `src/models/cnn.py`, `recurrent.py` | `CNN`; `Attention` (Eq. 8-9) and `ConvRNN` (LSTM, CNN-LSTM, CNN-BiLSTM, CLATT); `head` = attention / last / mean (default: attention for CLATT, last otherwise; same parameter creation order as before, so protocol A runs are byte-identical) |
| `src/models/__main__.py` | `python -m src.models` writes `results/tables/model_params.csv` |
| `src/baselines.py` | Persistence, MovingAverage (MA5), AR1, RidgeWindow; `BASELINES` dict |
| `src/metrics.py` | mse, rmse, mae, r2, mape (drops inf/nan terms, counted by `n_mape_dropped`), mape_excl, smape, n_near_zero, skill, r2_delta, dir_acc, `compute_all`. All take original-scale `y_true, y_pred, rho_prev` |
| `src/train.py` | `set_seed`, `load_config(model, overrides, tuned_group=None)` (with `tuned_group` merges `configs/tuned/<model>_g<g>.yaml` and adds `tuned_from`; baselines unchanged), `fit` (Adam, clip, early stopping), `train_one` (saves everything under `results/runs/<exp>/<model>/<pair>/seed<k>/`; passes `features` and `head` when present; refuses `final` + `force` once `preds_test.csv` exists); `fit` uses `cfg["loss"]` (mse or mae) and `build` gets `cfg["layers"]` |
| `src/run.py` | CLI: `--exp --models --pairs/--group --seeds --config-override --final --force --tuned --ablation A1..A4`; `--tuned` picks the group of each pair; `--final` needs a clean freeze; logs to `results/logs/<exp>.log`. `--exp` may be nested (`sweep_T/T20`) |
| `scripts/stage5.sh` | Stage 5 sweeps, CLATT, 12 pairs, seeds 0-2, `fixed_split_T=90`: `T <T...>` -> `sweep_T/T<T>`; `L <T*> <L...>` -> `sweep_L/L<L>`; `loss <T*> <L*> mae` -> `sweep_loss/mae` (L=2 and mse runs are reused from earlier sweeps) |
| `scripts/stage5_report.py` | Writes `timestep.csv`, `lstm_layers.csv`, `loss_compare.csv` (with paper Table 3/4 R²), `timestep_curve.png`, `configs/proposed_stage5.yaml`. Skips sweeps with fewer than 36 runs. Selection: lowest val MSE averaged over both groups; the paper's value (68, 2, mse) wins if within 1 std over seeds |
| `src/aggregate.py` | `python -m src.aggregate <exp>` writes `results/tables/<exp>_metrics_long.csv` and `<exp>_summary.csv` |
| `src/plots.py` | `python -m src.plots <exp>` writes `results/figures/<exp>_loss_curves.png`; `--final main_A main_B` writes `<exp>_pred_vs_true.png` (illustration pairs, first 100 test days, seed-mean) and `skill_by_pair.png`; `--ablation` writes `ablation.png` |
| `src/tune.py` | Stage 6 protocol B: `python -m src.tune --model M --group G` (20 trials, TPE seed 0, MedianPruner per epoch, no report (so no pruning) in each pair's first `patience` = 10 epochs, steps of pair i offset by i*80; SQLite `results/optuna/study.db`, resumable; a stale RUNNING trial is marked FAIL and rerun). Objective: mean original-scale val MSE over 2 representative pairs per group (max and min std of rho on **train rows**), seed 0, max_epochs 80. Trial 0 = Table 1 + lr 1e-3, run directly, not sampled (batch 23 and 62 are not in {16, 32, 64}). `--report` writes `configs/tuned/<model>_g<g>.yaml`, `tuning_summary.csv`, `tuning_history.png` |
| `scripts/stage5_6_chain.sh` | Unattended: wait for sweep_L -> L* -> sweep_loss -> loss* -> base.yaml -> 10 studies -> tune report + check 5 (stopped at stage 6 by the pruner fix) |
| `scripts/stage6_chain.sh` | Its stage 6 tail alone: 10 studies -> tune report + check 5 |
| `src/freeze.py` | `python -m src.freeze write|check`: sha256 of every `configs/**/*.yaml` in `configs/FROZEN.sha256`; `src.run --final` exits if `check` is not clean |
| `src/evaluate.py` | Stage 7.4: `python -m src.evaluate main_A`, `main_B --baselines main_A` -> `<exp>_test_by_pair/_by_group/_overall.csv` (by_group/overall: pair mean per seed, then mean/std over seeds; non-finite values dropped and counted in `n_dropped`; warns on missing runs). `--compare main_A main_B` -> `main_compare.csv`, `skill_by_pair.csv` |
| `src/stats.py` | Stage 7.5: `dm_test` (squared error, h=1, Harvey, t(n-1), two-sided), `compare(exp_a, model_a, exp_b, model_b)` (primary DM on seed-mean forecasts + per-seed share, Wilcoxon over 12 pairs, verdict = mean rel. skill > 0 and DM sig. on >= 7/12 pairs), Spearman vs paper Table 5 (MSE and MAE). `python -m src.stats main_A` / `main_B --baselines main_A` -> `<exp>_dm_by_pair.csv`, `<exp>_tests.csv`, `<exp>_spearman_table5.csv`; `python -m src.stats ablation` -> `ablation.csv`, `ablation_dm_by_pair.csv` |
| `configs/ablations.yaml`, `stage7_decisions.yaml` | Stage 8 switches A1 `features=[rho]`, A2 `head=last`, A3 `filters=null`, A4 `head=mean` (CLATT, protocol A); the user's stage 7 decisions (criterion, seeds A 0-4 / B 0-2, illustration pairs, accepted deviations) |
| `scripts/stage7_lane.sh N` | Lane 1: main_A baselines (seed 0) + 5 deep x 12 pairs x seeds 0-4, then abl_A1, abl_A2; lane 2: main_B group 1 (`--tuned`, seeds 0-2), waits for `results/logs/main_A.done`, then abl_A3, abl_A4; lane 3: main_B group 2. Env `SEEDS_A`, `SEEDS_B`, `PY`; `caffeinate`; resumable |
| `scripts/stage7_8_report.sh`, `run_all.sh` | Report: evaluate, stats, figures, freeze check, check 5. `run_all.sh`: pytest, data, params, leakage checks, freeze check, lanes 1, 3, 2 in sequence, report (`SEEDS_A=0 SEEDS_B=0` for a quick trial) |
| `scripts/leakage_checks.py` | Stage 4: the 9 checks of plan section 7 -> `results/tables/leakage_checks.csv` (check, pair, value, threshold, status, note). Checks 6-7 train CLATT on 2 pairs in memory (val only, nothing under `results/runs/`); check 7 is gated by Ridge, the CLATT run is MEASURE only; the CSV is rewritten after each pair. Check 9 (`nhan_khong_chong_lan`) passes only for the `future_window` label; the main `next` label shares 19 of 20 returns with X by design |
| `tests/` | `test_leakage.py` (checks 1-5 and 9), `test_pipeline.py` (data, `fixed_split_T` label dates for all 14 T), `test_models.py` (plan 4.4 + baselines), `test_metrics.py` (metrics vs sklearn, hand examples, `--final` guard, `lr=1e-3` override parsing, exact Persistence predictions and skip/config-diff checks via `train_one` into `tmp_path`) |
| `configs/` | `base.yaml` (T, k, lr, clip_norm, patience, num_threads, layers, loss, fixed_split_T), `models/<model>.yaml` (hidden, filters, dropout, batch_size, max_epochs), `tuned/<model>_g<group>.yaml` (stage 6 best, incl. lr and max_epochs 80; not yet read by `train_one`), `experiments/` |

## Commands

- Tests: `python -m pytest -q` (about 60-100 s, 80 tests; fixtures need the real data in `data/`). Use the repo venv: `source .venv/bin/activate` (Python 3.13 via `uv venv --python 3.13 .venv`; system python is 3.9 without torch)
- Stage 7-8 runs: one terminal per lane, `bash scripts/stage7_lane.sh 1|2|3`, then `bash scripts/stage7_8_report.sh`; everything at once: `bash scripts/run_all.sh`
- Leakage checks: `python -m scripts.leakage_checks` (about 10 min, mostly the 4 CLATT trainings)
- Smoke run: `python -m src.run --exp smoke --models CLATT CNN-BiLSTM Persistence --pairs 300015_300274 --seeds 0`
- Aggregate and plot: `python -m src.aggregate smoke`, `python -m src.plots smoke`
- Parameter table: `python -m src.models`; data pipeline: `python -m src.dataset`

## Conventions and decisions

- Model input `X` is `(B, T, F)` with rho in column 0. Label `y` is the scaled rho at `t+1`. `rho_prev` = rho(t), original scale.
- Predictions are scaled back with `y_pred * span + lo`, using the scaler's rho column. Metrics always use the original scale.
- Persistence bypasses that round trip: `train_one` sets `y_pred = rho_prev` exactly. Through float32 `X` it would differ by ~1e-8 and make `dir_acc` noise. So Persistence has `skill = 0`, `dir_acc = 0` (a zero predicted change never matches a sign) and a slightly negative `r2_delta` (mean Δρ on val is not 0).
- `hidden` means **units per direction** for BiLSTM. This is an interpretation, not stated by the paper (`[INFERENCE]`).
- Conv1D: kernel=3, padding="same" (group's own choice). LSTM has 2 layers. Without attention, models use the last state of the top layer (both directions for BiLSTM). CLATT uses the full sequence plus attention.
- "CLATT- (rho only)" is CLATT with `n_features=1`.
- Training: Adam lr 1e-3, MSE (or MAE via `loss`) on scaled y, clip_grad_norm 1.0, early stopping patience 10 on scaled val MSE, best weights restored. `batch_size` and `max_epochs` come from `configs/models/<name>.yaml`. With `loss: mae`, early stopping still uses val MSE (our choice; Keras would use val MAE).
- A run is complete only when `metrics.json` exists (it is written last). An existing run is skipped only if its saved `config.yaml` (minus `split_label_dates`) equals the new run config; any difference (e.g. `split_eval` from `--final`, an override) raises `ValueError` naming the keys. Use a new `--exp` or `--force`.
- `--config-override key=value` values are parsed as YAML; a string for a float key (`lr=1e-3`, which PyYAML reads as text) is cast to `float`. Unknown keys raise `KeyError`.
- `split_eval` may include `"test"` only with `--final`; otherwise `train_one` raises `ValueError`.
- `source` column in `model_params.csv` and config comments: `[PAPER]`, `[INFERENCE]` or `nhóm tự chọn` (the group's own choice). Never tag something `[PAPER]` if the paper does not state it clearly.
- `set_seed` sets random, numpy, torch, `CUBLAS_WORKSPACE_CONFIG` and `use_deterministic_algorithms(True)`. `torch.set_num_threads` comes from `cfg["num_threads"]`.
- Intentional shortcuts get a short comment: `shortcut: <the limit>, <when to upgrade>`.

## Smoke results (exp `smoke`, pair 300015_300274, seed 0, val only; for information, not for tuning)

| Model | MSE | MAE | R² | r2_delta | skill | dir_acc | best epoch |
|---|---|---|---|---|---|---|---|
| Persistence | 0.0074 | 0.0515 | 0.795 | -0.006 | 0 | 0 | - |
| CLATT | 0.0085 | 0.0654 | 0.765 | -0.154 | -0.147 | 0.571 | 15 / 25 |
| CNN-BiLSTM | 0.0088 | 0.0682 | 0.756 | -0.200 | -0.192 | 0.561 | 26 / 36 |

Loss curves converge normally. `--force` reruns give byte-identical `metrics.json` and `preds_val.csv` (apart from timings).

## Leakage check results (stage 4, `results/tables/leakage_checks.csv`, val only)

All gates PASS: checks 1-4 and `nhan_khong_chong_lan_ranh_gioi` 12/12, check 5 PASS, checks 6-7 2/2. A rerun with `-W error::RuntimeWarning` gives a byte-identical CSV.
- Persistence R² on val (check 8, the free baseline to beat): group 1 mean 0.817, group 2 0.830. With the `future_window` label: -2.46 and -1.54.
- CLATT copying y = rho(t) (`dich_nhan_clatt`, MEASURE): R² 0.967 and 0.994, so it is capacity-limited, not leaking.
- Deviations from the stage 4 spec (**not yet signed off by the user**): check 7 is gated by Ridge, not CLATT. Check 5 lists `preds_test.csv` in the note but no longer fails on it (stage 7 will create them).
- The scaler extra check reports "ok" on all 12 pairs, meaning val exceeds the train range, so a train+val-fit scaler would be caught.

## Measured timing (M-series, 8 cores, 4 threads, one pair, ~685 train samples)

Seconds per epoch: CNN 0.13, LSTM 0.23, CNN-LSTM 0.94, CNN-BiLSTM 2.7, CLATT 2.8 (one CLATT run 70-155 s).
Revised plan section 11 estimate (sequential): stage 5 T sweep 10-22 h, layers + loss 5-10 h, stage 6 Optuna 5-9 h,
stage 7 main 7.5-11.5 h, stage 8 A1-A4 5-10 h, A5-A8 6-13 h; about 40-75 h total. Two parallel processes would roughly halve it (untested; changes `num_threads`, recheck reproducibility).
Timing-only runs (3 epochs, CNN/LSTM/CNN-LSTM) are in `results/runs/timing/`.

## Pending work (from the latest review)

1. Done 2026-11-08: `tests/test_stage7.py::test_train_one_deep_model_reproducible` (CNN-LSTM, 2 epochs).
2. Done 2026-11-08: `mape` drops non-finite terms (`n_mape_dropped`); `smape` is only nan if y_true = y_pred = 0.
3. Old `smoke`/`timing` runs lack the new keys `layers`, `loss`, `fixed_split_T`; rerunning them without `--force` raises the config-diff error.
4. Done 2026-11-08: ruff I001 import-order issues fixed after the stage 7 runs.
5. Older review items not re-checked since stage 2: baselines' `structure`/`n_params` hardcoded in `src/models/__main__.py`; overfit test compares with `var(y)`; test for the BiLSTM last state; `results/tables` creation and `convert_dtypes()` in `__main__.py`.

## Working notes

- Extension 1 (after stage 7, user request 2026-11-09): `target=delta` config key (`src/dataset.py`, `src/train.py`, `tests/test_extension.py`), exp `ext_delta`, report `python -m scripts.ext_delta_report` -> `results/ext/delta/` (DESIGN.md written before the test read, RESULTS.md). `results/ext/` and `results/logs/ext_*.log` are git-ignored on purpose: the user keeps extension results local, uncommitted. Extension 2 (2026-11-09): `src/group.py` + `src.run --pooled` trains one model per group on the 6 pairs' pooled train samples (own scaler/split per pair, cross-pair date check), writes per-pair outputs in the usual layout; exps `ext_group`, `ext_group_delta`; report `python -m scripts.ext_group_report` -> `results/ext/group/`. Summary of both: `results/ext/SUMMARY.md`.

- `changes.diff` in the repo root is a temporary file. Do not commit it.
- Leakage check 4 (`dien_thieu`) greps `src/` and `scripts/` for `fillna|bfill|backfill|interpolate`, even in reporting code: use `.where(notna, x)` or `np.nan_to_num` instead.
- Parallel processes (num_threads 4 each) give byte-identical runs: the demo CLATT run repeated while pytest ran gave the same `metrics.json` and `preds_val.csv` (2026-11-08).
- `results/runs/` and `*.pt` are git-ignored; tables and figures under `results/` are not.
