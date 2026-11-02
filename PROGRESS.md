# Progress

Stage numbers follow section 10 of the experiment plan (`docs/`). How the code works is in `CLAUDE.md`.
This file records what each stage produced, how the data and code differ from the paper or plan, and what is still open.

| Stage | Status | Commit |
|---|---|---|
| 0. Download script | Done | b439a5e |
| 1. Data pipeline | Done | b439a5e |
| 2. Models and baselines | Done | a42ac39 |
| 3. Training loop and evaluation | Done | d0bf4f4 |
| 4. Leakage checks | Done (all gates PASS) | not committed |
| 5. Choose T, layers, loss | Done (T = 90, L = 2, MSE; written to base.yaml) | 135b659, ab10d6e, step 3 below |
| 6. Protocol B tuning (Optuna) | Running (restarted 2026-11-02 16:07 after the pruner fix) | |
| 7. Main test run (`--final`) | Not started | |
| 8. Ablations and packaging | Not started | |

## Stage 0: download script

`scripts/download_baostock.py` downloads 8 stocks, 2016-09-01 to 2022-01-28, `adjustflag="2"`. Run locally with baostock 0.9.4 (`data/raw/VERSION.txt`).

## Stage 1: data pipeline

Outputs: `results/tables/data_summary.csv`, `rho_stats.csv`, `split_dates.csv`, `results/figures/rho_series_*.png`.

How the data differs from the paper (report section 4.3):
- 2017-01-04 to 2022-01-28 has 1,235 market days, 8 more than the paper's 1,227. Stocks that never halted are at +8. The paper does not say which days it dropped.
- Traded days per stock: 1,209 to 1,235. sz.300450 halted from before the sample until 2017-02-07, so its 3 pairs start on 2017-02-08.
- Inner join leaves 1,204 to 1,235 days per pair. No unusual returns follow a halt; returns above 15% are all from August 2020 onward, when ChiNext's daily limit widened to 20%.

How the code differs from the plan:
- The k+T = 88-sample purge is removed first, then the rest is split 70/10/20. Splitting first would leave about 26 validation samples. The result is about 680 train, 97 validation and 195 test samples per pair (the plan estimated about 220 test).
- The scaler is fit on every row that train inputs and labels touch.

## Stage 2: models and baselines

Output: `results/tables/model_params.csv`. Parameters: CNN 135k, LSTM 13k, CNN-LSTM 173k, CNN-BiLSTM 534k, CLATT 1.13M. CLATT is about 2x CNN-BiLSTM, so ablation A2 matters.

Choices the paper does not state (all tagged in the CSV):
- `hidden` is units per direction for BiLSTM [INFERENCE].
- Dropout goes after Conv1D and after the LSTM in all models [INFERENCE].
- Conv kernel 3, padding same (nhóm tự chọn).
- The BiLSTM "last state" is the final state of each direction, as in Keras.

How the code differs from the plan:
- The overfit test (plan 4.4) checks a loss drop of at least 95% after 300 steps, not "near 0". At 300 steps the LSTM models are still at 0.08 to 0.18 x var(y); they reach about 0 at around 3,000 steps.

## Stage 3: training loop and evaluation

Smoke run (validation only, pair 300015_300274, seed 0; for information, not for tuning):

| Model | MSE | R² | r2_delta | skill | dir_acc |
|---|---|---|---|---|---|
| Persistence | 0.0074 | 0.795 | -0.006 | 0 | 0 |
| CLATT | 0.0085 | 0.765 | -0.154 | -0.147 | 0.571 |
| CNN-BiLSTM | 0.0088 | 0.756 | -0.200 | -0.192 | 0.561 |

- Both deep models are worse than persistence on this one pair and seed. Loss curves converge normally.
- `--force` reruns are byte-identical.
- Measured runtimes put stages 5 to 8 at about 40 to 75 CPU hours run one after another, about twice the plan's estimate (details in `CLAUDE.md`).

Bugs found and fixed during review:
- Persistence `dir_acc` came out as 0.54 because of float32 rounding; it is now exactly 0.
- Skipping an existing run ignored config changes; it now compares the saved `config.yaml`.
- `--config-override lr=1e-3` was read as a string; it is now a float.

## Stage 4: leakage checks

Code: `scripts/leakage_checks.py`, `tests/test_leakage.py`, `target="future_window"` option in `make_dataset`. Results: `results/tables/leakage_checks.csv` (rewritten after each pair).

Status per check (full run of `python -m scripts.leakage_checks`, no warnings):

| # | Check | Result |
|---|---|---|
| 1 | ranh_gioi | 12/12 PASS (pytest) |
| 2 | thu_tu_thoi_gian | 12/12 PASS (pytest) |
| 3 | scaler | 12/12 PASS (pytest; range derived from sample dates, also compared with the train+val range) |
| 4 | dien_thieu | 12/12 PASS (pytest; no bfill/backfill/interpolate/fillna in src/ or scripts/) |
| 5 | chon_sieu_tham_so | PASS (tune.py empty, rerun at stage 6; `preds_test.csv` files are listed in the note only) |
| 6 | xao_tron_nhan | 2/2 PASS (300015_300274 R² -0.012; 300014_300122 R² 0.0006) |
| 7 | dich_nhan | 2/2 PASS, gated by Ridge (R² 1.0000 on both pairs). CLATT as a MEASURE row (`dich_nhan_clatt`): R² 0.967 and 0.994 |
| 8 | chong_lan_cua_so | measurement only (status MEASURE, no threshold); computed in `static_checks`, before the training checks. Mean persistence R² on val: group 1 0.817, group 2 0.830 (with the `future_window` label: -2.46 and -1.54) |
| 9 | nhan_khong_chong_lan | the main `next` label shares 19 of 20 returns with X (by the paper's design); only the `future_window` label is disjoint, and that is what pytest checks (no shared return, matches a direct Pearson). Persistence R² on val is a consequence of the overlap |

Check 7 was first gated on CLATT copying `y = X[:, -1, 0]` and failed at R² 0.967 < 0.98. That tested model capacity, not the pipeline. The gate is now Ridge on the same y (threshold unchanged at 0.98), which reaches R² 1.0000 on both pairs, so the pipeline delivers the information. CLATT is kept as a non-gating measurement.

## Stage 5: choose T, LSTM layers, loss (val only)

Code: `fixed_split_T` in `make_dataset`, config keys `layers`, `loss`, `fixed_split_T` (base.yaml), `scripts/stage5.sh`, `scripts/stage5_report.py`. Tests: 43 passed; leakage check 5 still PASS.

- Fair comparison across T: splits are computed once with T_max = 90 and shifted, so train, val and test have the same label dates for every T (2017-05-24..2020-01-15, 2020-07-07..2020-11-18, 2021-05-06..2022-01-28 on 300015_300274). Tested for all 12 pairs and 14 T.
- Time estimate for the full T grid (14 T x 12 pairs x 3 seeds = 504 runs): about 11 h, over the 6 h limit. So only {20, 40, 60, 68, 72, 90} (216 runs, about 4-8 h) was started on 2026-11-01 21:27. The user decides on the other 8 T after the report.
- With `loss: mae`, early stopping still uses val MSE so both losses are selected on the same metric (our choice).
### Step 1: time step T (done 2026-11-02 03:10, 216 runs, no errors)

`results/tables/timestep.csv`, `results/figures/timestep_curve.png`. Val, mean ± std over 3 seeds of the 6-pair average:

| T | R² group 1 | R² group 2 | val MSE (both groups) |
|---|---|---|---|
| 20 | 0.805 ± 0.013 | 0.789 ± 0.004 | 0.00767 ± 0.00027 |
| 40 | 0.810 ± 0.007 | 0.791 ± 0.005 | 0.00748 ± 0.00016 |
| 60 | 0.813 ± 0.009 | 0.791 ± 0.005 | 0.00734 ± 0.00021 |
| 68 | 0.814 ± 0.010 | 0.787 ± 0.006 | 0.00730 ± 0.00016 |
| 72 | 0.820 ± 0.006 | 0.789 ± 0.003 | 0.00717 ± 0.00019 |
| 90 | 0.822 ± 0.004 | 0.794 ± 0.006 | 0.00702 ± 0.00015 |
| persistence | 0.841 | 0.810 | |

- T* = 90: lowest val MSE, and 68 is not within 1 std. This differs from the paper (68). The user confirmed 90 as the main setting and kept 68 as a paper check.
- No clear peak. Group 1 rises steadily with T (+0.017, more than the seed std); group 2 is flat (range 0.007, about one std). 90 is the edge of the grid, so the optimum may lie beyond it.
- CLATT is below persistence at every T (skill -0.13 to -0.25).
- The other 8 T values (63-67, 69-71) were skipped by the user: all are below 72, which already loses to 90.

### Step 2: LSTM layers (done 2026-11-02 14:40, L 1, 3, 4 at T=90: 108 new runs; L=2 reused from `sweep_T/T90`)

`results/tables/lstm_layers.csv`. Val, mean ± std over 3 seeds of the 6-pair average, next to the paper's Table 3 R² (test set, so only the ranking is comparable):

| L | R² group 1 | R² group 2 | val MSE g1 / g2 | paper R² g1 / g2 |
|---|---|---|---|---|
| 1 | 0.821 ± 0.004 | 0.789 ± 0.009 | 0.00779 / 0.00662 | 0.9467 / 0.8726 |
| 2 | **0.822 ± 0.004** | **0.794 ± 0.006** | **0.00758 / 0.00646** | **0.9655 / 0.9360** |
| 3 | 0.818 ± 0.006 | 0.791 ± 0.006 | 0.00796 / 0.00653 | 0.9561 / 0.9228 |
| 4 | 0.812 ± 0.009 | 0.747 ± 0.067 | 0.00814 / 0.00761 | 0.9397 / 0.8588 |

- L* = 2: lowest val MSE in both groups, the same as the paper. No other L is within 1 std.
- L = 4 is worst, as in the paper; group 2 is unstable across seeds (R² std 0.067). The order of L = 1 and 3 differs from the paper in group 1, but the gaps are within seed noise.
- Our R² is far below the paper's at every L (val vs their test; still below persistence, skill -0.13 to -0.37).

### Step 3: loss (done 2026-11-02 15:45, mae at T=90, L=2: 36 new runs; mse reused from `sweep_T/T90`)

`results/tables/loss_compare.csv`, next to the paper's Table 4 R²:

| loss | R² group 1 | R² group 2 | val MSE g1 / g2 | paper R² g1 / g2 |
|---|---|---|---|---|
| **mse** | **0.822 ± 0.004** | 0.794 ± 0.006 | **0.00758 / 0.00646** | **0.9825 / 0.9661** |
| mae | 0.822 ± 0.002 | 0.794 ± 0.005 | 0.00775 / 0.00651 | 0.9820 / 0.9644 |

- loss* = MSE: lowest val MSE, and MAE is within 1 std, so the paper's value wins under either reading. As in the paper, the two losses are nearly tied.
- The chain wrote `T: 90`, `layers: 2`, `loss: mse` into `configs/base.yaml` (comments name the paper values). `fixed_split_T` stays null: with T = 90 the splits are the same as in the sweeps.

**Stage 5 outcome:** T = 90 (paper 68), L = 2 (paper 2), MSE (paper MSE). Only T differs from the paper.

### Stage 6: unattended chain

On 2026-11-02 the user authorised Claude to decide and continue on its own, preferring the paper's values whenever the selection rule allows (L=2 and MSE win if within 1 std). `scripts/stage5_6_chain.sh` (started 07:25) waits for `sweep_L` (L 1, 3, 4 at T=90, started 07:20), picks L*, runs `sweep_loss` (mae), picks loss*, writes T=90, L*, loss* to `configs/base.yaml`, then runs the 10 stage 6 studies, the tuning report and leakage check 5. Log: `results/logs/stage5_6_chain.log`.

## Stage 6: protocol B tuning (Optuna)

Started 2026-11-02 15:45 by the chain with T=90, L=2, MSE. Order: CLATT, CNN-BiLSTM, CNN-LSTM, LSTM, CNN, each group 1 then 2. Representative pairs at T=90 (train std of rho): group 1 sz.300015_sz.300347 (max), sz.300347_sh.603799 (min). Logs: `results/logs/tune_<model>_g<group>.log`.

Studies done (val MSE on the original scale, mean over the 2 pairs; trial 0 = Table 1):

| Study | Time | Complete / pruned | Table 1 | Best (trial) | Gain | Best hp |
|---|---|---|---|---|---|---|
| CLATT g1 | 16:07-16:48 | 8 / 12 | 0.004365 | 0.004237 (18) | 2.9% | hidden 256, filters 104, dropout 0.10, batch 32, lr 5.6e-4 |
| CLATT g2 | 16:48-17:33 | 8 / 12 | 0.006317 | 0.005953 (19) | 5.8% | hidden 128, filters 96, dropout 0.10, batch 16, lr 9.5e-4 |
| CNN-BiLSTM g1 | 17:34-18:08 | 9 / 11 | 0.005723 | 0.004021 (18) | 29.7% | hidden 128, filters 80, dropout 0.17, batch 32, lr 1.8e-3 |
| CNN-BiLSTM g2 | 18:08-21:30 | 11 / 9 | 0.006550 | 0.005589 (14) | 14.7% | hidden 200, filters 112, dropout 0.29, batch 16, lr 6.8e-4 |
| CNN-LSTM g1 | 21:30-21:48 | 9 / 11 | 0.004758 | 0.004103 (11) | 13.8% | hidden 32, filters 128, dropout 0.21, batch 16, lr 2.9e-3 |
| CNN-LSTM g2 | 21:48-22:23 | 5 / 15 | 0.005920 | 0.005920 (0) | 0% | Table 1 (hidden 114, filters 32, dropout 0.2, batch 32, lr 1e-3) |
| LSTM g1 | 22:23-22:51 | 9 / 11 | 0.006807 | 0.003789 (4) | 44.3% | hidden 208, dropout 0.45, batch 16, lr 1.8e-3 |
| LSTM g2 | 22:51-23:08 | 6 / 14 | 0.007359 | 0.005507 (1) | 25.2% | hidden 152, dropout 0.39, batch 16, lr 1.3e-3 |

- After the warmup fix, pruned trials stop at epoch 11 with clearly worse val loss (scaled 0.0026-0.0113 against 0.0020-0.0025 for most complete trials), so pruning no longer looks like noise.
- Table 1 is already close to the best for CLATT (gains 2.9% and 5.8%). Both best configs use the lowest dropout region (0.10).
- CNN-BiLSTM g1: Table 1 is 31% worse than CLATT (0.005723 vs 0.004365), but tuned it beats tuned CLATT (0.004021 vs 0.004237). So the Table 1 gap is mostly a tuning gap, the point protocol B is meant to test.
- CNN-BiLSTM g2: tuned it also beats tuned CLATT (0.005589 vs 0.005953), so this holds in both groups (seed 0, 2 pairs each). The study took 3 h 22 min: batch 16 with hidden 200-232 ran 40-47 min per trial, and epochs were up to about 6 s, against 2.7 s in the stage 3 timing.
- LSTM g1: Table 1 (hidden 32) is the worst config so far, but tuned (hidden 208) it is the best of all models in group 1 (0.003789 vs CLATT 0.004237).

Code: `src/tune.py`, `tests/test_tune.py` (budget, resume, Table 1 trial, report on a tiny budget), `fit(on_epoch=...)` for pruning. Tests: 44 passed; check 5 PASS (tune.py never names the test split).

How the code differs from the stage 6 spec (decided by Claude under the user's authorisation):
- Trial 0 is Table 1 run directly, not via `enqueue_trial` with sampled params: Table 1 batch sizes 23 and 62 are not in {16, 32, 64}, and Optuna rejects them. TPE therefore does not learn from trial 0; it still counts as 1 of the 20 trials.
- Representative pairs use the std of rho over train rows, not the full-series std in `rho_stats.csv` (which includes val and test). With T=68 this changes the group 1 max pair (300015_300347 instead of 300015_300274).
- A study resumed after an interruption does not reproduce the uninterrupted TPE sequence.
- Pruner warmup (fix on 2026-11-02 16:07): the first CLATT g1 study (15:45-16:06) used MedianPruner with no warmup, and trials 5-19 were all pruned after 1-2 epochs (trial 5 at val 0.0144 against a median of about 0.0143), so only 5 trials trained fully. Epoch-1 val loss says little with patience 10. Fix: val loss is not reported (so no pruning) in each pair's first `patience` = 10 epochs. CLATT g1 and the partly run g2 were deleted from the study DB (backup: `results/optuna/study_noWarmup_backup.db`) and stage 6 restarted with `scripts/stage6_chain.sh`. Every model is tuned with the fixed pruner. Check 5 still PASS, 44 tests unchanged in count (`test_tune` also asserts nothing is reported at epoch 1).

## Open decisions (plan section 17)

- Inner join or forward fill when aligning pairs: inner join is implemented, but the choice is not formally signed off.
- Confirm or reject criteria (plan 6.4): to agree with the advisor before stage 7.
- PyTorch or Keras: PyTorch is in use.
- Run protocol B or only protocol A: not decided. This affects the stage 6 time budget.
- Overfit test criterion (stage 2): keep "95% drop" or require 3,000 steps.
- Run two processes in parallel to roughly halve the runtime: untested.

## Code to-dos

See "Pending work" in `CLAUDE.md`.
