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
| 5. Choose T, layers, loss | In progress (T sweep running) | not committed |
| 6. Protocol B tuning (Optuna) | Not started | |
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
- To do: report T table and figure, propose T*; then `sweep_L` (L 1, 3, 4; L=2 reused) and `sweep_loss` (mae; mse reused); write `configs/proposed_stage5.yaml`; base.yaml only after the user confirms.

## Open decisions (plan section 17)

- Inner join or forward fill when aligning pairs: inner join is implemented, but the choice is not formally signed off.
- Confirm or reject criteria (plan 6.4): to agree with the advisor before stage 7.
- PyTorch or Keras: PyTorch is in use.
- Run protocol B or only protocol A: not decided. This affects the stage 6 time budget.
- Overfit test criterion (stage 2): keep "95% drop" or require 3,000 steps.
- Stage 5: run the remaining 8 T values (63-67, 69-71) or not; confirm T*, L*, loss* before writing base.yaml.
- Run two processes in parallel to roughly halve the runtime: untested.

## Code to-dos

See "Pending work" in `CLAUDE.md`.
