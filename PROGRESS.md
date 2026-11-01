# Progress

Stage numbers follow section 10 of the experiment plan (`docs/`). How the code works is in `CLAUDE.md`.
This file records what each stage produced, how the data and code differ from the paper or plan, and what is still open.

| Stage | Status | Commit |
|---|---|---|
| 0. Download script | Done | b439a5e |
| 1. Data pipeline | Done | b439a5e |
| 2. Models and baselines | Done | a42ac39 |
| 3. Training loop and evaluation | Done | not committed |
| 4. Leakage checks | Not started | |
| 5. Choose T, layers, loss | Not started | |
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

## Open decisions (plan section 17)

- Inner join or forward fill when aligning pairs: inner join is implemented, but the choice is not formally signed off.
- Confirm or reject criteria (plan 6.4): to agree with the advisor before stage 7.
- PyTorch or Keras: PyTorch is in use.
- Run protocol B or only protocol A: not decided. This affects the stage 6 time budget.
- Overfit test criterion (stage 2): keep "95% drop" or require 3,000 steps.
- Run two processes in parallel to roughly halve the runtime: untested.

## Code to-dos

See "Pending work" in `CLAUDE.md`.
