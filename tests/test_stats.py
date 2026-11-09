"""Stage 7.4-7.5 on synthetic runs: DM test, comparisons, Spearman, evaluate tables (no real test split)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import evaluate, stats
from src.features import pairs

rng = np.random.default_rng(0)


def test_dm_same_distribution_not_significant():
    e1, e2 = rng.normal(0, 0.1, 400), rng.normal(0, 0.1, 400)
    dm, p = stats.dm_test(e1, e2)
    assert abs(dm) < 2 and p > 0.05


def test_dm_identical_errors():
    e = rng.normal(0, 0.1, 100)
    assert stats.dm_test(e, e) == (0.0, 1.0)


def test_dm_clearly_different():
    e1, e2 = rng.normal(0, 0.05, 300), rng.normal(0, 0.2, 300)
    dm, p = stats.dm_test(e1, e2)
    assert dm < 0 and p < 1e-6
    dm2, p2 = stats.dm_test(e2, e1)
    assert np.isclose(dm2, -dm) and np.isclose(p2, p)


def test_dm_harvey_correction_by_hand():
    e1, e2 = rng.normal(0, 0.1, 50), rng.normal(0, 0.12, 50)
    d = e1 ** 2 - e2 ** 2
    raw = d.mean() / np.sqrt(d.var() / len(d))
    dm, _ = stats.dm_test(e1, e2)
    assert np.isclose(dm, raw * np.sqrt((len(d) - 1) / len(d)))


N = 60


def write_run(root, exp, model, pair, seed, y, pred):
    out = root / exp / model / pair / f"seed{seed}"
    out.mkdir(parents=True)
    pd.DataFrame({"date": pd.date_range("2021-01-01", periods=len(y)).date, "y_true": y, "y_pred": pred,
                  "rho_prev": y}).to_csv(out / "preds_test.csv", index=False)
    mse = float(np.mean((pred - y) ** 2))
    (out / "metrics.json").write_text(json.dumps({"test": {"mse": mse, "mape": np.inf if seed == 1 else 0.3,
                                                           "skill": 0.1}, "train_seconds": 1.0}))


@pytest.fixture
def fake_runs(tmp_path, monkeypatch):
    for mod in ("src.stats.RUNS", "src.aggregate.RUNS"):
        monkeypatch.setattr(mod, tmp_path / "runs")
    monkeypatch.setattr("src.stats.TABLES", tmp_path)
    monkeypatch.setattr("src.evaluate.TABLES", tmp_path)
    for i, (_, a, b) in enumerate(pairs()):
        y = rng.uniform(-0.5, 0.8, N)
        pair = f"{a}_{b}"
        write_run(tmp_path / "runs", "fake", "Persistence", pair, 0, y, y + rng.normal(0, 0.10, N))
        for s in range(3):  # CLATT clearly better on every pair
            write_run(tmp_path / "runs", "fake", "CLATT", pair, s, y, y + rng.normal(0, 0.03, N))
    return tmp_path


def test_compare_and_verdict(fake_runs):
    per_pair, summary = stats.compare("fake", "CLATT", "fake", "Persistence")
    assert len(per_pair) == 12 and per_pair.a_better_sig.all()
    overall = summary[summary.group == "all"].iloc[0]
    assert overall.verdict_a_better and overall.n_dm_a_better == 12 and overall.wilcoxon_p < 0.01
    assert set(summary.group) == {"all", 1, 2}
    _, rev = stats.compare("fake", "Persistence", "fake", "CLATT")
    assert not rev[rev.group == "all"].iloc[0].verdict_a_better


def test_mean_forecast_checks_dates(fake_runs):
    preds = stats.load_preds("fake", "CLATT", "sz.300015_sz.300274")
    preds[1] = preds[1].assign(y_true=preds[1].y_true + 1)
    with pytest.raises(ValueError):
        stats.mean_forecast(preds)


def test_evaluate_tables(fake_runs):
    tables, gaps = evaluate.evaluate("fake")
    assert gaps == []
    o = tables["overall"].set_index(["model", "metric"])
    assert o.loc[("CLATT", "mse"), "n_seeds"] == 3 and o.loc[("CLATT", "mse"), "n_pairs"] == 12
    # seed 1 has mape = inf on every pair: dropped and counted, not averaged
    assert o.loc[("CLATT", "mape"), "n_dropped"] == 12 and np.isclose(o.loc[("CLATT", "mape"), "mean"], 0.3)
    assert np.isnan(o.loc[("Persistence", "mse"), "std"])  # one seed
    bp = tables["by_pair"]
    assert len(bp.query("model == 'CLATT' and metric == 'mse'")) == 12
    # by_group mean = mean over seeds of the pair mean = mean of pair means when every pair has every seed
    g1 = tables["by_group"].query("model == 'CLATT' and metric == 'mse' and group == 1")["mean"].item()
    assert np.isclose(g1, bp.query("model == 'CLATT' and metric == 'mse' and group == 1")["mean"].mean())


def test_evaluate_reports_missing_runs(fake_runs):
    import shutil
    shutil.rmtree(fake_runs / "runs" / "fake" / "CLATT" / "sz.300015_sz.300274" / "seed2")
    _, gaps = evaluate.evaluate("fake", seeds=[0, 1, 2])
    assert gaps == [("CLATT", "sz.300015_sz.300274", 2)]


def test_spearman_against_table5(tmp_path, monkeypatch):
    monkeypatch.setattr("src.stats.TABLES", tmp_path)
    rows = [{"model": m, "group": g, "metric": k, "mean": v}
            for g, paper in stats.TABLE5.items() for k, ref in paper.items() for m, v in ref.items()]
    pd.DataFrame(rows).to_csv(tmp_path / "x_test_by_group.csv", index=False)
    sp = stats.spearman("x")
    assert np.allclose(sp.spearman, 1.0) and len(sp) == 4


def test_report_tables_and_figures(fake_runs, monkeypatch):
    from src import plots
    (fake_runs / "tables").mkdir()
    (fake_runs / "figures").mkdir()
    monkeypatch.setattr("src.evaluate.TABLES", fake_runs / "tables")
    monkeypatch.setattr("src.stats.TABLES", fake_runs / "tables")
    monkeypatch.setattr("src.plots.RESULTS", fake_runs)
    evaluate.evaluate("fake")
    assert {"exp", "model", "group", "mse", "skill"} <= set(evaluate.compare(["fake"]).columns)
    assert plots.pred_vs_true("fake", models=("CLATT", "Persistence")).exists()
    assert plots.skill_by_pair().exists()
    # ablation table + figure, with the fake CLATT runs standing in for both the variant and main_A
    for v in ("A1", "A2"):
        (fake_runs / "runs" / f"abl_{v}").symlink_to(fake_runs / "runs" / "fake")
    table = stats.ablation(variants=("A1", "A2"), full="fake")
    assert np.allclose(table.mse_change_pct, 0) and len(table) == 6
    assert plots.ablation_bar().exists()
