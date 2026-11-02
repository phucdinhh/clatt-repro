"""Stage 6 tuner on a tiny budget: Table 1 is trial 0, the budget is never exceeded, an interrupted trial is rerun."""
import optuna
import yaml

import src.tune as tune
from src.models import TABLE1


def test_tune_budget_resume_and_report(tmp_path, monkeypatch):
    for d in ("tables", "figures"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(tune, "N_TRIALS", 3)
    monkeypatch.setattr(tune, "MAX_EPOCHS", 1)
    monkeypatch.setattr(tune, "STORAGE", tmp_path / "study.db")
    monkeypatch.setattr(tune, "TUNED", tmp_path / "tuned")
    monkeypatch.setattr(tune, "RESULTS", tmp_path)

    study = tune.study_for("LSTM", 1)
    study.enqueue_trial({}, user_attrs={"table1": True})
    study.ask()  # trial 0 left RUNNING, as if the process was killed
    tune.tune("LSTM", 1)
    tune.tune("LSTM", 1)  # resume: nothing left to do

    trials = tune.study_for("LSTM", 1).trials
    done = [t for t in trials if t.state in tune.DONE]
    assert len(done) == 3
    assert trials[0].state == optuna.trial.TrialState.FAIL
    t1 = next(t for t in done if t.user_attrs.get("table1"))
    assert t1.user_attrs["hp"] == {"hidden": 32, "dropout": 0.2, "batch_size": 64, "lr": 0.001}
    assert TABLE1["LSTM"]["hidden"] == 32 and "filters" not in done[-1].user_attrs["hp"]
    assert all(not t.intermediate_values for t in done)  # epoch 1 <= patience: never reported, so never pruned

    tune.report()
    cfg = yaml.safe_load((tmp_path / "tuned" / "lstm_g1.yaml").read_text())
    assert cfg["name"] == "LSTM" and cfg["max_epochs"] == 1
    assert (tmp_path / "tables" / "tuning_summary.csv").exists() and (tmp_path / "figures" / "tuning_history.png").exists()
