import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from src.metrics import compute_all, dir_acc, mae, mape, mse, r2, skill
from src.train import load_config, train_one

rng = np.random.default_rng(0)
Y, P = rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200)


def test_match_sklearn():
    assert np.isclose(mse(Y, P), mean_squared_error(Y, P))
    assert np.isclose(mae(Y, P), mean_absolute_error(Y, P))
    assert np.isclose(r2(Y, P), r2_score(Y, P))
    assert np.isclose(mape(Y, P), mean_absolute_percentage_error(Y, P))


def test_skill_by_hand():
    y, prev = np.array([0.5, 0.6]), np.array([0.3, 0.4])   # persistence MSE = 0.04
    assert np.isclose(skill(y, np.array([0.4, 0.5]), prev), 0.75)  # model MSE = 0.01
    assert skill(y, prev, prev) == 0


def test_dir_acc_by_hand():
    y = np.array([0.5, 0.2, 0.3, 0.4])
    prev = np.array([0.4, 0.3, 0.3, 0.3])      # true moves: up, down, none (dropped), up
    pred = np.array([0.45, 0.35, 0.9, 0.3])    # predicted: up (hit), up (miss), -, no move (miss)
    assert np.isclose(dir_acc(y, pred, prev), 1 / 3)


def test_compute_all_keys():
    assert set(compute_all(Y, P, P * 0)) == {"mse", "rmse", "mae", "r2", "mape", "mape_excl", "smape",
                                            "n_near_zero", "skill", "r2_delta", "dir_acc"}


def test_test_split_needs_final():
    with pytest.raises(ValueError):
        train_one("Persistence", "sz.300015_sz.300274", 0, load_config("Persistence"), split_eval=("val", "test"))


def test_override_parses_exponent_floats():
    assert load_config("CLATT", ["lr=1e-3"])["lr"] == 0.001


def test_persistence_predictions_exact(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.RUNS", tmp_path)
    metrics = train_one("Persistence", "sz.300015_sz.300274", 0, load_config("Persistence"), exp="tmp")
    df = pd.read_csv(tmp_path / "tmp" / "Persistence" / "sz.300015_sz.300274" / "seed0" / "preds_val.csv")
    np.testing.assert_array_equal(df["y_pred"].to_numpy(), df["rho_prev"].to_numpy())
    assert metrics["val"]["dir_acc"] == 0
    assert metrics["val"]["skill"] == 0
    # r2_delta = -mean(delta)^2 / var(delta): mean(y_true - rho_prev) != 0 on val, so it is not exactly 0
    delta = df["y_true"].to_numpy() - df["rho_prev"].to_numpy()
    assert np.isclose(metrics["val"]["r2_delta"], 1 - np.sum(delta ** 2) / np.sum((delta - delta.mean()) ** 2))


def test_skip_and_config_diff_check(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.RUNS", tmp_path)
    cfg = load_config("Persistence")
    pair = "sz.300015_sz.300274"
    exp = "tmp"

    # Lần 1: chạy bình thường
    res1 = train_one("Persistence", pair, 0, cfg, split_eval=("val",), exp=exp, force=True)
    # Lần 2: cùng tham số -> skip và trả về kết quả cũ
    res2 = train_one("Persistence", pair, 0, cfg, split_eval=("val",), exp=exp, force=False)
    assert res1 == res2

    # Lần 3: split_eval khác (thêm test), final=True, không force -> ValueError
    with pytest.raises(ValueError, match="existing run differs in.*split_eval"):
        train_one("Persistence", pair, 0, cfg, split_eval=("val", "test"), exp=exp, final=True, force=False)

    # Lần 4: cfg đổi (ví dụ lr=0.01) -> ValueError
    cfg_changed = {**cfg, "lr": 0.01}
    with pytest.raises(ValueError, match="existing run differs in.*lr"):
        train_one("Persistence", pair, 0, cfg_changed, split_eval=("val",), exp=exp, force=False)

    # Lần 5: force=True -> chạy được và ghi đè
    res5 = train_one("Persistence", pair, 0, cfg_changed, split_eval=("val",), exp=exp, force=True)
    assert res5["val"]["mse"] == res1["val"]["mse"]

