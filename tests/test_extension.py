"""Extension after stage 7: target=delta (forecast rho(t+1) - rho(t), add back rho(t))."""
import numpy as np
import pandas as pd
import pytest

from src.dataset import get_pair_data
from src.train import load_config, train_one

PAIR = "sz.300015_sz.300274"


@pytest.fixture(scope="module")
def both():
    return get_pair_data(PAIR), get_pair_data(PAIR, target="delta")


def test_delta_same_inputs_splits_and_dates(both):
    nxt, dlt = both
    assert np.array_equal(nxt["X"], dlt["X"])
    assert nxt["splits"] == dlt["splits"]
    assert nxt["y_date"].equals(dlt["y_date"])
    assert np.array_equal(nxt["y_raw"], dlt["y_raw"]) and np.array_equal(nxt["rho_prev"], dlt["rho_prev"])


def test_delta_label_is_change_and_inverts(both):
    _, d = both
    mu, sd = d["delta_scale"]
    back = d["y"].astype(np.float64) * sd + mu
    assert np.allclose(back, d["y_raw"] - d["rho_prev"], atol=1e-6)
    tr = d["splits"]["train"]
    assert abs(d["y"][tr].mean()) < 1e-5 and abs(d["y"][tr].std() - 1) < 1e-3  # standardized on train


def test_delta_scale_uses_train_labels_only(both):
    """Changing every rho after the train split leaves the label scaling unchanged."""
    from src import dataset
    _, d = both
    a, b = PAIR.split("_")
    df = dataset.pair_frame(dataset.returns_table([a, b]), a, b, dataset.K)
    last_train_label = d["y_date"][d["splits"]["train"]][-1]
    df2 = df.copy()
    df2.loc[df2.index > last_train_label, "rho"] = df2.loc[df2.index > last_train_label, "rho"] * 0.5 + 0.3
    d2 = dataset.make_dataset(df2, target="delta")
    assert d2["delta_scale"] == d["delta_scale"]


def test_default_target_unchanged():
    assert "target" not in load_config("CLATT")
    assert load_config("CLATT", ["target=delta"])["target"] == "delta"


def test_delta_train_one_adds_back_rho(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.RUNS", tmp_path)
    cfg = load_config("CLATT", ["target=delta", "max_epochs=1"])
    m = train_one("CLATT", PAIR, 0, cfg, exp="tmp")
    preds = pd.read_csv(tmp_path / "tmp" / "CLATT" / PAIR / "seed0" / "preds_val.csv")
    # forecast = rho(t) + small change: far closer to rho(t) than the spread of rho itself
    assert np.abs(preds.y_pred - preds.rho_prev).mean() < 0.2 * preds.rho_prev.std() + 0.05
    assert m["val"]["mse"] > 0 and np.isfinite(m["val"]["skill"])
