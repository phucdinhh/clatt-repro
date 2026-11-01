"""Automated checks listed at the end of plan section 3."""
import numpy as np
import pytest

from src.data import CODES
from src.dataset import T, make_dataset
from src.features import K, pair_frame, pairs, returns_table


@pytest.fixture(scope="module")
def frames():
    returns = returns_table(CODES)
    return {f"{a}_{b}": pair_frame(returns, a, b) for _, a, b in pairs()}


@pytest.fixture(scope="module")
def datasets(frames):
    return {name: (df, make_dataset(df)) for name, df in frames.items()}


def test_twelve_pairs(frames):
    assert len(frames) == 12


def test_no_nan_and_rho_in_range(frames):
    for name, df in frames.items():
        assert not df.isna().any().any(), name
        assert df.rho.between(-1, 1).all(), name


def test_rolling_corr_matches_numpy(frames):
    rng = np.random.default_rng(0)
    returns = returns_table(CODES)
    for name, df in frames.items():
        a, b = name.split("_")
        joined = returns[[a, b]].dropna()
        for date in rng.choice(df.index, 10, replace=False):
            w = joined.loc[:date].iloc[-K:]
            assert np.isclose(df.rho[date], np.corrcoef(w[a], w[b])[0, 1]), (name, date)


def test_window_ends_before_label(datasets):
    for name, (_, ds) in datasets.items():
        assert (ds["x_end"] < ds["y_date"]).all(), name


def test_purge_between_splits(datasets):
    for name, (df, ds) in datasets.items():
        s = ds["splits"]
        for prev, nxt in [("train", "val"), ("val", "test")]:
            last_used = s[prev].stop - 1 + T              # row of the last label in prev split
            first_used = s[nxt].start - K + 1             # first return under rho of next split's first input
            assert s[nxt].start - s[prev].stop >= K + T, name
            assert df.index[last_used] < df.index[first_used], (name, prev, nxt)


def test_scaler_fit_on_train_only(datasets):
    for name, (df, ds) in datasets.items():
        rows = df.iloc[: ds["splits"]["train"].stop + T]
        assert np.array_equal(ds["scaler"].data_min_, rows.min().to_numpy()), name
        assert np.array_equal(ds["scaler"].data_max_, rows.max().to_numpy()), name


def test_shapes(datasets):
    for name, (_, ds) in datasets.items():
        assert ds["X"].shape[1:] == (T, 3) and len(ds["X"]) == len(ds["y"]), name
