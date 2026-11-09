"""Stage 4 leakage checks 1-5 and 9 (plan section 7); checks 6-8 train or measure, see scripts/leakage_checks.py."""
import numpy as np
import pytest

from scripts.leakage_checks import (
    chon_sieu_tham_so,
    grep_fill,
    label_overlap,
    static_checks,
)
from src.data import CODES
from src.dataset import T, make_dataset
from src.features import K, pair_frame, pairs, returns_table


@pytest.fixture(scope="module")
def returns():
    return returns_table(CODES)


@pytest.fixture(scope="module")
def rows(returns):
    hits = grep_fill()
    return [r for _, a, b in pairs() for r in static_checks(returns, a, b, hits)]


@pytest.mark.parametrize("check", ["ranh_gioi", "thu_tu_thoi_gian", "scaler", "dien_thieu",
                                   "nhan_khong_chong_lan_ranh_gioi"])
def test_static_checks_pass(rows, check):
    got = [r for r in rows if r["check"] == check]
    assert len(got) == 12
    assert all(r["status"] == "PASS" for r in got), [r for r in got if r["status"] != "PASS"]


def test_no_test_predictions_or_tuning_on_test():
    assert chon_sieu_tham_so()["status"] == "PASS"


def test_check5_reads_tune_py_only(tmp_path):
    tune = tmp_path / "tune.py"
    assert chon_sieu_tham_so(tune)["status"] == "PASS"  # missing file
    tune.write_text("# finalize the study\n")
    assert chon_sieu_tham_so(tune)["status"] == "PASS"
    tune.write_text('df = load(split="test")\n')
    assert chon_sieu_tham_so(tune)["status"] == "FAIL"


def test_future_window_label_disjoint_from_X(returns):
    for _, a, b in pairs():
        df = pair_frame(returns, a, b)
        assert label_overlap(df, make_dataset(df)) == len(df) - T  # next-day label overlaps X by k-1 returns
        fw = make_dataset(df, target="future_window")
        assert label_overlap(df, fw) == 0, (a, b)
        assert len(fw["X"]) == len(fw["y"]) == len(fw["y_raw"]) == len(df) - T - K + 1
        for i in (0, len(fw["X"]) // 2, len(fw["X"]) - 1):  # y = Pearson of returns t+1..t+k
            w = df.iloc[i + T: i + T + K]
            assert w.index[0] > fw["x_end"][i]
            assert np.isclose(fw["y_raw"][i], np.corrcoef(w.r_A, w.r_B)[0, 1]), (a, b, i)
