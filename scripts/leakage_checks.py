"""Stage 4: the 9 leakage checks of plan section 7 -> results/tables/leakage_checks.csv.

Run from the repo root: python -m scripts.leakage_checks
Checks 6-7 train CLATT on 2 pairs (val only, nothing written under results/runs/); 7 is gated by Ridge.
"""
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baselines import RidgeWindow
from src.data import CODES, RESULTS, ROOT, load_raw, traded
from src.dataset import FEATURES, T, make_dataset
from src.features import K, pair_frame, pairs, returns_table
from src.metrics import r2, r2_delta
from src.models import build
from src.train import RUNS, fit, load_config, predict, set_seed

TRAIN_PAIRS = ["sz.300015_sz.300274", "sz.300014_sz.300122"]
FILL = re.compile(r"bfill|backfill|interpolate|fillna")
TUNE_TEST = re.compile(r"""["']test["']|--final|\bfinal\s*=""")


def row(check, pair, value, threshold, ok, note=""):
    status = "MEASURE" if ok is None else ("PASS" if ok else "FAIL")
    return {"check": check, "pair": pair, "value": value, "threshold": threshold, "status": status, "note": note}


def return_dates(joined, df, ds, k=K):
    """Per split, the return dates any sample touches: X starts k-1 returns before its first row (rho), y ends at row i+T-1+h."""
    p0 = joined.get_loc(df.index[0])
    out = {}
    for name, s in ds["splits"].items():
        lo, hi = p0 + s.start - k + 1, p0 + s.stop - 1 + T - 1 + ds["h"]
        assert lo >= 0, "rho warm-up missing"
        out[name] = joined[lo: hi + 1]
    return out


def ranh_gioi(joined, df, ds, check="ranh_gioi"):
    d = return_dates(joined, df, ds)
    overlap = sum(len(d[a].intersection(d[b])) for a, b in combinations(d, 2))
    note = " | ".join(f"{n} {v[0].date()}..{v[-1].date()}" for n, v in d.items())
    return row(check, None, overlap, "0 shared return dates", overlap == 0, note)


def thu_tu_thoi_gian(ds):
    frac = float(np.mean(ds["x_end"] < ds["y_date"]))
    return row("thu_tu_thoi_gian", None, frac, "1.0", frac == 1.0, f"n={len(ds['x_end'])}")


def scaler(df, ds):
    """Scaler range must equal the train rows' (up to the last train label date) and not the train+val rows'."""
    def span(split):
        rows = df.loc[:ds["y_date"][ds["splits"][split]][-1], FEATURES]
        return rows.min().to_numpy(), rows.max().to_numpy()
    (lo, hi), (lo_v, hi_v) = span("train"), span("val")
    s = ds["scaler"]
    err = max(np.abs(s.data_min_ - lo).max(), np.abs(s.data_max_ - hi).max())
    exceeds = bool((lo_v != lo).any() or (hi_v != hi).any())  # else the extra check cannot tell train from train+val
    saw_val = bool(np.allclose(s.data_min_, lo_v, rtol=0, atol=1e-12) and np.allclose(s.data_max_, hi_v, rtol=0, atol=1e-12))
    extra = "n/a (val inside train range)" if not exceeds else ("FAIL: equals train+val range" if saw_val else "ok")
    return row("scaler", None, float(err), "<=1e-12", err <= 1e-12 and not (exceeds and saw_val), f"val-range check: {extra}")


def grep_fill():
    files = [p for d in ("src", "scripts") for p in (ROOT / d).rglob("*")
             if p.suffix in (".py", ".sh") and p != Path(__file__).resolve()]
    return [f"{p.relative_to(ROOT)}:{i}" for p in files
            for i, line in enumerate(p.read_text().splitlines(), 1) if FILL.search(line)]


def dien_thieu(df, a, b, hits):
    both = traded(load_raw(a)).index.intersection(traded(load_raw(b)).index)
    outside = len(df.index.difference(both))
    note = f"grep hits: {', '.join(hits) or 'none'}"
    return row("dien_thieu", None, outside + len(hits), "0 dates outside A∩B, 0 grep hits",
               outside == 0 and not hits, note)


def chon_sieu_tham_so(tune_path=ROOT / "src" / "tune.py"):
    """Gate: tune.py never names the test split. preds_test.csv files are listed only (stage 7 creates them)."""
    preds_test = sorted(str(p.relative_to(RESULTS)) for p in RUNS.rglob("preds_test.csv"))
    tune = tune_path.read_text() if tune_path.exists() else ""
    bad = TUNE_TEST.findall(tune)
    note = (f"preds_test (info only): {preds_test or 'none'}; tune.py "
            + ("empty (rerun at stage 6)" if not tune.strip() else f"test refs: {bad or 'none'}"))
    return row("chon_sieu_tham_so", "all", len(bad), "0", not bad, note)


def chong_lan_cua_so(ds):
    va = ds["splits"]["val"]
    y, prev = ds["y_raw"][va], ds["rho_prev"][va]
    corr = float(np.corrcoef(y, prev)[0, 1])
    return row("chong_lan_cua_so", None, r2(y, prev), "-", None, f"corr(rho(t+1),rho(t))={corr:.4f}")


def label_overlap(df, ds):
    """Samples whose label returns (rows i+T-1+h-k+1..i+T-1+h) start on or before the window end x_end."""
    n = len(ds["x_end"])
    first_label_return = df.index[T - 1 + ds["h"] - K + 1:][:n]
    return int(np.sum(first_label_return <= ds["x_end"]))


def nhan_khong_chong_lan(df, ds):
    va = ds["splits"]["val"]
    overlap = label_overlap(df, ds)
    note = f"value = persistence R2 on val; samples with label∩X returns = {overlap}; n={len(ds['x_end'])}"
    return row("nhan_khong_chong_lan", None, r2(ds["y_raw"][va], ds["rho_prev"][va]), "-",
               None if overlap == 0 else False, note)


def static_checks(returns, a, b, hits):
    """Checks 1-4, 8, 9 for one pair."""
    pair = f"{a}_{b}"
    joined = returns[[a, b]].dropna().index
    df = pair_frame(returns, a, b)
    ds, fw = make_dataset(df), make_dataset(df, target="future_window")
    rows = [ranh_gioi(joined, df, ds), thu_tu_thoi_gian(ds), scaler(df, ds), dien_thieu(df, a, b, hits),
            chong_lan_cua_so(ds), nhan_khong_chong_lan(df, fw),
            ranh_gioi(joined, df, fw, "nhan_khong_chong_lan_ranh_gioi")]
    return [{**r, "pair": pair} for r in rows]


def train_clatt_val(ds, y, y_raw, seed=0, with_delta=False):
    """CLATT trained on scaled targets `y` with the normal fit(); returns (R2, r2_delta or None) on val, original scale."""
    cfg = load_config("CLATT")
    set_seed(seed)
    torch.set_num_threads(cfg["num_threads"])
    X, s = ds["X"], ds["splits"]
    model = build("CLATT", X.shape[2], T, **{k: cfg[k] for k in ("hidden", "filters", "dropout")})
    fit(model, X[s["train"]], y[s["train"]], X[s["val"]], y[s["val"]], cfg, seed)
    lo, span = ds["scaler"].data_min_[0], ds["scaler"].data_range_[0]
    va = s["val"]
    y_pred = predict(model, X[va]).astype(np.float64) * span + lo
    return r2(y_raw[va], y_pred), r2_delta(y_raw[va], y_pred, ds["rho_prev"][va]) if with_delta else None


def trained_checks(returns, pair):
    a, b = pair.split("_")
    ds = make_dataset(pair_frame(returns, a, b))
    tr = ds["splits"]["train"]
    y_perm = ds["y"].copy()
    y_perm[tr] = np.random.default_rng(0).permutation(y_perm[tr])  # val keeps the true labels
    r2_val, r2d = train_clatt_val(ds, y_perm, ds["y_raw"], with_delta=True)
    shuffled = row("xao_tron_nhan", pair, r2_val, "R2<=0.1 and r2_delta<=0.05", r2_val <= 0.1 and r2d <= 0.05,
                   f"r2_delta={r2d:.4f}")
    y_now = np.ascontiguousarray(ds["X"][:, -1, 0])  # scaled rho(t), already the last step of X
    X, va = ds["X"], ds["splits"]["val"]
    lo, span = ds["scaler"].data_min_[0], ds["scaler"].data_range_[0]
    ridge = RidgeWindow().fit(X[tr], y_now[tr], X[va], y_now[va])  # linear model: the pipeline must let it copy rho(t)
    r2_ridge = r2(ds["rho_prev"][va], ridge.predict(X[va]).astype(np.float64) * span + lo)
    shifted = row("dich_nhan", pair, r2_ridge, "R2>=0.98", r2_ridge >= 0.98, "Ridge, y = rho(t)")
    r2_clatt, _ = train_clatt_val(ds, y_now, ds["rho_prev"])
    clatt = row("dich_nhan_clatt", pair, r2_clatt, "-", None, "capacity measure, not a leak test")
    return [shuffled, shifted, clatt]


def main():
    returns = returns_table(CODES)
    hits = grep_fill()
    rows = [r for _, a, b in pairs() for r in static_checks(returns, a, b, hits)]
    rows.append(chon_sieu_tham_so())
    path = RESULTS / "tables" / "leakage_checks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    for pair in TRAIN_PAIRS:
        rows += trained_checks(returns, pair)
        print(*rows[-3:], sep="\n", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)  # rewritten per pair so a stopped run keeps its evidence
    out = pd.DataFrame(rows)

    print(out.groupby("check", sort=False).status.agg(lambda s: f"{(s == 'PASS').sum()} PASS / {(s == 'FAIL').sum()} FAIL / {len(s)}"))
    group = {f"{a}_{b}": g for g, a, b in pairs()}
    for check in ("chong_lan_cua_so", "nhan_khong_chong_lan"):
        m = out[out.check == check]
        print(check, "persistence R2 on val by group:", m.value.groupby(m.pair.map(group)).mean().round(4).to_dict())
    fails = out[out.status == "FAIL"]
    if len(fails):
        print("FAIL rows:\n", fails.to_string())


if __name__ == "__main__":
    main()
