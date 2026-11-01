"""Step 1 of plan section 3: read the Baostock CSVs and keep traded days."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

GROUPS = {
    1: ["sz.300015", "sz.300274", "sz.300347", "sh.603799"],
    2: ["sz.300014", "sz.300122", "sz.300207", "sz.300450"],
}
CODES = GROUPS[1] + GROUPS[2]
START, END = "2017-01-04", "2022-01-28"  # paper's sample period; earlier rows are label warm-up
PAPER_DAYS = 1227


def load_raw(code):
    return pd.read_csv(RAW / f"{code}.csv", parse_dates=["date"]).set_index("date")


def traded(df):
    return df[(df.tradestatus == 1) & (df.volume > 0)]
