"""Download daily bars for the 8 study stocks from baostock into data/raw/<code>.csv.

python -m scripts.download_baostock           the 8 stocks
python -m scripts.download_baostock --index   extension 3: ChiNext and CSI 300 indices into data/raw/index/
"""
import sys
from importlib.metadata import version
from pathlib import Path

import baostock as bs

CODES = ["sz.300015", "sz.300274", "sz.300347", "sh.603799",
         "sz.300014", "sz.300122", "sz.300207", "sz.300450"]
FIELDS = "date,code,open,high,low,close,volume,amount,tradestatus,pctChg"
START, END = "2016-09-01", "2022-01-28"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
INDEX_CODES = ["sz.399006", "sh.000300"]  # ChiNext Index, CSI 300


def main():
    index = "--index" in sys.argv[1:]
    out, codes = (OUT / "index", INDEX_CODES) if index else (OUT, CODES)
    out.mkdir(parents=True, exist_ok=True)
    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"login failed: {lg.error_msg}")
    try:
        for code in codes:
            rs = bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END,
                                              frequency="d", adjustflag="3" if index else "2")
            if rs.error_code != "0":
                raise SystemExit(f"{code}: {rs.error_msg}")
            df = rs.get_data()
            if df.empty:
                raise SystemExit(f"{code}: no rows returned")
            df.to_csv(out / f"{code}.csv", index=False)
            print(f"{code}: {len(df)} rows")
    finally:
        bs.logout()
    (out / "VERSION.txt").write_text(f"baostock {version('baostock')}\n")


if __name__ == "__main__":
    main()
