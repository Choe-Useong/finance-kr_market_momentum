import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marcap import marcap_data

os.chdir(ROOT)

START = "2014-01-01"
END = None  # None -> latest Date in marcap/data parquet files
MARCAP_DATA_DIR = ROOT / "marcap" / "data"

# Screening configuration
METRIC = "Marcap"  # "Marcap" or "Amount"
AMOUNT_WINDOW = 25 * 3  # rolling window (rows) for Amount
AMOUNT_MIN_PERIODS = 1
PICK_MODE = "N"  # "N" or "PCT"
TOP_N = 600
TOP_PCT_RANGE = (0.0, 0.8)  # keep ranks in (low, high] percent; e.g. top 10% = (0.0, 0.10)

# Scope configuration
SCOPES = ["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"] # SCOPES = ["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"]
SCOPE_MODE = "ALL"  # "BY_SCOPE" or "ALL"

OUT_UNIVERSE = str(BASE / "build_universe.parquet")


def latest_marcap_date(data_dir: Path) -> str:
    files = sorted(data_dir.glob("marcap-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No marcap parquet files in {data_dir}")

    latest = None
    for path in files:
        try:
            d = pd.read_parquet(path, columns=["Date"])
        except Exception:
            continue
        if d.empty:
            continue
        mx = pd.to_datetime(d["Date"]).max()
        if latest is None or mx > latest:
            latest = mx

    if latest is None:
        raise ValueError(f"No Date values found in {data_dir}")
    return latest.strftime("%Y-%m-%d")


def add_amount_avg(df: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    df = df.sort_values(["Code", "Date"]).copy()
    df["AmountAvg"] = (
        df.groupby("Code", sort=False)["Amount"]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def pick_top(df: pd.DataFrame, metric: str, pick_mode: str, top_n: int, pct_range: tuple[float, float]) -> pd.DataFrame:
    df = df.sort_values(metric, ascending=False).copy()
    # rank is 1..N
    df["Rank"] = df[metric].rank(ascending=False, method="min").astype(int)
    n = len(df)
    if n == 0:
        return df
    if pick_mode == "N":
        return df[df["Rank"] <= top_n]
    if pick_mode == "PCT":
        low, high = pct_range
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"Invalid TOP_PCT_RANGE: {pct_range}")
        # Convert percent range to rank bounds
        lo_rank = int(np.floor(low * n)) + 1
        hi_rank = int(np.ceil(high * n))
        lo_rank = max(1, lo_rank)
        hi_rank = max(lo_rank, hi_rank)
        return df[(df["Rank"] >= lo_rank) & (df["Rank"] <= hi_rank)]
    raise ValueError(f"Unknown PICK_MODE: {pick_mode}")


end_date = END if END is not None else latest_marcap_date(MARCAP_DATA_DIR)
print("marcap range:", START, "->", end_date)
df = marcap_data(START, end_date)
df["Code"] = df["Code"].astype(str).str.zfill(6)
df = df[df["Marcap"].notna()].copy()

df = df.copy()
df["Date"] = pd.to_datetime(df.index).normalize()
df = df.reset_index(drop=True)
df["Scope"] = df["Market"].replace({"KOSDAQ GLOBAL": "KOSDAQ"})

df = df[df["Market"].isin(SCOPES)].copy()

if METRIC == "Amount":
    df = add_amount_avg(df, AMOUNT_WINDOW, AMOUNT_MIN_PERIODS)
    metric_col = "AmountAvg"
elif METRIC == "Marcap":
    metric_col = "Marcap"
else:
    raise ValueError(f"Unknown METRIC: {METRIC}")

out_list = []

if SCOPE_MODE == "BY_SCOPE":
    group_cols = ["Date", "Scope"]
elif SCOPE_MODE == "ALL":
    group_cols = ["Date"]
else:
    raise ValueError(f"Unknown SCOPE_MODE: {SCOPE_MODE}")

for _, g in df.groupby(group_cols):
    picked = pick_top(g, metric_col, PICK_MODE, TOP_N, TOP_PCT_RANGE)
    out_list.append(picked)

out = pd.concat(out_list, ignore_index=True) if out_list else pd.DataFrame()
if not out.empty:
    out = out[["Date", "Scope", "Code", "Marcap", "Amount"]].copy()

out.to_parquet(OUT_UNIVERSE, index=False)
print("saved:", OUT_UNIVERSE, "rows:", len(out))
