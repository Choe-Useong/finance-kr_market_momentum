import pandas as pd
import numpy as np
import os
from pathlib import Path

from bt_core import (
    pick_rebal_dates,
    build_weights,
    add_amount_avg,
    add_marcap_avg,
    add_turnover_avg,
    add_amihud_avg,
    daily_rank_to_monthly,
    weekly_rank_to_monthly,
    momentum_12_1_log,
)

os.chdir(Path(__file__).resolve().parent)

UNIVERSE_FILE = "build_universe.parquet"
RAW_CLOSE_FILE = "raw_close.parquet"
OUT_TARGET = "build_target.parquet"

# Rebalance configuration
FREQ = "Q_END"   # "M_END" / "Q_END" / "Y_END"
SCOPES_FOR_ALL = ["KOSPI"]  # e.g. ["KOSPI", "KOSDAQ", "KONEX"]

# Rank momentum configuration
MOMENTUM_MODE = "RANK_DAILY"  # "RANK_DAILY" / "RANK_WEEKLY" / "MOM_12_1_LOG"
WEEKLY_FREQ = "W-FRI"
RANK_WINDOW = 9  # months (for rank momentum)
MOM_12M = 12
MOM_SKIP_1M = 1
PICK_MODE = "N"  # "N" or "PCT"
TOP_N = 10
TOP_PCT = 0.20  # top 10%

# Pre-filter configuration (before rank momentum)
PRE_FILTER_METRIC = "Marcap"  # "Marcap"/"Amount"/"MarcapAvg"/"AmountAvg"/"TurnoverAvg"/"AmihudAvg"
PRE_FILTER_MODE = "N"  # "N" or "PCT"
PRE_TOP_N = 200
PRE_TOP_PCT = 0.20
PRE_TOP_PCT_RANGE = None  # e.g. (0.1, 0.3) when PRE_FILTER_MODE == "PCT"

# Secondary pre-filter (AND/OR)
PRE_FILTER_COMBINE = "SINGLE"  # "SINGLE" / "AND" / "OR"
PRE_FILTER_METRIC_2 = "TurnoverAvg"
PRE_FILTER_MODE_2 = "PCT"  # "N" or "PCT"
PRE_TOP_N_2 = 100
PRE_TOP_PCT_2 = 0.8

# Shared metric window (Amount/Marcap/Turnover/Amihud etc.)
METRIC_AVG_WINDOW = 25 * 3
METRIC_AVG_MIN_PERIODS = 1

# Weighting configuration
WEIGHT_MODE = "EQUAL"  # "EQUAL" / "RANK" / "SCORE" / "RP"
RP_WINDOW = 120  # trading days for covariance
RP_MIN_PERIODS = 60  # minimum non-NaN observations per asset for RP


u = pd.read_parquet(UNIVERSE_FILE)
u["Date"] = pd.to_datetime(u["Date"]).dt.normalize()
u["Code"] = u["Code"].astype(str).str.zfill(6)

# Filter scopes and resolve duplicate Date+Code by larger Marcap
u = u[u["Scope"].isin(SCOPES_FOR_ALL)].copy()
if "Marcap" in u.columns:
    u = u.sort_values(["Date", "Marcap"], ascending=[True, False])
    u = u.drop_duplicates(subset=["Date", "Code"], keep="first")

all_dates = pd.DatetimeIndex(sorted(u["Date"].unique()))
rebal_dates = pick_rebal_dates(all_dates, FREQ)

# Load close data
close = pd.read_parquet(RAW_CLOSE_FILE)
close.index = pd.to_datetime(close.index).normalize()
close.columns = [str(c).zfill(6) for c in close.columns]

if "Amount" in u.columns:
    u = add_amount_avg(u, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)
if "Marcap" in u.columns:
    u = add_marcap_avg(u, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)
if "Amount" in u.columns and "Marcap" in u.columns:
    u = add_turnover_avg(u, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)
if "Amount" in u.columns:
    u = add_amihud_avg(u, close, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)

if PRE_FILTER_METRIC not in u.columns:
    raise ValueError(f"PRE_FILTER_METRIC not in universe: {PRE_FILTER_METRIC}")

if MOMENTUM_MODE == "RANK_DAILY":
    signal_m = daily_rank_to_monthly(close)
    rm = signal_m.rolling(RANK_WINDOW, min_periods=RANK_WINDOW).sum()
elif MOMENTUM_MODE == "RANK_WEEKLY":
    signal_m = weekly_rank_to_monthly(close, week_freq=WEEKLY_FREQ)
    rm = signal_m.rolling(RANK_WINDOW, min_periods=RANK_WINDOW).sum()
elif MOMENTUM_MODE == "MOM_12_1_LOG":
    rm = momentum_12_1_log(close, m=MOM_12M, skip=MOM_SKIP_1M)
else:
    raise ValueError(f"Unknown MOMENTUM_MODE: {MOMENTUM_MODE}")

if PRE_TOP_PCT_RANGE is not None:
    PRE_TOP_PCT = PRE_TOP_PCT_RANGE

w = build_weights(
    u, rm, rebal_dates,
    rank_window=RANK_WINDOW,
    pick_mode=PICK_MODE, top_n=TOP_N, top_pct=TOP_PCT,
    pre_mode=PRE_FILTER_MODE, pre_top_n=PRE_TOP_N, pre_top_pct=PRE_TOP_PCT,
    weight_mode=WEIGHT_MODE,
    pre_combine=PRE_FILTER_COMBINE,
    pre_metric_2=PRE_FILTER_METRIC_2,
    pre_mode_2=PRE_FILTER_MODE_2,
    pre_top_n_2=PRE_TOP_N_2,
    pre_top_pct_2=PRE_TOP_PCT_2,
    close=close,
    pre_filter_metric=PRE_FILTER_METRIC,
    rp_window=RP_WINDOW,
    rp_min_periods=RP_MIN_PERIODS,
)

w.to_parquet(OUT_TARGET)
print("saved:", OUT_TARGET)
if not w.empty:
    print("mode:", PICK_MODE, "top_n:", TOP_N, "top_pct:", TOP_PCT, "rank_window:", RANK_WINDOW)
    print("range:", w.index.min().date(), "->", w.index.max().date(), "cols:", w.shape[1])
    print(w.tail())
    print(len(w))
