import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from bt_core import run_backtest, metrics_from_equity, download_benchmark, equity_from_price, build_timing_series

os.chdir(Path(__file__).resolve().parent)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TARGET_FILE = "build_target.parquet"  # (date x asset) monthly target weights
RAW_FILE = "raw_close.parquet"        # (date x asset) daily close

FEES = 0.00025
INIT_CASH = 1.0

# Target holdings display (target weights by rebalance date)
TARGET_HOLDINGS_ENABLED = True
TARGET_HOLDINGS_DATE = None  # "YYYY-MM-DD" or None for latest rebalance
TARGET_HOLDINGS_MIN_WEIGHT = 0.00001
TARGET_HOLDINGS_TOP_N = None  # int or None

# Holdings display (target weights by rebalance date)
HOLDINGS_ENABLED = True
HOLDINGS_START = '2023-04-01'  # "YYYY-MM-DD" or None
HOLDINGS_END = '2023-05-01'    # "YYYY-MM-DD" or None
HOLDINGS_MIN_WEIGHT = 0.00001
HOLDINGS_TOP_N = None  # int or None
HOLDINGS_ONLY_CHANGES = True  # print only dates where holdings changed

# Market timing
TIMING_ENABLED = False
TIMING_TICKER = "069500.KS"
TIMING_SHORT_MA = 25
TIMING_LONG_MA = 200
TIMING_Z_WINDOW = TIMING_LONG_MA
TIMING_Z_MODE = "STD"  # "STD" or "MAD"
TIMING_K = 1.0
TIMING_BUFFER_DAYS = 60

w_m = pd.read_parquet(TARGET_FILE)
px = pd.read_parquet(RAW_FILE)

px.index = pd.to_datetime(px.index).normalize()
w_m.index = pd.to_datetime(w_m.index).normalize()
px = px.sort_index()
w_m = w_m.sort_index()

# Apply market timing on rebalance dates
if TIMING_ENABLED and not w_m.empty:
    timing = build_timing_series(
        TIMING_TICKER,
        w_m.index.min(),
        w_m.index.max(),
        TIMING_SHORT_MA,
        TIMING_LONG_MA,
        TIMING_Z_WINDOW,
        z_mode=TIMING_Z_MODE,
        k=TIMING_K,
        buffer_days=TIMING_BUFFER_DAYS,
    )
    timing_rebal = timing.reindex(w_m.index)
    valid = timing_rebal.notna()
    w_m = w_m.loc[valid]
    timing_rebal = timing_rebal.loc[valid]
    w_m = w_m.mul(timing_rebal, axis=0)

# align by columns
cols = px.columns.intersection(w_m.columns)
px = px.loc[:, cols]
w_m = w_m.loc[:, cols]

pf = run_backtest(px, w_m, fees=FEES, init_cash=INIT_CASH)

# equity and metrics
eq = pf.value().copy()
eq.index = pd.to_datetime(eq.index).normalize()
eq = eq.sort_index()

m = metrics_from_equity(eq)
print(m)

# Target holdings (rebalance weights)
if TARGET_HOLDINGS_ENABLED and not w_m.empty:
    if TARGET_HOLDINGS_DATE is None:
        dt = w_m.index.max()
    else:
        dt = pd.to_datetime(TARGET_HOLDINGS_DATE)
        if dt not in w_m.index:
            dt = w_m.index[w_m.index <= dt].max()
    s = w_m.loc[dt]
    s = s[s > TARGET_HOLDINGS_MIN_WEIGHT].sort_values(ascending=False)
    if TARGET_HOLDINGS_TOP_N is not None:
        s = s.head(TARGET_HOLDINGS_TOP_N)
    print("target_holdings_date:", dt.date(), "codes:", s.index.tolist(), "weights:", [float(x) for x in s.values])

# realized weight heatmap
av = pf.asset_value(group_by=False)
pv = pf.value()
w_real = av.div(pv, axis=0).fillna(0.0)

if HOLDINGS_ENABLED:
    h = w_real.copy()
    if HOLDINGS_START:
        h = h.loc[h.index >= pd.to_datetime(HOLDINGS_START)]
    if HOLDINGS_END:
        h = h.loc[h.index <= pd.to_datetime(HOLDINGS_END)]
    if HOLDINGS_ONLY_CHANGES and not h.empty:
        diff = h.diff().abs().sum(axis=1)
        diff.iloc[0] = 1.0
        h = h.loc[diff > 0]
    print("holdings_range:", h.index.min().date() if not h.empty else None, "->", h.index.max().date() if not h.empty else None)
    for dt, row in h.iterrows():
        s = row[row > HOLDINGS_MIN_WEIGHT].sort_values(ascending=False)
        if HOLDINGS_TOP_N is not None:
            s = s.head(HOLDINGS_TOP_N)
        codes = s.index.tolist()
        weights = [float(x) for x in s.values]
        print(dt.date(), "codes:", codes, "weights:", weights)

W = w_real.to_numpy().T  # (asset, time)

plt.figure(figsize=(14, 6))
plt.imshow(W, aspect="auto", interpolation="nearest")
plt.yticks(np.arange(len(w_real.columns)), w_real.columns)
if len(w_real.index) > 1:
    tick_count = min(10, len(w_real.index))
    xticks = np.linspace(0, len(w_real.index) - 1, num=tick_count, dtype=int)
    xlabels = w_real.index[xticks].strftime("%Y-%m-%d")
    plt.xticks(xticks, xlabels, rotation=45, ha="right")
plt.title("Asset Weight Heatmap (Realized)")
plt.xlabel("Time")
plt.ylabel("Asset")
plt.tight_layout()
plt.show()

# ===== Benchmark curves (yfinance) =====
BM_TICKERS = ["069500.KS", "229200.KS"]

start = eq.index.min().strftime("%Y-%m-%d")
end = (eq.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

bm_px = download_benchmark(start, end, BM_TICKERS)
bm_px = bm_px.reindex(eq.index).ffill()

# normalize names
bm_px = bm_px.rename(columns={"069500.KS": "KOSPI", "229200.KS": "KOSDAQ"})

kospi_eq = equity_from_price(bm_px["KOSPI"], init_cash=eq.iloc[0])
kosdaq_eq = equity_from_price(bm_px["KOSDAQ"], init_cash=eq.iloc[0])

plt.figure()
eq.plot(label="Strategy")
kospi_eq.plot(label="KOSPI")
kosdaq_eq.plot(label="KOSDAQ")
plt.grid(True)
plt.legend()
plt.title("Equity Curve vs Benchmarks")
plt.show()
