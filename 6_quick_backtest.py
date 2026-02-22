import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from bt_core import (
    pick_rebal_dates,
    run_backtest,
    metrics_from_equity,
    build_timing_series,
    apply_timing_weights,
    download_benchmark,
    benchmark_equity_for_index,
    add_amount_avg,
    add_marcap_avg,
    add_turnover_avg,
    add_amihud_avg,
    daily_rank_to_monthly,
    weekly_rank_to_monthly,
    build_weights,
    momentum_12_1_log,
    beat_ratio,
)

os.chdir(Path(__file__).resolve().parent)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ===== CONFIG =====
UNIVERSE_FILE = "build_universe.parquet"
RAW_CLOSE_FILE = "raw_close.parquet"

FREQ = "Q_END"   # "M_END" / "Q_END" / "H_END" / "Y_END"
SCOPES_FOR_ALL = ["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"]

RANK_WINDOW_LIST = [3, 6, 9, 12]
PICK_MODE = "PCT"  # "N" or "PCT" (N: top_n, PCT: top_pct; tuple = range)
TOP_N_LIST = [10, 20, 40, 50]
TOP_PCT_LIST = [0.05, 0.1, 0.2, (0.05, 0.15), (0.05, 0.1), (0.1, 0.2)]

# Momentum mode
MOMENTUM_MODE = "RANK_DAILY"  # "RANK_DAILY" / "RANK_WEEKLY" / "MOM_12_1_LOG"
WEEKLY_FREQ = "W-FRI"
MOM_12M = 12
MOM_SKIP_1M = 1

# Pre-filter (before rank momentum)
PRE_FILTER_METRIC = "MarcapAvg"  # "Marcap" / "Amount" / "MarcapAvg" / "AmountAvg" / "TurnoverAvg" / "AmihudAvg"
PRE_FILTER_MODE = "PCT"  # "N" or "PCT" (N: top_n, PCT: top_pct; tuple = range)
PRE_TOP_N_LIST = [100, 200, 400, 600, (1, 100), (50, 200), (150, 300), (250, 400), (350, 500), (450, 600)]
PRE_TOP_PCT_LIST = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (0, 0.9)]

# Secondary pre-filter (AND/OR)
PRE_FILTER_COMBINE = "AND"  # "AND" / "OR" / "SINGLE"
PRE_FILTER_METRIC_2 = "TurnoverAvg"  # when combine != SINGLE: "Marcap"/"Amount"/"MarcapAvg"/"AmountAvg"/"TurnoverAvg"/"AmihudAvg"
PRE_FILTER_MODE_2 = "PCT"  # "N" or "PCT" (N: top_n, PCT: top_pct; tuple = range)
PRE_TOP_N_2 = 100
PRE_TOP_PCT_2 = 0.8
# Shared metric window (Amount/Marcap/Turnover/Amihud etc.)
METRIC_AVG_WINDOW = 25 * 3
METRIC_AVG_MIN_PERIODS = 1

FEES = 0.00215
INIT_CASH = 1.0

# Weighting
WEIGHT_MODE_LIST = ["RANK"]  # "EQUAL" / "RANK" / "SCORE" / "RP"
RP_WINDOW = 120  # trading days for covariance
RP_MIN_PERIODS = 60

# Market timing
TIMING_ENABLED = False
TIMING_TICKER = "069500.KS"
TIMING_SHORT_MA = 25 * 6
TIMING_LONG_MA = 25 * 12
TIMING_Z_WINDOW = TIMING_LONG_MA
TIMING_Z_MODE = "MAD"  # "STD" / "MAD"
TIMING_K = 1.0
TIMING_BUFFER_DAYS = 60
TIMING_MODE = "ONOFF"  # "SCALE" / "ONOFF"
TIMING_ONOFF_Z = 0.0
TIMING_MIN_EXPOSURE = 0.6

# Alternative assets (used when TIMING_ENABLED is True)
ALT_TICKERS = []  # e.g. ["IEF", "GLD", 138230.KS]




# ===== LOAD DATA =====

u = pd.read_parquet(UNIVERSE_FILE)
u["Date"] = pd.to_datetime(u["Date"]).dt.normalize()
u["Code"] = u["Code"].astype(str).str.zfill(6)
if "Amount" in u.columns:
    u = add_amount_avg(u, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)
if "Marcap" in u.columns:
    u = add_marcap_avg(u, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)
if "Amount" in u.columns and "Marcap" in u.columns:
    u = add_turnover_avg(u, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)

u = u[u["Scope"].isin(SCOPES_FOR_ALL)].copy()
if "Marcap" in u.columns:
    u = u.sort_values(["Date", "Marcap"], ascending=[True, False])
    u = u.drop_duplicates(subset=["Date", "Code"], keep="first")

if PRE_FILTER_METRIC not in u.columns:
    raise ValueError(f"PRE_FILTER_METRIC not in universe: {PRE_FILTER_METRIC}")

all_dates = pd.DatetimeIndex(sorted(u["Date"].unique()))
rebal_dates = pick_rebal_dates(all_dates, FREQ)

close = pd.read_parquet(RAW_CLOSE_FILE)
close.index = pd.to_datetime(close.index).normalize()
close.columns = [str(c).zfill(6) for c in close.columns]

if ALT_TICKERS and not TIMING_ENABLED:
    raise ValueError("ALT_TICKERS requires TIMING_ENABLED=True")

if ALT_TICKERS:
    alt_start = close.index.min().strftime("%Y-%m-%d")
    alt_end = (close.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    alt_px = download_benchmark(alt_start, alt_end, ALT_TICKERS)
    alt_px = alt_px.reindex(close.index).ffill()
    alt_px = alt_px.dropna(how="all", axis=1)
    missing = set(ALT_TICKERS) - set(alt_px.columns)
    if missing:
        raise ValueError(f"Missing alt tickers in download: {missing}")
    overlap = set(close.columns) & set(alt_px.columns)
    if overlap:
        raise ValueError(f"ALT_TICKERS overlap with close columns: {overlap}")
    close = pd.concat([close, alt_px], axis=1)

if "Amount" in u.columns:
    u = add_amihud_avg(u, close, METRIC_AVG_WINDOW, METRIC_AVG_MIN_PERIODS)

timing_series = None
if TIMING_ENABLED:
    timing_series = build_timing_series(
        TIMING_TICKER,
        rebal_dates.min(),
        rebal_dates.max(),
        TIMING_SHORT_MA,
        TIMING_LONG_MA,
        TIMING_Z_WINDOW,
        z_mode=TIMING_Z_MODE,
        k=TIMING_K,
        buffer_days=TIMING_BUFFER_DAYS,
    )

if MOMENTUM_MODE == "RANK_DAILY":
    signal_m = daily_rank_to_monthly(close)
elif MOMENTUM_MODE == "RANK_WEEKLY":
    signal_m = weekly_rank_to_monthly(close, week_freq=WEEKLY_FREQ)
elif MOMENTUM_MODE == "MOM_12_1_LOG":
    signal_m = momentum_12_1_log(close, m=MOM_12M, skip=MOM_SKIP_1M)
else:
    raise ValueError(f"Unknown MOMENTUM_MODE: {MOMENTUM_MODE}")

# ===== GRID SEARCH =====

results = []
portfolios = {}

if PICK_MODE == "N":
    pick_top_n_list = TOP_N_LIST
    pick_top_pct_list = [None]
elif PICK_MODE == "PCT":
    pick_top_n_list = [None]
    pick_top_pct_list = TOP_PCT_LIST
else:
    raise ValueError(f"Unknown PICK_MODE: {PICK_MODE}")

if PRE_FILTER_MODE == "N":
    pre_top_n_list = PRE_TOP_N_LIST
    pre_top_pct_list = [None]
elif PRE_FILTER_MODE == "PCT":
    pre_top_n_list = [None]
    pre_top_pct_list = PRE_TOP_PCT_LIST
else:
    raise ValueError(f"Unknown PRE_FILTER_MODE: {PRE_FILTER_MODE}")

rank_window_list = RANK_WINDOW_LIST if MOMENTUM_MODE == "RANK_DAILY" else [MOM_12M]

for weight_mode in WEIGHT_MODE_LIST:
    for rank_window in rank_window_list:
        if MOMENTUM_MODE == "RANK_DAILY":
            rm = signal_m.rolling(rank_window, min_periods=rank_window).sum()
        else:
            rm = signal_m
        for top_n in pick_top_n_list:
            for top_pct in pick_top_pct_list:
                for pre_top_n in pre_top_n_list:
                    for pre_top_pct in pre_top_pct_list:
                        w_m = build_weights(
                            u, rm, rebal_dates,
                            rank_window, PICK_MODE, top_n, top_pct,
                            PRE_FILTER_MODE, pre_top_n or 0, pre_top_pct or 0.0,
                            weight_mode,
                            pre_combine=PRE_FILTER_COMBINE,
                            pre_metric_2=PRE_FILTER_METRIC_2,
                            pre_mode_2=PRE_FILTER_MODE_2,
                            pre_top_n_2=PRE_TOP_N_2,
                            pre_top_pct_2=PRE_TOP_PCT_2,
                        )
                        if w_m.empty:
                            continue
                        if TIMING_ENABLED:
                            w_m, timing_rebal = apply_timing_weights(
                                w_m,
                                timing_series,
                                mode=TIMING_MODE,
                                k=TIMING_K,
                                onoff_z=TIMING_ONOFF_Z,
                                min_exposure=TIMING_MIN_EXPOSURE,
                                return_timing=True,
                            )
                            if w_m.empty:
                                continue
                            if ALT_TICKERS:
                                w_alt = 1.0 - timing_rebal
                                alt_w = w_alt / len(ALT_TICKERS)
                                for t in ALT_TICKERS:
                                    w_m[t] = alt_w
                        holdings_cols = [c for c in w_m.columns if c not in ALT_TICKERS]
                        if holdings_cols:
                            counts = (w_m[holdings_cols] > 0).sum(axis=1)
                            hold_avg = float(counts.mean())
                            hold_min = int(counts.min())
                            hold_max = int(counts.max())
                        else:
                            hold_avg = np.nan
                            hold_min = np.nan
                            hold_max = np.nan
                        pf = run_backtest(close, w_m, fees=FEES, init_cash=INIT_CASH)

                        record_pre_top_n = pre_top_n
                        key = (
                            weight_mode,
                            rank_window,
                            PICK_MODE,
                            top_n,
                            top_pct,
                            PRE_FILTER_MODE,
                            record_pre_top_n,
                            pre_top_pct,
                        )
                        portfolios[key] = pf
                        results.append({
                            "weight_mode": weight_mode,
                            "rank_window": rank_window,
                            "pick_mode": PICK_MODE,
                            "top_n": top_n,
                            "top_pct": top_pct,
                            "pre_mode": PRE_FILTER_MODE,
                            "pre_top_n": record_pre_top_n,
                            "pre_top_pct": pre_top_pct,
                            "total_return": np.nan,
                            "sharpe": np.nan,
                            "cagr": np.nan,
                            "calmar": np.nan,
                            "max_drawdown": np.nan,
                            "rolling_12m_return_std": np.nan,
                            "hold_avg": hold_avg,
                            "hold_min": hold_min,
                            "hold_max": hold_max,
                        })

res = pd.DataFrame(
    results,
    columns=[
        "weight_mode", "rank_window", "pick_mode", "top_n", "top_pct",
        "pre_mode", "pre_top_n", "pre_top_pct",
        "total_return", "sharpe", "cagr", "calmar", "max_drawdown", "rolling_12m_return_std",
        "hold_avg", "hold_min", "hold_max",
    ],
)
# Align to common start date for fair comparison
if portfolios:
    common_start = max(pf.value().index.min() for pf in portfolios.values())
else:
    common_start = None

bm_eq_full = None
if common_start is not None and portfolios:
    any_pf = next(iter(portfolios.values()))
    eq_any = any_pf.value().copy()
    eq_any.index = pd.to_datetime(eq_any.index).normalize()
    eq_any = eq_any.sort_index()
    eq_any = eq_any.loc[eq_any.index >= common_start]
    if not eq_any.empty:
        bm_eq_full = benchmark_equity_for_index(
            eq_any.index,
            "069500.KS",
            init_cash=eq_any.iloc[0],
            name="KOSPI",
        )

if common_start is not None:
    for i, row in res.iterrows():
        key = (
            row["weight_mode"], row["rank_window"], row["pick_mode"], row["top_n"], row["top_pct"],
            row["pre_mode"], row["pre_top_n"], row["pre_top_pct"],
        )
        pf = portfolios[key]
        eq = pf.value().copy()
        eq.index = pd.to_datetime(eq.index).normalize()
        eq = eq.sort_index()
        eq = eq.loc[eq.index >= common_start]
        m = metrics_from_equity(eq)
        beat_q = np.nan
        if bm_eq_full is not None:
            beat_q = beat_ratio(eq, bm_eq_full, freq="QE")
        res.loc[i, "total_return"] = m["total_return"]
        res.loc[i, "sharpe"] = m["sharpe"]
        res.loc[i, "cagr"] = m["cagr"]
        res.loc[i, "calmar"] = m["calmar"]
        res.loc[i, "max_drawdown"] = m["max_drawdown"]
        res.loc[i, "rolling_12m_return_std"] = m["rolling_12m_return_std"]
        res.loc[i, "beat_ratio_q"] = beat_q

res = res.sort_values("total_return", ascending=False)
if common_start is not None:
    print("common_start:", common_start.date())

# ===== TOP 5 PLOTS =====

top5 = res.head(5)

plt.figure()

all_eq = []
for _, row in top5.iterrows():
    key = (
        row["weight_mode"], row["rank_window"], row["pick_mode"], row["top_n"], row["top_pct"],
        row["pre_mode"], row["pre_top_n"], row["pre_top_pct"],
    )
    pf = portfolios[key]
    eq = pf.value().copy()
    eq.index = pd.to_datetime(eq.index).normalize()
    eq = eq.sort_index()
    if common_start is not None:
        eq = eq.loc[eq.index >= common_start]
    if not eq.empty:
        eq = eq / eq.iloc[0]
    label = f"rw{row['rank_window']}_{row['pick_mode']}"
    eq.plot(label=label)
    all_eq.append(eq)

if all_eq:
    eq_ref = all_eq[0]
    kospi_eq = benchmark_equity_for_index(
        eq_ref.index,
        "069500.KS",
        init_cash=eq_ref.iloc[0],
        name="KOSPI",
    )
    if not kospi_eq.empty:
        kospi_eq = kospi_eq / kospi_eq.iloc[0]
        kospi_eq.plot(label="KOSPI")

    # Add benchmark metrics on common_start period
    if common_start is not None:
        kospi_eq_m = kospi_eq.loc[kospi_eq.index >= common_start]
    else:
        kospi_eq_m = kospi_eq
    bm_metrics = metrics_from_equity(kospi_eq_m)
    bench_row = {
        "weight_mode": "BENCH",
        "rank_window": "KOSPI",
        "pick_mode": "",
        "top_n": np.nan,
        "top_pct": np.nan,
        "pre_mode": "",
        "pre_top_n": np.nan,
        "pre_top_pct": np.nan,
        "total_return": bm_metrics["total_return"],
        "sharpe": bm_metrics["sharpe"],
        "cagr": bm_metrics["cagr"],
        "calmar": bm_metrics["calmar"],
        "max_drawdown": bm_metrics["max_drawdown"],
        "rolling_12m_return_std": bm_metrics["rolling_12m_return_std"],
        "beat_ratio_q": np.nan,
    }
    res = pd.concat([res, pd.DataFrame([bench_row])], ignore_index=True)
    res = res.sort_values("total_return", ascending=False)

print(res.head(20))

plt.grid(True)
plt.legend()
plt.title("Top 5 Strategies vs Benchmarks")
plt.show()
