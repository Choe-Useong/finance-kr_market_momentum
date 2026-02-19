import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
import os
from pathlib import Path
import yfinance as yf

os.chdir(Path(__file__).resolve().parent)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ===== CONFIG =====
UNIVERSE_FILE = "build_universe.parquet"
RAW_CLOSE_FILE = "raw_close.parquet"

FREQ = "Q_END"   # "M_END" / "Q_END" / "Y_END"
SCOPES_FOR_ALL = ["KOSPI"]

RANK_WINDOW_LIST = [3, 6, 9, 12]
PICK_MODE = "N"  # "N" or "PCT"
TOP_N_LIST = [10, 20, 30, 40, 50]
TOP_PCT_LIST = []

# Pre-filter (before rank momentum)
PRE_FILTER_METRIC = "Marcap"  # "Marcap" or "Amount"
PRE_FILTER_MODE = "N"  # "N" or "PCT"
PRE_TOP_N_LIST = [50, 100, 200, 400]
PRE_TOP_PCT_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Optional: define explicit ranges as pairs and expand to list
PRE_TOP_N_PAIRS = []  # e.g. [(50, 150), (200, 400)]
PRE_TOP_PCT_PAIRS = []  # e.g. [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]

FEES = 0.00025
INIT_CASH = 1.0

# Weighting
WEIGHT_MODE_LIST = ["EQUAL"]  # "EQUAL" or "RP"
RP_WINDOW = 120  # trading days for covariance
RP_MIN_PERIODS = 60

# Market timing
TIMING_ENABLED = True
TIMING_TICKER = "069500.KS"
TIMING_SHORT_MA = 25*2
TIMING_LONG_MA = 25*10
TIMING_Z_WINDOW = TIMING_LONG_MA
TIMING_Z_MODE = "STD"  # "STD" or "MAD"
TIMING_K = 1.0
TIMING_BUFFER_DAYS = 60

# ===== FUNCTIONS =====

def pick_rebal_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    s = pd.Series(dates, index=dates)
    if freq == "M_END":
        return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().values)
    if freq == "Q_END":
        return pd.DatetimeIndex(s.groupby(s.index.to_period("Q")).max().values)
    if freq == "Y_END":
        return pd.DatetimeIndex(s.groupby(s.index.to_period("Y")).max().values)
    raise ValueError(freq)


def pick_top_by_mode(s: pd.Series, mode: str, top_n: int, top_pct: float) -> pd.Index:
    s = s.dropna().sort_values(ascending=False)
    if s.empty:
        return pd.Index([])
    if mode == "N":
        return s.head(top_n).index
    if mode == "PCT":
        n = len(s)
        k = int(np.ceil(n * top_pct))
        k = max(1, k)
        return s.head(k).index
    raise ValueError(mode)

def pick_top_by_pct_range(s: pd.Series, pct_range: tuple[float, float]) -> pd.Index:
    low, high = pct_range
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"Invalid pct range: {pct_range}")
    s = s.dropna().sort_values(ascending=False)
    if s.empty:
        return pd.Index([])
    n = len(s)
    lo_rank = int(np.floor(low * n)) + 1
    hi_rank = int(np.ceil(high * n))
    lo_rank = max(1, lo_rank)
    hi_rank = max(lo_rank, hi_rank)
    ranks = s.rank(ascending=False, method="min").astype(int)
    return ranks[(ranks >= lo_rank) & (ranks <= hi_rank)].index


def risk_parity_weights(ret: pd.DataFrame, max_iter: int = 200, tol: float = 1e-8) -> pd.Series:
    n = ret.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    cov = ret.cov(min_periods=RP_MIN_PERIODS)
    w = np.ones(n) / n
    for _ in range(max_iter):
        mrc = cov.values @ w
        rc = w * mrc
        avg_rc = rc.mean()
        if np.all(np.abs(rc - avg_rc) < tol):
            break
        grad = (rc - avg_rc)
        w = w * np.exp(-grad)
        w = np.clip(w, 1e-8, None)
        w = w / w.sum()
    return pd.Series(w, index=ret.columns)

def download_timing_price(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    df = yf.download(
        tickers=TIMING_TICKER,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"Timing price download failed: {TIMING_TICKER}")
    px = df["Close"].copy()
    if isinstance(px, pd.DataFrame):
        if px.shape[1] == 1:
            px = px.iloc[:, 0]
        else:
            px = px.iloc[:, 0]
    px.index = pd.to_datetime(px.index).normalize()
    px = px.sort_index()
    return px


def build_timing_series(min_rebal: pd.Timestamp, max_rebal: pd.Timestamp) -> pd.Series:
    lookback = TIMING_LONG_MA + TIMING_Z_WINDOW + TIMING_BUFFER_DAYS
    start = pd.Timestamp(min_rebal) - pd.tseries.offsets.BDay(lookback)
    end = pd.Timestamp(max_rebal)

    px = download_timing_price(start, end)
    ma_s = px.rolling(TIMING_SHORT_MA, min_periods=TIMING_SHORT_MA).mean()
    ma_l = px.rolling(TIMING_LONG_MA, min_periods=TIMING_LONG_MA).mean()
    ratio = (ma_s - ma_l) / ma_l

    if TIMING_Z_MODE == "STD":
        mu = ratio.rolling(TIMING_Z_WINDOW, min_periods=TIMING_Z_WINDOW).mean()
        sigma = ratio.rolling(TIMING_Z_WINDOW, min_periods=TIMING_Z_WINDOW).std(ddof=0)
        z = (ratio - mu) / sigma.replace(0, np.nan)
    elif TIMING_Z_MODE == "MAD":
        med = ratio.rolling(TIMING_Z_WINDOW, min_periods=TIMING_Z_WINDOW).median()
        mad = (ratio - med).abs().rolling(TIMING_Z_WINDOW, min_periods=TIMING_Z_WINDOW).median()
        z = (ratio - med) / (1.4826 * mad.replace(0, np.nan))
    else:
        raise ValueError(f"Unknown TIMING_Z_MODE: {TIMING_Z_MODE}")

    z = z.replace([np.inf, -np.inf], np.nan)
    timing = 1.0 / (1.0 + np.exp(-TIMING_K * z))
    return timing


def build_weights(u: pd.DataFrame, rm: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
                  rank_window: int, pick_mode: str, top_n: int, top_pct: float,
                  pre_mode: str, pre_top_n: int, pre_top_pct: float,
                  weight_mode: str) -> pd.DataFrame:
    rm_month = rm.copy()
    rm_month["Month"] = rm_month.index.to_period("M")

    picked_rows = []
    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        month = dt.to_period("M")
        month_rows = rm_month[rm_month["Month"] == month]
        if month_rows.empty:
            continue
        month_end = month_rows.index.max()

        g = u[u["Date"] == dt]
        if g.empty:
            continue

        # Pre-filter by metric per Date
        s_pre = g.set_index("Code")[PRE_FILTER_METRIC]
        if pre_mode == "PCT" and isinstance(pre_top_pct, tuple):
            pre_codes = pick_top_by_pct_range(s_pre, pre_top_pct)
        else:
            pre_codes = pick_top_by_mode(s_pre, pre_mode, pre_top_n, pre_top_pct)
        if len(pre_codes) == 0:
            continue

        codes = g[g["Code"].isin(pre_codes)]["Code"].tolist()
        if not codes:
            continue

        scores = rm.loc[month_end, codes]
        pick_codes = pick_top_by_mode(scores, pick_mode, top_n, top_pct)
        if len(pick_codes) == 0:
            continue

        if weight_mode == "EQUAL":
            w = 1.0 / len(pick_codes)
            for code in pick_codes:
                picked_rows.append((dt, code, w))
        elif weight_mode == "RP":
            end = pd.Timestamp(dt)
            start = end - pd.tseries.offsets.BDay(RP_WINDOW * 2)
            ret = np.log(close.loc[(close.index > start) & (close.index <= end), pick_codes]).diff()
            ret = ret.tail(RP_WINDOW)
            ret = ret.dropna(how="all")
            enough = ret.count() >= RP_MIN_PERIODS
            ret = ret.loc[:, enough]
            if ret.empty or ret.shape[0] < 2 or ret.shape[1] < 2:
                w = 1.0 / len(pick_codes)
                for code in pick_codes:
                    picked_rows.append((dt, code, w))
            else:
                w_rp = risk_parity_weights(ret)
                for code, w in w_rp.items():
                    picked_rows.append((dt, code, float(w)))
        else:
            raise ValueError(f"Unknown WEIGHT_MODE: {weight_mode}")

    if picked_rows:
        out = pd.DataFrame(picked_rows, columns=["Date", "Code", "Weight"])
        return out.pivot_table(index="Date", columns="Code", values="Weight", fill_value=0.0)
    return pd.DataFrame()


def run_backtest(px: pd.DataFrame, w_m: pd.DataFrame) -> vbt.Portfolio:
    px = px.sort_index()
    w_m = w_m.sort_index()
    cols = px.columns.intersection(w_m.columns)
    px = px.loc[:, cols]
    w_m = w_m.loc[:, cols]
    if not w_m.empty:
        px = px.loc[px.index >= w_m.index.min()]
    px = px.dropna(how="all")
    w = w_m.reindex(px.index)
    w_on_px = w.reindex(px.index)
    w_exec = w_on_px.shift(1)
    pf = vbt.Portfolio.from_orders(
        close=px,
        size=w_exec,
        size_type="TargetPercent",
        init_cash=INIT_CASH,
        cash_sharing=True,
        fees=FEES,
        freq="1D",
        call_seq="auto",
    )
    return pf


def metrics_from_equity(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"total_return": np.nan, "sharpe": np.nan, "calmar": np.nan, "max_drawdown": np.nan}
    ret = np.log(eq).diff().dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann_factor = 252
    sharpe = np.nan
    if ret.std() != 0:
        sharpe = float(ret.mean() / ret.std() * np.sqrt(ann_factor))
    # max drawdown
    cummax = eq.cummax()
    dd = (eq / cummax - 1.0)
    max_dd = float(dd.min())
    years = len(ret) / ann_factor
    cagr = np.nan
    if years > 0:
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)
    calmar = np.nan
    if max_dd != 0 and not np.isnan(cagr):
        calmar = cagr / abs(max_dd)
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_dd,
    }


# ===== LOAD DATA =====

u = pd.read_parquet(UNIVERSE_FILE)
u["Date"] = pd.to_datetime(u["Date"]).dt.normalize()
u["Code"] = u["Code"].astype(str).str.zfill(6)

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

timing_series = None
if TIMING_ENABLED:
    timing_series = build_timing_series(rebal_dates.min(), rebal_dates.max())

close_m = close.resample("M").last()
ret_m = np.log(close_m).diff()

rank_m = ret_m.rank(axis=1, ascending=True, method="min")
count_m = ret_m.notna().sum(axis=1)
rank_norm = (rank_m.sub(1, axis=0)).div((count_m - 1).replace(0, np.nan), axis=0)

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
    if PRE_TOP_N_PAIRS:
        pre_top_n_list = sorted({x for a, b in PRE_TOP_N_PAIRS for x in range(a, b + 1)})
    else:
        pre_top_n_list = PRE_TOP_N_LIST
    pre_top_pct_list = [None]
elif PRE_FILTER_MODE == "PCT":
    pre_top_n_list = [None]
    if PRE_TOP_PCT_PAIRS:
        pre_top_pct_list = PRE_TOP_PCT_PAIRS
    else:
        pre_top_pct_list = PRE_TOP_PCT_LIST
else:
    raise ValueError(f"Unknown PRE_FILTER_MODE: {PRE_FILTER_MODE}")

for weight_mode in WEIGHT_MODE_LIST:
    for rank_window in RANK_WINDOW_LIST:
        rm = rank_norm.rolling(rank_window, min_periods=rank_window).mean()
        for top_n in pick_top_n_list:
            for top_pct in pick_top_pct_list:
                for pre_top_n in pre_top_n_list:
                    for pre_top_pct in pre_top_pct_list:
                        w_m = build_weights(
                            u, rm, rebal_dates,
                            rank_window, PICK_MODE, top_n or 0, top_pct or 0.0,
                            PRE_FILTER_MODE, pre_top_n or 0, pre_top_pct or 0.0,
                            weight_mode,
                        )
                        if w_m.empty:
                            continue
                        if TIMING_ENABLED:
                            timing_rebal = timing_series.reindex(w_m.index)
                            valid = timing_rebal.notna()
                            w_m = w_m.loc[valid]
                            timing_rebal = timing_rebal.loc[valid]
                            w_m = w_m.mul(timing_rebal, axis=0)
                            if w_m.empty:
                                continue
                        pf = run_backtest(close, w_m)

                        key = (weight_mode, rank_window, PICK_MODE, top_n, top_pct, PRE_FILTER_MODE, pre_top_n, pre_top_pct)
                        portfolios[key] = pf
                        results.append({
                            "weight_mode": weight_mode,
                            "rank_window": rank_window,
                            "pick_mode": PICK_MODE,
                            "top_n": top_n,
                            "top_pct": top_pct,
                            "pre_mode": PRE_FILTER_MODE,
                            "pre_top_n": pre_top_n,
                            "pre_top_pct": pre_top_pct,
                            "total_return": np.nan,
                            "sharpe": np.nan,
                            "calmar": np.nan,
                            "max_drawdown": np.nan,
                        })

res = pd.DataFrame(
    results,
    columns=[
        "weight_mode", "rank_window", "pick_mode", "top_n", "top_pct",
        "pre_mode", "pre_top_n", "pre_top_pct",
        "total_return", "sharpe", "calmar", "max_drawdown",
    ],
)
# Align to common start date for fair comparison
if portfolios:
    common_start = max(pf.value().index.min() for pf in portfolios.values())
else:
    common_start = None

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
        res.loc[i, "total_return"] = m["total_return"]
        res.loc[i, "sharpe"] = m["sharpe"]
        res.loc[i, "calmar"] = m["calmar"]
        res.loc[i, "max_drawdown"] = m["max_drawdown"]

res = res.sort_values("total_return", ascending=False)
if common_start is not None:
    print("common_start:", common_start.date())
print(res.head(20))

# ===== TOP 5 PLOTS =====

top5 = res.head(5)

BM_TICKERS = ["069500.KS", "229200.KS"]

def download_benchmark(start, end):
    df = yf.download(
        tickers=BM_TICKERS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].copy()
    px.index = pd.to_datetime(px.index).normalize()
    px = px.sort_index()
    px = px.rename(columns={"069500.KS": "KOSPI", "229200.KS": "KOSDAQ"})
    return px

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
    label = f"rw{row['rank_window']}_{row['pick_mode']}"
    eq.plot(label=label)
    all_eq.append(eq)

if all_eq:
    eq_ref = all_eq[0]
    start = eq_ref.index.min().strftime("%Y-%m-%d")
    end = (eq_ref.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    bm_px = download_benchmark(start, end)
    bm_px = bm_px.reindex(eq_ref.index).ffill()

    def equity_from_price(px: pd.Series, init_cash=1.0):
        r = np.log(px).diff().fillna(0.0)
        return init_cash * (1.0 + r).cumprod()

    kospi_eq = equity_from_price(bm_px["KOSPI"], init_cash=eq_ref.iloc[0])
    kosdaq_eq = equity_from_price(bm_px["KOSDAQ"], init_cash=eq_ref.iloc[0])

    kospi_eq.plot(label="KOSPI")
    kosdaq_eq.plot(label="KOSDAQ")

plt.grid(True)
plt.legend()
plt.title("Top 5 Strategies vs Benchmarks")
plt.show()
