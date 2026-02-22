import pandas as pd
import numpy as np
import vectorbt as vbt
import yfinance as yf


def pick_rebal_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    s = pd.Series(dates, index=dates)
    if freq == "M_END":
        return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().values)
    if freq == "Q_END":
        return pd.DatetimeIndex(s.groupby(s.index.to_period("Q")).max().values)
    if freq == "H_END":
        return pd.DatetimeIndex(s.groupby(s.index.to_period("2Q")).max().values)
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


def pick_top_by_n_range(s: pd.Series, n_range: tuple[int, int]) -> pd.Index:
    low, high = n_range
    if not (1 <= low <= high):
        raise ValueError(f"Invalid N range: {n_range}")
    s = s.dropna().sort_values(ascending=False)
    if s.empty:
        return pd.Index([])
    n = len(s)
    lo_rank = max(1, low)
    hi_rank = min(n, high)
    ranks = s.rank(ascending=False, method="min").astype(int)
    return ranks[(ranks >= lo_rank) & (ranks <= hi_rank)].index


def add_amount_avg(df: pd.DataFrame, window: int, min_periods: int = 1) -> pd.DataFrame:
    df = df.sort_values(["Code", "Date"]).copy()
    df["AmountAvg"] = (
        df.groupby("Code", sort=False)["Amount"]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_marcap_avg(df: pd.DataFrame, window: int, min_periods: int = 1) -> pd.DataFrame:
    df = df.sort_values(["Code", "Date"]).copy()
    df["MarcapAvg"] = (
        df.groupby("Code", sort=False)["Marcap"]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_turnover_avg(df: pd.DataFrame, window: int, min_periods: int = 1) -> pd.DataFrame:
    df = df.sort_values(["Code", "Date"]).copy()
    ratio = df["Amount"] / df["Marcap"]
    df["TurnoverAvg"] = (
        ratio.groupby(df["Code"], sort=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_amihud_avg(
    df: pd.DataFrame,
    close: pd.DataFrame,
    window: int,
    min_periods: int = 1,
) -> pd.DataFrame:
    df = df.sort_values(["Code", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Code"] = df["Code"].astype(str)

    close = close.copy()
    close.index = pd.to_datetime(close.index).normalize()
    close.columns = close.columns.astype(str)
    ret = np.log(close).diff()
    ret_s = ret.stack()

    df_idx = df.set_index(["Date", "Code"])
    amt = df_idx["Amount"].replace(0, np.nan)
    df_idx["Amihud"] = ret_s.abs()
    df_idx["Amihud"] = df_idx["Amihud"] / amt
    df_idx["AmihudAvg"] = (
        df_idx.groupby(level="Code", sort=False)["Amihud"]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df = df_idx.reset_index()
    df = df.drop(columns=["Amihud"])
    return df


def daily_rank_to_monthly(close: pd.DataFrame) -> pd.DataFrame:
    ret_d = np.log(close).diff()
    rank_d = ret_d.rank(axis=1, ascending=True, method="min")
    count_d = ret_d.notna().sum(axis=1)
    rank_norm_d = (rank_d.sub(1, axis=0)).div((count_d - 1).replace(0, np.nan), axis=0)
    rank_m = rank_norm_d.resample("ME").mean()
    return rank_m


def weekly_rank_to_monthly(close: pd.DataFrame, week_freq: str = "W-FRI") -> pd.DataFrame:
    close_w = close.resample(week_freq).last()
    ret_w = np.log(close_w).diff()
    rank_w = ret_w.rank(axis=1, ascending=True, method="min")
    count_w = ret_w.notna().sum(axis=1)
    rank_norm_w = (rank_w.sub(1, axis=0)).div((count_w - 1).replace(0, np.nan), axis=0)
    rank_m = rank_norm_w.resample("ME").mean()
    return rank_m


def rank_momentum_from_daily(close: pd.DataFrame, rank_window: int, agg: str = "sum") -> pd.DataFrame:
    rank_m = daily_rank_to_monthly(close)
    if agg == "sum":
        return rank_m.rolling(rank_window, min_periods=rank_window).sum()
    if agg == "mean":
        return rank_m.rolling(rank_window, min_periods=rank_window).mean()
    raise ValueError(f"Unknown agg: {agg}")


def combine_prefilter(pre_codes: pd.Index, pre_codes_2: pd.Index, mode: str = "AND") -> pd.Index:
    if mode == "AND":
        return pre_codes.intersection(pre_codes_2)
    if mode == "OR":
        return pre_codes.union(pre_codes_2)
    if mode == "SINGLE":
        return pre_codes
    raise ValueError(f"Unknown mode: {mode}")


def momentum_12_1_log(close: pd.DataFrame, m: int = 12, skip: int = 1) -> pd.DataFrame:
    close_m = close.resample("ME").last()
    return np.log(close_m / close_m.shift(m)) - np.log(close_m / close_m.shift(skip))


def risk_parity_weights(
    ret: pd.DataFrame,
    rp_min_periods: int = 60,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> pd.Series:
    n = ret.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    cov = ret.cov(min_periods=rp_min_periods)
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


def build_weights(
    u: pd.DataFrame,
    rm: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    rank_window: int,
    pick_mode: str,
    top_n: int,
    top_pct: float,
    pre_mode: str,
    pre_top_n: int,
    pre_top_pct: float,
    weight_mode: str,
    pre_combine: str = "SINGLE",
    pre_metric_2: str | None = None,
    pre_mode_2: str = "N",
    pre_top_n_2: int = 0,
    pre_top_pct_2: float = 0.0,
    pre_top_pct_range_2: tuple[float, float] | None = None,
    pre_top_n_range: tuple[int, int] | None = None,
    close: pd.DataFrame | None = None,
    pre_filter_metric: str = "Marcap",
    pre_top_pct_range: tuple[float, float] | None = None,
    top_pct_range: tuple[float, float] | None = None,
    rp_window: int = 120,
    rp_min_periods: int = 60,
) -> pd.DataFrame:
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

        s_pre = g.set_index("Code")[pre_filter_metric]
        if pre_mode == "N" and pre_top_n_range is not None:
            pre_codes = pick_top_by_n_range(s_pre, pre_top_n_range)
        elif pre_mode == "PCT" and pre_top_pct_range is not None:
            pre_codes = pick_top_by_pct_range(s_pre, pre_top_pct_range)
        elif pre_mode == "PCT" and isinstance(pre_top_pct, tuple):
            pre_codes = pick_top_by_pct_range(s_pre, pre_top_pct)
        else:
            pre_codes = pick_top_by_mode(s_pre, pre_mode, pre_top_n, pre_top_pct)

        if pre_combine != "SINGLE":
            if pre_metric_2 not in g.columns:
                raise ValueError(f"pre_metric_2 not in universe: {pre_metric_2}")
            s_pre2 = g.set_index("Code")[pre_metric_2]
            if pre_mode_2 == "PCT" and pre_top_pct_range_2 is not None:
                pre_codes_2 = pick_top_by_pct_range(s_pre2, pre_top_pct_range_2)
            else:
                pre_codes_2 = pick_top_by_mode(s_pre2, pre_mode_2, pre_top_n_2, pre_top_pct_2)
            pre_codes = combine_prefilter(pre_codes, pre_codes_2, mode=pre_combine)
        if len(pre_codes) == 0:
            continue

        codes = g[g["Code"].isin(pre_codes)]["Code"].tolist()
        if not codes:
            continue

        scores = rm.loc[month_end, codes]
        if pick_mode == "PCT" and top_pct_range is not None:
            pick_codes = pick_top_by_pct_range(scores, top_pct_range)
        else:
            pick_codes = pick_top_by_mode(scores, pick_mode, top_n, top_pct)
        if len(pick_codes) == 0:
            continue

        if weight_mode == "EQUAL":
            w = 1.0 / len(pick_codes)
            for code in pick_codes:
                picked_rows.append((dt, code, w))
        elif weight_mode == "RANK":
            ranks = scores.loc[pick_codes].rank(ascending=False, method="average")
            n = len(ranks)
            weights = (n - ranks + 1).astype(float)
            weights = weights / weights.sum()
            for code, w in weights.items():
                picked_rows.append((dt, code, float(w)))
        elif weight_mode == "SCORE":
            sc = scores.loc[pick_codes].astype(float)
            min_s = float(sc.min())
            weights = sc - min_s + 1e-12
            weights = weights / weights.sum()
            for code, w in weights.items():
                picked_rows.append((dt, code, float(w)))
        elif weight_mode == "RP":
            if close is None:
                raise ValueError("close is required for RP weights")
            end = pd.Timestamp(dt)
            start = end - pd.tseries.offsets.BDay(rp_window * 2)
            ret = np.log(close.loc[(close.index > start) & (close.index <= end), pick_codes]).diff()
            ret = ret.tail(rp_window)
            ret = ret.dropna(how="all")
            enough = ret.count() >= rp_min_periods
            ret = ret.loc[:, enough]
            if ret.empty or ret.shape[0] < 2 or ret.shape[1] < 2:
                w = 1.0 / len(pick_codes)
                for code in pick_codes:
                    picked_rows.append((dt, code, w))
            else:
                w_rp = risk_parity_weights(ret, rp_min_periods=rp_min_periods)
                for code, w in w_rp.items():
                    picked_rows.append((dt, code, float(w)))
        else:
            raise ValueError(f"Unknown WEIGHT_MODE: {weight_mode}")

    if picked_rows:
        out = pd.DataFrame(picked_rows, columns=["Date", "Code", "Weight"])
        return out.pivot_table(index="Date", columns="Code", values="Weight", fill_value=0.0)
    return pd.DataFrame()


def run_backtest(
    px: pd.DataFrame,
    w_m: pd.DataFrame,
    fees: float = 0.00025,
    init_cash: float = 1.0,
) -> vbt.Portfolio:
    px = px.sort_index()
    w_m = w_m.sort_index()
    cols = px.columns.intersection(w_m.columns)
    px = px.loc[:, cols]
    w_m = w_m.loc[:, cols]
    if not w_m.empty:
        px = px.loc[px.index >= w_m.index.min()]
    px = px.dropna(how="all")
    w_on_px = w_m.reindex(px.index)
    w_exec = w_on_px.shift(1)
    pf = vbt.Portfolio.from_orders(
        close=px,
        size=w_exec,
        size_type="TargetPercent",
        init_cash=init_cash,
        cash_sharing=True,
        fees=fees,
        freq="1D",
        call_seq="auto",
    )
    return pf


def metrics_from_equity(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {
            "total_return": np.nan,
            "sharpe": np.nan,
            "cagr": np.nan,
            "calmar": np.nan,
            "max_drawdown": np.nan,
            "rolling_12m_return_std": np.nan,
        }
    ret = np.log(eq).diff().dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann_factor = 252
    sharpe = np.nan
    if ret.std() != 0:
        sharpe = float(ret.mean() / ret.std() * np.sqrt(ann_factor))
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

    # 12M rolling total return std (month-end)
    eq_m = eq.resample("ME").last()
    if len(eq_m) > 12:
        r12 = eq_m / eq_m.shift(12) - 1.0
        rolling_12m_std = float(r12.std(skipna=True))
    else:
        rolling_12m_std = np.nan
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "cagr": cagr,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "rolling_12m_return_std": rolling_12m_std,
    }


def download_timing_price(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    df = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"Timing price download failed: {ticker}")
    px = df["Close"].copy()
    if isinstance(px, pd.DataFrame):
        if px.shape[1] == 1:
            px = px.iloc[:, 0]
        else:
            px = px.iloc[:, 0]
    px.index = pd.to_datetime(px.index).normalize()
    px = px.sort_index()
    return px


def build_timing_series(
    ticker: str,
    min_rebal: pd.Timestamp,
    max_rebal: pd.Timestamp,
    short_ma: int,
    long_ma: int,
    z_window: int,
    z_mode: str = "STD",
    k: float = 1.0,
    buffer_days: int = 60,
) -> pd.Series:
    lookback = long_ma + z_window + buffer_days
    start = pd.Timestamp(min_rebal) - pd.tseries.offsets.BDay(lookback)
    end = pd.Timestamp(max_rebal)

    px = download_timing_price(ticker, start, end)
    ma_s = px.rolling(short_ma, min_periods=short_ma).mean()
    ma_l = px.rolling(long_ma, min_periods=long_ma).mean()
    ratio = (ma_s - ma_l) / ma_l

    if z_mode == "STD":
        mu = ratio.rolling(z_window, min_periods=z_window).mean()
        sigma = ratio.rolling(z_window, min_periods=z_window).std(ddof=0)
        z = (ratio - mu) / sigma.replace(0, np.nan)
    elif z_mode == "MAD":
        med = ratio.rolling(z_window, min_periods=z_window).median()
        mad = (ratio - med).abs().rolling(z_window, min_periods=z_window).median()
        z = (ratio - med) / (1.4826 * mad.replace(0, np.nan))
    else:
        raise ValueError(f"Unknown z_mode: {z_mode}")

    z = z.replace([np.inf, -np.inf], np.nan)
    timing = 1.0 / (1.0 + np.exp(-k * z))
    return timing


def apply_timing_weights(
    w_m: pd.DataFrame,
    timing_series: pd.Series,
    mode: str = "SCALE",
    k: float = 1.0,
    onoff_z: float = 0.0,
    min_exposure: float = 0.2,
    return_timing: bool = False,
) -> pd.DataFrame:
    if w_m.empty:
        if return_timing:
            return w_m, pd.Series(dtype=float)
        return w_m
    if timing_series is None:
        raise ValueError("timing_series is None")
    timing_rebal = timing_series.reindex(w_m.index)
    if mode == "ONOFF":
        t = timing_rebal.clip(lower=1e-6, upper=1 - 1e-6)
        z = np.log(t / (1 - t)) / k
        on = (z > onoff_z).astype(float)
        timing_rebal = on * (1.0 - min_exposure) + min_exposure
    elif mode == "SCALE":
        if min_exposure > 0.0:
            timing_rebal = timing_rebal * (1.0 - min_exposure) + min_exposure
    else:
        raise ValueError(f"Unknown TIMING_MODE: {mode}")
    valid = timing_rebal.notna()
    w_m = w_m.loc[valid]
    timing_rebal = timing_rebal.loc[valid]
    w_m = w_m.mul(timing_rebal, axis=0)
    if return_timing:
        return w_m, timing_rebal
    return w_m


def download_benchmark(start: str, end: str, tickers: list[str]) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].copy()
    px.index = pd.to_datetime(px.index).normalize()
    px = px.sort_index()
    return px


def equity_from_price(px: pd.Series, init_cash: float = 1.0) -> pd.Series:
    r = px.pct_change().fillna(0.0)
    return init_cash * (1.0 + r).cumprod()


def benchmark_equity_for_index(
    index: pd.Index,
    ticker: str,
    init_cash: float = 1.0,
    name: str | None = None,
) -> pd.Series:
    if index is None or len(index) == 0:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(index).normalize()
    idx = pd.DatetimeIndex(idx).sort_values().unique()
    start = idx.min().strftime("%Y-%m-%d")
    end = (idx.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bm_px = download_benchmark(start, end, [ticker])
    col = name or ticker
    if ticker in bm_px.columns:
        bm_px = bm_px.rename(columns={ticker: col})
    elif bm_px.columns.size > 0:
        col = bm_px.columns[0]
    bm_px = bm_px.reindex(idx).ffill()
    if bm_px.empty or col not in bm_px.columns:
        return pd.Series(dtype=float)
    return equity_from_price(bm_px[col], init_cash=init_cash)


def beat_ratio(eq: pd.Series, bm_eq: pd.Series, freq: str = "QE") -> float:
    eq = eq.dropna()
    bm_eq = bm_eq.dropna()
    idx = eq.index.intersection(bm_eq.index)
    if idx.empty:
        return np.nan
    eq = eq.loc[idx]
    bm_eq = bm_eq.loc[idx]

    eq_r = np.log(eq).resample(freq).last().diff()
    bm_r = np.log(bm_eq).resample(freq).last().diff()
    aligned = pd.concat([eq_r, bm_r], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float((aligned.iloc[:, 0] > aligned.iloc[:, 1]).mean())
