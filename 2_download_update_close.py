import os
import time
from pathlib import Path

import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf


BASE = Path(__file__).resolve().parent
os.chdir(BASE)

UNIVERSE_FILE = "build_universe.parquet"
RAW_FILE = "raw_close.parquet"
SLEEP_SEC = 0.1

SOURCE = "FDR"   # "FDR" / "YF" / "YF_FDR"
YF_SUFFIXES = [".KS", ".KQ"]
YF_BATCH = 80


def yf_download_close(tickers, start, end) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    close = df["Close"].copy() if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].copy()
    close.index = pd.to_datetime(close.index).normalize()
    return close.sort_index()


u = pd.read_parquet(UNIVERSE_FILE)
u["Code"] = u["Code"].astype(str).str.zfill(6)
codes = sorted(u["Code"].unique().tolist())

start_all = pd.to_datetime(u["Date"]).min().normalize()
end_all = pd.to_datetime(u["Date"]).max().normalize()

print("source:", SOURCE)
print("universe codes:", len(codes))
print("full range:", start_all.date(), "->", end_all.date())

failed = []
merged = pd.DataFrame()

if SOURCE == "FDR":
    frames = []
    for i, code in enumerate(codes, 1):
        try:
            df = fdr.DataReader(code, start_all, end_all)
            if df is not None and not df.empty and "Close" in df.columns:
                s = df["Close"].copy()
                s.index = pd.to_datetime(s.index).normalize()
                s.name = code
                frames.append(s)
            else:
                failed.append(code)
        except Exception:
            failed.append(code)

        if i % 50 == 0:
            print("done", i, "/", len(codes))
        time.sleep(SLEEP_SEC)

    merged = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

elif SOURCE in ("YF", "YF_FDR"):
    # 6자리 코드 -> 6자리.KS / 6자리.KQ 둘 다 후보로 만들어서 다운로드
    candidates = []
    meta = []  # (yf_ticker, code)
    for code in codes:
        for suf in YF_SUFFIXES:
            t = f"{code}{suf}"
            candidates.append(t)
            meta.append((t, code))

    # 배치 다운로드
    parts = []
    for i in range(0, len(candidates), YF_BATCH):
        batch = candidates[i:i + YF_BATCH]
        close = yf_download_close(batch, start_all, end_all)
        if not close.empty:
            parts.append(close)
        print("done", min(i + YF_BATCH, len(candidates)), "/", len(candidates))

    close_all = pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()

    # yf ticker -> 원 code로 매핑 (KS/KQ 중 먼저 성공한 컬럼을 사용)
    out = {}
    for t, code in meta:
        if t in close_all.columns and code not in out:
            s = close_all[t].copy()
            s.name = code
            out[code] = s

    merged = pd.concat(out.values(), axis=1).sort_index() if out else pd.DataFrame()

    # 실패 코드 = 어떤 suffix로도 못 받은 것
    got = set(merged.columns)
    failed = [c for c in codes if c not in got]
    # Fallback to FDR for failed codes
    if SOURCE == "YF_FDR" and failed:
        frames = []
        for i, code in enumerate(failed, 1):
            try:
                df = fdr.DataReader(code, start_all, end_all)
                if df is not None and not df.empty and "Close" in df.columns:
                    s = df["Close"].copy()
                    s.index = pd.to_datetime(s.index).normalize()
                    s.name = code
                    frames.append(s)
            except Exception:
                pass
            if i % 50 == 0:
                print("fdr fallback done", i, "/", len(failed))
            time.sleep(SLEEP_SEC)
        if frames:
            fdr_merged = pd.concat(frames, axis=1).sort_index()
            merged = pd.concat([merged, fdr_merged], axis=1)

else:
    raise ValueError(SOURCE)

if not merged.empty:
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    keep_cols = [c for c in codes if c in merged.columns]
    merged = merged.loc[:, keep_cols]

merged.to_parquet(RAW_FILE)
print("saved:", RAW_FILE)
if not merged.empty:
    print("range:", merged.index.min().date(), "->", merged.index.max().date(), "cols:", merged.shape[1])
print("failed codes:", len(set(failed)))
