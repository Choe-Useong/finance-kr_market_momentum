import subprocess
import sys
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent

MARCAP_UPDATE = PROJECT_ROOT / "marcap" / "marcap_update.py"

RUN_MARCAP_UPDATE = True
RUN_BUILD_UNIVERSE = True
RUN_DOWNLOAD_CLOSE = True
RUN_BUILD_TARGET = True

UNIVERSE_FILE = BASE / "build_universe.parquet"
RAW_CLOSE_FILE = BASE / "raw_close.parquet"
TARGET_FILE = BASE / "build_target.parquet"

MIN_WEIGHT = 0.00001
TOP_N = None


def run_script(path: Path) -> None:
    print("\n== run:", path.name)
    subprocess.run([sys.executable, str(path)], cwd=str(path.parent), check=True)


def print_parquet_range(path: Path, date_col: str | None = None) -> None:
    if not path.exists():
        print("missing:", path.name)
        return
    df = pd.read_parquet(path)
    print(path.name, "rows:", len(df), "cols:", len(df.columns))
    if df.empty:
        return
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col])
        print("range:", d.min().date(), "->", d.max().date())
    elif isinstance(df.index, pd.DatetimeIndex):
        print("range:", df.index.min().date(), "->", df.index.max().date())


def print_latest_target() -> None:
    if not TARGET_FILE.exists():
        print("missing:", TARGET_FILE.name)
        return

    w = pd.read_parquet(TARGET_FILE)
    if w.empty:
        print("target is empty")
        return

    w.index = pd.to_datetime(w.index).normalize()
    w = w.sort_index()
    dt = w.index.max()
    s = w.loc[dt]
    s = s[s > MIN_WEIGHT].sort_values(ascending=False)
    if TOP_N is not None:
        s = s.head(TOP_N)

    print("\n== latest target")
    print("date:", dt.date())
    print("codes:", s.index.tolist())
    print("weights:", [float(x) for x in s.values])


if __name__ == "__main__":
    if RUN_MARCAP_UPDATE:
        run_script(MARCAP_UPDATE)
    if RUN_BUILD_UNIVERSE:
        run_script(BASE / "1_build_universe2.py")
        print_parquet_range(UNIVERSE_FILE, date_col="Date")
    if RUN_DOWNLOAD_CLOSE:
        run_script(BASE / "2_download_update_close.py")
        print_parquet_range(RAW_CLOSE_FILE)
    if RUN_BUILD_TARGET:
        run_script(BASE / "3_build_inputs2.py")
        print_parquet_range(TARGET_FILE)

    print_latest_target()
