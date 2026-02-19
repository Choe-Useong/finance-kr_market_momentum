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
END = "2026-02-10"  # None -> use full available range

OUT_UNIVERSE = str(BASE / "build_universe.parquet")

df = marcap_data(START, END)
df["Code"] = df["Code"].astype(str).str.zfill(6)
df = df[df["Marcap"].notna()].copy()

df = df.copy()
df["Date"] = pd.to_datetime(df.index).normalize()
df["Scope"] = df["Market"].replace({"KOSDAQ GLOBAL": "KOSDAQ"})

# Keep only columns needed downstream; screening happens in 3_build_inputs.py
out = df[["Date", "Scope", "Code", "Marcap", "Amount"]].copy()
out.to_parquet(OUT_UNIVERSE, index=False)
print("saved:", OUT_UNIVERSE, "rows:", len(out))
