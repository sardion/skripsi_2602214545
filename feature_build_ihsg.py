#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_build_ihsg.py

Build IHSG contextual features:
- date
- ihsg_close
- ihsg_return_lag_1         (daily return vs previous close; formatted percentage string, e.g. '0.647%')
- ihsg_std_dev_7            (rolling std of close, window=7; starts after full window)
- ihsg_std_dev_14           (rolling std of close, window=14; starts after full window)
- ihsg_std_dev_21           (rolling std of close, window=21; starts after full window)

Input:
- data/raw/ihsg_raw.csv
  Columns: Trading Date, Open, High, Low, Close, Adj Close, Volume

Output:
- data/raw/ihsg_features_raw.csv

Notes:
- Input may be in any order; will be sorted ascending by date.
- ihsg_return_lag_1 stored as string with '%' suffix (for consistency with macroeconomic % features).
- Std dev columns remain numeric.
- All numeric outputs rounded to 3 decimal places for readability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import pandas as pd


# =========================
# Config (relative paths)
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"

INPUT_FILENAME: Final[str] = "ihsg_raw.csv"
OUTPUT_FILENAME: Final[str] = "ihsg_features_raw.csv"

INPUT_CSV: Final[Path] = RAW_DIR / INPUT_FILENAME
OUTPUT_CSV: Final[Path] = RAW_DIR / OUTPUT_FILENAME


# =========================
# Helpers
# =========================
def _clean_numeric(s: pd.Series) -> pd.Series:
    """Remove thousand separators/spaces and cast to float."""
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .replace({"": None})
         .astype(float)
    )


def read_and_prepare_raw_data(path: Path) -> pd.DataFrame:
    """
    Read raw IHSG CSV, normalize date, clean Close, sort ascending,
    and return a base DataFrame with columns: date, ihsg_close.
    """
    df: pd.DataFrame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    required = {"Trading Date", "Close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Input must contain columns: {sorted(required)}")

    # Parse date (normalize to midnight)
    dt = pd.to_datetime(df["Trading Date"], errors="raise").dt.normalize()

    # Clean close
    close = _clean_numeric(df["Close"])

    base = pd.DataFrame({
        "date": dt,
        "ihsg_close": close
    }).sort_values("date", ascending=True).reset_index(drop=True)

    return base


def build_features(base: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ihsg_return_lag_1 (percentage with % sign) and rolling std devs on close.
    Std dev values start only after full window; earlier rows filled with 0.
    Round all numeric outputs to 3 decimal places.
    """
    out = base.copy()

    # Daily return vs previous close (percentage numeric)
    pct_return = ((out["ihsg_close"] / out["ihsg_close"].shift(1)) - 1.0) * 100.0

    # Rolling std dev of close price; enforce full window before computing
    out["ihsg_std_dev_7"]  = out["ihsg_close"].rolling(window=7,  min_periods=7).std()
    out["ihsg_std_dev_14"] = out["ihsg_close"].rolling(window=14, min_periods=14).std()
    out["ihsg_std_dev_21"] = out["ihsg_close"].rolling(window=21, min_periods=21).std()

    # Fill NaNs (initial periods before full window and first return) with 0
    out = out.fillna(0.0)
    pct_return = pct_return.fillna(0.0)

    # Round numeric columns to 3 decimal places
    pct_return = pct_return.round(3)
    out["ihsg_std_dev_7"]  = out["ihsg_std_dev_7"].round(3)
    out["ihsg_std_dev_14"] = out["ihsg_std_dev_14"].round(3)
    out["ihsg_std_dev_21"] = out["ihsg_std_dev_21"].round(3)

    # Convert percentage column to string with '%' sign
    out["ihsg_return_lag_1"] = pct_return.astype(str) + "%"

    # Keep only required columns in specified order
    out = out[[
        "date",
        "ihsg_close",
        "ihsg_return_lag_1",
        "ihsg_std_dev_7",
        "ihsg_std_dev_14",
        "ihsg_std_dev_21",
    ]]

    return out


# =========================
# Main
# =========================
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    base = read_and_prepare_raw_data(INPUT_CSV)
    feats = build_features(base)

    # Save to data/raw per the procedure
    feats.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {OUTPUT_FILENAME} with {len(feats)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
