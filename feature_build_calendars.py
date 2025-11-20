    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_build_calendars.py

Build base calendar feature scaffold from an integrated ticker file:
- date
- is_sunday, is_monday, is_tuesday, is_wednesday, is_thursday, is_friday, is_saturday
- is_trading_day (set to 1 for all Trading Date rows)
- next_is_trading_day
- is_month_start, is_month_end

Input:
- data/raw/integrated_ticker_summary_csv/integrated_BBCA.csv
  Required column: Trading Date

Output:
- data/raw/calendar_features_raw.csv

Notes:
- Only marks actual trading dates for now; other columns are left blank ("")
  and will be filled when constructing the full calendar-day time series later.
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
INTEGRATED_DIR: Final[Path] = RAW_DIR / "integrated_ticker_summary_csv"

INPUT_FILENAME: Final[str] = "integrated_BBCA.csv"
OUTPUT_FILENAME: Final[str] = "calendar_features_raw.csv"

INPUT_CSV: Final[Path] = INTEGRATED_DIR / INPUT_FILENAME
OUTPUT_CSV: Final[Path] = RAW_DIR / OUTPUT_FILENAME


# =========================
# Core
# =========================
def read_trading_days(path: Path) -> pd.Series:
    """
    Read the integrated_BBCA.csv and return a sorted, unique datetime index
    based on 'Trading Date'.
    """
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Trading Date" not in df.columns:
        raise ValueError("Input must contain 'Trading Date' column.")

    dates = pd.to_datetime(df["Trading Date"], errors="raise").dt.normalize()
    dates = pd.Series(dates).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    return dates


def build_calendar_scaffold(trading_dates: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame with required columns. Set is_trading_day=1; others blank.
    """
    out = pd.DataFrame({"date": trading_dates})

    # Blank columns to be filled later (use empty string "")
    blank_cols = [
        "is_sunday", "is_monday", "is_tuesday", "is_wednesday",
        "is_thursday", "is_friday", "is_saturday",
        "next_is_trading_day", "is_month_start", "is_month_end",
    ]
    for c in blank_cols:
        out[c] = ""

    # Mark actual trading days now
    out["is_trading_day"] = 1

    # Order columns exactly as specified (+ date first for mergeability)
    out = out[
        [
            "date",
            "is_sunday", "is_monday", "is_tuesday", "is_wednesday",
            "is_thursday", "is_friday", "is_saturday",
            "is_trading_day",
            "next_is_trading_day",
            "is_month_start", "is_month_end",
        ]
    ]
    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    trading_dates = read_trading_days(INPUT_CSV)
    cal_df = build_calendar_scaffold(trading_dates)

    cal_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {OUTPUT_FILENAME} with {len(cal_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
