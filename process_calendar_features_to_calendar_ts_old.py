#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_calendar_features_to_calendar_ts.py

Convert calendar/trading-day flags into a full calendar-day time series:
- calendar_date (index)
- is_sunday, is_monday, is_tuesday, is_wednesday, is_thursday, is_friday, is_saturday
- is_trading_day
- next_is_trading_day
- is_month_start, is_month_end

Input:
- data/raw/calendar_features_raw.csv
  Columns:
    date, is_sunday, is_monday, is_tuesday, is_wednesday,
    is_thursday, is_friday, is_saturday,
    is_trading_day, next_is_trading_day, is_month_start, is_month_end

Output:
- data/processed/calendar_features_ts_cal.csv

Calendar window:
- 2020-01-02 .. 2025-08-31

Rules:
- Keep is_trading_day=1 for dates present in the raw file; set 0 for other calendar days.
- Compute weekday flags from calendar_date (1/0).
- next_is_trading_day reflects the next day's is_trading_day (1 if tomorrow is trading day, else 0).
- is_month_start/is_month_end computed from calendar_date (handles leap years).
- Handle tz-aware dates by converting to tz-naive and normalizing to midnight.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Final, List
import pandas as pd


# =========================
# Config (relative paths)
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"

INPUT_FILENAME: Final[str] = "calendar_features_raw.csv"
OUTPUT_FILENAME: Final[str] = "calendar_features_ts_cal.csv"

INPUT_CSV: Final[Path] = RAW_DIR / INPUT_FILENAME
OUTPUT_CSV: Final[Path] = PROCESSED_DIR / OUTPUT_FILENAME

CAL_START: Final[date] = date(2020, 1, 2)
CAL_END: Final[date] = date(2025, 8, 31)


# =========================
# Helpers
# =========================
def build_calendar(start: date, end: date) -> pd.DataFrame:
    """Create continuous daily calendar DataFrame (tz-naive)."""
    days: List[date] = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return pd.DataFrame({"calendar_date": pd.to_datetime(days)})


def to_tznaive_normalized(s: pd.Series) -> pd.Series:
    """Convert any tz-aware datetime Series to tz-naive and normalize to midnight."""
    dt = pd.to_datetime(s, errors="raise")
    if getattr(dt.dtype, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    return dt.dt.normalize()


def read_raw_calendar_flags(path: Path) -> pd.DataFrame:
    """Read the raw calendar flags file and normalize date dtype."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns or "is_trading_day" not in df.columns:
        raise ValueError("Input must contain 'date' and 'is_trading_day' columns.")

    df["date"] = to_tznaive_normalized(df["date"])

    # Keep only what we need from raw; we will recompute weekday/month flags and next_is_trading_day
    df = df[["date", "is_trading_day"]].sort_values("date").reset_index(drop=True)

    # Normalize is_trading_day to int {0,1}
    df["is_trading_day"] = (
        pd.to_numeric(df["is_trading_day"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )

    return df


def build_flags_for_calendar(calendar_df: pd.DataFrame, trading_dates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge calendar with trading dates, set weekday flags, month flags,
    and next_is_trading_day according to the rules.
    """
    # Start with full calendar
    out = calendar_df.copy()

    # Prepare trading-day map
    td = trading_dates_df[["date", "is_trading_day"]].drop_duplicates()
    out = out.merge(td, left_on="calendar_date", right_on="date", how="left").drop(columns=["date"])

    # For days not present in raw trading dates, set 0
    out["is_trading_day"] = out["is_trading_day"].fillna(0).astype(int)

    # Weekday flags (Monday=0 ... Sunday=6)
    dow = out["calendar_date"].dt.dayofweek
    out["is_monday"] = (dow == 0).astype(int)
    out["is_tuesday"] = (dow == 1).astype(int)
    out["is_wednesday"] = (dow == 2).astype(int)
    out["is_thursday"] = (dow == 3).astype(int)
    out["is_friday"] = (dow == 4).astype(int)
    out["is_saturday"] = (dow == 5).astype(int)
    out["is_sunday"] = (dow == 6).astype(int)

    # Month flags
    out["is_month_start"] = out["calendar_date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["calendar_date"].dt.is_month_end.astype(int)

    # next_is_trading_day = next day's is_trading_day (shift -1), fill last day with 0
    out["next_is_trading_day"] = out["is_trading_day"].shift(-1).fillna(0).astype(int)

    # Order columns as specified (index will be calendar_date)
    out = out[
        [
            "calendar_date",
            "is_sunday", "is_monday", "is_tuesday", "is_wednesday",
            "is_thursday", "is_friday", "is_saturday",
            "is_trading_day",
            "next_is_trading_day",
            "is_month_start", "is_month_end",
        ]
    ]

    return out


# =========================
# Main
# =========================
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_flags = read_raw_calendar_flags(INPUT_CSV)
    cal_df = build_calendar(CAL_START, CAL_END)
    filled = build_flags_for_calendar(cal_df, raw_flags)

    out = filled.set_index("calendar_date")
    out.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"[OK] Saved {OUTPUT_FILENAME} with {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
