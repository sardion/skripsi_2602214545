#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_calendar_features_to_calendar_ts.py

Output:
- data/processed/calendar_features_ts_cal.csv
  with a canonical join key: calendar_date (YYYY-MM-DD string, not index)

Calendar window:
- 2020-01-02 .. 2025-08-31
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
    """Create continuous daily calendar DataFrame."""
    days: List[date] = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    cal = pd.DataFrame({"calendar_date_dt": pd.to_datetime(days)})
    return cal

def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Robustly parse date strings that may be mixed (DD/MM/YYYY, YYYY-MM-DD, etc.).
    Returns tz-naive normalized datetimes.
    """
    raw = s.astype(str).str.strip()

    # First attempt: mixed formats, day-first
    dt = pd.to_datetime(raw, errors="coerce", dayfirst=True, format="mixed")

    # Fallback: mixed formats, month-first
    mask = dt.isna()
    if mask.any():
        dt_mf = pd.to_datetime(raw[mask], errors="coerce", dayfirst=False, format="mixed")
        dt.loc[mask] = dt_mf

    # Final explicit fallbacks if any leftovers remain
    mask = dt.isna()
    if mask.any():
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
            try_dt = pd.to_datetime(raw[mask], errors="coerce", format=fmt)
            dt.loc[mask & try_dt.notna()] = try_dt
            mask = dt.isna()
            if not mask.any():
                break

    if dt.isna().any():
        bad = raw[dt.isna()].unique().tolist()[:5]
        raise ValueError(f"Unparseable date strings (sample): {bad}")

    return dt.dt.normalize()

def read_raw_calendar_flags(path: Path) -> pd.DataFrame:
    """Read the raw calendar flags file and normalize types we actually use."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns or "is_trading_day" not in df.columns:
        raise ValueError("Input must contain 'date' and 'is_trading_day' columns.")
    df["date_dt"] = parse_date_series(df["date"])
    # Keep only date + is_trading_day from raw; we will recompute the rest
    df = df[["date_dt", "is_trading_day"]].sort_values("date_dt").reset_index(drop=True)
    df["is_trading_day"] = (
        pd.to_numeric(df["is_trading_day"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    )
    return df

def build_flags_for_calendar(calendar_df: pd.DataFrame, trading_dates_df: pd.DataFrame) -> pd.DataFrame:
    """Merge calendar with trading dates and compute weekday/month/next flags."""
    out = calendar_df.copy()  # has 'calendar_date_dt'

    # Map trading days
    td = trading_dates_df.rename(columns={"date_dt": "calendar_date_dt"})[["calendar_date_dt", "is_trading_day"]]
    out = out.merge(td, on="calendar_date_dt", how="left")
    out["is_trading_day"] = out["is_trading_day"].fillna(0).astype(int)

    # Weekday flags (Mon=0..Sun=6)
    dow = out["calendar_date_dt"].dt.dayofweek
    out["is_monday"] = (dow == 0).astype(int)
    out["is_tuesday"] = (dow == 1).astype(int)
    out["is_wednesday"] = (dow == 2).astype(int)
    out["is_thursday"] = (dow == 3).astype(int)
    out["is_friday"] = (dow == 4).astype(int)
    out["is_saturday"] = (dow == 5).astype(int)
    out["is_sunday"] = (dow == 6).astype(int)

    # Month flags
    out["is_month_start"] = out["calendar_date_dt"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["calendar_date_dt"].dt.is_month_end.astype(int)

    # Next trading day flag (shift -1)
    out["next_is_trading_day"] = out["is_trading_day"].shift(-1).fillna(0).astype(int)

    # Finalize calendar_date as canonical string key
    out["calendar_date"] = out["calendar_date_dt"].dt.strftime("%Y-%m-%d")

    # Order & return as regular columns (no index)
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
    out = build_flags_for_calendar(cal_df, raw_flags)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {OUTPUT_FILENAME} with {len(out)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
