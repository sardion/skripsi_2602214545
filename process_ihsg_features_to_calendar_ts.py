#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_ihsg_features_to_calendar_ts.py

Build a daily calendar time series for IHSG features with:
- calendar_date (index)
- ihsg_return_lag_1
- ihsg_std_dev_7
- ihsg_std_dev_14
- ihsg_std_dev_21

Input:
- data/raw/ihsg_features_raw.csv
  Columns: date, ihsg_close, ihsg_return_lag_1, ihsg_std_dev_7, ihsg_std_dev_14, ihsg_std_dev_21

Output:
- data/processed/ihsg_features_ts_cal.csv

Calendar window:
- 2020-01-02 .. 2025-08-31

Rules:
- Source data may contain tz-aware datetimes (e.g., +07:00); these are converted to tz-naive.
- Ignore ihsg_close column.
- Forward-fill between missing calendar days (weekends/holidays).
- If the earliest 2020 date is later than 2020-01-02 but a Dec-2019 value exists, use that for initial forward fill.
- No retrospective fill before the first historical record.
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

INPUT_FILENAME: Final[str] = "ihsg_features_raw.csv"
OUTPUT_FILENAME: Final[str] = "ihsg_features_ts_cal.csv"

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
    return pd.DataFrame({"calendar_date": pd.to_datetime(days)})


def to_tznaive_normalized(s: pd.Series) -> pd.Series:
    """Convert any tz-aware datetime Series to tz-naive and normalize to midnight."""
    dt = pd.to_datetime(s, errors="raise")
    if getattr(dt.dtype, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    return dt.dt.normalize()


def read_and_prepare_raw(path: Path) -> pd.DataFrame:
    """Read raw IHSG features and clean columns, making dates tz-naive."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    required = {
        "date",
        "ihsg_return_lag_1",
        "ihsg_std_dev_7",
        "ihsg_std_dev_14",
        "ihsg_std_dev_21",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns. Required: {sorted(required)}")

    df["date"] = to_tznaive_normalized(df["date"])

    # Keep relevant cols only, sort ascending
    df = df[
        ["date", "ihsg_return_lag_1", "ihsg_std_dev_7", "ihsg_std_dev_14", "ihsg_std_dev_21"]
    ].sort_values("date").reset_index(drop=True)

    return df


def merge_and_forward_fill(calendar_df: pd.DataFrame, ihsg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge onto full calendar, fill missing values forward.
    If earliest 2020 record is after CAL_START and a Dec-2019 record exists,
    use that record to fill from CAL_START up to that first date.
    """
    # identify latest pre-2020 record
    pre_start = ihsg_df.loc[ihsg_df["date"] < pd.Timestamp(CAL_START)]
    if not pre_start.empty:
        latest_pre = pre_start.iloc[-1]
        # if earliest in 2020 > CAL_START, prepend a pseudo-row at CAL_START
        first_in_2020 = ihsg_df.loc[ihsg_df["date"] >= pd.Timestamp(CAL_START)].head(1)
        if not first_in_2020.empty and first_in_2020.iloc[0]["date"].date() > CAL_START:
            new_row = pd.DataFrame(
                {
                    "date": [pd.Timestamp(CAL_START)],
                    "ihsg_return_lag_1": [latest_pre["ihsg_return_lag_1"]],
                    "ihsg_std_dev_7": [latest_pre["ihsg_std_dev_7"]],
                    "ihsg_std_dev_14": [latest_pre["ihsg_std_dev_14"]],
                    "ihsg_std_dev_21": [latest_pre["ihsg_std_dev_21"]],
                }
            )
            ihsg_df = pd.concat([ihsg_df, new_row], ignore_index=True)

    merged = pd.merge(calendar_df, ihsg_df, left_on="calendar_date", right_on="date", how="left")
    merged = merged.drop(columns=["date"], errors="ignore").sort_values("calendar_date")

    # forward fill
    for col in ["ihsg_return_lag_1", "ihsg_std_dev_7", "ihsg_std_dev_14", "ihsg_std_dev_21"]:
        merged[col] = merged[col].ffill()

    return merged


# =========================
# Main
# =========================
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = read_and_prepare_raw(INPUT_CSV)
    cal_df = build_calendar(CAL_START, CAL_END)
    filled_df = merge_and_forward_fill(cal_df, raw_df)

    out = filled_df.set_index("calendar_date")
    out.to_csv(OUTPUT_CSV, encoding="utf-8-sig", index=True)
    print(f"[OK] Saved {OUTPUT_FILENAME} with {len(out)} rows")


if __name__ == "__main__":
    main()
