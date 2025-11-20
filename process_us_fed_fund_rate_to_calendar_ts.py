#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_us_fed_fund_rate_to_calendar_ts.py

Build a daily calendar time series for the US Fed Funds Rate with:
- calendar_date
- us_fed_fund_rate (string, preserves % as in raw CSV)
- days_since_us_fed_fund_rate (int)

Input:
- data/raw/investing_us_fed_fund_rate_2020-2025_raw.csv
  Columns: Date, Time, Currency, Impact, Event, Actual, Forecast, Previous

Output:
- data/processed/us_fed_fund_rate_ts_cal.csv

Calendar window:
- 2020-01-01 .. 2025-08-31

Rules:
- On release dates: rate = Actual (string, keep %)
- Forward-fill Actual until next release date
- Before first release: retrospective-fill rate using first row's Previous (string, keep %)
- days_since_us_fed_fund_rate:
    * On each release date = 0
    * Increases by 1 each day after a release until the next release
    * Before first release: at CAL_START equals the gap in days to the first release, then decrements to 0 on that release date

Notes:
- Multiple rows on the same release date → keep the LAST one by (Date, Time) order
- Slash dates in raw are parsed as MONTH/DAY/YEAR (e.g., 5/8/2025 = May 8, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple
import re

import pandas as pd


# =========================
# Config (relative paths)
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"

INPUT_FILENAME: Final[str] = "investing_us_fed_fund_rate_2020-2025_raw.csv"
OUTPUT_FILENAME: Final[str] = "us_fed_fund_rate_ts_cal.csv"

INPUT_CSV: Final[Path] = RAW_DIR / INPUT_FILENAME
OUTPUT_CSV: Final[Path] = PROCESSED_DIR / OUTPUT_FILENAME

CAL_START: Final[date] = date(2020, 1, 2)
CAL_END: Final[date] = date(2025, 8, 31)


# =========================
# Parsing helpers
# =========================
_SLASH_MDY_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$")

def parse_time_str(s: str) -> dtime:
    """
    Returns a comparable time; treat 'All Day' or blank as earliest (00:00).
    Supports 'HH:MM' (24h) and 'h:mm AM/PM'.
    """
    if not isinstance(s, str):
        return dtime(0, 0)
    s_clean = s.strip()
    if not s_clean or s_clean.lower() == "all day":
        return dtime(0, 0)

    for fmt in ("%H:%M", "%I:%M %p"):
        try:
            return datetime.strptime(s_clean, fmt).time()
        except ValueError:
            continue
    return dtime(0, 0)


@dataclass(frozen=True)
class ReleaseRow:
    rel_date: date
    rel_time: dtime
    actual: str
    previous: str


# =========================
# Core functions
# =========================
def read_release_rows(csv_path: Path) -> List[ReleaseRow]:
    """
    Read raw Investing.com CSV, parse month-first slash dates,
    sort by (date, time), and keep the LAST row per date.
    """
    df: pd.DataFrame = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    # Normalize headers
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    for col in ("Date", "Actual", "Previous"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Date parsing: FORCE month/day/year for slash dates ---
    # Fast path: parse the whole column as %m/%d/%Y; non-matching rows become NaT
    parsed_mdy = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    # If any rows didn't match the slash format, handle them with a fallback
    na_mask = parsed_mdy.isna()

    if na_mask.any():
        # For truly non-slash rows (rare here), try general parsing (still month-first by default in pandas)
        fallback = pd.to_datetime(df.loc[na_mask, "Date"], errors="raise")
        parsed_mdy.loc[na_mask] = fallback

    df["__date"] = parsed_mdy.dt.date

    # --- Time parsing (optional column) ---
    if "Time" in df.columns:
        df["__time"] = df["Time"].map(parse_time_str)
    else:
        df["__time"] = dtime(0, 0)

    # Sort by date then time; take LAST row per date
    df_sorted: pd.DataFrame = df.sort_values(["__date", "__time"])
    dedup: pd.DataFrame = df_sorted.groupby("__date", as_index=False).last()

    releases: List[ReleaseRow] = []
    for _, row in dedup.iterrows():
        d: date = row["__date"]
        t: dtime = row["__time"]
        a: str = str(row["Actual"]).strip()
        p: str = str(row["Previous"]).strip()
        releases.append(ReleaseRow(d, t, a, p))

    releases.sort(key=lambda r: (r.rel_date, r.rel_time))
    return releases


def build_calendar(start: date, end: date) -> List[date]:
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]


def compute_days_since_series(calendar: List[date], release_dates: List[date]) -> List[int]:
    """
    Compute days_since_us_fed_fund_rate:

    - Before the first release date D0:
        days_since at CAL_START = (D0 - CAL_START).days
        then decrement by 1 each day until it reaches 0 on D0.
    - On each release date: 0
    - After each release date: increases by 1 daily until the next release.
    """
    days_since: List[int] = [0] * len(calendar)
    if not release_dates:
        # No releases at all: monotonic increasing from 0
        counter = 0
        for i in range(len(calendar)):
            days_since[i] = counter
            counter += 1
        return days_since

    rel_set = set(release_dates)
    first_rel = release_dates[0]

    # 1) Before first release: start at gap and decrement to 0
    if calendar[0] < first_rel:
        gap = (first_rel - calendar[0]).days
        for i, d in enumerate(calendar):
            if d < first_rel:
                days_since[i] = gap - i
            else:
                break

    # 2) From first release onward: 0 on release days, +1 otherwise
    started = False
    counter = 0
    for i, d in enumerate(calendar):
        if d < first_rel:
            continue
        if d in rel_set:
            days_since[i] = 0
            counter = 0
            started = True
        else:
            if started:
                counter += 1
                days_since[i] = counter
            else:
                days_since[i] = 0
    return days_since


def build_rate_series(calendar: List[date], releases: List[ReleaseRow]) -> List[str]:
    """
    Build us_fed_fund_rate as strings (preserve '%').
    - On release dates: Actual string from that date.
    - Forward fill between releases.
    - Before first release: retrospective fill using first Previous string.
    """
    rate_by_day: List[Optional[str]] = [None] * len(calendar)
    cal_index: Dict[date, int] = {d: i for i, d in enumerate(calendar)}

    if not releases:
        return ["" for _ in calendar]

    # Only keep releases inside the calendar window
    first_day, last_day = calendar[0], calendar[-1]
    in_window: List[ReleaseRow] = [r for r in releases if first_day <= r.rel_date <= last_day]

    if not in_window:
        return ["" for _ in calendar]

    # Mark Actual on release dates
    for r in in_window:
        idx = cal_index[r.rel_date]
        rate_by_day[idx] = r.actual  # string, keep %

    # Forward-fill
    last_seen: Optional[str] = None
    for i in range(len(calendar)):
        if rate_by_day[i] is not None:
            last_seen = rate_by_day[i]
        else:
            if last_seen is not None:
                rate_by_day[i] = last_seen

    # Retrospective-fill before first in-window release with that release's Previous
    first_rel = in_window[0]
    first_idx = cal_index[first_rel.rel_date]
    for i in range(0, first_idx):
        rate_by_day[i] = first_rel.previous

    # Replace any remaining None (shouldn't happen) with empty string
    return [r if r is not None else "" for r in rate_by_day]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Read & dedup releases
    releases: List[ReleaseRow] = read_release_rows(INPUT_CSV)
    if not releases:
        raise RuntimeError("No releases found in input CSV.")

    # 2) Build daily calendar
    calendar_days: List[date] = build_calendar(CAL_START, CAL_END)

    # 3) Build rate series (strings, with %)
    rate_series: List[str] = build_rate_series(calendar_days, releases)

    # 4) Build days_since series (ints)
    release_dates: List[date] = sorted({r.rel_date for r in releases if CAL_START <= r.rel_date <= CAL_END})
    days_since_series: List[int] = compute_days_since_series(calendar_days, release_dates)

    # 5) Save CSV
    out_df: pd.DataFrame = pd.DataFrame({
        "calendar_date": [d.isoformat() for d in calendar_days],
        "us_fed_fund_rate": rate_series,
        "days_since_us_fed_fund_rate": days_since_series,
    })
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
