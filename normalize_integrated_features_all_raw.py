#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_integrated_features_all_raw.py

Normalize the per-ticker integrated feature datasets according to Table 3.5.

For each ticker:
1) Read data/raw_features/{TICKER}_integrated_features_all_raw.csv
2) Clean symbols: strip spaces, remove '%' (keep negatives), remove commas
3) Convert to numeric (calendar_date excluded)
4) Fill missing: ffill then bfill (binary flags -> fill 0)
5) Normalize per category:
   - Prices & trading structure: Min-Max (0..1)
   - Macro main values: Z-Score
   - Macro days_since_*: Min-Max (0..1)
   - Global risk/market/commodity + IHSG stats: Z-Score
   - Technical indicators: Min-Max (0..1)
   - Calendar flags (is_* / next_is_trading_day): keep 0/1 (no scaling)
6) Write normalized CSV:
   data/raw_features/{TICKER}_integrated_features_all_normalized.csv
7) Write sidecar JSON documenting rules + per-column stats + fill counts

Author: (your name)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Final, List, Dict, Tuple

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR

RAW_FEATURES_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw_features"
TICKERS: Final[List[str]] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]

# Column groups by rule (explicit names used where possible; robust checks below)
PRICE_VOL_COLS: Final[List[str]] = [
    "open_price", "high_price", "low_price", "close_price",
    "volume", "bid_volume", "offer_volume", "foreign_buy", "foreign_sell",
]

MACRO_MAIN_COLS: Final[List[str]] = [
    # US
    "us_fed_fund_rate", "us_core_cpi_mom", "us_gdp_qoq",
    # ID
    "id_bi_rate", "id_core_inflation_yoy", "id_gdp_qoq",
    "id_inflation_mom", "id_retail_sales_yoy",
]

# any column starting with 'days_since_' will be Min-Max
DAYS_SINCE_PREFIX: Final[str] = "days_since_"

GLOBAL_RISK_COLS: Final[List[str]] = [
    # FX / market / risk / commodities
    # Accept either usd_idr or usd_idr_rate depending on earlier pipeline
    "usd_idr", "usd_idr_rate",
    "usd_index_dxy", "sp500_vix", "crude_oil_wti", "gold_futures",
    # IHSG contextual stats
    "ihsg_return_lag_1", "ihsg_std_dev_7", "ihsg_std_dev_14", "ihsg_std_dev_21",
]

TECHNICAL_COLS: Final[List[str]] = [
    "rsi_7", "rsi_14",
    "sma_7", "sma_14", "sma_21", "sma_100",
    "std_dev_7", "std_dev_14", "std_dev_21", "std_dev_100",
]

# Calendar flags: startswith('is_') or equals 'next_is_trading_day'
CALENDAR_FLAG_NAME: Final[str] = "next_is_trading_day"
CALENDAR_PREFIX: Final[str] = "is_"


# =========================
# Helpers
# =========================
def minmax_scale(x: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    arr = x.astype(float).to_numpy(dtype=np.float64)
    xmin = np.nanmin(arr)
    xmax = np.nanmax(arr)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        # empty or all NaN -> return zeros
        return pd.Series(np.zeros_like(arr, dtype=np.float32), index=x.index), {"min": float("nan"), "max": float("nan")}
    rng = xmax - xmin
    if rng == 0:
        # constant column -> zeros
        return pd.Series(np.zeros_like(arr, dtype=np.float32), index=x.index), {"min": float(xmin), "max": float(xmax)}
    scaled = (arr - xmin) / rng
    return pd.Series(scaled.astype(np.float32), index=x.index), {"min": float(xmin), "max": float(xmax)}


def zscore_scale(x: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    arr = x.astype(float).to_numpy(dtype=np.float64)
    mean = np.nanmean(arr)
    std = np.nanstd(arr, ddof=0)
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
        # empty/all NaN or constant -> zeros
        return pd.Series(np.zeros_like(arr, dtype=np.float32), index=x.index), {"mean": float("nan") if not np.isfinite(mean) else float(mean), "std": float(std)}
    scaled = (arr - mean) / std
    return pd.Series(scaled.astype(np.float32), index=x.index), {"mean": float(mean), "std": float(std)}


def is_calendar_flag(col: str) -> bool:
    return col.startswith(CALENDAR_PREFIX) or col == CALENDAR_FLAG_NAME


def clean_to_numeric(s: pd.Series) -> pd.Series:
    """
    Remove '%' and commas, strip spaces.
    Keep leading '-' (negatives).
    """
    cleaned = (
        s.astype(str)
         .str.strip()
         .str.replace("%", "", regex=False)
         .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def group_columns(df_cols: List[str]) -> Dict[str, List[str]]:
    """
    Build column groups present in the dataframe (based on known lists/prefix).
    """
    cols = set(df_cols)

    price_vol = [c for c in PRICE_VOL_COLS if c in cols]
    macro_main = [c for c in MACRO_MAIN_COLS if c in cols]
    days_since = [c for c in df_cols if c.startswith(DAYS_SINCE_PREFIX)]
    global_risk = [c for c in GLOBAL_RISK_COLS if c in cols]
    technical = [c for c in TECHNICAL_COLS if c in cols]
    calendar_flags = [c for c in df_cols if is_calendar_flag(c)]

    return {
        "price_volume_minmax": price_vol,
        "macro_main_zscore": macro_main,
        "macro_days_since_minmax": days_since,
        "global_risk_zscore": global_risk,
        "technical_minmax": technical,
        "calendar_flags_binary": calendar_flags,
    }


def normalize_ticker(ticker: str) -> None:
    in_path = RAW_FEATURES_DIR / f"{ticker}_integrated_features_all_raw.csv"
    out_csv = RAW_FEATURES_DIR / f"{ticker}_integrated_features_all_normalized.csv"
    out_meta = RAW_FEATURES_DIR / f"{ticker}_integrated_features_all_normalized_meta.json"

    if not in_path.exists():
        print(f"[WARN] Input missing for {ticker}: {in_path.name} — skipping.")
        return

    # Read as strings to preserve original symbols before cleaning
    df_raw = pd.read_csv(in_path, dtype=str, encoding="utf-8-sig")
    if "calendar_date" not in df_raw.columns:
        # assume first column is date if mislabeled
        df_raw = df_raw.rename(columns={df_raw.columns[0]: "calendar_date"})

    # Preserve original column order
    col_order = df_raw.columns.tolist()

    # Separate date
    calendar_date = df_raw["calendar_date"].copy()

    # Clean and convert all non-date columns
    work = df_raw.drop(columns=["calendar_date"]).apply(clean_to_numeric)

    # Missing value handling
    # 1) Calendar flags -> 0
    groups = group_columns(work.columns.tolist())
    for c in groups["calendar_flags_binary"]:
        if c in work.columns:
            work[c] = work[c].fillna(0).astype(np.int8)

    # 2) For other numeric columns -> ffill then bfill
    numeric_cols = [c for c in work.columns if c not in groups["calendar_flags_binary"]]
    work[numeric_cols] = work[numeric_cols].ffill().bfill()

    # Build metadata container
    meta: Dict[str, object] = {
        "ticker": ticker,
        "source_file": in_path.name,
        "output_file": out_csv.name,
        "timestamp": datetime.now().isoformat(),
        "normalization_rules": {
            "price_volume": "MinMax (0..1)",
            "macro_days_since": "MinMax (0..1)",
            "macro_main": "Z-Score",
            "global_risk": "Z-Score",
            "technical": "MinMax (0..1)",
            "calendar": "Binary (0/1, unchanged)",
        },
        "columns": {},
        "missing_value_summary": {},
    }

    # Record missing summary before scaling (after fill for flags, before ffill/bfill for others)
    # To provide a more informative report, compute NaN counts before the ffill/bfill we just did.
    # Recompute on a fresh cleaned copy:
    cleaned_for_report = df_raw.drop(columns=["calendar_date"]).apply(clean_to_numeric)
    mv_summary = cleaned_for_report.isna().sum().to_dict()
    meta["missing_value_summary"] = {k: int(v) for k, v in mv_summary.items()}

    # Apply scaling by groups
    scaled = work.copy()

    # MinMax groups
    for c in groups["price_volume_minmax"] + groups["macro_days_since_minmax"] + groups["technical_minmax"]:
        if c in scaled.columns:
            scaled[c], stats = minmax_scale(scaled[c])
            meta["columns"][c] = {"scaler": "minmax", **stats}
    # Z-Score groups
    for c in groups["macro_main_zscore"] + groups["global_risk_zscore"]:
        if c in scaled.columns:
            scaled[c], stats = zscore_scale(scaled[c])
            meta["columns"][c] = {"scaler": "zscore", **stats}
    # Calendar flags: ensure 0/1 ints
    for c in groups["calendar_flags_binary"]:
        if c in scaled.columns:
            scaled[c] = scaled[c].fillna(0).astype(np.int8)
            meta["columns"][c] = {"scaler": "binary", "values": "0/1"}

    # Reassemble with calendar_date first, keep original order afterwards
    out_df = pd.concat([calendar_date, scaled], axis=1)
    # ensure same column order as input (date first)
    ordered_cols = ["calendar_date"] + [c for c in col_order if c != "calendar_date"]
    out_df = out_df.reindex(columns=ordered_cols)

    # Dtypes: floats for scaled, ints for flags
    for c in groups["calendar_flags_binary"]:
        if c in out_df.columns:
            out_df[c] = out_df[c].astype(np.int8)
    # everything else numeric -> float32 (skip calendar_date)
    for c in out_df.columns:
        if c == "calendar_date" or c in groups["calendar_flags_binary"]:
            continue
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce").astype(np.float32)

    # Write outputs
    RAW_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Small console report
    print(f"[OK] Normalized {ticker}:")
    print(f"     -> {out_csv.name} ({len(out_df)} rows, {out_df.shape[1]} cols)")
    print(f"     -> {out_meta.name} (rules + stats)")
    # Quick sanity spot-check
    check_cols = (
        groups["price_volume_minmax"][:2]
        + groups["macro_main_zscore"][:2]
        + groups["technical_minmax"][:2]
        + groups["global_risk_zscore"][:2]
    )
    check_cols = [c for c in check_cols if c in out_df.columns][:6]
    if check_cols:
        desc = out_df[check_cols].describe().loc[["mean", "std", "min", "max"]]
        print(desc.to_string())
    else:
        print("     (No sample columns available for quick stats.)")


def main() -> None:
    for t in TICKERS:
        normalize_ticker(t)


if __name__ == "__main__":
    main()
