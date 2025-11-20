#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert all IDX 'Stock Summary-YYYYMMDD.xlsx' files into CSV with an added 'Trading Date' column.

Behavior:
- Reads every .xlsx file in ./data/raw/idx_stock_summary_xlsx
- Extracts the date (YYYYMMDD) from the filename
- Inserts a new column 'Trading Date' RIGHT AFTER the 'No' column
  and fills it with that date for all rows
- Saves into ./data/raw/idx_stock_summary_csv with the same base name but .csv

This script is intentionally self-contained (no external helpers, no config modules)
to maximize transparency and reproducibility for academic review.
"""

from __future__ import annotations  # allows forward references in type hints

import os
import sys
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame


# =====================================================================================
# HARD-CODED PATHS (relative to where this script lives)
# =====================================================================================

# Resolve the folder where this script is currently located.
# In Jupyter/Colab, __file__ won't exist, so we fall back to os.getcwd().
SCRIPT_DIR: str
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

INPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "idx_stock_summary_xlsx")
OUTPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "idx_stock_summary_csv")

# Ensure output directory exists
_ensure_dir_ok: None = os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================================
# INLINE LOGIC
# =====================================================================================

def extract_trading_date_from_filename(filename: str) -> datetime:
    """
    Given a filename like 'Stock Summary-20200106.xlsx',
    extract the '20200106' part and parse it into a datetime.

    Assumptions:
    - There's exactly one dash before the date segment.
    - Date is in YYYYMMDD format.

    Returns:
        datetime: naive datetime (no timezone info)
    Raises:
        ValueError: if filename doesn't match the expected pattern.
    """
    base_name: str = os.path.basename(filename)

    # Remove extension
    if base_name.lower().endswith(".xlsx"):
        base_no_ext: str = base_name[:-5]  # strip ".xlsx"
    else:
        # We only expect .xlsx but guard anyway
        base_no_ext = os.path.splitext(base_name)[0]

    # Split at the last '-' and assume the last part is the date.
    # Example: "Stock Summary-20200106" -> ["Stock Summary", "20200106"]
    parts: List[str] = base_no_ext.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot split date from filename: {filename}")

    date_str: str = parts[1].strip()

    # Parse YYYYMMDD -> datetime
    # Example: "20200106" -> datetime(2020, 1, 6)
    if len(date_str) != 8 or (not date_str.isdigit()):
        raise ValueError(
            f"Date segment '{date_str}' in filename '{filename}' "
            f"is not in YYYYMMDD format"
        )

    year: int = int(date_str[0:4])
    month: int = int(date_str[4:6])
    day: int = int(date_str[6:8])

    parsed_dt: datetime = datetime(year, month, day)
    return parsed_dt


def insert_trading_date_column(df: DataFrame, trading_date_dt: datetime) -> DataFrame:
    """
    Insert a new column 'Trading Date' right AFTER the column 'No'.

    - trading_date_dt will be inserted as a pandas-friendly datetime64[ns]
      (same value repeated).
    - If 'No' doesn't exist, we will still add 'Trading Date' as the first column.

    Returns:
        DataFrame: new DataFrame with 'Trading Date' inserted.
    """
    # Repeat the same timestamp for all rows
    trading_date_series: List[datetime] = [trading_date_dt] * len(df.index)

    if "No" in df.columns:
        no_idx: int = df.columns.get_loc("No")  # integer position of "No"

        # Build new columns order manually:
        # left_part + ["Trading Date"] + right_part
        new_cols: List[str] = []
        for i, col in enumerate(df.columns):
            new_cols.append(col)
            if i == no_idx:
                new_cols.append("Trading Date")

        # Create augmented DataFrame
        df_out: DataFrame = df.copy()
        df_out["Trading Date"] = trading_date_series

        # Reorder columns so Trading Date appears after "No"
        df_out = df_out[new_cols]
    else:
        # Fallback: put Trading Date as the first column
        df_out = df.copy()
        df_out["Trading Date"] = trading_date_series

        reordered_cols: List[str] = ["Trading Date"] + [c for c in df.columns]
        df_out = df_out[reordered_cols]

    return df_out


def convert_single_file(xlsx_path: str, output_dir: str) -> None:
    """
    Read one XLSX, add Trading Date column, write CSV.

    Args:
        xlsx_path (str): full path to source XLSX file
        output_dir (str): directory where CSV should be written
    """
    # Extract date from filename
    trading_date_dt: datetime = extract_trading_date_from_filename(xlsx_path)

    # Read Excel into DataFrame
    # We let pandas infer headers exactly as in the original file.
    # If the Excel file has multiple sheets, we assume the first sheet.
    try:
        df_raw: DataFrame = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception as e:
        print(f"[ERROR] Failed to read '{xlsx_path}': {e}")
        return

    # Insert Trading Date column after 'No'
    df_aug: DataFrame = insert_trading_date_column(df_raw, trading_date_dt)

    # Prepare output filename
    base_name: str = os.path.basename(xlsx_path)
    base_no_ext: str
    _ext: str
    base_no_ext, _ext = os.path.splitext(base_name)

    csv_name: str = base_no_ext + ".csv"
    csv_path: str = os.path.join(output_dir, csv_name)

    # Write CSV (UTF-8, include header, exclude index)
    try:
        _none_result: None = df_aug.to_csv(csv_path, index=False)
        print(f"[OK] {base_name} -> {csv_name} (rows: {len(df_aug)})")
    except Exception as e:
        print(f"[ERROR] Failed to write '{csv_path}': {e}")


# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main() -> None:
    # Basic sanity checks
    if not os.path.isdir(INPUT_DIR):
        print(f"[FATAL] INPUT_DIR does not exist: {INPUT_DIR}")
        sys.exit(1)

    if not os.path.isdir(OUTPUT_DIR):
        print(f"[INFO] OUTPUT_DIR does not exist, creating: {OUTPUT_DIR}")
        _mk: None = os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List all files in input dir
    all_files: List[str] = sorted(os.listdir(INPUT_DIR))

    # Filter only .xlsx
    xlsx_files: List[str] = [f for f in all_files if f.lower().endswith(".xlsx")]

    if not xlsx_files:
        print(f"[WARN] No .xlsx files found in {INPUT_DIR}")
        sys.exit(0)

    # Process each file
    for fname in xlsx_files:
        full_path: str = os.path.join(INPUT_DIR, fname)
        convert_single_file(full_path, OUTPUT_DIR)

    print("[DONE] Conversion completed.")


if __name__ == "__main__":
    main()
