#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrate IDX structural data and Yahoo Finance price data per ticker
on pure calendar date (YYYY-MM-DD), ignoring time / timezone.

Adds diagnostics: for each ticker, prints which dates are only in IDX,
only in Yahoo, and confirms final row counts.
"""

from __future__ import annotations

import os
import sys
from typing import List, Set
import pandas as pd
from pandas import DataFrame, Series


# =====================================================================================
# HARD-CODED CONFIG
# =====================================================================================

SCRIPT_DIR: str
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

IDX_INPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "idx_ticker_summary_csv")
YF_INPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "yf_ticker_summary_csv")
OUTPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "integrated_ticker_summary_csv")
_make_out_dir_none: None = os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TICKERS: List[str] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]

IDX_DATE_COL: str = "Trading Date"
YF_DATE_COL: str = "Trading Date"

FINAL_COL_ORDER: List[str] = [
    IDX_DATE_COL,
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Bid Volume",
    "Offer Volume",
    "Foreign Sell",
    "Foreign Buy",
]


# =====================================================================================
# DATE NORMALIZATION HELPERS
# =====================================================================================

def _normalize_idx_date_col_to_date(series: Series) -> Series:
    dt_parsed: Series = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    only_date: Series = dt_parsed.dt.date
    return only_date


def _normalize_yf_date_col_to_date(series: Series) -> Series:
    dt_utc: Series = pd.to_datetime(series, errors="coerce", utc=True)
    dt_local: Series = dt_utc.dt.tz_convert("Asia/Jakarta")
    dt_naive_local: Series = dt_local.dt.tz_localize(None)
    only_date: Series = dt_naive_local.dt.date
    return only_date


# =====================================================================================
# LOADING HELPERS
# =====================================================================================

def _load_idx_history(ticker: str) -> DataFrame:
    idx_path: str = os.path.join(IDX_INPUT_DIR, f"idx_{ticker}.csv")

    if not os.path.isfile(idx_path):
        print(f"[WARN] IDX file not found for {ticker}: {idx_path}")
        return pd.DataFrame()

    df_idx: DataFrame = pd.read_csv(idx_path)

    if IDX_DATE_COL not in df_idx.columns and "Unnamed: 0" in df_idx.columns:
        df_idx = df_idx.rename(columns={"Unnamed: 0": IDX_DATE_COL})

    if IDX_DATE_COL not in df_idx.columns:
        print(f"[WARN] '{IDX_DATE_COL}' not found in {idx_path}")
        return pd.DataFrame()

    df_idx[IDX_DATE_COL] = _normalize_idx_date_col_to_date(df_idx[IDX_DATE_COL])
    df_idx = df_idx.dropna(subset=[IDX_DATE_COL])
    df_idx = df_idx.sort_values(by=[IDX_DATE_COL], ascending=True)

    return df_idx


def _load_yf_history(ticker: str) -> DataFrame:
    yf_path: str = os.path.join(YF_INPUT_DIR, f"yf_{ticker}_raw.csv")

    if not os.path.isfile(yf_path):
        print(f"[WARN] Yahoo Finance file not found for {ticker}: {yf_path}")
        return pd.DataFrame()

    df_yf: DataFrame = pd.read_csv(yf_path)

    if YF_DATE_COL not in df_yf.columns and "Unnamed: 0" in df_yf.columns:
        df_yf = df_yf.rename(columns={"Unnamed: 0": YF_DATE_COL})
    if YF_DATE_COL not in df_yf.columns and "Date" in df_yf.columns:
        df_yf = df_yf.rename(columns={"Date": YF_DATE_COL})

    if YF_DATE_COL not in df_yf.columns:
        print(f"[WARN] '{YF_DATE_COL}' not found in {yf_path}")
        return pd.DataFrame()

    df_yf[YF_DATE_COL] = _normalize_yf_date_col_to_date(df_yf[YF_DATE_COL])
    df_yf = df_yf.dropna(subset=[YF_DATE_COL])
    df_yf = df_yf.sort_values(by=[YF_DATE_COL], ascending=True)

    if "Volume" in df_yf.columns:
        df_yf = df_yf.rename(columns={"Volume": "Volume_yf"})

    return df_yf


# =====================================================================================
# MERGE + SAVE HELPERS
# =====================================================================================

def _merge_idx_yf(df_idx: DataFrame, df_yf: DataFrame) -> DataFrame:
    yf_keep_cols_all: List[str] = [YF_DATE_COL, "Open", "High", "Low", "Close"]
    yf_keep_cols: List[str] = [c for c in yf_keep_cols_all if c in df_yf.columns]
    df_yf_sub: DataFrame = df_yf[yf_keep_cols].copy()

    idx_keep_cols_all: List[str] = [
        IDX_DATE_COL,
        "Volume",
        "Bid Volume",
        "Offer Volume",
        "Foreign Sell",
        "Foreign Buy",
    ]
    idx_keep_cols: List[str] = [c for c in idx_keep_cols_all if c in df_idx.columns]
    df_idx_sub: DataFrame = df_idx[idx_keep_cols].copy()

    df_merged: DataFrame = pd.merge(
        left=df_yf_sub,
        right=df_idx_sub,
        left_on=YF_DATE_COL,
        right_on=IDX_DATE_COL,
        how="inner",
    )

    # unify Trading Date
    if "Trading Date_x" in df_merged.columns and "Trading Date_y" in df_merged.columns:
        df_merged[IDX_DATE_COL] = df_merged["Trading Date_x"]
        df_merged = df_merged.drop(columns=["Trading Date_x", "Trading Date_y"])
    elif "Trading Date_x" in df_merged.columns:
        df_merged[IDX_DATE_COL] = df_merged["Trading Date_x"]
        df_merged = df_merged.drop(columns=["Trading Date_x"])
    elif "Trading Date_y" in df_merged.columns:
        df_merged[IDX_DATE_COL] = df_merged["Trading Date_y"]
        df_merged = df_merged.drop(columns=["Trading Date_y"])
    elif IDX_DATE_COL not in df_merged.columns and YF_DATE_COL in df_merged.columns:
        df_merged = df_merged.rename(columns={YF_DATE_COL: IDX_DATE_COL})

    for col_name in FINAL_COL_ORDER:
        if col_name not in df_merged.columns:
            df_merged[col_name] = pd.NA

    df_final: DataFrame = df_merged[FINAL_COL_ORDER].copy()
    df_final = df_final.sort_values(by=[IDX_DATE_COL], ascending=True)

    return df_final


def _save_integrated_csv(ticker: str, df_final: DataFrame) -> None:
    df_out: DataFrame = df_final.set_index(IDX_DATE_COL)

    out_name: str = f"integrated_{ticker}.csv"
    out_path: str = os.path.join(OUTPUT_DIR, out_name)

    try:
        df_out.to_csv(out_path, index=True)
        print(f"[OK] Saved {out_name} with {len(df_out)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to save {out_path}: {e}")


# =====================================================================================
# MAIN PIPE FOR ONE TICKER (WITH DIAGNOSTICS)
# =====================================================================================

def process_single_ticker(ticker: str) -> None:
    print(f"[TICKER] Integrating {ticker} ...")

    df_idx: DataFrame = _load_idx_history(ticker)
    df_yf: DataFrame = _load_yf_history(ticker)

    if df_idx.empty:
        print(f"[WARN] Skipping ticker {ticker}: IDX data empty.")
        return
    if df_yf.empty:
        print(f"[WARN] Skipping ticker {ticker}: Yahoo data empty.")
        return

    # --- Diagnostics before merge ---
    idx_dates_set: Set = set(df_idx[IDX_DATE_COL].tolist())
    yf_dates_set: Set = set(df_yf[YF_DATE_COL].tolist())

    intersection_dates_set: Set = idx_dates_set.intersection(yf_dates_set)
    idx_only_dates_set: Set = idx_dates_set.difference(yf_dates_set)
    yf_only_dates_set: Set = yf_dates_set.difference(idx_dates_set)

    print(f"[DEBUG] {ticker}: IDX rows            = {len(df_idx)}")
    print(f"[DEBUG] {ticker}: YF rows             = {len(df_yf)}")
    print(f"[DEBUG] {ticker}: Intersection rows   = {len(intersection_dates_set)}")
    print(f"[DEBUG] {ticker}: IDX-only dates      = {sorted(list(idx_only_dates_set))[:10]}")
    print(f"[DEBUG] {ticker}: YF-only dates       = {sorted(list(yf_only_dates_set))[:10]}")

    # Merge
    df_final: DataFrame = _merge_idx_yf(df_idx=df_idx, df_yf=df_yf)

    if df_final.empty:
        print(f"[WARN] No overlapping Trading Date for {ticker}, nothing to save.")
        return

    print(f"[DEBUG] {ticker}: Final merged rows  = {len(df_final)}")

    _save_integrated_csv(ticker, df_final)


# =====================================================================================
# SCRIPT MAIN
# =====================================================================================

def main() -> None:
    if not os.path.isdir(IDX_INPUT_DIR):
        print(f"[FATAL] IDX_INPUT_DIR does not exist: {IDX_INPUT_DIR}")
        sys.exit(1)

    if not os.path.isdir(YF_INPUT_DIR):
        print(f"[FATAL] YF_INPUT_DIR does not exist: {YF_INPUT_DIR}")
        sys.exit(1)

    if not os.path.isdir(OUTPUT_DIR):
        print(f"[INFO] OUTPUT_DIR does not exist, creating: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ticker in TARGET_TICKERS:
        process_single_ticker(ticker)

    print("[DONE] IDX + Yahoo Finance integration complete.")


if __name__ == "__main__":
    main()
