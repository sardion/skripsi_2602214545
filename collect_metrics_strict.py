#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict metrics collector.

CSV columns (exact order):
arch,scenario,target,split,ticker,rmse,mae,mape,cpu_util_pct,peak_cpu_mem_mb,total_elapsed_s,mtime_iso
"""

from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ARCH_HINTS = {"lstm", "tcn", "tft"}
SCENARIO_HINTS = {"stopping_embargo", "stopping_no_embargo", "efficiency"}

def find_arch_and_scenario(p: Path) -> Tuple[str, str]:
    parts = [x.lower() for x in p.parts]
    arch = next((x for x in parts if x in ARCH_HINTS), "unknown")
    scenario = next((x for x in parts if x in SCENARIO_HINTS), "unknown")
    return arch, scenario

def parse_split(split_field: str) -> Tuple[str, str]:
    # e.g. "close_val" -> ("close","val"); "mean_test" -> ("mean","test")
    if "_" in split_field:
        a, b = split_field.split("_", 1)
        target = a.lower().strip()
        split = b.lower().strip()
    else:
        target = "unknown"
        split = split_field.lower().strip()
    if target not in {"close", "mean"}:
        target = "unknown"
    if split not in {"val", "test"}:
        split = "unknown"
    return target, split

def read_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def try_read_float(p: Path) -> Optional[float]:
    try:
        s = p.read_text(encoding="utf-8").strip()
        if not s:
            return None
        return float(s.split()[0])
    except Exception:
        return None

def mtime_iso(p: Path) -> str:
    try:
        ts = p.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return ""

def scenario_root_from_metrics(metrics_path: Path) -> Optional[Path]:
    # Walk up until we hit .../results; scenario root is its parent.
    for parent in metrics_path.parents:
        if parent.name == "results":
            return parent.parent
    return None

def efficiency_fields(metrics_path: Path, ticker: str, target: str, scenario: str) -> Tuple[str, str, str]:
    """
    Return (cpu_util_pct, peak_cpu_mem_mb, total_elapsed_s) as strings for CSV.
    Only populated for 'efficiency' scenario; otherwise "".
    """
    if scenario != "efficiency":
        return "", "", ""

    scen_root = scenario_root_from_metrics(metrics_path)
    if scen_root is None:
        return "", "", ""

    logs = scen_root / "results" / "logs"

    cpu = try_read_float(logs / f"{ticker}_efficiency_{target}_cpu_utilization_pct.txt")
    peak = try_read_float(logs / f"{ticker}_efficiency_{target}_peak_cpu_memory_mb.txt")
    total = try_read_float(logs / f"{ticker}_efficiency_{target}_total_elapsed_s.txt")

    # Convert Nones to ""
    def _fmt(x: Optional[float]) -> str:
        return "" if x is None else f"{x:.2f}"

    return _fmt(cpu), _fmt(peak), _fmt(total)

def collect_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for j in root.rglob("*_metrics.json"):
        data = read_json(j)
        if not data:
            continue

        split_field = str(data.get("split", ""))
        target, split = parse_split(split_field)

        arch, scenario = find_arch_and_scenario(j)
        ticker = str(data.get("ticker", "")).upper()

        # metrics (write as raw numbers -> str for CSV)
        try:
            rmse = f"{float(data.get('rmse', float('nan'))):.6f}"
        except Exception:
            rmse = ""
        try:
            mae = f"{float(data.get('mae', float('nan'))):.6f}"
        except Exception:
            mae = ""
        try:
            mape = f"{float(data.get('mape', float('nan'))):.6f}"
        except Exception:
            mape = ""

        cpu_util_pct, peak_cpu_mem_mb, total_elapsed_s = efficiency_fields(j, ticker, target, scenario)
        row = {
            "arch": arch,
            "scenario": scenario,
            "target": target,
            "split": split,
            "ticker": ticker,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "cpu_util_pct": cpu_util_pct,
            "peak_cpu_mem_mb": peak_cpu_mem_mb,
            "total_elapsed_s": total_elapsed_s,
            "mtime_iso": mtime_iso(j),
        }
        rows.append(row)

    # stable ordering
    rows.sort(key=lambda r: (r["ticker"], r["target"], r["split"], r["arch"], r["scenario"]))
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root to scan.")
    ap.add_argument("--out", default="all_metrics_summary.csv", help="Output CSV path.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = collect_rows(root)

    cols = [
        "arch","scenario","target","split","ticker",
        "rmse","mae","mape",
        "cpu_util_pct","peak_cpu_mem_mb","total_elapsed_s",
        "mtime_iso",
    ]
    outp = Path(args.out).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"[collector] wrote {outp}")

if __name__ == "__main__":
    main()
