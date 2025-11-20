#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM stopping_embargo: test script for MEAN target (IDR-scale metrics).

Matches train_mean.py / train_close.py conventions:
- Deterministic make_dataloaders(...)
- y_min/y_max attach for MEAN target (from metadata if present; else RAW fallback)
- RAW fallback computes MEAN as (open+high+low+close)/4 when column is absent
- Safe checkpoint load (weights_only=True with fallback)
- Single [SPLIT] line (silences compute_splits' internal prints)
- Quiet save_metrics; a single artifact summary trailer (console + log)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from typing import Any, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from config import Config
from dataloader import load_dataset
from splitter import compute_splits
from model import LSTMRegressor
from metrics import evaluate, save_metrics
from figures import plot_pred_vs_true


# ----------------------------- helpers ----------------------------- #

def make_dataloaders(
    ds: Dataset[Any],
    splits: Tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders from contiguous index ranges,
    mirroring train_mean.py (deterministic shuffle for train).
    """
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = splits

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_dl = DataLoader(
        Subset(ds, range(tr_s, tr_e)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        generator=g,
    )
    val_dl = DataLoader(
        Subset(ds, range(va_s, va_e)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        Subset(ds, range(te_s, te_e)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_dl, val_dl, test_dl


def attach_y_minmax_for_mean(ds: Dataset[Any], ticker: str, project_root: Path) -> None:
    """
    Attach y_min/y_max (IDR) for MEAN PRICE so metrics.evaluate can denormalize.

    Priority:
      1) normalization metadata JSON: columns.mean_price.min/max (if present)
      2) RAW CSV:
         - If 'mean_price' column exists, use its min/max
         - Else compute mean_price = (open+high+low+close)/4 then use its min/max
    """
    # Try metadata first (some pipelines may store 'mean_price' stats)
    meta_candidates = [
        project_root / "data" / "raw_features" / f"{ticker}_normalization_report.json",
        project_root / "data" / "raw_features" / f"{ticker}_integrated_features_all_normalized_meta.json",
        project_root / "data" / "raw_features" / f"{ticker}_integrated_features_all_meta.json",
        project_root / "data" / "raw_features" / f"{ticker}_normalization_meta.json",
    ]
    for p in meta_candidates:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                cols = meta.get("columns", {})
                # Prefer explicit mean_price stats if available
                mean_info = cols.get("mean_price", {})
                y_min = mean_info.get("min", None)
                y_max = mean_info.get("max", None)
                if y_min is not None and y_max is not None and float(y_max) > float(y_min):
                    ds.y_min = float(y_min)
                    ds.y_max = float(y_max)
                    print(f"[attach_y_minmax] (metadata) {ticker} MEAN: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                    return
            except Exception:
                pass  # fall back to RAW

    # RAW fallback
    raw_candidates = [
        project_root / "data" / "raw_features" / f"{ticker}_integrated_features_all_raw.csv",
        project_root / "data" / "raw" / f"{ticker}_integrated_features_all_raw.csv",
        project_root / "data" / "raw" / f"{ticker}_integrated_features_all.csv",
    ]
    for p in raw_candidates:
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_csv(p)
                if "mean_price" in df.columns:
                    s = df["mean_price"].astype(float).to_numpy()
                else:
                    # Compute arithmetic mean of OHLC if dedicated column is absent
                    for col in ("open_price", "high_price", "low_price", "close_price"):
                        if col not in df.columns:
                            raise KeyError(f"Missing column '{col}' in RAW for mean computation: {p}")
                    s = (
                        df["open_price"].astype(float)
                        + df["high_price"].astype(float)
                        + df["low_price"].astype(float)
                        + df["close_price"].astype(float)
                    ).to_numpy() / 4.0

                y_min = float(np.min(s))
                y_max = float(np.max(s))
                if y_max > y_min:
                    ds.y_min = y_min
                    ds.y_max = y_max
                    src_col = "mean_price" if "mean_price" in df.columns else "OHLC/4"
                    print(f"[attach_y_minmax] (raw) {ticker} MEAN: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name} ({src_col})")
                    return
            except Exception:
                pass

    print(f"[attach_y_minmax] WARNING: could not determine y_min/y_max for {ticker} MEAN. "
          f"Metrics stay on normalized scale for this run.")


# ----------------------------- main ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock code, e.g., BBCA")
    args = parser.parse_args()

    cfg = Config()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticker = args.ticker
    target_kind = "mean"

    start_msg = f"[LSTM stopping_embargo][test_mean] Running on device: {device}"
    print(start_msg)

    # Dirs & test log path (ensure exists before writing)
    results_dir = Path("results")
    for sub in ("metrics", "figures", "logs", "checkpoints"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs" / f"{ticker}_test_mean_terminal_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(start_msg + "\n")

    # Dataset
    ds = load_dataset(ticker, target_kind, cfg)

    # Attach y_min/y_max for IDR-scale metrics (mean-aware)
    project_root = Path(__file__).resolve().parents[2]
    attach_y_minmax_for_mean(ds, ticker, project_root)
    debug_msg = (f"[debug] has y_min/y_max? {hasattr(ds,'y_min')} {hasattr(ds,'y_max')} "
                 f"vals: {getattr(ds,'y_min',None)} {getattr(ds,'y_max',None)}")
    print(debug_msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(debug_msg + "\n")

    # Splits (silence internal prints; print once)
    with contextlib.redirect_stdout(io.StringIO()):
        splits = compute_splits(
            n_samples=len(ds),
            embargo_days=cfg.embargo_days,
            train_ratio=0.7,
            val_ratio=0.15,
        )
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = splits
    split_msg = (f"[SPLIT] n={len(ds)} | embargo={cfg.embargo_days} | "
                 f"train=({tr_s},{tr_e}) len={tr_e-tr_s} | "
                 f"val=({va_s},{va_e}) len={va_e-va_s} | "
                 f"test=({te_s},{te_e}) len={te_e-te_s}")
    print(split_msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(split_msg + "\n")

    # Dataloaders (reuse same helper)
    _, val_dl, test_dl = make_dataloaders(
        ds, splits, cfg.batch_size, num_workers=cfg.num_workers, seed=getattr(cfg, "random_seed", 42)
    )

    # Model (shape must match training)
    X0, _ = ds[0]
    input_dim = int(X0.shape[-1])
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    # Load best checkpoint safely
    ckpt_path = results_dir / "checkpoints" / f"{ticker}_mean_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    ckpt_msg = f"[test_mean] Loaded checkpoint: {ckpt_path.name}"
    print(ckpt_msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(ckpt_msg + "\n")

    # Evaluate (IDR scale via explicit y_min/y_max)
    y_true_val, y_pred_val, m_val = evaluate(
        model, val_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "mean_val", m_val, ticker=ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_val, y_pred_val, "mean_val", ticker=ticker)

    y_true_test, y_pred_test, m_test = evaluate(
        model, test_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "mean_test", m_test, ticker=ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_test, y_pred_test, "mean_test", ticker=ticker)

    # Final summary (Windows-safe ASCII)
    val_line = f"[test_mean] VAL  RMSE:{m_val.rmse:.2f} | MAE:{m_val.mae:.2f} | MAPE:{m_val.mape:.2f}%"
    test_line = f"[test_mean] TEST RMSE:{m_test.rmse:.2f} | MAE:{m_test.mae:.2f} | MAPE:{m_test.mape:.2f}%"
    print(val_line)
    print(test_line)

    done_msg = "[test_mean] Done."
    print(done_msg)

    # One-time artifact summary (console + log trailer)
    metrics_dir = (results_dir / "metrics").resolve()

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(val_line + "\n")
        f.write(test_line + "\n")
        f.write(done_msg + "\n")
        f.write(f"[test_mean] Done on device: {device}\n")
        f.write(f"[save_metrics] Final result artifacts in: {metrics_dir}\n")
        f.write(f"[save_metrics] Files: "
                f"{ticker}_mean_val_metrics.(json|txt), "
                f"{ticker}_mean_test_metrics.(json|txt)\n")
        f.write(f"{'='*80}\n\n")

    print(f"[save_metrics] Final result artifacts in: {metrics_dir}")
    print(f"[save_metrics] Files: "
          f"{ticker}_mean_val_metrics.(json|txt), "
          f"{ticker}_mean_test_metrics.(json|txt)")


if __name__ == "__main__":
    main()
