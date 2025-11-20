#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for the LSTM (stopping_no_embargo): MEAN-price (AMP) target.

Conventions aligned with train_close.py / train_mean.py:
- Deterministic DataLoader shuffling for train split (though not used here)
- y_min/y_max attach for mean/AMP price (from metadata or RAW fallback)
- Single [SPLIT] print (silence compute_splits internal prints)
- Safe checkpoint load (weights_only=True with fallback)
- IDR-scale RMSE/MAE + stabilized MAPE via metrics.evaluate(...)
- Quiet save_metrics; one-time artifact trailer to console and log
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


def attach_y_minmax_from_metadata_or_raw(ds: Dataset[Any], ticker: str, project_root: Path) -> None:
    """
    Attach y_min/y_max (IDR) for MEAN/AMP PRICE so metrics.evaluate can denormalize.

    Priority:
      1) normalization metadata JSON: take min/max from a plausible mean/AMP field
         (tries: mean_price, amp_price, arithmetic_mean_price, ohlc_mean)
      2) RAW CSV: compute min/max from mean column if present; otherwise compute
         on-the-fly from (open+high+low+close)/4
    """
    # ---- 1) Try metadata JSONs ----
    meta_candidates = [
        project_root / "data" / "raw_features" / f"{ticker}_normalization_report.json",
        project_root / "data" / "raw_features" / f"{ticker}_integrated_features_all_normalized_meta.json",
        project_root / "data" / "raw_features" / f"{ticker}_integrated_features_all_meta.json",
        project_root / "data" / "raw_features" / f"{ticker}_normalization_meta.json",
    ]
    mean_keys = ("mean_price", "amp_price", "arithmetic_mean_price", "ohlc_mean")

    for p in meta_candidates:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                cols = meta.get("columns", {})
                for key in mean_keys:
                    if key in cols:
                        info = cols.get(key, {})
                        y_min = info.get("min", None)
                        y_max = info.get("max", None)
                        if y_min is not None and y_max is not None and float(y_max) > float(y_min):
                            ds.y_min = float(y_min)
                            ds.y_max = float(y_max)
                            print(f"[attach_y_minmax] (metadata) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name} [{key}]")
                            return
            except Exception:
                pass

    # ---- 2) Fallback: RAW CSV ----
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
                # Prefer direct mean/AMP column if present
                for key in mean_keys:
                    if key in df.columns:
                        s = df[key].astype(float).to_numpy()
                        y_min = float(np.min(s)); y_max = float(np.max(s))
                        if y_max > y_min:
                            ds.y_min = y_min; ds.y_max = y_max
                            print(f"[attach_y_minmax] (raw) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name} [{key}]")
                            return
                # Otherwise compute AMP from OHLC
                needed = {"open_price", "high_price", "low_price", "close_price"}
                if needed.issubset(df.columns):
                    s = (df["open_price"].astype(float) +
                         df["high_price"].astype(float) +
                         df["low_price"].astype(float) +
                         df["close_price"].astype(float)) / 4.0
                    y_min = float(np.min(s)); y_max = float(np.max(s))
                    if y_max > y_min:
                        ds.y_min = y_min; ds.y_max = y_max
                        print(f"[attach_y_minmax] (raw, computed AMP) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                        return
            except Exception:
                pass

    print(f"[attach_y_minmax] WARNING: could not determine y_min/y_max for {ticker} (mean). "
          f"Metrics stay on normalized scale for this run.")


# ----------------------------- main ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock code (e.g., BBCA)")
    args = parser.parse_args()

    cfg = Config()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticker = args.ticker
    target_kind = "mean"

    start_msg = f"[LSTM stopping_no_embargo][validate_mean] Running on device: {device}"
    print(start_msg)

    # Dirs & log (ensure exists before writing)
    results_dir = Path("results")
    for sub in ("metrics", "figures", "logs", "checkpoints"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs" / f"{ticker}_validate_mean_terminal_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(start_msg + "\n")

    # Dataset
    ds = load_dataset(ticker, target_kind, cfg)

    # Attach y_min/y_max for IDR-scale metrics (mean/AMP)
    project_root = Path(__file__).resolve().parents[2]
    attach_y_minmax_from_metadata_or_raw(ds, ticker, project_root)
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

    # Loaders (we only need val here, but keep helper parity)
    _, val_dl, _ = make_dataloaders(
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

    # Load best checkpoint safely (MEAN target naming)
    ckpt_path = results_dir / "checkpoints" / f"{ticker}_mean_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    ckpt_msg = f"[validate_mean] Loaded checkpoint: {ckpt_path.name}"
    print(ckpt_msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(ckpt_msg + "\n")

    # Evaluate on validation (IDR scale via explicit y_min/y_max)
    y_true_val, y_pred_val, m_val = evaluate(
        model, val_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "mean_val", m_val, ticker=ticker, verbose=False)

    # Final summary (Windows-safe ASCII)
    val_line = f"[validate_mean] VAL RMSE:{m_val.rmse:.2f} | MAE:{m_val.mae:.2f} | MAPE:{m_val.mape:.2f}%"
    print(val_line)

    done_msg = "[validate_mean] Done."
    print(done_msg)

    # One-time artifact summary (console + log trailer)
    metrics_dir = (results_dir / "metrics").resolve()

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(val_line + "\n")
        f.write(done_msg + "\n")
        f.write(f"[validate_mean] Done on device: {device}\n")
        f.write(f"[save_metrics] Final result artifacts in: {metrics_dir}\n")
        f.write(f"[save_metrics] Files: "
                f"{ticker}_mean_val_metrics.(json|txt)\n")
        f.write(f"{'='*80}\n\n")

    print(f"[save_metrics] Final result artifacts in: {metrics_dir}")
    print(f"[save_metrics] Files: "
          f"{ticker}_mean_val_metrics.(json|txt)")


if __name__ == "__main__":
    main()
