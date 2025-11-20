#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for TFT (stopping_embargo): CLOSE-price target.

Conventions aligned with LSTM/TFTLite:
- Deterministic DataLoader shuffling seed for parity
- y_min/y_max attach for close price (metadata JSON → RAW fallback)
- Single [SPLIT] print (silence compute_splits internal prints)
- Safe checkpoint load (weights_only=True with fallback)
- IDR-scale RMSE/MAE + stabilized MAPE via metrics.evaluate(...)
- Quiet save_metrics; one-time artifact trailer
- Uses _EvalAdapter so metrics.evaluate(model, loader, ...) works with dict batches
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import warnings
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
    module="torch.nn.modules.rnn"
)

from config import Config
from dataloader import load_dataset_tft
from splitter import compute_splits
from model import TFTLite
from metrics import evaluate, save_metrics


# ----------------------------- helpers ----------------------------- #

def make_dataloaders(
    ds: Dataset[Any],
    splits: Tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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


class _EvalAdapter(torch.nn.Module):
    """Adapter so metrics.evaluate(model, loader, ...) — which calls model(X) — works with dict batches."""
    def __init__(self, core: TFTLite):
        super().__init__()
        self.core = core

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.core(batch["past"], batch["future"], batch["static"])


def attach_y_minmax_from_metadata_or_raw(ds: Dataset[Any], ticker: str, project_root: Path) -> None:
    """
    Attach y_min/y_max (IDR) for CLOSE PRICE so metrics.evaluate can denormalize.

    Priority:
      1) normalization metadata JSON: columns.close_price.min/max
      2) RAW CSV: compute min/max from 'close_price'
    """
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
                close_info = cols.get("close_price", {})
                y_min = close_info.get("min", None)
                y_max = close_info.get("max", None)
                if y_min is not None and y_max is not None and float(y_max) > float(y_min):
                    ds.y_min = float(y_min)
                    ds.y_max = float(y_max)
                    print(f"[attach_y_minmax] (metadata) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                    return
            except Exception:
                pass

    raw_candidates = [
        project_root / "data" / "raw_features" / f"{ticker}_integrated_features_all_raw.csv",
        project_root / "data" / "raw" / f"{ticker}_integrated_features_all_raw.csv",
        project_root / "data" / "raw" / f"{ticker}_integrated_features_all.csv",
    ]
    for p in raw_candidates:
        if p.exists():
            try:
                import pandas as pd
                s = pd.to_numeric(pd.read_csv(p)["close_price"], errors="coerce").dropna().to_numpy()
                if s.size == 0:
                    continue
                y_min = float(np.min(s)); y_max = float(np.max(s))
                if y_max > y_min:
                    ds.y_min = y_min; ds.y_max = y_max
                    print(f"[attach_y_minmax] (raw) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                    return
            except Exception:
                pass

    print(f"[attach_y_minmax] WARNING: could not determine y_min/y_max for {ticker}. "
          f"Metrics stay on normalized scale for this run.")


# ----------------------------- main ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock code (e.g., BBCA)")
    args = parser.parse_args()

    cfg = Config()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticker = args.ticker

    start_msg = f"[TFT stopping_embargo][validate_close] Running on device: {device}"
    print(start_msg)

    # Dirs & log
    results_dir = Path("results")
    for sub in ("metrics", "figures", "logs", "checkpoints"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs" / f"{ticker}_validate_close_terminal_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(start_msg + "\n")

    # Dataset (TFT-specific)
    target_kind = "close"
    ds = load_dataset_tft(ticker, target_kind, cfg)

    # Attach y_min/y_max for IDR-scale metrics
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

    # Loaders (validation only needed, keep helper for parity)
    _, val_dl, _ = make_dataloaders(
        ds, splits, cfg.batch_size, num_workers=cfg.num_workers, seed=getattr(cfg, "random_seed", 42)
    )

    # Infer dims and build model + adapter
    sample_X, _ = ds[0]
    d_past   = int(getattr(ds, "d_past",   sample_X["past"].shape[-1]))
    d_fut    = int(getattr(ds, "d_fut",    sample_X["future"].shape[-1]))
    d_static = int(getattr(ds, "d_static", sample_X["static"].shape[-1]))

    core = TFTLite(
        d_past=d_past,
        d_fut=d_fut,
        d_static=d_static,
        hidden=cfg.hidden_size,
        n_heads=cfg.attention_heads,
        enc_layers=cfg.enc_layers,
        dropout=cfg.dropout,
    ).to(device)
    model = _EvalAdapter(core)

    # Load best checkpoint safely
    ckpt_path = results_dir / "checkpoints" / f"{ticker}_close_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)
    core.load_state_dict(state_dict)
    ckpt_msg = f"[validate_close] Loaded checkpoint: {ckpt_path.name}"
    print(ckpt_msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(ckpt_msg + "\n")

    # Evaluate on validation (IDR scale via explicit y_min/y_max)
    y_true_val, y_pred_val, m_val = evaluate(
        model, val_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "close_val", m_val, ticker=ticker, verbose=False)

    # Final summary
    val_line = f"[validate_close] VAL RMSE:{m_val.rmse:.2f} | MAE:{m_val.mae:.2f} | MAPE:{m_val.mape:.2f}%"
    print(val_line)
    print("[validate_close] Done.")

    # One-time artifact summary (console + log trailer)
    metrics_dir = (results_dir / "metrics").resolve()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(val_line + "\n")
        f.write("[validate_close] Done.\n")
        f.write(f"[validate_close] Done on device: {device}\n")
        f.write(f"[save_metrics] Final result artifacts in: {metrics_dir}\n")
        f.write(f"[save_metrics] Files: {ticker}_close_val_metrics.(json|txt)\n")
        f.write(f"{'='*80}\n\n")

    print(f"[save_metrics] Final result artifacts in: {metrics_dir}")
    print(f"[save_metrics] Files: {ticker}_close_val_metrics.(json|txt)")


if __name__ == "__main__":
    main()
