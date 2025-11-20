#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM training script (stopping_no_embargo): close-price target.

Changes in this version:
- Suppress duplicate split prints (silence compute_splits; print once here).
- Quiet save_metrics during loop & plots; print a single artifact summary at the end.
- Labeled val line: val=RMSE:...|MAE:...|MAPE:...%
- Denormalization-aware evaluation via y_min/y_max attach (from JSON or RAW CSV).
- Safe checkpoint loading (weights_only=True with fallback).
- Reproducible DataLoader shuffling (torch.Generator).
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
from torch import nn, optim, amp
from torch.utils.data import DataLoader, Subset, Dataset

from config import Config
from dataloader import load_dataset
from splitter import compute_splits
from model import LSTMRegressor
from metrics import EvalResult, save_metrics, evaluate
from figures import plot_pred_vs_true, plot_loss_curve


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


def forward_pass(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    scaler: amp.GradScaler | None,
) -> torch.Tensor:
    X = X.to(device)
    enabled = (device.type == "cuda") and (scaler is not None) and getattr(scaler, "is_enabled", lambda: False)()
    with amp.autocast(device_type="cuda", enabled=enabled):
        yhat = model(X).squeeze(-1)
    return yhat


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
                s = pd.read_csv(p)["close_price"].astype(float).to_numpy()
                y_min = float(np.min(s))
                y_max = float(np.max(s))
                if y_max > y_min:
                    ds.y_min = y_min
                    ds.y_max = y_max
                    print(f"[attach_y_minmax] (raw) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                    return
            except Exception:
                pass

    print(f"[attach_y_minmax] WARNING: could not determine y_min/y_max for {ticker}. "
          f"Metrics stay on normalized scale for this run.")


# ----------------------------- main ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock code, e.g., BBCA")
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(getattr(cfg, "random_seed", 42))
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_msg = (f"[train_close][LSTM stopping_no_embargo] Running on device: {device}")
    print(start_msg)

    # Dirs & log
    results_dir = Path("results")
    for sub in ("metrics", "figures", "checkpoints", "logs"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs" / f"{args.ticker}_train_close_terminal_log.txt"

    # Dataset
    target_kind: str = "close"
    ds = load_dataset(args.ticker, target_kind, cfg)

    # Attach y_min/y_max for denorm
    project_root = Path(__file__).resolve().parents[2]
    attach_y_minmax_from_metadata_or_raw(ds, args.ticker, project_root)
    debug_msg = (f"[debug] has y_min/y_max? {hasattr(ds,'y_min')} {hasattr(ds,'y_max')} "
          f"vals: {getattr(ds,'y_min',None)} {getattr(ds,'y_max',None)}")
    print(debug_msg)

    n_samples: int = len(ds)

    # Splits (suppress internal prints)
    with contextlib.redirect_stdout(io.StringIO()):
        splits = compute_splits(
            n_samples=n_samples,
            embargo_days=cfg.embargo_days,
            train_ratio=0.7,
            val_ratio=0.15,
        )
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = splits
    
    # Log the start_msg
    with log_path.open("a", encoding="utf-8") as f:
            f.write(start_msg + "\n")
    # Log the debug_msg
    with log_path.open("a", encoding="utf-8") as f:
            f.write(debug_msg + "\n")
    
    split_msg = (f"[SPLIT] n={n_samples} | embargo={cfg.embargo_days} | "
          f"train=({tr_s},{tr_e}) len={tr_e-tr_s} | "
          f"val=({va_s},{va_e}) len={va_e-va_s} | "
          f"test=({te_s},{te_e}) len={te_e-te_s}")
    print(split_msg)

    # log the split line
    with log_path.open("a", encoding="utf-8") as f:
            f.write(split_msg + "\n")


    train_dl, val_dl, test_dl = make_dataloaders(
        ds, splits, cfg.batch_size, num_workers=cfg.num_workers, seed=getattr(cfg, "random_seed", 42)
    )

    # Input dimension
    X0, _ = ds[0]
    input_dim: int = int(X0.shape[-1])

    # Model / Optim / Loss
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()
    scaler: amp.GradScaler | None = amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val: float = float("inf")
    best_path: Path = results_dir / "checkpoints" / f"{args.ticker}_close_best.pt"
    patience: int = cfg.early_stopping_patience
    epochs_no_improve: int = 0

    train_losses: List[float] = []
    val_rmses: List[float] = []
    start_time_all: float = time.time()

    # --------------------- TRAIN LOOP ---------------------
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running: float = 0.0

        for X, y in train_dl:
            y = y.to(device).float().squeeze(-1)
            opt.zero_grad(set_to_none=True)

            yhat = forward_pass(model, X, device, scaler)
            loss = mse(yhat, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

            running += float(loss.item()) * X.size(0)

        train_loss: float = running / max(1, len(train_dl.dataset))

        # Validation (denorm via y_min/y_max)
        _, _, m_val = evaluate(
            model, val_dl, device,
            y_min=getattr(ds, "y_min", None),
            y_max=getattr(ds, "y_max", None),
        )
        train_losses.append(train_loss)
        val_rmses.append(m_val.rmse)
        save_metrics(results_dir, "close_val", m_val, ticker=args.ticker, verbose=False)

        elapsed: float = time.time() - start_time_all
        epoch_msg = (
            f"[train_close] Epoch {epoch:03d} | train={train_loss:.6f} | "
            f"val=RMSE:{m_val.rmse:.2f}|MAE:{m_val.mae:.2f}|MAPE:{m_val.mape:.2f}% | elapsed={elapsed:.2f}s"
        )
        print(epoch_msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")

        # Early stopping by price-scale RMSE
        if cfg.use_early_stopping:
            if m_val.rmse + 1e-12 < best_val:
                best_val = m_val.rmse
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stop_msg = f"[train_close] Early stopping triggered at epoch {epoch}."
                    print(stop_msg)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(stop_msg + "\n")
                    break

    # --------------------- POST TRAINING ---------------------
    plot_loss_curve(results_dir, train_losses, val_rmses, ticker=f"{args.ticker}_close")

    # Load best checkpoint safely
    if best_path.exists():
        try:
            state_dict = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(state_dict)
        msg = f"[train_close] Loaded best checkpoint safely: {best_path.name} (weights_only)"
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Validation (final)
    y_true_val, y_pred_val, m_val = evaluate(
        model, val_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "close_val", m_val, ticker=args.ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_val, y_pred_val, "close_val", ticker=args.ticker)

    # Test (final)
    y_true_test, y_pred_test, m_test = evaluate(
        model, test_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "close_test", m_test, ticker=args.ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_test, y_pred_test, "close_test", ticker=args.ticker)

    total_elapsed: float = time.time() - start_time_all
    done_msg: str = (
        f"[train_close] Done. Total elapsed: {total_elapsed:.2f}s\n"
        f"[train_close] Test RMSE:{m_test.rmse:.2f} | MAE:{m_test.mae:.2f} | MAPE:{m_test.mape:.2f}%"
    )
    print(done_msg)
    
    # One-time artifact summary
    metrics_dir = (results_dir / "metrics").resolve()

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Ticker: {args.ticker}\n")
        f.write(done_msg + "\n")
        f.write(f"[train_close] [LSTM stopping_no_embargo] Done on device: {device}\n")
        f.write(f"[save_metrics] Final result artifacts in: {metrics_dir}\n")
        f.write(f"[save_metrics] Files: "
          f"{args.ticker}_close_val_metrics.(json|txt), "
          f"{args.ticker}_close_test_metrics.(json|txt)\n")
        f.write(f"{'='*80}\n\n")

    
    print(f"[save_metrics] Final result artifacts in: {metrics_dir}")
    print(f"[save_metrics] Files: "
          f"{args.ticker}_close_val_metrics.(json|txt), "
          f"{args.ticker}_close_test_metrics.(json|txt)")


if __name__ == "__main__":
    main()
