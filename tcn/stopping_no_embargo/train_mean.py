#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCN training script (stopping_no_embargo): mean-price target.

Conventions aligned with LSTM/TCN close:
- Suppress duplicate split prints (silence compute_splits; print once here).
- Quiet save_metrics during loop & plots; print a single artifact summary at the end.
- Labeled val line: val=RMSE:...|MAE:...|MAPE:...%
- Denormalization-aware evaluation via y_min/y_max attach (from JSON or RAW CSV).
- Safe checkpoint loading (weights_only=True with fallback).
- Reproducible DataLoader shuffling (torch.Generator).

TCN-specific:
- Dataloader yields (B, T, F); TCN expects (B, F, T). We transpose per-batch.
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
from model import TemporalConvNet
from figures import plot_pred_vs_true, plot_loss_curve
from metrics import save_metrics  # reuse artifact writers (JSON + TXT)


# ----------------------------- helpers ----------------------------- #

def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, F) -> (B, F, T) for Conv1d."""
    return x.transpose(1, 2).contiguous()


def make_dataloaders(
    ds: Dataset[Any],
    splits: Tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders with deterministic shuffle (like LSTM)."""
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


def forward_pass_tcn(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    scaler: amp.GradScaler | None,
) -> torch.Tensor:
    """
    TCN forward pass:
    - move to device
    - transpose (B,T,F)->(B,F,T)
    - AMP only on CUDA
    """
    X = _to_channels_first(X.to(device))
    enabled = (device.type == "cuda") and (scaler is not None) and getattr(scaler, "is_enabled", lambda: False)()
    with amp.autocast(device_type="cuda", enabled=enabled):
        yhat = model(X).squeeze(-1)
    return yhat


def attach_y_minmax_from_metadata_or_raw(ds: Dataset[Any], ticker: str, project_root: Path) -> None:
    """
    Attach y_min/y_max (IDR) for MEAN PRICE so evaluation can denormalize.

    Priority:
      1) normalization metadata JSON: columns.mean_price.min/max
      2) RAW CSV:
         - if column 'mean_price' exists: use its min/max
         - else compute arithmetic mean of OHLC: (open+high+low+close)/4
    """
    # Try metadata first
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
                mean_info = cols.get("mean_price", {})
                y_min = mean_info.get("min", None)
                y_max = mean_info.get("max", None)
                if y_min is not None and y_max is not None and float(y_max) > float(y_min):
                    ds.y_min = float(y_min)
                    ds.y_max = float(y_max)
                    print(f"[attach_y_minmax] (metadata) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                    return
            except Exception:
                pass

    # Fallback to RAW CSV(s)
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
                    # compute arithmetic mean of OHLC if mean_price not present
                    cols = ["open_price", "high_price", "low_price", "close_price"]
                    if all(c in df.columns for c in cols):
                        s = (df["open_price"].astype(float)
                             + df["high_price"].astype(float)
                             + df["low_price"].astype(float)
                             + df["close_price"].astype(float)) / 4.0
                        s = s.to_numpy()
                    else:
                        raise KeyError("Required columns for mean_price not found.")
                y_min = float(np.min(s))
                y_max = float(np.max(s))
                if y_max > y_min:
                    ds.y_min = y_min
                    ds.y_max = y_max
                    print(f"[attach_y_minmax] (raw) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                    return
            except Exception:
                pass

    print(f"[attach_y_minmax] WARNING: could not determine y_min/y_max for {ticker} (mean). "
          f"Metrics stay on normalized scale for this run.")



def evaluate_tcn(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Evaluation that mirrors the close version, but keeps the TCN transpose.
    Returns (y_true, y_pred, {'rmse':..,'mae':..,'mape':..}).
    """
    model.eval()
    ys: List[np.ndarray] = []
    yhats: List[np.ndarray] = []
    with torch.no_grad():
        for X, y in dl:
            X = _to_channels_first(X.to(device))
            y = y.to(device).float().squeeze(-1)
            yhat = model(X).squeeze(-1)
            ys.append(np.atleast_1d(y.detach().cpu().numpy()).reshape(-1))
            yhats.append(np.atleast_1d(yhat.detach().cpu().numpy()).reshape(-1))

    if not ys:
        y_true = np.array([]); y_pred = np.array([])
    else:
        y_true = np.concatenate(ys, axis=0)
        y_pred = np.concatenate(yhats, axis=0)

    # Denorm to IDR if possible (explicit beats autodetect).
    if y_true.size and (y_min is not None) and (y_max is not None) and (y_max > y_min):
        scale = float(y_max - y_min); offset = float(y_min)
        y_true = (y_true * scale) + offset
        y_pred = (y_pred * scale) + offset

    if y_true.size:
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        denom_floor = 0.01 * float(np.median(np.abs(y_true)))
        denom = np.maximum(np.abs(y_true), max(denom_floor, 1e-8))
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    else:
        rmse = mae = mape = float("nan")

    return y_true, y_pred, {"rmse": rmse, "mae": mae, "mape": mape}


# ----------------------------- main ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock code, e.g., BBCA")
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(getattr(cfg, "random_seed", 42))
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_msg = (f"[train_mean][TCN stopping_no_embargo] Running on device: {device}")
    print(start_msg)

    # Dirs & log
    results_dir = Path("results")
    for sub in ("metrics", "figures", "checkpoints", "logs"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "logs" / f"{args.ticker}_train_mean_terminal_log.txt"

    # Dataset
    target_kind: str = "mean"
    ds = load_dataset(args.ticker, target_kind, cfg)

    # Attach y_min/y_max for IDR-scale metrics (mean price)
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

    # Log early lines
    with log_path.open("a", encoding="utf-8") as f:
        f.write(start_msg + "\n")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(debug_msg + "\n")

    split_msg = (f"[SPLIT] n={n_samples} | embargo={cfg.embargo_days} | "
                 f"train=({tr_s},{tr_e}) len={tr_e-tr_s} | "
                 f"val=({va_s},{va_e}) len={va_e-va_s} | "
                 f"test=({te_s},{te_e}) len={te_e-te_s}")
    print(split_msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(split_msg + "\n")

    train_dl, val_dl, test_dl = make_dataloaders(
        ds, splits, cfg.batch_size, num_workers=cfg.num_workers, seed=getattr(cfg, "random_seed", 42)
    )

    # Input dimension (F) from a sample (B,T,F)
    X0, _ = ds[0]
    input_dim: int = int(X0.shape[-1])

    # Model / Optim / Loss
    model = TemporalConvNet(
        in_channels=input_dim,
        channels=cfg.channels,
        num_blocks=cfg.residual_blocks,
        kernel_size=cfg.kernel_size,
        dilations=tuple(cfg.dilations),
        dropout=cfg.dropout,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()
    scaler: amp.GradScaler | None = amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val: float = float("inf")
    best_path: Path = results_dir / "checkpoints" / f"{args.ticker}_mean_best.pt"
    patience: int = cfg.early_stopping_patience
    epochs_no_improve: int = 0

    train_losses: List[float] = []
    val_rmses: List[float] = []
    start_time_all: float = time.time()

    # --------------------- TRAIN LOOP --------------------- #
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running: float = 0.0

        for X, y in train_dl:
            y = y.to(device).float().squeeze(-1)
            opt.zero_grad(set_to_none=True)

            yhat = forward_pass_tcn(model, X, device, scaler)
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

        # Validation on IDR scale (explicit y_min/y_max)
        _, _, m_val = evaluate_tcn(
            model, val_dl, device,
            y_min=getattr(ds, "y_min", None),
            y_max=getattr(ds, "y_max", None),
        )
        train_losses.append(train_loss)
        val_rmses.append(m_val["rmse"])
        save_metrics(results_dir, "mean_val",
                     result=type("EvalResultLike", (), m_val)(),
                     ticker=args.ticker,
                     verbose=False)

        elapsed: float = time.time() - start_time_all
        epoch_msg = (
            f"[train_mean] Epoch {epoch:03d} | train={train_loss:.6f} | "
            f"val=RMSE:{m_val['rmse']:.2f}|MAE:{m_val['mae']:.2f}|MAPE:{m_val['mape']:.2f}% | elapsed={elapsed:.2f}s"
        )
        print(epoch_msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")

        # Early stopping by price-scale RMSE
        if cfg.use_early_stopping:
            if m_val["rmse"] + 1e-12 < best_val:
                best_val = m_val["rmse"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stop_msg = f"[train_mean] Early stopping triggered at epoch {epoch}."
                    print(stop_msg)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(stop_msg + "\n")
                    break

    # --------------------- POST TRAINING --------------------- #
    plot_loss_curve(results_dir, train_losses, val_rmses, ticker=f"{args.ticker}_mean")

    # Load best checkpoint safely
    if best_path.exists():
        try:
            state_dict = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(state_dict)
        msg = f"[train_mean] Loaded best checkpoint safely: {best_path.name} (weights_only)"
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Validation (final, IDR scale)
    y_true_val, y_pred_val, m_val = evaluate_tcn(
        model, val_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "mean_val",
                 result=type("EvalResultLike", (), m_val)(),
                 ticker=args.ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_val, y_pred_val, "mean_val", ticker=args.ticker)

    # Test (final, IDR scale)
    y_true_test, y_pred_test, m_test = evaluate_tcn(
        model, test_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "mean_test",
                 result=type("EvalResultLike", (), m_test)(),
                 ticker=args.ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_test, y_pred_test, "mean_test", ticker=args.ticker)

    total_elapsed: float = time.time() - start_time_all
    done_msg: str = (
        f"[train_mean] Done. Total elapsed: {total_elapsed:.2f}s\n"
        f"[train_mean] Test RMSE:{m_test['rmse']:.2f} | MAE:{m_test['mae']:.2f} | MAPE:{m_test['mape']:.2f}%"
    )
    print(done_msg)

    # One-time artifact summary (match LSTM/TCN trailer style)
    metrics_dir = (results_dir / "metrics").resolve()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Ticker: {args.ticker}\n")
        f.write(done_msg + "\n")
        f.write(f"[train_mean] [TCN stopping_no_embargo] Done on device: {device}\n")
        f.write(f"[save_metrics] Final result artifacts in: {metrics_dir}\n")
        f.write(f"[save_metrics] Files: "
                f"{args.ticker}_mean_val_metrics.(json|txt), "
                f"{args.ticker}_mean_test_metrics.(json|txt)\n")
        f.write(f"{'='*80}\n\n")

    print(f"[save_metrics] Final result artifacts in: {metrics_dir}")
    print(f"[save_metrics] Files: "
          f"{args.ticker}_mean_val_metrics.(json|txt), "
          f"{args.ticker}_mean_test_metrics.(json|txt)")


if __name__ == "__main__":
    main()
