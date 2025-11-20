#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency scenario trainer (no early stopping, fixed epochs) for TFT MEAN (AMP) target.

Artifacts (all ticker-prefixed, under results/):
- logs/{TICKER}_train_mean_efficiency_terminal_log.txt
- logs/{TICKER}_trace_{cpu|cuda}.json
- logs/{TICKER}_profiler_{cpu|cuda}.txt
- logs/{TICKER}_efficiency_mean_total_elapsed_s.txt
- logs/{TICKER}_efficiency_mean_peak_{cpu|cuda}_memory_mb.txt
- logs/{TICKER}_efficiency_mean_cpu_utilization_pct.txt
- figures: pred_vs_true (mean_val, mean_test) + loss curves
- metrics: mean_val.(json|txt), mean_test.(json|txt)

Notes
-----
- Fixed epochs, no early stopping (compute-efficiency scenario).
- Uses TFTLite with dict-batches: {"past","future","static"}.
- AMP enabled only if CUDA is available AND cfg.use_amp is True.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path
from typing import Tuple, List, Any

import numpy as np
import psutil
import torch
from torch import nn, optim, amp
from torch.utils.data import DataLoader, Subset, Dataset
import torch.profiler as tprof
import warnings
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
    module="torch.nn.modules.rnn",
)
warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")

from config import Config
from dataloader import load_dataset_tft
from splitter import compute_splits
from model import TFTLite
from metrics import evaluate, save_metrics  # denorm-aware via y_min/y_max
from figures import plot_pred_vs_true, plot_loss_curve


# ---------------- helpers ---------------- #

def make_dataloaders(
    ds: Dataset[Any],
    splits: Tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Deterministic train shuffle to match baseline conventions."""
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


def _compute_amp_from_raw_df(df) -> np.ndarray:
    cols = {c.lower(): c for c in df.columns}
    need = ["open_price", "high_price", "low_price", "close_price"]
    if all(k in cols for k in need):
        o = df[cols["open_price"]].astype(float).to_numpy()
        h = df[cols["high_price"]].astype(float).to_numpy()
        l = df[cols["low_price"]].astype(float).to_numpy()
        c = df[cols["close_price"]].astype(float).to_numpy()
        return (o + h + l + c) / 4.0
    alt = ["open", "high", "low", "close"]
    if all(k in cols for k in alt):
        o = df[cols["open"]].astype(float).to_numpy()
        h = df[cols["high"]].astype(float).to_numpy()
        l = df[cols["low"]].astype(float).to_numpy()
        c = df[cols["close"]].astype(float).to_numpy()
        return (o + h + l + c) / 4.0
    raise KeyError("Cannot compute AMP: missing OHLC columns")


def attach_y_minmax_from_metadata_or_raw_mean(ds: Any, ticker: str, project_root: Path) -> None:
    """
    Attach y_min/y_max (IDR) for MEAN PRICE so metrics.evaluate can denormalize.

    Priority:
      1) normalization metadata JSON: columns.mean_price.min/max
      2) RAW CSV: min/max from 'mean_price' or AMP(OHLC) fallback
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
                cols_lower = {c.lower(): c for c in df.columns}
                if "mean_price" in cols_lower:
                    s = pd.to_numeric(df[cols_lower["mean_price"]], errors="coerce").dropna().to_numpy()
                else:
                    s = _compute_amp_from_raw_df(df)
                if s.size == 0:
                    continue
                y_min = float(np.min(s)); y_max = float(np.max(s))
                if y_max > y_min:
                    ds.y_min = y_min; ds.y_max = y_max
                    print(f"[attach_y_minmax] (raw:mean) {ticker}: y_min={ds.y_min:.2f} y_max={ds.y_max:.2f} from {p.name}")
                    return
            except Exception:
                pass

    print(f"[attach_y_minmax] WARNING: could not determine y_min/y_max for {ticker}. "
          f"Metrics will remain on normalized scale for this run.")


class _EvalAdapter(torch.nn.Module):
    """Adapter so metrics.evaluate(model, loader, ...) — which calls model(X) — works with dict batches."""
    def __init__(self, core: TFTLite):
        super().__init__()
        self.core = core

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.core(batch["past"], batch["future"], batch["static"])


# ---------------- main ---------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock code, e.g., BBCA")
    args = parser.parse_args()
    ticker = args.ticker
    target_kind: str = "mean"
    base_tag = f"{ticker}_{target_kind}"

    cfg = Config()  # efficiency scenario: fixed epochs, no early stopping
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_msg = (f"[train_mean][TFT efficiency] Running on device: {device}")
    print(start_msg)

    results_dir = Path("results")
    for sub in ("metrics", "figures", "logs", "checkpoints"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    # AMP / Device
    use_amp_now: bool = torch.cuda.is_available() and getattr(cfg, "use_amp", False)
    scaler: amp.GradScaler | None = amp.GradScaler(enabled=use_amp_now)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]
        device_msg = ("[train_mean][TFT efficiency] CUDA available, running on GPU (AMP enabled, torch profiling).")
    else:
        device_msg = ("[train_mean][TFT efficiency] CUDA not available, running on CPU (AMP disabled, CPU-only profiling).")
    print(device_msg)

    # Profiler
    acts = [tprof.ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(tprof.ProfilerActivity.CUDA)

    trace_path = results_dir / "logs" / f"{base_tag}_trace_{device.type}.json"
    table_path = results_dir / "logs" / f"{base_tag}_profiler_{device.type}.txt"

    def on_trace_ready(p: tprof.profile) -> None:
        p.export_chrome_trace(str(trace_path))
        table = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=200)
        table_path.write_text(table, encoding="utf-8")

    prof = tprof.profile(
        activities=acts,
        schedule=tprof.schedule(wait=1, warmup=1, active=10, repeat=1),
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=True,
    )
    prof.__enter__()

    # Dataset & splits (TFT dict-batch)
    ds = load_dataset_tft(ticker, target_kind, cfg)

    # Attach y_min/y_max for denorm-aware metrics
    project_root = Path(__file__).resolve().parents[2]
    attach_y_minmax_from_metadata_or_raw_mean(ds, ticker, project_root)

    # Split (silence internal prints; print once)
    with contextlib.redirect_stdout(io.StringIO()):
        splits = compute_splits(len(ds), embargo_days=cfg.embargo_days, train_ratio=0.7, val_ratio=0.15)
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = splits
    split_msg = (f"[SPLIT] n={len(ds)} | embargo={cfg.embargo_days} | "
                 f"train=({tr_s},{tr_e}) len={tr_e-tr_s} | "
                 f"val=({va_s},{va_e}) len={va_e-va_s} | "
                 f"test=({te_s},{te_e}) len={te_e-te_s}")
    print(split_msg)

    train_dl, val_dl, test_dl = make_dataloaders(
        ds, splits, cfg.batch_size, num_workers=cfg.num_workers, seed=getattr(cfg, "random_seed", 42)
    )

    # Model / Optim / Loss (infer dims from sample or ds attrs)
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
    model_for_eval = _EvalAdapter(core)
    opt = optim.Adam(core.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()

    # CPU process baseline (for utilization)
    proc = psutil.Process(os.getpid())
    cpu_times_start = proc.cpu_times()

    # Terminal log file
    log_path = results_dir / "logs" / f"{ticker}_train_mean_efficiency_terminal_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(start_msg + "\n")
        f.write(device_msg + "\n")
        f.write(split_msg + "\n")

    # Trackers
    train_losses: List[float] = []
    val_rmses: List[float] = []

    # Training (fixed epochs)
    start_all = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        core.train()
        running: float = 0.0
        epoch_start = time.time()

        for batch, y in train_dl:
            # dict batch: {"past","future","static"}
            past   = batch["past"].to(device)
            future = batch["future"].to(device)
            static = batch["static"].to(device)
            y = y.to(device).float().squeeze(-1)
            opt.zero_grad(set_to_none=True)

            if use_amp_now:
                with amp.autocast(device_type="cuda", enabled=True):
                    yhat = core(past, future, static)
                    loss = mse(yhat, y)
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(core.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                yhat = core(past, future, static)
                loss = mse(yhat, y)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(core.parameters(), cfg.grad_clip)
                opt.step()

            running += float(loss.item()) * y.size(0)

        train_loss: float = running / max(1, len(train_dl.dataset))
        train_losses.append(train_loss)

        # Per-epoch validation in IDR scale (denorm via y_min/y_max)
        _, _, m_val_epoch = evaluate(
            model_for_eval, val_dl, device,
            y_min=getattr(ds, "y_min", None),
            y_max=getattr(ds, "y_max", None),
        )
        val_rmses.append(float(m_val_epoch.rmse))
        prof.step()

        epoch_msg = (
            f"[efficiency_mean] Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | val=RMSE:{m_val_epoch.rmse:.2f}|MAE:{m_val_epoch.mae:.2f}|MAPE:{m_val_epoch.mape:.2f}% | "
            f"epoch_time={time.time()-epoch_start:.2f}s"
        )
        print(epoch_msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")

    total_elapsed: float = time.time() - start_all

    # Time
    (results_dir / "logs" / f"{ticker}_efficiency_mean_total_elapsed_s.txt").write_text(
        f"{total_elapsed:.2f}\n", encoding="utf-8"
    )

    # Peak memory
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)  # type: ignore[attr-defined]
        (results_dir / "logs" / f"{ticker}_efficiency_mean_peak_cuda_memory_mb.txt").write_text(
            f"{peak_mem_mb:.2f}\n", encoding="utf-8"
        )
    else:
        mem_info = proc.memory_info()
        peak_mb = getattr(mem_info, "peak_wset", mem_info.rss) / (1024 ** 2)
        (results_dir / "logs" / f"{ticker}_efficiency_mean_peak_cpu_memory_mb.txt").write_text(
            f"{peak_mb:.2f}\n", encoding="utf-8"
        )

    # CPU utilization over training
    cpu_times_end = proc.cpu_times()
    cpu_time_used = (cpu_times_end.user - cpu_times_start.user) + (cpu_times_end.system - cpu_times_start.system)
    logical_cores = max(1, os.cpu_count() or 1)
    cpu_util_pct = (cpu_time_used / (total_elapsed * logical_cores)) * 100.0 if total_elapsed > 0 else 0.0
    (results_dir / "logs" / f"{ticker}_efficiency_mean_cpu_utilization_pct.txt").write_text(
        f"{cpu_util_pct:.2f}\n", encoding="utf-8"
    )

    # Final evaluation (IDR scale)
    y_true_val, y_pred_val, m_val = evaluate(
        model_for_eval, val_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "mean_val", m_val, ticker=ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_val, y_pred_val, "mean_val", ticker=ticker)

    y_true_test, y_pred_test, m_test = evaluate(
        model_for_eval, test_dl, device,
        y_min=getattr(ds, "y_min", None),
        y_max=getattr(ds, "y_max", None),
    )
    save_metrics(results_dir, "mean_test", m_test, ticker=ticker, verbose=False)
    plot_pred_vs_true(results_dir, y_true_test, y_pred_test, "mean_test", ticker=ticker)

    # Loss curves
    tl_path, vr_path = plot_loss_curve(results_dir, train_losses, val_rmses, ticker=f"{ticker}_mean")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"Saved loss curves: {tl_path.name}, {vr_path.name}\n")

    prof.__exit__(None, None, None)

    done_msg = (
        f"[efficiency_mean] Done. Total elapsed: {total_elapsed:.2f}s\n"
        f"[efficiency_mean] Test RMSE={m_test.rmse:.2f} MAE={m_test.mae:.2f} MAPE={m_test.mape:.2f}%"
    )
    print(done_msg)

    metrics_dir = (results_dir / "metrics").resolve()

    with log_path.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(done_msg + "\n")
        f.write(f"[train_mean][TFT efficiency] Done on device: {device}\n")
        f.write(f"[save_metrics] Final metrics in: {metrics_dir}\n")
        f.write(f"[save_metrics] Files: {ticker}_mean_val_metrics.(json|txt), {ticker}_mean_test_metrics.(json|txt)\n")
        f.write("=" * 80 + "\n\n")

    print(f"[save_metrics] Final metrics in: {metrics_dir}")
    print(f"[save_metrics] Files: {ticker}_mean_val_metrics.(json|txt), {ticker}_mean_test_metrics.(json|txt)")


if __name__ == "__main__":
    main()
