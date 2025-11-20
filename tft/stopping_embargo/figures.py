#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_pred_vs_true(results_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, split_name: str, ticker: str | None = None) -> Path:
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    prefix = f"{ticker}_" if ticker else ""
    figpath = results_dir / "figures" / f"{prefix}pred_vs_true_{split_name}.png"
    plt.figure()
    plt.plot(y_true, label="true")
    plt.plot(y_pred, label="pred")
    plt.legend()
    plt.title(f"Pred vs True — {split_name}")
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()
    return figpath

def plot_loss_curve(results_dir: Path, train_losses: list[float], val_rmses: list[float], ticker: str | None = None) -> tuple[Path, Path]:
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    prefix = f"{ticker}_" if ticker else ""

    tl_path = results_dir / "figures" / f"{prefix}training_loss_curve.png"
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Epoch")
    plt.savefig(tl_path, dpi=150, bbox_inches="tight")
    plt.close()

    vr_path = results_dir / "figures" / f"{prefix}validation_rmse_curve.png"
    plt.figure()
    plt.plot(range(1, len(val_rmses)+1), val_rmses)
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE per Epoch")
    plt.savefig(vr_path, dpi=150, bbox_inches="tight")
    plt.close()

    return tl_path, vr_path
