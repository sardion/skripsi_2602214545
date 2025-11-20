#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure utilities for model evaluation and training visualization.

- plot_pred_vs_true: Predicted vs true line plot (single PNG)
- plot_loss_curve:   Two separate figures:
    1) training_loss_curve.png
    2) validation_rmse_curve.png

Both functions save under results/figures, with optional ticker prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_pred_vs_true(
    results_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    ticker: Optional[str] = None,
) -> Path:
    """
    Plot predicted vs true values for a dataset split.

    Parameters
    ----------
    results_dir : Path
        Root output directory; figures are saved under results/figures.
    y_true : np.ndarray
        Ground-truth targets.
    y_pred : np.ndarray
        Model predictions.
    split_name : str
        Split tag (e.g., 'close_val', 'mean_test').
    ticker : Optional[str]
        Optional ticker prefix for filenames.

    Returns
    -------
    Path
        Path to the saved figure.
    """
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


def plot_loss_curve(
    results_dir: Path,
    train_losses: list[float],
    val_rmses: list[float],
    ticker: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Plot and save two separate curves:
      1) Training loss per epoch
      2) Validation RMSE per epoch

    Parameters
    ----------
    results_dir : Path
        Root output directory; figures are saved under results/figures.
    train_losses : list[float]
        Training losses per epoch.
    val_rmses : list[float]
        Validation RMSE values per epoch.
    ticker : Optional[str]
        Optional ticker prefix for filenames.

    Returns
    -------
    (Path, Path)
        Paths to (training_loss_curve.png, validation_rmse_curve.png).
    """
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    prefix = f"{ticker}_" if ticker else ""

    # Training loss curve
    tl_path = results_dir / "figures" / f"{prefix}training_loss_curve.png"
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Epoch")
    plt.savefig(tl_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Validation RMSE curve
    vr_path = results_dir / "figures" / f"{prefix}validation_rmse_curve.png"
    plt.figure()
    plt.plot(range(1, len(val_rmses) + 1), val_rmses)
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE per Epoch")
    plt.savefig(vr_path, dpi=150, bbox_inches="tight")
    plt.close()

    return tl_path, vr_path
