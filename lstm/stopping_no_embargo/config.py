#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for LSTM training (stopping+no embargo scenario).

This module centralizes hyperparameters and directory layout assumptions used by
the LSTM pipeline. It is intentionally minimal and immutable (frozen dataclasses)
to promote reproducibility and straightforward auditability in the thesis context.

Directory assumptions
---------------------
- This file lives under: <PROJECT_ROOT>/lstm/<scenario>/config.py
- `PROJECT_ROOT` is computed as `SCRIPT_DIR.parents[1]`
- Feature files are expected under: <PROJECT_ROOT>/data/features

Exports
-------
- SCRIPT_DIR, PROJECT_ROOT, FEATURES_DIR: resolved `pathlib.Path` constants.
- TargetKind: typing alias for the two supported targets ("close", "mean").
- BaseConfig, Config: frozen dataclasses with training hyperparameters.
- feature_filename(): helper to derive a canonical feature file stem.

No side effects occur at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

__all__ = [
    "SCRIPT_DIR",
    "PROJECT_ROOT",
    "FEATURES_DIR",
    "TargetKind",
    "BaseConfig",
    "Config",
    "feature_filename",
]

# =========================
# Paths (resolved constants)
# =========================

#: Absolute path to the directory containing this file.
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent

#: Project root inferred from the expected layout:
#: <PROJECT_ROOT>/lstm/<scenario>/config.py  -> parents[1] == <PROJECT_ROOT>
PROJECT_ROOT: Final[Path] = SCRIPT_DIR.parents[1]

#: Directory containing prebuilt sliding-window feature files.
FEATURES_DIR: Final[Path] = PROJECT_ROOT / "data" / "features"

# =========================
# Typing aliases
# =========================

#: Allowed prediction target kinds.
TargetKind = Literal["close", "mean"]

# =========================
# Config dataclasses
# =========================


@dataclass(frozen=True)
class BaseConfig:
    """
    Base immutable configuration for training.

    Attributes
    ----------
    random_seed : int
        Seed for PyTorch/NumPy to encourage deterministic behavior.
    batch_size : int
        Mini-batch size used by the DataLoader.
    learning_rate : float
        Initial learning rate for the optimizer.
    dropout : float
        Dropout probability applied inside the LSTM/FC layers (if used by model).
    hidden_dim : int
        Hidden dimension size for the LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    grad_clip : float
        Max gradient norm for clipping (<= 0 disables clipping).
    use_early_stopping : bool
        Enable early stopping on validation metric.
    early_stopping_patience : int
        Number of epochs without improvement before stopping.
    embargo_days : int
        Purge/embargo gap (in trading days) between train/val/test splits.
    max_epochs : int
        Maximum number of training epochs.
    use_amp : bool
        Enable automatic mixed precision (AMP) when CUDA is available.
    loss_name : str
        Loss identifier; expected by the training script ("mse" by default).
    """

    random_seed: int = 42
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout: float = 0.2
    hidden_dim: int = 64
    num_layers: int = 1
    grad_clip: float = 0.5

    # Data loading
    num_workers: int = 0

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10

    # Embargo / splitting
    embargo_days: int = 0

    # Training loop
    max_epochs: int = 100

    # Precision / loss
    use_amp: bool = False
    loss_name: str = "mse"

    


@dataclass(frozen=True)
class Config(BaseConfig):
    """
    Scenario-specific configuration extending :class:`BaseConfig`.

    Attributes
    ----------
    window_size : int
        Lookback sequence length (timesteps) for supervised sliding-window
        features in accuracy scenarios.
    """

    window_size: int = 120  # Lookback (sequence length) for accuracy scenarios


# =========================
# Helpers
# =========================

def feature_filename(ticker: str, target: TargetKind) -> str:
    """
    Return the canonical feature file stem for a given ticker/target.

    The returned string is a filename **stem** without extension and is
    consistent with the sliding-window generation stage (w=120).

    Parameters
    ----------
    ticker : str
        Stock code (e.g., "BBCA", "ANTM").
    target : TargetKind
        Target type: "close" for close price or "mean" for arithmetic mean price.

    Returns
    -------
    str
        Filename stem such as "BBCA_sliding_window_close_price_w120".

    Raises
    ------
    AssertionError
        If `target` is not a supported literal (guard for static analyzers).
    """
    # make sure only close or mean
    assert target in ("close", "mean"), "Unsupported target kind."

    if target == "close":
        return f"{ticker}_sliding_window_close_price_w120"
    else:
        return f"{ticker}_sliding_window_mean_price_w120"
