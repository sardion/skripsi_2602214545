#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for TFT training (efficiency scenario).

This mirrors the LSTM efficiency config style:
- Fixed epochs, no early stopping
- No embargo (embargo_days = 0)
- Shorter lookback (window_size = 30)
- Identical API/exports for drop-in interchangeability
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
#: <PROJECT_ROOT>/tft/<scenario>/config.py  -> parents[1] == <PROJECT_ROOT>
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
    Base immutable configuration for TFT efficiency runs.

    Efficiency scenario:
    - No early stopping.
    - No embargo (embargo_days = 0).
    - AMP disabled by default; scripts enable it only when CUDA is available.
    """

    # Optimization / model
    random_seed: int = 42
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout: float = 0.2

    # TFT-specific widths/depths
    hidden_size: int = 32
    enc_layers: int = 1
    dec_layers: int = 1
    attention_heads: int = 4

    # Stability
    grad_clip: float = 0.5

    # Data loading
    num_workers: int = 0

    # Early stopping (off)
    use_early_stopping: bool = False
    early_stopping_patience: int = 10  # kept for API consistency

    # Embargo / splitting
    embargo_days: int = 0

    # Training loop
    max_epochs: int = 100

    # AMP (disabled by default; enabled dynamically if CUDA available)
    use_amp: bool = False

    # Loss
    loss_name: str = "mse"


@dataclass(frozen=True)
class Config(BaseConfig):
    """
    Scenario-specific configuration extending :class:`BaseConfig`.

    Attributes
    ----------
    window_size : int
        Lookback sequence length (timesteps) for supervised sliding-window
        features in the efficiency scenario (w=30 by design).
    """
    window_size: int = 30


# =========================
# Helpers
# =========================

def feature_filename(ticker: str, target: TargetKind) -> str:
    """
    Return the canonical feature file stem for a given ticker/target.

    Consistent with the sliding-window generation stage for efficiency runs (w=30).

    Parameters
    ----------
    ticker : str
        Stock code (e.g., "BBCA", "ANTM").
    target : TargetKind
        "close" for close price or "mean" for arithmetic mean price.

    Returns
    -------
    str
        Filename stem such as "BBCA_sliding_window_close_price_w30".

    Raises
    ------
    AssertionError
        If `target` is not one of {"close", "mean"}.
    """
    assert target in ("close", "mean"), "Unsupported target kind."

    if target == "close":
        return f"{ticker}_sliding_window_close_price_w30"
    else:
        return f"{ticker}_sliding_window_mean_price_w30"
