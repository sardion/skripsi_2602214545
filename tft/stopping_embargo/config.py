#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for TFT training (stopping + embargo scenario).

This mirrors the style/structure used in LSTM/TCN configs to keep a unified API:
- Early stopping enabled by default
- Embargo gap between splits (default = 7)
- AMP disabled by default (can be toggled at runtime if needed)
- window_size = 120 for accuracy scenarios

Exports
-------
- SCRIPT_DIR, PROJECT_ROOT, FEATURES_DIR
- TargetKind
- BaseConfig, Config
- feature_filename()
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
FEATURES_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw_features"

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
    Base immutable configuration for TFT training.

    Notes
    -----
    - Stopping + embargo scenario (accuracy-focused):
      * Early stopping: ON
      * Embargo gap: default 7 samples
      * window_size: defined in `Config` (120 by default)
    - `num_workers` defaults to 0 to encourage determinism; you may increase it
      on GPU instances if you need faster input pipelines.
    """

    # Optimization / model
    random_seed: int = 42
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout: float = 0.2
    grad_clip: float = 0.5

    # TFT-specific architecture
    hidden_size: int = 32
    enc_layers: int = 1
    dec_layers: int = 1
    attention_heads: int = 4

    # Data loading
    num_workers: int = 0

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10

    # Embargo / splitting
    embargo_days: int = 7

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
        Lookback sequence length (timesteps) for sliding-window features
        in accuracy scenarios (120 by design).
    """

    window_size: int = 120  # Lookback for accuracy scenarios


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
        "close" for close price or "mean" for arithmetic mean price.

    Returns
    -------
    str
        Filename stem like "BBCA_sliding_window_close_price_w120".
    """
    assert target in ("close", "mean"), "Unsupported target kind."

    if target == "close":
        return f"{ticker}_sliding_window_close_price_w120"
    else:
        return f"{ticker}_sliding_window_mean_price_w120"


