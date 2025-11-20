#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility for embargo-aware dataset splitting into train/val/test segments.

Used consistently across all scenarios:
- stopping_embargo
- stopping_no_embargo
- efficiency (embargo_days = 0)

The function ensures index safety via clipping and reports
segment boundaries for reproducibility.
"""

from __future__ import annotations
from typing import Tuple


def _clip_pair(a: int, b: int, n: int) -> tuple[int, int]:
    """
    Clip a pair of integer indices (a, b) so they lie within [0, n].
    Ensures that b >= a after clipping.
    """
    a = max(0, min(a, n))
    b = max(0, min(b, n))
    if b < a:
        b = a
    return a, b


def compute_splits(
    n_samples: int,
    embargo_days: int = 7,   # set default=7 to match stopping_embargo expectation
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Compute embargo-aware train/val/test index splits.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    embargo_days : int, default=7
        Gap (in samples) between consecutive splits to avoid leakage.
        Set to 0 for no-embargo / efficiency scenarios.
    train_ratio : float, default=0.7
        Fraction of samples used for training.
    val_ratio : float, default=0.15
        Fraction of samples used for validation.

    Returns
    -------
    (train_range, val_range, test_range) : Tuple of tuples[int, int]
        Each range is an inclusive-exclusive index pair (start, end).
    """
    assert (
        0.0 < train_ratio < 1.0
        and 0.0 < val_ratio < 1.0
        and train_ratio + val_ratio < 1.0
    ), "train_ratio + val_ratio must be < 1.0"

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    tr_s, tr_e = 0, n_train
    va_s = tr_e + embargo_days
    va_e = va_s + n_val
    te_s = va_e + embargo_days
    te_e = n_samples

    tr_s, tr_e = _clip_pair(tr_s, tr_e, n_samples)
    va_s, va_e = _clip_pair(va_s, va_e, n_samples)
    te_s, te_e = _clip_pair(te_s, te_e, n_samples)

    # Handle degenerate validation/test regions
    if va_e - va_s <= 0:
        va_s, va_e = va_e, va_e
        te_s, te_e = _clip_pair(tr_e + embargo_days, n_samples, n_samples)

    print(
        f"[SPLIT] n={n_samples} | embargo={embargo_days} | "
        f"train=({tr_s},{tr_e}) len={tr_e - tr_s} | "
        f"val=({va_s},{va_e}) len={va_e - va_s} | "
        f"test=({te_s},{te_e}) len={te_e - te_s}"
    )

    return (tr_s, tr_e), (va_s, va_e), (te_s, te_e)
