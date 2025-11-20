# metrics.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List

import json
import numpy as np

# Optional import-time torch typing (avoids hard dependency at import)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Container for evaluation metrics.

    Attributes
    ----------
    rmse : float
        Root Mean Squared Error (in IDR if denormalized).
    mae : float
        Mean Absolute Error (in IDR if denormalized).
    mape : float
        Mean Absolute Percentage Error (in percent, stabilized).
    """
    rmse: float
    mae: float
    mape: float


# ---------------------------------------------------------------------
# Internal helpers (denormalization and robust MAPE)
# ---------------------------------------------------------------------

def _unwrap_dataset(ds: Any) -> Any:
    """
    Peel off nested wrappers (e.g., torch.utils.data.Subset) until reaching
    the base dataset that may carry y_scaler / y_min / y_max.
    """
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _get_denorm_fn(dataset: Any) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    Build a vectorized inverse transform function from dataset metadata if available.

    Precedence:
      1) dataset.y_scaler.inverse_transform (sklearn-like)
      2) dataset.y_min / dataset.y_max (min-max scaling)
    """
    ds = _unwrap_dataset(dataset)

    # sklearn-like scaler
    y_scaler = getattr(ds, "y_scaler", None)
    if y_scaler is not None and hasattr(y_scaler, "inverse_transform"):
        return lambda arr: y_scaler.inverse_transform(arr.reshape(-1, 1)).ravel()

    # min-max fallback
    y_min = getattr(ds, "y_min", None)
    y_max = getattr(ds, "y_max", None)
    if isinstance(y_min, (int, float)) and isinstance(y_max, (int, float)) and y_max > y_min:
        scale = float(y_max - y_min)
        offset = float(y_min)
        return lambda arr: (arr * scale) + offset

    return None


def _stable_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute a stabilized MAPE in percent to avoid explosion near zero.

    Denominator: max(|y_true|, floor), where floor is 1% of median(|y_true|) or 1e-8.
    """
    if y_true.size == 0:
        return float("nan")
    floor = 0.01 * float(np.median(np.abs(y_true)))
    denom = np.maximum(np.abs(y_true), max(floor, 1e-8))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def evaluate(
    model: Any,
    dl: Any,  # torch.utils.data.DataLoader
    device: Any,
    *,
    denormalize: bool = True,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_transform: Optional[Callable[[Any], Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, EvalResult]:
    """
    Evaluate a model on a DataLoader and compute RMSE/MAE/MAPE.

    This function optionally denormalizes predictions and targets back to the
    original monetary scale for RMSE/MAE and for stable MAPE computation.

    Denormalization precedence:
        1) Explicit y_min/y_max passed as arguments (if valid)
        2) Dataset-provided inverse transform via y_scaler.inverse_transform
        3) Dataset-provided y_min/y_max (min-max)

    Parameters
    ----------
    model : Any
        PyTorch model with .eval() and callable forward pass.
    dl : Any
        torch.utils.data.DataLoader yielding (X, y).
    device : Any
        torch device used for inference.
    denormalize : bool, optional
        If True (default), attempt to convert normalized values back to IDR.
    y_min : Optional[float], optional
        Explicit minimum for min-max inverse transform (takes precedence).
    y_max : Optional[float], optional
        Explicit maximum for min-max inverse transform (takes precedence).
    x_transform : Optional[Callable[[Any], Any]], optional
        Optional function applied to each batch input X *after* moving to device,
        but *before* forward(). Use this to adapt shapes per-architecture, e.g.:
            - TCN: lambda X: X.transpose(1, 2).contiguous()  # (B,T,F)->(B,F, T)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, EvalResult]
        (y_true, y_pred, EvalResult) where arrays are 1-D numpy vectors
        in IDR scale if denormalized, otherwise in dataset scale.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for evaluate().")

    model.eval()
    ys: List[np.ndarray] = []
    yhats: List[np.ndarray] = []

    with torch.no_grad():
        for X, y in dl:
            X = X.to(device)
            if x_transform is not None:
                X = x_transform(X)
            y = y.to(device).float().squeeze(-1)
            yhat = model(X).squeeze(-1)
            ys.append(y.detach().cpu().numpy().reshape(-1))
            yhats.append(yhat.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(yhats) if yhats else np.array([])

    # Denormalize if requested and data exists
    if denormalize and y_true.size:
        if (y_min is not None) and (y_max is not None) and (y_max > y_min):
            scale = float(y_max - y_min)
            offset = float(y_min)
            y_true = (y_true * scale) + offset
            y_pred = (y_pred * scale) + offset
        else:
            denorm_fn = _get_denorm_fn(dl.dataset)
            if denorm_fn is not None:
                y_true = denorm_fn(y_true)
                y_pred = denorm_fn(y_pred)

    if y_true.size:
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = _stable_mape(y_true, y_pred)
    else:
        rmse = mae = mape = float("nan")

    return y_true, y_pred, EvalResult(rmse=rmse, mae=mae, mape=mape)


def save_metrics(
    results_dir: Path,
    split_name: str,
    result: Any,
    ticker: str,
    verbose: bool = False,
) -> None:
    """
    Persist evaluation metrics (RMSE, MAE, MAPE) in both JSON and TXT formats.

    Output directory: results/metrics/
    """
    out_dir = Path(results_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    # JSON
    json_path = out_dir / f"{ticker}_{split_name}_metrics.json"
    data = {
        "ticker": ticker,
        "split": split_name,
        "timestamp": timestamp,
        "rmse": float(result.rmse),
        "mae": float(result.mae),
        "mape": float(result.mape),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # TXT (human-friendly summary)
    txt_path = out_dir / f"{ticker}_{split_name}_metrics.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== Evaluation Metrics Summary ===\n")
        f.write(f"Ticker        : {ticker}\n")
        f.write(f"Data Split    : {split_name}\n")
        f.write(f"Timestamp     : {timestamp}\n")
        f.write(f"RMSE (IDR)    : {result.rmse:.4f}\n")
        f.write(f"MAE (IDR)     : {result.mae:.4f}\n")
        f.write(f"MAPE (%)      : {result.mape:.2f}\n")
        f.write("==================================\n")

    if verbose:
        print(f"[save_metrics] Saved to:\n - {json_path}\n - {txt_path}")
