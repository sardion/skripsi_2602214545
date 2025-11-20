#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-based regressor for financial time-series forecasting.

Used consistently across all scenarios (stopping_embargo, stopping_no_embargo, efficiency).
Predicts a scalar target (e.g., close or mean price) from a windowed input sequence.

Architecture:
- Stacked LSTM encoder (batch_first)
- Two-layer feedforward head with ReLU + dropout
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LSTMRegressor(nn.Module):
    """
    LSTM-based sequence regressor.

    Parameters
    ----------
    input_dim : int
        Number of features per time step.
    hidden_dim : int, default=64
        Hidden state dimension of the LSTM.
    num_layers : int, default=1
        Number of stacked LSTM layers.
    dropout : float, default=0.2
        Dropout rate applied to LSTM (if num_layers > 1) and the feedforward head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Feedforward regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Optional initialization for stability
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the LSTM and regression head.

        Parameters
        ----------
        x : Tensor
            Input of shape (batch_size, sequence_length, input_dim).

        Returns
        -------
        Tensor
            Predicted scalar output of shape (batch_size,).
        """
        out, _ = self.lstm(x)          # (B, T, H)
        last_hidden = out[:, -1, :]    # Take the last time step
        yhat = self.head(last_hidden)  # (B, 1)
        return yhat.squeeze(-1)
