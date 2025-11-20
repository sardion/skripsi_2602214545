#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    Lightweight LSTM regressor: last-step pooling + small MLP head.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)       # (B, T, H)
        last = out[:, -1, :]        # (B, H)
        return self.head(last).squeeze(-1)
