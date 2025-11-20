#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Convolutional Network (TCN) for univariate regression.

This module provides:
- CausalConv1d: left-padded 1D convolution to preserve causality.
- TemporalBlock: two-layer causal residual block with dropout.
- TemporalConvNet: stack of TemporalBlocks + linear head.

Input/Output conventions
------------------------
- Model expects inputs shaped (B, F, T), where:
  B = batch size
  F = number of features (channels)
  T = sequence length (timesteps)
- The forward pass returns a 1-D prediction per sequence: shape (B,).

Notes
-----
- This implementation is architecture-agnostic with respect to the rest of the
  pipeline; it only assumes that the dataloader supplies (B, F, T).
- Dilations are taken from the provided sequence; if `num_blocks` exceeds the
  length of `dilations`, the last dilation value is reused for remaining blocks.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution implemented via left-padding.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size (time dimension).
    dilation : int
        Dilation factor.
    bias : bool, optional
        Whether to include bias in the convolution, by default True.

    Notes
    -----
    - Uses ConstantPad1d to pad only on the left by `(kernel_size - 1) * dilation`
      to ensure outputs at time t do not depend on future inputs (> t).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        pad_left: int = (kernel_size - 1) * dilation
        self.pad_layer = nn.ConstantPad1d((pad_left, 0), 0.0)  # (left, right)
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # padding handled explicitly for causality
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, T).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, T).
        """
        return self.conv(self.pad_layer(x))


class TemporalBlock(nn.Module):
    """
    Residual temporal block with two causal convolutions.

    Structure
    ---------
    [CausalConv1d -> ReLU -> Dropout] x 2  + residual (1x1 conv if channels change)

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel_size : int
        Kernel size for causal convolutions.
    dilation : int
        Dilation factor used in both convolutions of this block.
    dropout : float
        Dropout probability applied after each ReLU.

    Notes
    -----
    - If `in_ch != out_ch`, a 1x1 Conv1d is applied to the residual path to
      match channel dimensions before addition.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample: nn.Module = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, T).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, T).
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = self.downsample(x)
        return out + res  # residual connection


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network backbone for sequence regression.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (F).
    channels : int
        Base channel width for each residual block.
    num_blocks : int
        Number of residual blocks to stack (depth).
    kernel_size : int
        Temporal kernel size for all blocks (e.g., 3).
    dilations : Tuple[int, ...]
        Dilation factors. If `num_blocks` > len(dilations), the last value is repeated.
    dropout : float
        Dropout probability for blocks.

    Notes
    -----
    - The head is a linear layer applied to the final timestep's features to
      yield a scalar regression output per sequence.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_blocks: int,
        kernel_size: int,
        dilations: Tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        layers = []
        c_in = in_channels
        for i in range(num_blocks):
            d = dilations[i] if i < len(dilations) else dilations[-1]
            layers.append(TemporalBlock(c_in, channels, kernel_size, d, dropout))
            c_in = channels

        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, F, T).

        Returns
        -------
        torch.Tensor
            Predicted values of shape (B,).
        """
        y = self.network(x)     # (B, C, T)
        last = y[:, :, -1]      # (B, C)
        out = self.head(last)   # (B, 1)
        return out.squeeze(-1)
