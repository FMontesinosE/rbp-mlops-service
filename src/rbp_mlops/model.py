"""Neural network model for RBP binding site prediction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize Squeeze-Excitation block.

        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for bottleneck (default 4).
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, length).

        Returns:
            Recalibrated tensor of same shape.
        """
        batch, channels, _ = x.size()

        # Squeeze: global pooling
        squeeze = self.pool(x).view(batch, channels)

        # Excitation: FC layers
        excitation = self.fc1(squeeze)
        excitation = self.activation(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        # Recalibrate
        excitation = excitation.view(batch, channels, 1)
        return x * excitation


class RBPNet1D(nn.Module):
    """1D-CNN + BiGRU model for RBP binding site prediction.

    Architecture:
    - Conv1D(4→64, k=9) + BN + GELU + Dropout(0.2)
    - Conv1D(64→128, k=5) + BN + GELU + Dropout(0.2)
    - SqueezeExcitation(128)
    - BiGRU(128→64 hidden, bidirectional)
    - Linear(128→1, raw logits)
    """

    def __init__(
        self,
        input_channels: int = 4,
        conv1_out: int = 64,
        conv2_out: int = 128,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        """Initialize RBPNet1D model.

        Args:
            input_channels: Number of input channels (default 4 for ACGT).
            conv1_out: Output channels for first conv layer.
            conv2_out: Output channels for second conv layer.
            gru_hidden: Hidden size for GRU (bidirectional, so output is 2*gru_hidden).
            dropout: Dropout probability.
        """
        super().__init__()

        # First conv block
        self.conv1 = nn.Conv1d(input_channels, conv1_out, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv block
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        # Squeeze-Excitation
        self.se = SqueezeExcitation(conv2_out)

        # Bidirectional GRU
        self.gru = nn.GRU(
            conv2_out,
            gru_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # Output layer (raw logits)
        self.fc = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 4, 101).

        Returns:
            Output tensor of shape (batch, 1) with raw logits.
        """
        # Conv blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)

        # Squeeze-Excitation
        x = self.se(x)

        # Reshape for GRU: (batch, channels, length) -> (batch, length, channels)
        x = x.transpose(1, 2)

        # BiGRU
        x, _ = self.gru(x)

        # Global max pooling over sequence length
        x, _ = torch.max(x, dim=1)

        # Output layer
        x = self.fc(x)

        return x


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: For full determinism on CUDA, uncomment below
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def export_torchscript(model: nn.Module, path: Path) -> None:
    """Export model to TorchScript for production.

    Args:
        model: PyTorch model to export.
        path: Output path for TorchScript file.
    """
    model.eval()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(path))
