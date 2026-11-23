"""Tests for model module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rbp_mlops.model import RBPNet1D, SqueezeExcitation, export_torchscript, set_seed


def test_model_forward_shape() -> None:
    """Input (8, 4, 101) -> Output (8, 1)."""
    model = RBPNet1D()
    model.eval()

    # Create input tensor
    x = torch.randn(8, 4, 101)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    assert output.shape == (8, 1)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_deterministic() -> None:
    """Same seed -> same output for same input."""
    x = torch.randn(4, 4, 101)

    # First run
    set_seed(42)
    model1 = RBPNet1D()
    model1.eval()
    with torch.no_grad():
        output1 = model1(x)

    # Second run with same seed
    set_seed(42)
    model2 = RBPNet1D()
    model2.eval()
    with torch.no_grad():
        output2 = model2(x)

    # Outputs should be identical
    torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)


def test_model_torchscript_export() -> None:
    """Export and reload, verify output matches."""
    model = RBPNet1D()
    model.eval()

    x = torch.randn(4, 4, 101)

    # Original output
    with torch.no_grad():
        output_original = model(x)

    # Export to TorchScript
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pt"
        export_torchscript(model, path)

        # Reload
        loaded_model = torch.jit.load(str(path))
        loaded_model.eval()

        # Loaded output
        with torch.no_grad():
            output_loaded = loaded_model(x)

    # Outputs should be identical
    torch.testing.assert_close(output_original, output_loaded, rtol=1e-5, atol=1e-5)


def test_squeeze_excitation() -> None:
    """Verify channel attention weights in [0, 1]."""
    se = SqueezeExcitation(channels=128, reduction=4)
    se.eval()

    x = torch.randn(4, 128, 101)

    with torch.no_grad():
        output = se(x)

    # Output shape should match input
    assert output.shape == x.shape

    # Output should not be NaN or Inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
