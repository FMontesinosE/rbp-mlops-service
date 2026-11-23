"""Tests for API module."""

import pytest


def test_predict_valid_sequences() -> None:
    """POST /predict with valid 101bp sequences."""
    # Note: This test requires a trained model to be registered
    # Skipping in unit tests, covered by e2e test
    pytest.skip("Requires trained model, covered in e2e test")


def test_predict_invalid_length() -> None:
    """POST /predict with wrong length -> 422."""
    from rbp_mlops.api import PredictRequest

    with pytest.raises(ValueError, match="Sequence length must be 101bp"):
        PredictRequest(sequences=["ACGT" * 20])  # 80bp, not 101


def test_predict_invalid_characters() -> None:
    """POST /predict with invalid chars -> validation error."""
    from rbp_mlops.api import PredictRequest

    with pytest.raises(ValueError, match="must contain only ACGTN"):
        PredictRequest(sequences=["X" * 101])


def test_predict_batch_size_limits() -> None:
    """POST /predict enforces min/max batch size."""
    from pydantic import ValidationError

    from rbp_mlops.api import PredictRequest

    # Empty list should fail
    with pytest.raises(ValidationError):
        PredictRequest(sequences=[])

    # Too many sequences should fail
    with pytest.raises(ValidationError):
        PredictRequest(sequences=["A" * 101 for _ in range(257)])


def test_health_endpoint_structure() -> None:
    """Health response has correct structure."""
    from rbp_mlops.api import HealthResponse

    response = HealthResponse(
        model_name="test",
        model_version="1",
        device="cpu",
        status="healthy",
    )

    assert response.model_name == "test"
    assert response.model_version == "1"
    assert response.device == "cpu"
    assert response.status == "healthy"


def test_predict_response_structure() -> None:
    """Predict response has correct structure."""
    from rbp_mlops.api import PredictResponse

    response = PredictResponse(
        predictions=[0.5, 0.8],
        binary_calls=[False, True],
        drift_js_divergence=0.05,
        drift_gc_zscore=1.2,
        drift_detected=False,
        latency_ms=10.5,
    )

    assert len(response.predictions) == 2
    assert len(response.binary_calls) == 2
    assert response.drift_js_divergence == 0.05
    assert not response.drift_detected
