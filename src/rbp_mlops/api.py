"""FastAPI inference server for RBP binding prediction."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from rbp_mlops.data import one_hot_encode
from rbp_mlops.drift import compute_drift

# Global state
model: nn.Module | None = None
baseline: dict[str, Any] | None = None
model_version: str = "unknown"
device: str = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model and baseline on startup."""
    global model, baseline, model_version, device

    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model from MLflow registry
        client = mlflow.MlflowClient()

        try:
            # Try to load model with champion alias
            model_uri = "models:/rbp-binding-classifier@champion"
            model = mlflow.pytorch.load_model(model_uri)
            model.eval()
            model = model.to(device)

            # Get version info
            alias_info = client.get_model_version_by_alias("rbp-binding-classifier", "champion")
            model_version = alias_info.version
            run_id = alias_info.run_id

            # Load baseline stats
            artifact_path = client.download_artifacts(run_id, "baseline_stats.json")
            with open(artifact_path) as f:
                baseline = json.load(f)

        except Exception:
            # Fallback: try to load latest version
            latest_versions = client.get_latest_versions("rbp-binding-classifier")
            if latest_versions:
                model_uri = f"models:/rbp-binding-classifier/{latest_versions[0].version}"
                model = mlflow.pytorch.load_model(model_uri)
                model.eval()
                model = model.to(device)
                model_version = latest_versions[0].version
                run_id = latest_versions[0].run_id

                # Load baseline stats
                artifact_path = client.download_artifacts(run_id, "baseline_stats.json")
                with open(artifact_path) as f:
                    baseline = json.load(f)
            else:
                raise RuntimeError("No model found in registry") from None

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    yield

    # Cleanup
    model = None
    baseline = None


app = FastAPI(lifespan=lifespan, title="RBP MLOps API", version="0.1.0")


class HealthResponse(BaseModel):
    """Health check response."""

    model_name: str
    model_version: str
    device: str
    status: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        model_name="rbp-binding-classifier",
        model_version=model_version,
        device=device,
        status="healthy",
    )


class PredictRequest(BaseModel):
    """Prediction request."""

    sequences: list[str] = Field(..., min_length=1, max_length=256)

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: list[str]) -> list[str]:
        """Validate sequences."""
        for seq in v:
            if len(seq) != 101:
                raise ValueError(f"Sequence length must be 101bp, got {len(seq)}")
            seq_upper = seq.upper()
            if not all(c in "ACGTN" for c in seq_upper):
                raise ValueError("Sequences must contain only ACGTN characters")
        return [s.upper() for s in v]


class PredictResponse(BaseModel):
    """Prediction response."""

    predictions: list[float]
    binary_calls: list[bool]
    drift_js_divergence: float
    drift_gc_zscore: float
    drift_detected: bool
    latency_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict binding probabilities."""
    if model is None or baseline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()

    # Encode sequences
    tensors = torch.stack([torch.from_numpy(one_hot_encode(seq)) for seq in request.sequences]).to(
        device
    )

    # Predict
    with torch.no_grad():
        logits = model(tensors).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().tolist()

    # Binary calls (threshold 0.5)
    binary_calls = [p > 0.5 for p in probs]

    # Drift detection
    drift = compute_drift(request.sequences, baseline)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return PredictResponse(
        predictions=probs,
        binary_calls=binary_calls,
        drift_js_divergence=drift.js_divergence_3mer,
        drift_gc_zscore=drift.gc_zscore,
        drift_detected=drift.is_drifted,
        latency_ms=latency_ms,
    )
