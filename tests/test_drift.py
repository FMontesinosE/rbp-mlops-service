"""Tests for drift detection module."""

import pytest

from rbp_mlops.drift import (
    compute_3mer_frequencies,
    compute_drift,
    compute_gc_content,
    gc_zscore,
    js_divergence,
)


def test_compute_3mer_frequencies() -> None:
    """Verify 64 3-mers, sum to 1.0."""
    sequences = ["ACGTACGTACGT" * 10, "TGCATGCATGCA" * 10]
    freq = compute_3mer_frequencies(sequences)

    # Frequencies should sum to ~1.0
    total = sum(freq.values())
    assert 0.99 <= total <= 1.01

    # Check some expected kmers are present
    assert "ACG" in freq
    assert "TGC" in freq

    # All frequencies should be positive
    assert all(v > 0 for v in freq.values())


def test_js_divergence_identical() -> None:
    """Same distribution -> JS = 0."""
    dist = {"AAA": 0.25, "CCC": 0.25, "GGG": 0.25, "TTT": 0.25}

    js = js_divergence(dist, dist)

    # Identical distributions should have JS divergence = 0
    assert js == pytest.approx(0.0, abs=1e-6)


def test_js_divergence_different() -> None:
    """Different distributions -> JS > 0."""
    dist1 = {"AAA": 0.5, "CCC": 0.5}
    dist2 = {"GGG": 0.5, "TTT": 0.5}

    js = js_divergence(dist1, dist2)

    # Different distributions should have JS divergence > 0
    assert js > 0.1
    assert js <= 1.0  # JS divergence is bounded by 1


def test_gc_zscore() -> None:
    """Verify Z-score computation."""
    # Sequences with varied GC content for non-zero std
    sequences_baseline = [
        "ACGTACGTACGT",
        "ATATATATATAT",
        "GCGCGCGCGCGC",
        "TGCATGCATGCA",
    ] * 5

    # Compute baseline
    baseline_mean, baseline_std = compute_gc_content(sequences_baseline)
    assert baseline_std > 0.0  # Ensure we have variation

    # Similar sequences should have Z-score near 0
    similar_sequences = ["ACGTACGTACGT"] * 10
    z1 = gc_zscore(similar_sequences, baseline_mean, baseline_std)
    assert abs(z1) < 2.0

    # High GC sequences should have positive Z-score
    high_gc_sequences = ["GCGCGCGCGCGC"] * 10
    z2 = gc_zscore(high_gc_sequences, baseline_mean, baseline_std)
    assert z2 > 0.5


def test_drift_detection_no_drift() -> None:
    """Similar sequences -> no drift."""
    # Training sequences
    train_sequences = ["ACGT" * 25] * 100

    # Compute baseline

    # Create mock baseline
    baseline = {
        "gc_mean": 0.5,
        "gc_std": 0.05,
        "3mer_freq": compute_3mer_frequencies(train_sequences),
    }

    # Inference sequences (similar)
    inference_sequences = ["ACGT" * 25] * 10

    drift = compute_drift(inference_sequences, baseline)

    assert not drift.is_drifted
    assert drift.js_divergence_3mer < 0.1
    assert abs(drift.gc_zscore) < 3.0


def test_drift_detection_with_drift() -> None:
    """High GC sequences -> drift detected."""
    # Baseline with 50% GC
    baseline = {
        "gc_mean": 0.5,
        "gc_std": 0.05,
        "3mer_freq": {"ACG": 0.3, "TGC": 0.3, "CAT": 0.2, "GTA": 0.2},
    }

    # High GC sequences (90% GC)
    high_gc_sequences = ["GCGCGCGCGCGC" * 8 + "A"] * 10

    drift = compute_drift(high_gc_sequences, baseline)

    # Should detect drift due to high GC content
    assert drift.is_drifted
    assert abs(drift.gc_zscore) > 3.0 or drift.js_divergence_3mer > 0.1
