"""Drift detection for sequence data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_3mer_frequencies(sequences: list[str]) -> dict[str, float]:
    """Count 3-mer frequencies across sequences.

    Args:
        sequences: List of DNA sequences.

    Returns:
        Dictionary mapping 3-mers to their frequencies (sum to 1.0).
    """
    kmer_counts: dict[str, int] = {}
    total_kmers = 0

    for seq in sequences:
        seq_upper = seq.upper()
        for i in range(len(seq_upper) - 2):
            kmer = seq_upper[i : i + 3]
            # Only count kmers with valid bases
            if all(b in "ACGT" for b in kmer):
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                total_kmers += 1

    # Normalize to frequencies
    if total_kmers == 0:
        return {}

    return {k: v / total_kmers for k, v in kmer_counts.items()}


def compute_gc_content(sequences: list[str]) -> tuple[float, float]:
    """Return (mean, std) of GC content.

    Args:
        sequences: List of DNA sequences.

    Returns:
        Tuple of (mean, std) GC content.
    """
    gc_contents = []
    for seq in sequences:
        seq_upper = seq.upper()
        gc_count = seq_upper.count("G") + seq_upper.count("C")
        total_count = sum(1 for b in seq_upper if b in "ACGT")
        if total_count > 0:
            gc_contents.append(gc_count / total_count)

    if not gc_contents:
        return (0.0, 0.0)

    gc_array = np.array(gc_contents)
    return (float(np.mean(gc_array)), float(np.std(gc_array)))


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two 3-mer distributions.

    Args:
        p: First distribution (3-mer frequencies).
        q: Second distribution (3-mer frequencies).

    Returns:
        JS divergence value in [0, 1].
    """
    # Get all kmers from both distributions
    all_kmers = sorted(set(p.keys()) | set(q.keys()))

    if not all_kmers:
        return 0.0

    # Build aligned probability vectors
    p_vec = np.array([p.get(k, 0.0) for k in all_kmers])
    q_vec = np.array([q.get(k, 0.0) for k in all_kmers])

    # Normalize to ensure they sum to 1
    p_sum = p_vec.sum()
    q_sum = q_vec.sum()
    if p_sum > 0:
        p_vec = p_vec / p_sum
    if q_sum > 0:
        q_vec = q_vec / q_sum

    # Compute JS divergence
    return float(jensenshannon(p_vec, q_vec))


def gc_zscore(
    sequences: list[str],
    baseline_mean: float,
    baseline_std: float,
) -> float:
    """Z-score of incoming GC vs. baseline.

    Args:
        sequences: List of DNA sequences.
        baseline_mean: Baseline mean GC content.
        baseline_std: Baseline std GC content.

    Returns:
        Z-score of incoming GC content.
    """
    incoming_mean, _ = compute_gc_content(sequences)

    if baseline_std == 0.0:
        return 0.0

    return (incoming_mean - baseline_mean) / baseline_std


@dataclass
class DriftMetrics:
    """Drift detection metrics."""

    js_divergence_3mer: float
    gc_zscore: float
    is_drifted: bool


def compute_drift(
    sequences: list[str],
    baseline: dict[str, Any],
) -> DriftMetrics:
    """Compute drift metrics against training baseline.

    Args:
        sequences: List of DNA sequences to check.
        baseline: Baseline statistics dictionary with keys:
            - '3mer_freq': 3-mer frequency distribution
            - 'gc_mean': Mean GC content
            - 'gc_std': Std GC content

    Returns:
        DriftMetrics object with JS divergence, GC Z-score, and drift flag.
    """
    # Compute 3-mer JS divergence
    incoming_3mer = compute_3mer_frequencies(sequences)
    baseline_3mer = baseline.get("3mer_freq", {})
    js_div = js_divergence(incoming_3mer, baseline_3mer)

    # Compute GC Z-score
    baseline_gc_mean = baseline.get("gc_mean", 0.5)
    baseline_gc_std = baseline.get("gc_std", 0.1)
    gc_z = gc_zscore(sequences, baseline_gc_mean, baseline_gc_std)

    # Drift detection thresholds
    is_drifted = (js_div > 0.1) or (abs(gc_z) > 3.0)

    return DriftMetrics(
        js_divergence_3mer=js_div,
        gc_zscore=gc_z,
        is_drifted=is_drifted,
    )
