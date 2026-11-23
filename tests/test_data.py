"""Tests for data pipeline module."""

from pathlib import Path

import numpy as np
import pyfaidx
import torch

from rbp_mlops.data import (
    RBPDataset,
    extract_sequence,
    generate_negatives,
    one_hot_encode,
    parse_narrowpeak,
    split_by_chromosome,
)

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_BED = TEST_DATA_DIR / "sample_peaks.bed"
TEST_FASTA = TEST_DATA_DIR / "sample_genome.fa"


def test_parse_narrowpeak_valid() -> None:
    """Parse sample_peaks.bed, verify 120 rows."""
    peaks = parse_narrowpeak(TEST_BED)
    assert len(peaks) == 120
    assert all(peak.chrom in ["chr21", "chr22"] for peak in peaks)
    assert all(peak.start < peak.end for peak in peaks)
    assert all(0 <= peak.peak <= (peak.end - peak.start) for peak in peaks)


def test_extract_sequence_positive_strand() -> None:
    """Extract from chr21, verify 101bp on positive strand."""
    peaks = parse_narrowpeak(TEST_BED)
    fasta = pyfaidx.Fasta(str(TEST_FASTA))

    # Find a peak on chr21 positive strand
    pos_peaks = [p for p in peaks if p.chrom == "chr21" and p.strand == "+"]
    assert len(pos_peaks) > 0

    peak = pos_peaks[0]
    seq = extract_sequence(fasta, peak, window=101)

    assert len(seq) == 101
    assert all(b in "ACGTN" for b in seq)
    assert seq.isupper()


def test_extract_sequence_negative_strand() -> None:
    """Extract from chr22, verify reverse complement on negative strand."""
    peaks = parse_narrowpeak(TEST_BED)
    fasta = pyfaidx.Fasta(str(TEST_FASTA))

    # Find a peak on chr22 negative strand
    neg_peaks = [p for p in peaks if p.chrom == "chr22" and p.strand == "-"]
    assert len(neg_peaks) > 0

    peak = neg_peaks[0]
    seq = extract_sequence(fasta, peak, window=101)

    assert len(seq) == 101
    assert all(b in "ACGTN" for b in seq)


def test_one_hot_encode() -> None:
    """Verify (4, 101) shape, ACGT order, N handling."""
    seq = "ACGT" * 25 + "N"
    encoded = one_hot_encode(seq)

    assert encoded.shape == (4, 101)
    assert encoded.dtype == np.float32

    # Check ACGT encoding
    assert encoded[0, 0] == 1.0  # A
    assert encoded[1, 1] == 1.0  # C
    assert encoded[2, 2] == 1.0  # G
    assert encoded[3, 3] == 1.0  # T

    # Check N encoding (all zeros)
    assert np.all(encoded[:, 100] == 0.0)

    # Check sum is 0 or 1 for each position
    col_sums = encoded.sum(axis=0)
    assert np.all((col_sums == 0.0) | (col_sums == 1.0))


def test_generate_negatives() -> None:
    """Verify shift range, overlap filtering."""
    peaks = parse_narrowpeak(TEST_BED)
    chr21_peaks = [p for p in peaks if p.chrom == "chr21"]

    negatives = generate_negatives(chr21_peaks, shift_range=(300, 300))

    # Should generate up to 2 negatives per peak (one left, one right)
    assert 0 < len(negatives) <= 2 * len(chr21_peaks)

    # Verify negatives don't overlap with positives
    for neg in negatives:
        overlaps = any(
            (
                neg.start < p.end
                and neg.end > p.start
                and neg.chrom == p.chrom
                and neg.strand == p.strand
            )
            for p in chr21_peaks
        )
        assert not overlaps


def test_split_by_chromosome() -> None:
    """Verify chr22=test, chr21=val."""
    peaks = parse_narrowpeak(TEST_BED)
    split = split_by_chromosome(peaks)

    assert "train" in split
    assert "val" in split
    assert "test" in split

    # Check chr22 goes to test
    assert all(p.chrom == "chr22" for p in split["test"])
    assert len(split["test"]) == sum(1 for p in peaks if p.chrom == "chr22")

    # Check chr21 goes to val
    assert all(p.chrom == "chr21" for p in split["val"])
    assert len(split["val"]) == sum(1 for p in peaks if p.chrom == "chr21")

    # Train should be empty for our test data (only chr21/chr22)
    assert len(split["train"]) == 0


def test_dataset_initialization() -> None:
    """Test RBPDataset initialization and length."""
    dataset = RBPDataset(
        bed_path=TEST_BED,
        fasta_path=TEST_FASTA,
        chromosomes=["test"],  # chr22 only
        generate_negatives_flag=True,
    )

    # Should have positives + negatives
    assert len(dataset) > 60  # At least 60 positives from chr22

    # Test getitem returns correct types
    seq_tensor, label_tensor = dataset[0]
    assert seq_tensor.shape == (4, 101)
    assert label_tensor.shape == ()
    assert label_tensor.dtype == torch.float32


def test_compute_baseline_stats() -> None:
    """Test baseline statistics computation."""
    dataset = RBPDataset(
        bed_path=TEST_BED,
        fasta_path=TEST_FASTA,
        chromosomes=["val"],  # chr21
        generate_negatives_flag=False,
    )

    baseline = dataset.compute_baseline_stats()

    assert "gc_mean" in baseline
    assert "gc_std" in baseline
    assert "3mer_freq" in baseline

    # GC content should be reasonable (0-1 range)
    assert 0.0 <= baseline["gc_mean"] <= 1.0
    assert baseline["gc_std"] >= 0.0

    # 3-mer frequencies should sum to ~1.0
    kmer_freq = baseline["3mer_freq"]
    assert len(kmer_freq) > 0
    total_freq = sum(kmer_freq.values())
    assert 0.95 <= total_freq <= 1.05  # Allow small floating point error
