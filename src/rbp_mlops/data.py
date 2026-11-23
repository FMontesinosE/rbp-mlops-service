"""Data pipeline for RNA-binding protein binding site prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyfaidx
import torch
from intervaltree import Interval, IntervalTree
from torch.utils.data import Dataset


@dataclass
class Peak:
    """ENCODE narrowPeak representation."""

    chrom: str
    start: int
    end: int
    name: str
    score: float
    strand: str
    signal_value: float
    p_value: float
    q_value: float
    peak: int  # Offset from start to summit


def parse_narrowpeak(bed_path: Path) -> list[Peak]:
    """Parse ENCODE 10-column narrowPeak format.

    Args:
        bed_path: Path to narrowPeak BED file.

    Returns:
        List of Peak objects.
    """
    peaks = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) < 10:
                continue

            try:
                start = int(fields[1])
                end = int(fields[2])
                peak_offset = int(fields[9])

                # Validate coordinates
                if start < 0 or end <= start:
                    continue
                if peak_offset < 0 or peak_offset > (end - start):
                    peak_offset = (end - start) // 2  # Use midpoint as fallback

                peak = Peak(
                    chrom=fields[0],
                    start=start,
                    end=end,
                    name=fields[3],
                    score=float(fields[4]),
                    strand=fields[5] if fields[5] in ["+", "-"] else "+",
                    signal_value=float(fields[6]),
                    p_value=float(fields[7]),
                    q_value=float(fields[8]),
                    peak=peak_offset,
                )
                peaks.append(peak)
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

    return peaks


def _reverse_complement(seq: str) -> str:
    """Reverse complement DNA sequence."""
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement.get(b.upper(), "N") for b in reversed(seq))


def extract_sequence(fasta: pyfaidx.Fasta, peak: Peak, window: int = 101) -> str:
    """Extract strand-aware 101nt sequence centered on peak summit.

    Args:
        fasta: pyfaidx.Fasta object for genome access.
        peak: Peak object with summit information.
        window: Window size (default 101bp).

    Returns:
        DNA sequence string (uppercase). Reverse complemented if strand is '-'.
        Returns empty string if extraction fails.
    """
    try:
        summit = peak.start + peak.peak
        half_window = window // 2
        start = max(0, summit - half_window)  # Ensure non-negative
        end = summit + half_window + 1  # +1 for inclusive end

        # Check if chromosome exists
        if peak.chrom not in fasta:
            return ""

        # Get chromosome length
        chrom_len = len(fasta[peak.chrom])

        # Adjust end if beyond chromosome length
        if end > chrom_len:
            end = chrom_len

        # Extract sequence
        seq = str(fasta[peak.chrom][start:end])

        # Pad if sequence is shorter than window (edge cases)
        if len(seq) < window:
            # Pad with N's
            seq = "N" * (window - len(seq)) + seq if start == 0 else seq + "N" * (window - len(seq))

        # Ensure exactly window length
        seq = seq[:window]

        # Reverse complement if negative strand
        if peak.strand == "-":
            seq = _reverse_complement(seq)

        return seq.upper()
    except (KeyError, IndexError):
        # Return empty string on error
        return ""


def generate_negatives(peaks: list[Peak], shift_range: tuple[int, int] = (300, 300)) -> list[Peak]:
    """Generate negative intervals by shifting peaks.

    Shifts peaks +/- shift_range[0] and shift_range[1] on same chromosome/strand.
    Filters out negatives overlapping with any positive peak.

    Args:
        peaks: List of positive peaks.
        shift_range: Tuple of (left_shift, right_shift) in bp.

    Returns:
        List of negative Peak objects.
    """
    # Build interval tree for overlap detection
    trees: dict[str, IntervalTree] = {}
    for peak in peaks:
        key = f"{peak.chrom}:{peak.strand}"
        if key not in trees:
            trees[key] = IntervalTree()
        trees[key].add(Interval(peak.start, peak.end))

    negatives = []
    for peak in peaks:
        key = f"{peak.chrom}:{peak.strand}"
        tree = trees[key]

        # Generate negative by shifting
        for shift in [-shift_range[0], shift_range[1]]:
            neg_start = peak.start + shift
            neg_end = peak.end + shift

            # Skip if negative or would go out of reasonable bounds
            if neg_start < 0:
                continue

            # Check for overlaps
            if not tree.overlap(neg_start, neg_end):
                neg_peak = Peak(
                    chrom=peak.chrom,
                    start=neg_start,
                    end=neg_end,
                    name=f"{peak.name}_neg_{shift:+d}",
                    score=0.0,
                    strand=peak.strand,
                    signal_value=0.0,
                    p_value=1.0,
                    q_value=1.0,
                    peak=peak.peak,  # Keep same summit offset
                )
                negatives.append(neg_peak)

    return negatives


def one_hot_encode(sequence: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding.

    Args:
        sequence: DNA sequence string.

    Returns:
        NumPy array of shape (4, len(sequence)) with dtype float32.
        Order: ACGT. N bases are encoded as zeros.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, len(sequence)), dtype=np.float32)

    for i, base in enumerate(sequence.upper()):
        idx = mapping.get(base, -1)
        if idx >= 0:
            encoded[idx, i] = 1.0

    return encoded


def split_by_chromosome(peaks: list[Peak]) -> dict[str, list[Peak]]:
    """Split peaks by chromosome for train/val/test.

    Chromosome assignment:
    - test: chr22
    - val: chr21
    - train: all others

    Args:
        peaks: List of all peaks.

    Returns:
        Dictionary with keys 'train', 'val', 'test' and lists of peaks.
    """
    split: dict[str, list[Peak]] = {"train": [], "val": [], "test": []}

    for peak in peaks:
        if peak.chrom == "chr22":
            split["test"].append(peak)
        elif peak.chrom == "chr21":
            split["val"].append(peak)
        else:
            split["train"].append(peak)

    return split


class RBPDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for RBP binding site prediction."""

    def __init__(
        self,
        bed_path: Path,
        fasta_path: Path,
        chromosomes: list[str],
        generate_negatives_flag: bool = True,
        window: int = 101,
    ) -> None:
        """Initialize RBP dataset.

        Args:
            bed_path: Path to narrowPeak BED file.
            fasta_path: Path to FASTA genome file.
            chromosomes: List of chromosome splits to include ('train', 'val', 'test').
            generate_negatives_flag: Whether to generate negative examples.
            window: Sequence window size (default 101bp).
        """
        self.bed_path = bed_path
        self.fasta_path = fasta_path
        self.chromosomes = chromosomes
        self.window = window

        # Load genome
        self.fasta = pyfaidx.Fasta(str(fasta_path))

        # Parse peaks and split by chromosome
        all_peaks = parse_narrowpeak(bed_path)
        split_peaks = split_by_chromosome(all_peaks)

        # Collect peaks for requested chromosomes
        self.peaks: list[Peak] = []
        for chrom_set in chromosomes:
            if chrom_set in split_peaks:
                self.peaks.extend(split_peaks[chrom_set])

        # Generate negatives if requested
        self.labels: list[int] = [1] * len(self.peaks)
        if generate_negatives_flag:
            negatives = generate_negatives(self.peaks)
            self.peaks.extend(negatives)
            self.labels.extend([0] * len(negatives))

        # Cache sequences
        self.sequences: list[str] = []
        for peak in self.peaks:
            try:
                seq = extract_sequence(self.fasta, peak, window=self.window)
                self.sequences.append(seq)
            except (KeyError, IndexError):
                # Skip peaks that can't be extracted (at contig boundaries)
                self.sequences.append("N" * self.window)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.peaks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (sequence_tensor, label_tensor).
        """
        seq = self.sequences[idx]
        encoded = one_hot_encode(seq)
        label = self.labels[idx]

        return (
            torch.from_numpy(encoded),
            torch.tensor(label, dtype=torch.float32),
        )

    def compute_baseline_stats(self) -> dict[str, Any]:
        """Compute baseline statistics for drift detection.

        Returns:
            Dictionary with keys:
            - 'gc_mean': Mean GC content across sequences.
            - 'gc_std': Standard deviation of GC content.
            - '3mer_freq': Dictionary of 3-mer frequencies.
        """
        gc_contents = []
        kmer_counts: dict[str, int] = {}
        total_kmers = 0

        for seq in self.sequences:
            # GC content
            gc_count = seq.count("G") + seq.count("C")
            gc_contents.append(gc_count / len(seq))

            # 3-mer frequencies
            for i in range(len(seq) - 2):
                kmer = seq[i : i + 3]
                if all(b in "ACGT" for b in kmer):
                    kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                    total_kmers += 1

        # Normalize 3-mer counts to frequencies
        kmer_freq = {k: v / total_kmers for k, v in kmer_counts.items()} if total_kmers > 0 else {}

        gc_array = np.array(gc_contents)
        return {
            "gc_mean": float(np.mean(gc_array)),
            "gc_std": float(np.std(gc_array)),
            "3mer_freq": kmer_freq,
        }
