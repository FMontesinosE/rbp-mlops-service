"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def test_bed_file(test_data_dir: Path) -> Path:
    """Return path to test BED file."""
    return test_data_dir / "sample_peaks.bed"


@pytest.fixture
def test_fasta_file(test_data_dir: Path) -> Path:
    """Return path to test FASTA file."""
    return test_data_dir / "sample_genome.fa"
