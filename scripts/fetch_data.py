"""Fetch ENCODE eCLIP data and reference genome."""

import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import requests
import typer

app = typer.Typer()

DATA_DIR = Path("data/raw")
CHECKSUM_FILE = DATA_DIR / "checksums.json"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_with_retry(url: str, output_path: Path, max_retries: int = 3) -> None:
    """Download file with retry logic."""
    for attempt in range(max_retries):
        try:
            typer.echo(f"Downloading {url} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            typer.echo(f"Downloaded {output_path.name}")
            return

        except (OSError, requests.RequestException) as e:
            typer.echo(f"Download failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                typer.echo(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise


@app.command()
def main(
    force: bool = typer.Option(False, help="Force re-download even if files exist"),
) -> None:
    """Fetch ENCODE RBFOX2 eCLIP data and GRCh38 reference genome."""
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing checksums
    checksums = {}
    if CHECKSUM_FILE.exists():
        with open(CHECKSUM_FILE) as f:
            checksums = json.load(f)

    # Download chr21 and chr22 FASTA
    chromosomes = ["chr21", "chr22"]
    fasta_files = []

    for chrom in chromosomes:
        url = f"http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{chrom}.fa.gz"
        gz_path = DATA_DIR / f"{chrom}.fa.gz"
        fa_path = DATA_DIR / f"{chrom}.fa"

        # Check if already downloaded and verified
        if not force and fa_path.exists() and str(fa_path) in checksums:
            current_checksum = compute_sha256(fa_path)
            if current_checksum == checksums[str(fa_path)]:
                typer.echo(f"{fa_path.name} already exists and verified, skipping")
                fasta_files.append(fa_path)
                continue

        # Download
        download_with_retry(url, gz_path)

        # Decompress
        typer.echo(f"Decompressing {gz_path.name}...")
        with gzip.open(gz_path, "rb") as f_in, open(fa_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Compute checksum
        checksum = compute_sha256(fa_path)
        checksums[str(fa_path)] = checksum

        # Remove .gz file
        gz_path.unlink()

        fasta_files.append(fa_path)

    # Merge FASTA files
    merged_fasta = DATA_DIR / "GRCh38.fa"
    typer.echo(f"Merging FASTA files into {merged_fasta.name}...")
    with open(merged_fasta, "wb") as outfile:
        for fa_path in fasta_files:
            with open(fa_path, "rb") as infile:
                outfile.write(infile.read())

    checksums[str(merged_fasta)] = compute_sha256(merged_fasta)

    # Note: For narrowPeak download, users should manually download from ENCODE
    # due to API complexity and authentication requirements
    typer.echo("\n" + "=" * 60)
    typer.echo("FASTA download complete!")
    typer.echo("\nTo download RBFOX2 eCLIP narrowPeak file:")
    typer.echo("1. Visit: https://www.encodeproject.org/search/?type=Experiment")
    typer.echo("2. Search for: assay_title=eCLIP AND target.label=RBFOX2")
    typer.echo("3. Download a replicated narrowPeak file")
    typer.echo(f"4. Save as: {DATA_DIR}/rbfox2.bed")
    typer.echo("=" * 60)

    # Save checksums
    with open(CHECKSUM_FILE, "w") as f:
        json.dump(checksums, f, indent=2)

    typer.echo(f"\nChecksums saved to {CHECKSUM_FILE}")


if __name__ == "__main__":
    app()
