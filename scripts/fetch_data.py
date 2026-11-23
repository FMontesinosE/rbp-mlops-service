"""Fetch ENCODE eCLIP data and reference genome."""

import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import requests
import typer
from tqdm import tqdm

app = typer.Typer()

DATA_DIR = Path("data/raw")
CHECKSUM_FILE = DATA_DIR / "checksums.json"
ENCODE_API_BASE = "https://www.encodeproject.org"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_with_retry(
    url: str, output_path: Path, max_retries: int = 3, show_progress: bool = True
) -> None:
    """Download file with retry logic and progress bar."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192

            with open(output_path, "wb") as f:
                if show_progress and total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=output_path.name,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=block_size):
                        f.write(chunk)

            return

        except (OSError, requests.RequestException) as e:
            typer.echo(f"Download failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                typer.echo(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise


def query_encode_eclip(target_label: str | None = None) -> list[dict]:
    """Query ENCODE API for eCLIP experiments.

    Args:
        target_label: Specific RBP target (e.g., "RBFOX2"). If None, fetch all eCLIP.

    Returns:
        List of experiment metadata dictionaries.
    """
    search_url = f"{ENCODE_API_BASE}/search/"
    params = {
        "type": "Experiment",
        "assay_title": "eCLIP",
        "status": "released",
        "files.file_type": "bed narrowPeak",
        "replicates.library.biosample.donor.organism.scientific_name": "Homo sapiens",
        "limit": "all",
        "format": "json",
    }

    if target_label:
        params["target.label"] = target_label

    typer.echo(f"Querying ENCODE for eCLIP experiments (target={target_label or 'all'})...")
    response = requests.get(search_url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    experiments = data.get("@graph", [])
    typer.echo(f"Found {len(experiments)} experiments")

    return experiments


def download_narrowpeak_files(
    experiments: list[dict], output_dir: Path, limit: int | None = None
) -> list[Path]:
    """Download narrowPeak BED files from experiments.

    Args:
        experiments: List of experiment metadata from ENCODE API.
        output_dir: Directory to save files.
        limit: Maximum number of files to download (None = all).

    Returns:
        List of downloaded file paths.
    """
    downloaded_files = []
    experiment_subset = experiments[:limit] if limit else experiments

    typer.echo(f"\nDownloading narrowPeak files for {len(experiment_subset)} experiments...")

    for exp in tqdm(experiment_subset, desc="Experiments"):
        exp_id = exp.get("accession", "unknown")
        target = exp.get("target", {})
        target_label = target.get("label", "unknown") if isinstance(target, dict) else "unknown"

        # Get files from experiment
        files_url = f"{ENCODE_API_BASE}{exp['@id']}"
        response = requests.get(files_url, params={"format": "json"}, timeout=60)
        response.raise_for_status()
        exp_data = response.json()

        # Find replicated narrowPeak files (preferred) or any narrowPeak
        files = exp_data.get("files", [])
        narrowpeak_files = [
            f
            for f in files
            if f.get("file_type") == "bed narrowPeak"
            and f.get("status") == "released"
            and f.get("output_type") in ["replicated peaks", "peaks"]
        ]

        if not narrowpeak_files:
            continue

        # Prefer replicated peaks, fallback to first peaks file
        replicated = [f for f in narrowpeak_files if f.get("output_type") == "replicated peaks"]
        file_to_download = replicated[0] if replicated else narrowpeak_files[0]

        file_id = file_to_download.get("accession", "unknown")
        download_url = f"{ENCODE_API_BASE}{file_to_download['href']}"
        output_path = output_dir / f"{target_label}_{exp_id}_{file_id}.bed"

        try:
            download_with_retry(download_url, output_path, show_progress=False)
            downloaded_files.append(output_path)
        except Exception as e:
            typer.echo(f"Failed to download {file_id}: {e}")

    typer.echo(f"Downloaded {len(downloaded_files)} narrowPeak files")
    return downloaded_files


@app.command()
def main(
    force: bool = typer.Option(False, help="Force re-download even if files exist"),
    target: str | None = typer.Option(
        None, help="Specific RBP target (e.g., RBFOX2). If not set, downloads all eCLIP."
    ),
    limit: int | None = typer.Option(
        None, help="Limit number of experiments to download (useful for testing)"
    ),
    chromosomes_only: bool = typer.Option(
        False, help="Download only reference genome, skip narrowPeak files"
    ),
    test_mode: bool = typer.Option(False, help="Quick test mode (chr21-22 only)"),
) -> None:
    """Fetch ENCODE eCLIP data and GRCh38 reference genome.

    Examples:
        # Quick test with chr21-22 only
        python scripts/fetch_data.py --test-mode

        # Download all eCLIP experiments (WARNING: large dataset)
        python scripts/fetch_data.py

        # Download specific RBP target
        python scripts/fetch_data.py --target RBFOX2

        # Download genome only
        python scripts/fetch_data.py --chromosomes-only
    """
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing checksums
    checksums = {}
    if CHECKSUM_FILE.exists():
        with open(CHECKSUM_FILE) as f:
            checksums = json.load(f)

    # Determine chromosome list
    if test_mode:
        chromosomes = ["chr21", "chr22"]
        typer.echo("Test mode: downloading chr21 and chr22 only")
    else:
        chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        typer.echo(f"Full mode: downloading {len(chromosomes)} chromosomes")

    # Download FASTA files
    fasta_files = []
    typer.echo("\n" + "=" * 60)
    typer.echo("Downloading reference genome from UCSC...")
    typer.echo("=" * 60)

    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        url = f"http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{chrom}.fa.gz"
        gz_path = DATA_DIR / f"{chrom}.fa.gz"
        fa_path = DATA_DIR / f"{chrom}.fa"

        # Check if already downloaded and verified
        if not force and fa_path.exists() and str(fa_path) in checksums:
            current_checksum = compute_sha256(fa_path)
            if current_checksum == checksums[str(fa_path)]:
                fasta_files.append(fa_path)
                continue

        # Download
        download_with_retry(url, gz_path, show_progress=False)

        # Decompress
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
    typer.echo(f"\nMerging {len(fasta_files)} FASTA files into {merged_fasta.name}...")
    with open(merged_fasta, "wb") as outfile:
        for fa_path in tqdm(fasta_files, desc="Merging", leave=False):
            with open(fa_path, "rb") as infile:
                outfile.write(infile.read())

    checksums[str(merged_fasta)] = compute_sha256(merged_fasta)
    typer.echo(f"Created {merged_fasta} ({merged_fasta.stat().st_size / 1e9:.2f} GB)")

    # Download narrowPeak files from ENCODE
    if not chromosomes_only:
        typer.echo("\n" + "=" * 60)
        typer.echo("Downloading narrowPeak files from ENCODE...")
        typer.echo("=" * 60)

        try:
            experiments = query_encode_eclip(target)
            if experiments:
                bed_files = download_narrowpeak_files(experiments, DATA_DIR, limit)
                typer.echo(f"\nDownloaded {len(bed_files)} BED files to {DATA_DIR}")

                # Compute checksums for BED files
                for bed_file in tqdm(bed_files, desc="Computing checksums", leave=False):
                    checksums[str(bed_file)] = compute_sha256(bed_file)
            else:
                typer.echo("No experiments found matching criteria")
        except Exception as e:
            typer.echo(f"Error downloading narrowPeak files: {e}")
            typer.echo("You can retry with --force or download manually from ENCODE")

    # Save checksums
    with open(CHECKSUM_FILE, "w") as f:
        json.dump(checksums, f, indent=2)

    typer.echo("\n" + "=" * 60)
    typer.echo("Download complete!")
    typer.echo(f"Checksums saved to {CHECKSUM_FILE}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
