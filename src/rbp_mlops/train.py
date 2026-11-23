"""Training CLI for RBP binding classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import typer
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from rbp_mlops.data import RBPDataset
from rbp_mlops.model import RBPNet1D, set_seed

app = typer.Typer()


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train for one epoch.

    Args:
        model: Neural network model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for sequences, labels in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        # Forward pass
        logits = model(sequences).squeeze(-1)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """Validate model.

    Args:
        model: Neural network model.
        loader: Validation data loader.
        criterion: Loss function.

    Returns:
        Tuple of (loss, AUPRC, AUROC).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc="Validating", leave=False):
            # Forward pass
            logits = model(sequences).squeeze(-1)
            loss = criterion(logits, labels)

            # Collect predictions
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    # Compute metrics
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        auprc = average_precision_score(all_labels, all_probs)
        auroc = roc_auc_score(all_labels, all_probs)
    else:
        auprc = 0.0
        auroc = 0.0

    return avg_loss, auprc, auroc


@app.command()
def train(
    bed: Annotated[Path, typer.Option(help="Path to narrowPeak BED")],
    fasta: Annotated[Path, typer.Option(help="Path to FASTA genome")],
    epochs: Annotated[int, typer.Option(help="Training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 0.001,
    tracking_uri: Annotated[
        str, typer.Option(help="MLflow tracking URI")
    ] = "sqlite:///mlruns/mlflow.db",
    seed: int = 42,
) -> None:
    """Train RBP binding classifier with MLflow tracking."""
    # Setup
    set_seed(seed)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rbp-binding")

    # Create datasets
    typer.echo("Loading datasets...")
    train_ds = RBPDataset(
        bed_path=bed,
        fasta_path=fasta,
        chromosomes=["train"],
        generate_negatives_flag=True,
    )
    val_ds = RBPDataset(
        bed_path=bed,
        fasta_path=fasta,
        chromosomes=["val"],
        generate_negatives_flag=True,
    )

    if len(train_ds) == 0:
        typer.echo("Warning: No training data found. Using validation set for training.")
        train_ds = val_ds

    typer.echo(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Initialize model
    model = RBPNet1D()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop with MLflow tracking
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
            }
        )

        best_auprc = 0.0
        best_model_state = None
        patience = 10
        patience_counter = 0

        typer.echo("Starting training...")
        for epoch in tqdm(range(epochs), desc="Epochs"):
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer)

            # Validate
            val_loss, val_auprc, val_auroc = validate(model, val_loader, criterion)

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auprc": val_auprc,
                    "val_auroc": val_auroc,
                },
                step=epoch,
            )

            typer.echo(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_auprc={val_auprc:.4f}, "
                f"val_auroc={val_auroc:.4f}"
            )

            # Early stopping
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    typer.echo(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step()

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save best model
        torch.save(model.state_dict(), "best_model.pth")
        mlflow.log_artifact("best_model.pth")

        # Compute and save baseline stats
        typer.echo("Computing baseline statistics...")
        baseline = train_ds.compute_baseline_stats()
        with open("baseline_stats.json", "w") as f:
            json.dump(baseline, f, indent=2)
        mlflow.log_artifact("baseline_stats.json")

        # Register model
        typer.echo("Registering model in MLflow...")
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="rbp-binding-classifier",
        )

        # Set champion alias
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions("rbp-binding-classifier")
        if latest_versions:
            latest_version = latest_versions[0].version
            client.set_registered_model_alias(
                "rbp-binding-classifier",
                "champion",
                latest_version,
            )
            typer.echo(f"Model version {latest_version} registered as 'champion'")

    typer.echo("Training complete!")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
