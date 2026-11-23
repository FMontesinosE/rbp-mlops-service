"""End-to-end smoke test."""

from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.pytorch
import pytest
import torch

# Mark as slow test
pytestmark = pytest.mark.slow

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_BED = TEST_DATA_DIR / "sample_peaks.bed"
TEST_FASTA = TEST_DATA_DIR / "sample_genome.fa"


def test_e2e_pipeline() -> None:
    """Full pipeline: train -> register -> load -> predict."""
    from rbp_mlops.data import RBPDataset, one_hot_encode
    from rbp_mlops.model import RBPNet1D, set_seed
    from rbp_mlops.train import train_epoch, validate

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Setup MLflow with temp DB
        mlflow_db = tmpdir_path / "mlflow.db"
        mlflow_uri = f"sqlite:///{mlflow_db}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test-rbp-binding")

        # Train for 2 epochs
        set_seed(42)

        # Create dataset
        train_ds = RBPDataset(
            bed_path=TEST_BED,
            fasta_path=TEST_FASTA,
            chromosomes=["val"],  # Use val set for quick test
            generate_negatives_flag=True,
        )

        assert len(train_ds) > 0

        # Create data loader
        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        # Initialize model
        model = RBPNet1D()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Train for 2 epochs
        with mlflow.start_run():
            for epoch in range(2):
                train_loss = train_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_auprc, val_auroc = validate(model, train_loader, criterion)

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auprc": val_auprc,
                        "val_auroc": val_auroc,
                    },
                    step=epoch,
                )

            # Save baseline
            import json

            baseline = train_ds.compute_baseline_stats()
            baseline_path = tmpdir_path / "baseline_stats.json"
            with open(baseline_path, "w") as f:
                json.dump(baseline, f)
            mlflow.log_artifact(str(baseline_path))

            # Register model
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="rbp-binding-classifier",
            )

            # Set champion alias
            client = mlflow.MlflowClient()
            latest_versions = client.get_latest_versions("rbp-binding-classifier")
            assert len(latest_versions) > 0
            latest_version = latest_versions[0].version
            client.set_registered_model_alias(
                "rbp-binding-classifier",
                "champion",
                latest_version,
            )

        # Load model via champion alias
        loaded_model = mlflow.pytorch.load_model("models:/rbp-binding-classifier@champion")
        loaded_model.eval()

        # Predict on test sequence
        test_seq = "ACGT" * 25 + "A"  # 101bp
        tensor = torch.from_numpy(one_hot_encode(test_seq)).unsqueeze(0)

        with torch.no_grad():
            logit = loaded_model(tensor)
            prob = torch.sigmoid(logit).item()

        # Verify prediction is valid probability
        assert 0.0 <= prob <= 1.0

        # Verify MLflow DB was created
        assert mlflow_db.exists()
