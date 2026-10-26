"""Generate a competition-valid submission from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from scripts.config import TrainConfig
from scripts.data import build_test_loader
from scripts.model import build_model
from scripts.utils import discover_data_dir


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, data, and output locations."""

    parser = argparse.ArgumentParser(description="Create Kaggle submission")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/convnext_tiny_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submissions/submission_convnext.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[list[str], list[int]]:
    """Predict ordered labels and preserve test identifiers from each batch."""

    model.eval()
    identifiers: list[str] = []
    labels: list[int] = []
    for images, batch_ids in tqdm(loader, desc="predict"):
        logits = model(images.to(device, non_blocking=True))
        identifiers.extend(list(batch_ids))
        labels.extend(logits.argmax(dim=1).cpu().tolist())
    return identifiers, labels


def main() -> None:
    """Load the best model, infer the test set, and validate submission schema."""

    args = parse_args()
    data_dir = discover_data_dir(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {})
    config = TrainConfig(
        data_dir=data_dir,
        batch_size=args.batch_size,
        image_size=int(saved.get("image_size", 224)),
        num_workers=int(saved.get("num_workers", 2)),
        pretrained=False,
    )
    test_frame = pd.read_csv(data_dir / "test.csv")
    loader = build_test_loader(test_frame, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    identifiers, labels = predict(model, loader, device)
    predictions = pd.DataFrame({"id": identifiers, "label": labels})
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    submission = sample[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")
    if submission["label"].isna().any() or not submission["label"].between(0, 15).all():
        raise ValueError("Submission gagal validasi: label hilang atau di luar rentang 0-15.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Submission tersimpan: {args.output} ({len(submission):,} baris)")


if __name__ == "__main__":
    main()
