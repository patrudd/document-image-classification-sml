"""Command-line entry point for leakage-safe model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from scripts.config import TrainConfig
from scripts.data import build_loaders, class_weights, stratified_split
from scripts.engine import evaluate, fit
from scripts.model import build_model, parameter_groups
from scripts.utils import discover_data_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse configurable paths and core training hyperparameters."""

    parser = argparse.ArgumentParser(description="Train ConvNeXt on document images")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train, select the best checkpoint, and persist holdout diagnostics."""

    args = parse_args()
    config = TrainConfig(
        data_dir=discover_data_dir(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
    )
    seed_everything(config.seed)
    train_frame = pd.read_csv(config.data_dir / "train.csv")
    train_split, val_split = stratified_split(train_frame, config.val_size, config.seed)
    train_loader, val_loader = build_loaders(train_split, val_split, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.num_classes, config.pretrained).to(device)
    weights = class_weights(train_split, config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.backbone_lr, config.head_lr),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    checkpoint_path = config.output_dir / "convnext_tiny_best.pt"
    history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config.epochs,
        config.patience,
        checkpoint_path,
        {"config": config.to_dict(), "class_names": _class_names(train_frame)},
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, val_accuracy, targets, predictions = evaluate(model, val_loader, criterion, device)
    report = classification_report(targets, predictions, output_dict=True, zero_division=0)
    metrics = {
        "best_val_accuracy": val_accuracy,
        "history": history,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
        "train_rows": len(train_split),
        "validation_rows": len(val_split),
    }
    save_json(metrics, Path("reports/validation_metrics.json"))
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Validation accuracy: {val_accuracy:.4f}")


def _class_names(frame: pd.DataFrame) -> list[str]:
    """Return class names ordered by numeric label for checkpoint metadata."""

    mapping = frame[["label", "class_name"]].drop_duplicates().sort_values("label")
    return mapping["class_name"].astype(str).tolist()


if __name__ == "__main__":
    main()
