"""Training, validation, early stopping, and checkpoint routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    """Optimize a model for one epoch with optional CUDA mixed precision."""

    model.train()
    total_loss = 0.0
    sample_count = 0
    predictions: list[int] = []
    targets: list[int] = []
    amp_enabled = device.type == "cuda"
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        sample_count += images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    return total_loss / sample_count, accuracy_score(targets, predictions)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Measure validation loss and accuracy without changing model state."""

    model.eval()
    total_loss = 0.0
    sample_count = 0
    predictions: list[int] = []
    targets: list[int] = []
    for images, labels in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        sample_count += images.size(0)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return total_loss / sample_count, accuracy_score(targets, predictions), targets, predictions


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, float]]:
    """Train with early stopping and save the best validation-accuracy checkpoint."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} | val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_accuracy": best_accuracy,
                    "history": history,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping: tidak membaik selama {patience} epoch.")
                break
    return history

