"""Small utilities shared by training and prediction scripts."""

from __future__ import annotations

import json
import os
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_data_dir(explicit: str | Path | None = None) -> Path:
    """Locate competition data across local and common Kaggle paths."""

    candidates = [
        Path(explicit) if explicit else None,
        Path(os.environ["KAGGLE_INPUT_PATH"]) if os.getenv("KAGGLE_INPUT_PATH") else None,
        Path("/kaggle/input/document-image-classification-sd234504"),
        Path("/kaggle/input/competitions/document-image-classification-sd234504"),
        Path("data"),
    ]
    for candidate in candidates:
        if candidate and (candidate / "train.csv").exists():
            return candidate.resolve()
    searched = "\n".join(f"- {path}" for path in candidates if path)
    raise FileNotFoundError(f"train.csv tidak ditemukan. Lokasi yang diperiksa:\n{searched}")


def extract_archives(source_dir: Path, destination: Path) -> Path:
    """Extract train/test archives when Kaggle exposes ZIP files only."""

    destination.mkdir(parents=True, exist_ok=True)
    for name in ("train.zip", "test.zip"):
        archive = source_dir / name
        target_name = archive.stem
        if archive.exists() and not (destination / target_name).exists():
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(destination)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        source = source_dir / name
        target = destination / name
        if source.exists() and not target.exists():
            target.write_bytes(source.read_bytes())
    return destination.resolve()


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Persist metrics as readable JSON, creating parent folders as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
