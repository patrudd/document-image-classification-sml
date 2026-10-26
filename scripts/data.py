"""Dataset, transforms, validation split, and dataloader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scripts.config import TrainConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Pad a PIL image to a square without cropping document content."""

    def __init__(self, fill: int = 255) -> None:
        """Initialize the padding color, using white for scanned paper."""

        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return a centered square image while preserving the aspect ratio."""

        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, (left, top, right, bottom), fill=self.fill)


class DocumentDataset(Dataset[tuple[torch.Tensor, int | str]]):
    """Read TIFF document images referenced by the competition CSV files."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        inference: bool = False,
    ) -> None:
        """Store metadata and validate the columns required by each mode."""

        required = {"id", "filename"} if inference else {"id", "filename", "label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.inference = inference

    def __len__(self) -> int:
        """Return the number of metadata rows."""

        return len(self.frame)

    def _resolve_path(self, filename: str) -> Path:
        """Resolve paths whether CSV filenames include train/test prefixes or not."""

        raw = Path(filename)
        candidates = [self.data_dir / raw, self.data_dir / raw.name]
        if len(raw.parts) > 1:
            candidates.append(self.data_dir / Path(*raw.parts[1:]))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Citra tidak ditemukan untuk filename={filename}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int | str]:
        """Load one RGB image and return its label or test identifier."""

        row = self.frame.iloc[index]
        path = self._resolve_path(str(row["filename"]))
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        tensor = self.transform(image)
        target: int | str = str(row["id"]) if self.inference else int(row["label"])
        return tensor, target


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Create train and validation transforms suited to scanned documents."""

    common = [
        PadToSquare(fill=255),
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    ]
    train_transform = transforms.Compose(
        common
        + [
            transforms.RandomRotation(4, fill=255),
            transforms.RandomAutocontrast(p=0.25),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        common + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return train_transform, eval_transform


def stratified_split(frame: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a leakage-safe stratified holdout split using only training rows."""

    train_frame, val_frame = train_test_split(
        frame,
        test_size=val_size,
        random_state=seed,
        stratify=frame["label"],
    )
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)


def class_weights(frame: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency weights normalized around one."""

    counts = frame["label"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError("Semua kelas 0-15 harus tersedia pada training split.")
    weights = len(frame) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """Construct deterministic validation and shuffled training dataloaders."""

    train_transform, eval_transform = build_transforms(config.image_size)
    train_dataset = DocumentDataset(train_frame, config.data_dir, train_transform)
    val_dataset = DocumentDataset(val_frame, config.data_dir, eval_transform)
    use_cuda = torch.cuda.is_available()
    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader


def build_test_loader(frame: pd.DataFrame, config: TrainConfig) -> DataLoader:
    """Construct an ordered test loader so predictions align with sample submission."""

    _, eval_transform = build_transforms(config.image_size)
    dataset = DocumentDataset(frame, config.data_dir, eval_transform, inference=True)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

