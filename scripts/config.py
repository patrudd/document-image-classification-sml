"""Central configuration for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    """Store reproducible experiment settings in one serializable object."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("models")
    model_name: str = "convnext_tiny"
    num_classes: int = 16
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 8
    backbone_lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    val_size: float = 0.15
    num_workers: int = 2
    seed: int = 42
    patience: int = 3
    pretrained: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert paths and scalar values into a checkpoint-safe dictionary."""

        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload
