"""Model factory for document image classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def build_model(num_classes: int = 16, pretrained: bool = True) -> nn.Module:
    """Create ConvNeXt-Tiny and replace its final classifier for 16 classes."""

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list[dict]:
    """Use a smaller learning rate for pretrained features than the new head."""

    head_params = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]

