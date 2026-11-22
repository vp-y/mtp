"""Stage 2 models: a pretrained timm backbone + a light prediction head (paper section "Classification")."""
from __future__ import annotations

import torch
import torch.nn as nn

BACKBONES = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "mobilenetv3_small": "mobilenetv3_small_100",
    "vit_b16": "vit_base_patch16_224",
}


class Classifier(nn.Module):
    """Backbone -> head. head='linear' or 'mlp' (hidden sizes 128, 64). Outputs one logit (malignant)."""

    def __init__(self, backbone: str, head: str = "linear", dropout: float = 0.0, pretrained: bool = True,
                 img_size: int = 224, num_outputs: int = 1, freeze_bn: bool = True):
        super().__init__()
        import timm

        name = BACKBONES.get(backbone, backbone)
        kw = {"img_size": img_size} if name.startswith("vit") else {}
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, **kw)
        self.backbone.eval()
        with torch.no_grad():  # feature size differs between timm versions; measure it
            dim = self.backbone(torch.zeros(1, 3, img_size, img_size)).shape[1]
        if head == "linear":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_outputs))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(64, num_outputs))
        else:
            raise ValueError(f"head must be 'linear' or 'mlp', got {head!r}")
        self.freeze_bn = freeze_bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn:  # frozen BatchNorm layers keep their pretrained statistics
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and \
                        not any(p.requires_grad for p in m.parameters(recurse=False)):
                    m.eval()
        return self


def set_trainable(model: Classifier, unfreeze_k) -> int:
    """Freeze the backbone, then unfreeze its last K parameter tensors (paper: K in {0, 10}).

    unfreeze_k: int K >= 0, or 'all' / -1 for full fine-tuning. The head is always trainable.
    Returns the number of trainable parameters.
    """
    params = list(model.backbone.parameters())
    for p in params:
        p.requires_grad = False
    if unfreeze_k in ("all", -1):
        for p in params:
            p.requires_grad = True
    elif int(unfreeze_k) > 0:
        for p in params[-int(unfreeze_k):]:
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
