"""Stage 1 models: U-Net / FPN / DeepLabV3+ (segmentation_models_pytorch) and the paper's plain U-Net.

`base_channels` is the paper's "Base Channel" column. The paper does not define it for the
library models, so we map it to decoder width (b = 32 reproduces each library default for
U-Net and FPN, b = 64 reproduces it for DeepLabV3+):
  U-Net       decoder_channels = (8b, 4b, 2b, b, b/2)
  FPN         pyramid = 8b, segmentation = 4b
  DeepLabV3+  decoder_channels = 4b
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _double_conv(cin: int, cout: int, batch_norm: bool) -> nn.Sequential:
    layers = []
    for c in (cin, cout):
        layers.append(nn.Conv2d(c, cout, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class VanillaUNet(nn.Module):
    """The 'traditional' U-Net described in the paper: 4 encoder blocks (2 x conv3x3 + ReLU,
    'same' padding, 2x2 max-pool), a 2-conv bottleneck, and 4 decoder blocks with 2x2
    transposed convolutions and skip connections."""

    def __init__(self, in_channels: int = 1, base: int = 32, depth: int = 4, batch_norm: bool = False):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        c = in_channels
        for co in ch:
            self.down.append(_double_conv(c, co, batch_norm))
            c = co
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(ch[-1], ch[-1] * 2, batch_norm)
        self.ups = nn.ModuleList(nn.ConvTranspose2d(ch[i] * 2, ch[i], 2, stride=2) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(_double_conv(ch[i] * 2, ch[i], batch_norm) for i in reversed(range(depth)))
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def _norm_arch(name: str) -> str:
    return name.lower().replace("+", "plus").replace("-", "").replace("_", "")


def build_seg_model(cfg: dict, in_channels: int = 1) -> nn.Module:
    arch = _norm_arch(cfg["arch"])
    b = int(cfg.get("base_channels", 32))
    if arch in ("vanillaunet", "plainunet"):
        return VanillaUNet(in_channels, base=b, batch_norm=bool(cfg.get("batch_norm", False)))

    import segmentation_models_pytorch as smp

    kw = dict(encoder_name=cfg.get("encoder", "resnet34"),
              encoder_weights=cfg.get("encoder_weights", "imagenet"),
              in_channels=in_channels, classes=1)
    if arch == "unet":
        return smp.Unet(decoder_channels=(8 * b, 4 * b, 2 * b, b, max(b // 2, 8)), **kw)
    if arch == "fpn":
        return smp.FPN(decoder_pyramid_channels=8 * b, decoder_segmentation_channels=4 * b, **kw)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(decoder_channels=4 * b, **kw)
    raise ValueError(f"unknown segmentation arch {cfg['arch']!r}")
