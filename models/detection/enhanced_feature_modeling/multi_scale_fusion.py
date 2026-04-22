from typing import Sequence

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class RefinementBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.channel_gate = ChannelGate(channels, reduction=reduction)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.act(self.pointwise(self.depthwise(x)))
        return x * (1.0 + self.channel_gate(x))


class MultiScaleEventFusion(nn.Module):
    """Lightweight adjacent-scale fusion tailored to event features."""

    def __init__(self, in_channels: Sequence[int], channel_gate_reduction: int = 4, use_residual: bool = True):
        super().__init__()
        assert len(in_channels) >= 2
        self.use_residual = use_residual
        self.top_down = nn.ModuleList(
            nn.Conv2d(in_channels[idx + 1], in_channels[idx], kernel_size=1, bias=False)
            for idx in range(len(in_channels) - 1)
        )
        self.bottom_up = nn.ModuleList(
            nn.Conv2d(in_channels[idx], in_channels[idx + 1], kernel_size=1, bias=False)
            for idx in range(len(in_channels) - 1)
        )
        self.refine = nn.ModuleList(
            RefinementBlock(channels, reduction=channel_gate_reduction) for channels in in_channels
        )

    def forward(self, features: Sequence[th.Tensor]):
        outputs = []
        num_levels = len(features)

        for idx, feat in enumerate(features):
            fused = feat
            if idx < num_levels - 1:
                td = self.top_down[idx](features[idx + 1])
                td = F.interpolate(td, size=feat.shape[-2:], mode='nearest-exact')
                fused = fused + td
            if idx > 0:
                bu = self.bottom_up[idx - 1](features[idx - 1])
                bu = F.adaptive_avg_pool2d(bu, output_size=feat.shape[-2:])
                fused = fused + bu

            refined = self.refine[idx](fused)
            outputs.append(feat + refined if self.use_residual else refined)

        return outputs
