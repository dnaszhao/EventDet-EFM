from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualGate(nn.Module):
    """Selectively injects previous-step hidden states into current features."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, feat: th.Tensor, prev_hidden: Optional[th.Tensor]) -> th.Tensor:
        if prev_hidden is None:
            return feat

        prev_hidden = prev_hidden.to(dtype=feat.dtype)
        if prev_hidden.shape[-2:] != feat.shape[-2:]:
            prev_hidden = F.interpolate(prev_hidden, size=feat.shape[-2:], mode='nearest')

        alpha = th.sigmoid(self.gate(th.cat([feat, prev_hidden], dim=1)))
        return feat + alpha * self.project(prev_hidden)
