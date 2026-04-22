import torch as th
import torch.nn as nn
import torch.nn.functional as F


class EventDensityGate(nn.Module):
    """Reweights features using a lightweight event-density prior."""

    def __init__(self, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.local_context = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.project = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)

    def forward(self, feat: th.Tensor, ev_repr: th.Tensor) -> th.Tensor:
        density = ev_repr.abs().sum(dim=1, keepdim=True)
        density = density / density.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        density = F.interpolate(density, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        gate = th.sigmoid(self.project(self.local_context(density)))
        return feat * (1.0 + gate)
