from typing import Dict, Optional, Sequence, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig

from data.utils.types import BackboneFeatures, LstmStates

from .density_gate import EventDensityGate
from .multi_scale_fusion import MultiScaleEventFusion
from .temporal_gate import TemporalResidualGate


class EventFeatureEnhancer(nn.Module):
    """Feature enhancement stack for event-based object detection."""

    def __init__(self, cfg: DictConfig, in_stages: Sequence[int], in_channels: Sequence[int]):
        super().__init__()
        assert len(in_stages) == len(in_channels)

        self.stage_ids: Tuple[int, ...] = tuple(in_stages)

        density_cfg = cfg.density_gate
        temporal_cfg = cfg.temporal_gate
        fusion_cfg = cfg.multi_scale_fusion

        self.use_density_gate = density_cfg.enable
        self.use_temporal_gate = temporal_cfg.enable
        self.use_multi_scale_fusion = fusion_cfg.enable

        if self.use_density_gate:
            self.density_gates = nn.ModuleDict({
                str(stage_id): EventDensityGate(
                    out_channels=channels,
                    kernel_size=density_cfg.kernel_size,
                )
                for stage_id, channels in zip(self.stage_ids, in_channels)
            })
        else:
            self.density_gates = None

        if self.use_temporal_gate:
            self.temporal_gates = nn.ModuleDict({
                str(stage_id): TemporalResidualGate(channels=channels)
                for stage_id, channels in zip(self.stage_ids, in_channels)
            })
        else:
            self.temporal_gates = None

        self.multi_scale_fusion = MultiScaleEventFusion(
            in_channels=in_channels,
            channel_gate_reduction=fusion_cfg.channel_gate_reduction,
            use_residual=fusion_cfg.use_residual,
        ) if self.use_multi_scale_fusion else None

    @staticmethod
    def _get_prev_hidden(prev_states: Optional[LstmStates], stage_id: int) -> Optional[th.Tensor]:
        if prev_states is None or len(prev_states) < stage_id:
            return None
        state = prev_states[stage_id - 1]
        if state is None:
            return None
        return state[0]

    def forward(
            self,
            backbone_features: BackboneFeatures,
            ev_repr: th.Tensor,
            prev_states: Optional[LstmStates] = None) -> BackboneFeatures:
        enhanced: Dict[int, th.Tensor] = dict(backbone_features)
        selected_features = []

        for stage_id in self.stage_ids:
            feat = backbone_features[stage_id]
            if self.use_density_gate:
                feat = self.density_gates[str(stage_id)](feat, ev_repr)
            if self.use_temporal_gate:
                feat = self.temporal_gates[str(stage_id)](
                    feat, self._get_prev_hidden(prev_states, stage_id))
            selected_features.append(feat)

        if self.multi_scale_fusion is not None:
            selected_features = self.multi_scale_fusion(selected_features)

        for stage_id, feat in zip(self.stage_ids, selected_features):
            enhanced[stage_id] = feat
        return enhanced
