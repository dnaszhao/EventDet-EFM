from typing import Optional, Tuple

from omegaconf import DictConfig

from .feature_enhancer import EventFeatureEnhancer


def build_feature_enhancer(
        enhancer_cfg: Optional[DictConfig],
        in_stages: Tuple[int, ...],
        in_channels: Tuple[int, ...]):
    if enhancer_cfg is None or not enhancer_cfg.enable:
        return None
    return EventFeatureEnhancer(cfg=enhancer_cfg, in_stages=in_stages, in_channels=in_channels)
