"""Frozen RT-DETRv4-X detection teacher for RaCoFD.

This module instantiates the detection network defined by the RT-DETRv4-X
configuration, NOT the DINOv3 teacher in that configuration.  Only the trained
RT-DETRv4 detection checkpoint is loaded. DINOv3 supervision for the student
remains a separate, unchanged teacher in the RaCoFD training loop.
"""
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _detector_state_dict(checkpoint: Any, prefer_ema: bool) -> dict:
    """Read model / EMA parameters from an official RT-DETRv4 solver checkpoint."""
    if not isinstance(checkpoint, dict):
        raise TypeError('RT-DETRv4-X checkpoint must contain a state dictionary')
    state = None
    if prefer_ema and isinstance(checkpoint.get('ema'), dict):
        state = checkpoint['ema'].get('module', checkpoint['ema'])
    if state is None:
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise TypeError('No usable RT-DETRv4-X model/ema state found')
    result = {}
    for name, value in state.items():
        if torch.is_tensor(value):
            key = name.removeprefix('module.')
            result[key] = value
    if not result:
        raise RuntimeError('RT-DETRv4-X checkpoint contains no model tensors')
    return result


class RaCoFDDetectionTeacher(nn.Module):
    """A fixed RT-DETRv4-X teacher, returning detection outputs for training KD.

    `config_path` and `checkpoint_path` MUST match the same RT-DETRv4-X model.
    The wrapper checks parameter coverage to prevent silently using random
    model parameters. `return_features=True` exposes the deepest encoder output
    for the optional spatial-energy feature-KD experiment.
    """

    def __init__(self, config_path: str, checkpoint_path: str,
                 checkpoint_prefer_ema: bool = True,
                 minimum_checkpoint_coverage: float = 0.95,
                 return_features: bool = False):
        super().__init__()
        from ..core.yaml_config import YAMLConfig

        if not Path(config_path).is_file():
            raise FileNotFoundError(f'RT-DETRv4-X config not found: {config_path}')
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f'RT-DETRv4-X checkpoint not found: {checkpoint_path}. '
                'Download a trained RT-DETRv4-X detector checkpoint.'
            )
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.return_features = bool(return_features)

        teacher_cfg = YAMLConfig(config_path)
        # Validate actual architecture before constructing it. This is not
        # interchangeable with RT-DETRv2-X or its ResNet-101-vd checkpoint.
        teacher_model_name = teacher_cfg.yaml_cfg.get('model')
        if teacher_model_name != 'RTv4':
            raise ValueError(f'Expected RTv4 teacher model; got {teacher_model_name!r}')
        if not str(config_path).replace('\\', '/').endswith('/rtv4_hgnetv2_x_coco.yml'):
            raise ValueError('RaCoFD paper setting requires configs/rtv4/rtv4_hgnetv2_x_coco.yml')
        if teacher_cfg.yaml_cfg.get('HGNetv2', {}).get('name') != 'B5':
            raise ValueError('Expected HGNetv2 B5 backbone for RT-DETRv4-X')
        # An already trained full checkpoint does not need backbone download.
        if 'HGNetv2' in teacher_cfg.yaml_cfg:
            teacher_cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        # Student augmentations may use sizes other than 640x640. Positional
        # encodings and decoder anchors must therefore be computed dynamically.
        teacher_cfg.yaml_cfg['eval_spatial_size'] = None

        # Only access YAMLConfig.model; its separate DINOv3TeacherModel property
        # is NOT instantiated, so this does not run an unwanted third teacher.
        self.model = teacher_cfg.model
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = _detector_state_dict(checkpoint, checkpoint_prefer_ema)

        model_state = self.model.state_dict()
        matched = {key: value for key, value in state.items()
                   if key in model_state and model_state[key].shape == value.shape}
        trainable_keys = set(dict(self.model.named_parameters()))
        covered = sum(model_state[key].numel() for key in matched if key in trainable_keys)
        total = sum(model_state[key].numel() for key in trainable_keys)
        ratio = covered / max(total, 1)
        if ratio < minimum_checkpoint_coverage:
            raise RuntimeError(
                f'RT-DETRv4-X checkpoint/config mismatch: {ratio:.2%} parameter coverage; '
                f'expected >= {minimum_checkpoint_coverage:.2%}. '
                'Do not use RT-DETRv2-X weights with RT-DETRv4-X config.'
            )
        self.model.load_state_dict(matched, strict=False)
        self.loaded_coverage = float(ratio)
        self.model.eval()
        self.model.requires_grad_(False)
        print(f'[RaCoFD] Frozen RT-DETRv4-X teacher loaded: {checkpoint_path}; '
              f'coverage={ratio:.2%}')

    def train(self, mode: bool = True):
        # External trainer .train() calls must not activate BN/dropout in teacher.
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict:
        self.model.eval()
        if self.return_features:
            encoded = self.model.encoder(self.model.backbone(images))
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            if not isinstance(encoded, (list, tuple)) or not encoded:
                raise TypeError('RT-DETRv4-X encoder must return multi-scale feature maps')
            outputs = self.model.decoder(encoded, targets=None)
            feature_f5 = encoded[-1]
        else:
            outputs = self.model(images)
            feature_f5 = None

        if not isinstance(outputs, dict) or not all(
            key in outputs for key in ('pred_logits', 'pred_boxes')
        ):
            raise TypeError('RT-DETRv4-X must return pred_logits and pred_boxes')
        result = {
            'pred_logits': outputs['pred_logits'].detach(),
            'pred_boxes': outputs['pred_boxes'].detach(),
        }
        if feature_f5 is not None:
            result['feature_f5'] = feature_f5.detach()
        return result
