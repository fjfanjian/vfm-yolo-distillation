#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyyaml>=6.0",
#   "torch>=2.2",
#   "ultralytics>=8.4",
# ]
# ///
# How to run:
#   uv run scripts/train_dinov3_objectness_aux_multilayer.py --config <CONFIG>

from __future__ import annotations

import types
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import torch
from torch import nn
from train_dinov3_objectness_aux import (
    AuxTargetMode,
    AuxTargetSettings,
    DinoObjectnessAuxTrainer,
    aux_target_settings_from_config,
    aux_trainer_overrides,
)
from train_dinov3_objectness_pretrain import (
    BatchValue,
    TrainSettings,
    expect_mapping,
    image_files,
    parse_args,
    read_yaml,
    settings_from_config,
)
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from vfm_yolo_distillation.objectness_multilayer import (
    MultiLayerSettings,
    parse_multilayer_settings,
)

if TYPE_CHECKING:
    from ultralytics.nn.tasks import DetectionModel


class DinoObjectnessMultiLayerAuxTrainer(DinoObjectnessAuxTrainer):
    loss_names = ("box_loss", "cls_loss", "dfl_loss", "dino_objectness_loss")

    def __init__(
        self,
        settings: TrainSettings,
        aux_settings: AuxTargetSettings,
        layer_settings: MultiLayerSettings,
        overrides: dict[str, str | int | float | bool],
    ) -> None:
        self.layer_settings = layer_settings
        super().__init__(settings=settings, aux_settings=aux_settings, overrides=overrides)

    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ) -> DetectionModel:
        model = DetectionTrainer.get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        model.dino_objectness_heads = nn.ModuleDict(
            {
                str(spec.layer): nn.Conv2d(spec.dim, 1, kernel_size=1)
                for spec in self.layer_settings.layers
            }
        )
        for spec in self.layer_settings.layers:
            model.model[spec.layer].register_forward_hook(self._capture_student_feature)
        model.loss = types.MethodType(self._detection_objectness_loss, model)
        return model

    def _objectness_aux_loss(
        self, model: DetectionModel, batch: dict[str, BatchValue], images: torch.Tensor
    ) -> torch.Tensor:
        heads = getattr(model, "dino_objectness_heads", None)
        if not isinstance(heads, nn.ModuleDict):
            return images.sum() * 0.0
        losses: list[torch.Tensor] = []
        for spec in self.layer_settings.layers:
            layer = model.model[spec.layer]
            feature = getattr(layer, "dino_objectness_feature", None)
            head = heads[str(spec.layer)]
            if isinstance(feature, torch.Tensor) and isinstance(head, nn.Conv2d):
                losses.append(self._loss_for_logits(head(feature), batch, images))
        if not losses:
            return images.sum() * 0.0
        return torch.stack(losses).mean()

    def _loss_for_logits(
        self, logits: torch.Tensor, batch: dict[str, BatchValue], images: torch.Tensor
    ) -> torch.Tensor:
        mode = self.aux_settings.mode
        match mode:
            case "small_crop_soft":
                return self._small_crop_soft_loss(logits, batch, images)
            case "small_tile_soft":
                return self._small_tile_soft_loss(logits, batch, images)
            case "soft" | "ignore_aware" | "small_gt_weighted_soft" | "peak_ignore_aware":
                return self._teacher_target_loss(mode, logits, batch, images)
            case unreachable:
                assert_never(unreachable)

    def _teacher_target_loss(
        self,
        mode: AuxTargetMode,
        logits: torch.Tensor,
        batch: dict[str, BatchValue],
        images: torch.Tensor,
    ) -> torch.Tensor:
        targets = self._teacher_targets(
            images, image_files(batch, images.shape[0]), tuple(logits.shape[-2:])
        )
        match mode:
            case "soft":
                return self._soft_loss(logits, targets)
            case "ignore_aware":
                return self._ignore_aware_loss(logits, targets)
            case "small_gt_weighted_soft":
                return self._small_gt_weighted_loss(logits, targets, batch, images)
            case "peak_ignore_aware":
                return self._peak_ignore_aware_loss(logits, targets)
            case "small_crop_soft" | "small_tile_soft" as unreachable:
                assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)

    def save_model(self) -> bool:
        models = [unwrap_model(self.model)]
        if self.ema is not None:
            models.append(unwrap_model(self.ema.ema))
        saved_loss = [model.__dict__.pop("loss", None) for model in models]
        saved_heads: list[nn.Module | None] = []
        saved_hooks = []
        for model in models:
            saved_heads.append(model._modules.pop("dino_objectness_heads", None))
            hooks_for_model = []
            for spec in self.layer_settings.layers:
                layer = model.model[spec.layer]
                hooks_for_model.append((layer, layer._forward_hooks.copy()))
                layer._forward_hooks = OrderedDict()
                layer.__dict__.pop("dino_objectness_feature", None)
            saved_hooks.append(hooks_for_model)
        try:
            return bool(DetectionTrainer.save_model(self))
        finally:
            for model, loss_method, hooks_for_model, heads in zip(
                models, saved_loss, saved_hooks, saved_heads, strict=True
            ):
                for layer, hooks in hooks_for_model:
                    layer._forward_hooks = hooks
                if heads is not None:
                    model.dino_objectness_heads = heads
                if loss_method is not None:
                    model.loss = loss_method


def layer_settings_from_config(path: Path) -> MultiLayerSettings:
    raw = read_yaml(path)
    objectness = expect_mapping(raw["objectness"], "objectness")
    return parse_multilayer_settings(objectness)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    settings = settings_from_config(config_path, args)
    aux_settings = aux_target_settings_from_config(config_path)
    layer_settings = layer_settings_from_config(config_path)
    trainer = DinoObjectnessMultiLayerAuxTrainer(
        settings=settings,
        aux_settings=aux_settings,
        layer_settings=layer_settings,
        overrides=aux_trainer_overrides(settings),
    )
    trainer.train()


if __name__ == "__main__":
    main()
