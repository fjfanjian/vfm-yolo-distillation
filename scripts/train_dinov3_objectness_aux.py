#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml>=6.0",
#   "torch>=2.2",
#   "ultralytics>=8.4",
# ]
# ///
from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, assert_never

import torch
from torch import nn
from torch.nn import functional
from train_dinov3_objectness_pretrain import (
    BatchValue,
    ConfigError,
    DinoObjectnessPretrainTrainer,
    TrainSettings,
    dice_loss,
    expect_mapping,
    image_files,
    parse_args,
    read_yaml,
    settings_from_config,
    tensor_value,
)
from ultralytics.models.yolo.detect import DetectionTrainer

from vfm_yolo_distillation.objectness_targets import (
    AreaThresholds,
    PeakIgnoreSettings,
    SmallGtWeightMapSettings,
    SmallGtWeights,
    build_peak_ignore_targets,
    build_small_gt_weight_map,
    weighted_soft_objectness_loss,
)

if TYPE_CHECKING:
    from ultralytics.nn.tasks import DetectionModel

AuxTargetMode = Literal[
    "soft",
    "ignore_aware",
    "small_gt_weighted_soft",
    "peak_ignore_aware",
]


@dataclass(frozen=True, slots=True)
class AuxTargetSettings:
    mode: AuxTargetMode
    positive_quantile: float
    negative_quantile: float
    small_gt: SmallGtWeightMapSettings
    peak: PeakIgnoreSettings


class DinoObjectnessAuxTrainer(DinoObjectnessPretrainTrainer):
    loss_names = ("box_loss", "cls_loss", "dfl_loss", "dino_objectness_loss")

    def __init__(
        self,
        settings: TrainSettings,
        aux_settings: AuxTargetSettings,
        overrides: dict[str, str | int | float | bool],
    ) -> None:
        self.aux_settings = aux_settings
        super().__init__(settings=settings, overrides=overrides)

    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ) -> DetectionModel:
        model = DetectionTrainer.get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        model.dino_objectness_head = nn.Conv2d(
            self.objectness_settings.student_dim, 1, kernel_size=1
        )
        model.model[self.objectness_settings.student_layer].register_forward_hook(
            self._capture_student_feature
        )
        model.loss = types.MethodType(self._detection_objectness_loss, model)
        return model

    @staticmethod
    def _capture_student_feature(
        module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        module.dino_objectness_feature = output

    def _detection_objectness_loss(
        self,
        model: DetectionModel,
        batch: dict[str, BatchValue],
        preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = tensor_value(batch, "img")
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        if preds is None:
            preds = model.forward(images)
        detection_loss, detection_items = model.criterion(preds, batch)
        objectness_loss = self._objectness_aux_loss(model, batch, images)
        total_loss = detection_loss + objectness_loss
        return total_loss, torch.cat((detection_items, objectness_loss.detach().reshape(1)))

    def _objectness_aux_loss(
        self, model: DetectionModel, batch: dict[str, BatchValue], images: torch.Tensor
    ) -> torch.Tensor:
        layer = model.model[self.objectness_settings.student_layer]
        feature = getattr(layer, "dino_objectness_feature", None)
        head = getattr(model, "dino_objectness_head", None)
        if not isinstance(feature, torch.Tensor) or not isinstance(head, nn.Conv2d):
            return images.sum() * 0.0
        logits = head(feature)
        targets = self._teacher_targets(
            images, image_files(batch, images.shape[0]), tuple(logits.shape[-2:])
        )
        match self.aux_settings.mode:
            case "soft":
                return self._soft_loss(logits, targets)
            case "ignore_aware":
                return self._ignore_aware_loss(logits, targets)
            case "small_gt_weighted_soft":
                return self._small_gt_weighted_loss(logits, targets, batch, images)
            case "peak_ignore_aware":
                return self._peak_ignore_aware_loss(logits, targets)
            case unreachable:
                assert_never(unreachable)

    def _soft_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(torch.sigmoid(logits), targets)
        return (bce + dice) * 0.5 * self.objectness_settings.lambda_objectness

    def _ignore_aware_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        labels, weights = ignore_aware_targets(targets, self.aux_settings)
        return self._weighted_label_loss(logits, labels, weights)

    def _peak_ignore_aware_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        labels, weights = build_peak_ignore_targets(targets, self.aux_settings.peak)
        return self._weighted_label_loss(logits, labels, weights)

    def _small_gt_weighted_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch: dict[str, BatchValue],
        images: torch.Tensor,
    ) -> torch.Tensor:
        weights = build_small_gt_weight_map(
            targets=targets,
            batch_idx=tensor_value(batch, "batch_idx"),
            bboxes_xywhn=tensor_value(batch, "bboxes"),
            image_size_hw=(images.shape[-2], images.shape[-1]),
            settings=self.aux_settings.small_gt,
        )
        return (
            weighted_soft_objectness_loss(logits, targets, weights)
            * self.objectness_settings.lambda_objectness
        )

    def _weighted_label_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weighted_bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
        weighted_dice = masked_dice_loss(torch.sigmoid(logits), labels, weights)
        return (weighted_bce + weighted_dice) * 0.5 * self.objectness_settings.lambda_objectness


def ignore_aware_targets(
    targets: torch.Tensor, settings: AuxTargetSettings
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = targets.flatten(start_dim=1)
    positive_threshold = torch.quantile(flat, settings.positive_quantile, dim=1)
    negative_threshold = torch.quantile(flat, settings.negative_quantile, dim=1)
    positive = targets >= positive_threshold.view(-1, 1, 1, 1)
    negative = targets <= negative_threshold.view(-1, 1, 1, 1)
    labels = positive.to(dtype=targets.dtype)
    weights = (positive | negative).to(dtype=targets.dtype)
    return labels, weights


def masked_dice_loss(
    probs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    dims = (1, 2, 3)
    masked_probs = probs * weights
    masked_labels = labels * weights
    intersection = (masked_probs * masked_labels).sum(dim=dims)
    denominator = masked_probs.sum(dim=dims) + masked_labels.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def aux_target_mode(value: str) -> AuxTargetMode:
    match value:
        case "soft" | "ignore_aware" | "small_gt_weighted_soft" | "peak_ignore_aware":
            return value
        case _:
            message = f"Unsupported auxiliary target mode: {value}"
            raise ConfigError(message)


def aux_target_settings_from_config(path: Path) -> AuxTargetSettings:
    raw = read_yaml(path)
    objectness = expect_mapping(raw["objectness"], "objectness")
    return AuxTargetSettings(
        mode=aux_target_mode(str(objectness.get("target_mode", "soft"))),
        positive_quantile=float(objectness.get("positive_quantile", 0.85)),
        negative_quantile=float(objectness.get("negative_quantile", 0.45)),
        small_gt=SmallGtWeightMapSettings(
            thresholds=AreaThresholds(
                small_area_px=float(objectness.get("small_area_px", 1024.0)),
                medium_area_px=float(objectness.get("medium_area_px", 9216.0)),
            ),
            weights=SmallGtWeights(
                background=float(objectness.get("background_weight", 0.25)),
                small=float(objectness.get("small_weight", 3.0)),
                medium=float(objectness.get("medium_weight", 1.0)),
                large=float(objectness.get("large_weight", 0.5)),
                false_positive=float(objectness.get("false_positive_weight", 0.05)),
            ),
            false_positive_quantile=float(objectness.get("false_positive_quantile", 0.85)),
            box_expand_cells=int(objectness.get("box_expand_cells", 1)),
        ),
        peak=PeakIgnoreSettings(
            positive_quantile=float(objectness.get("positive_quantile", 0.85)),
            negative_quantile=float(objectness.get("negative_quantile", 0.45)),
            peak_kernel=int(objectness.get("peak_kernel", 3)),
        ),
    )


def aux_trainer_overrides(settings: TrainSettings) -> dict[str, str | int | float | bool]:
    return {
        "task": "detect",
        "mode": "train",
        "model": settings.model,
        "data": settings.data.as_posix(),
        "imgsz": settings.image_size,
        "epochs": settings.epochs,
        "batch": settings.batch,
        "device": settings.device,
        "workers": settings.workers,
        "seed": settings.seed,
        "fraction": settings.fraction,
        "project": settings.project.as_posix(),
        "name": settings.name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "patience": 100,
        "close_mosaic": 10,
        "amp": True,
        "plots": True,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    settings = settings_from_config(config_path, args)
    aux_settings = aux_target_settings_from_config(config_path)
    trainer = DinoObjectnessAuxTrainer(
        settings=settings,
        aux_settings=aux_settings,
        overrides=aux_trainer_overrides(settings),
    )
    trainer.train()


if __name__ == "__main__":
    main()
