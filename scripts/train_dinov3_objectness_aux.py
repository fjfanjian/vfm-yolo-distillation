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
    objectness_score,
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
    "small_crop_soft",
    "small_tile_soft",
]


@dataclass(frozen=True, slots=True)
class AuxTargetSettings:
    mode: AuxTargetMode
    positive_quantile: float
    negative_quantile: float
    small_gt: SmallGtWeightMapSettings
    peak: PeakIgnoreSettings
    small_crop: SmallCropTargetSettings
    small_tile: SmallTileTargetSettings


@dataclass(frozen=True, slots=True)
class SmallCropTargetSettings:
    small_area_px: float
    context_scale: float
    min_crop_size: int
    max_crops_per_image: int
    weight: float


@dataclass(frozen=True, slots=True)
class SmallTileTargetSettings:
    small_area_px: float
    tile_size: int
    tile_stride: int
    max_tiles_per_image: int
    min_small_boxes: int
    weight: float


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
        if self.aux_settings.mode == "small_crop_soft":
            return self._small_crop_soft_loss(logits, batch, images)
        if self.aux_settings.mode == "small_tile_soft":
            return self._small_tile_soft_loss(logits, batch, images)
        targets = self._teacher_targets(
            images, image_files(batch, images.shape[0]), tuple(logits.shape[-2:])
        )
        match self.aux_settings.mode:
            case "soft":
                loss = self._soft_loss(logits, targets)
            case "ignore_aware":
                loss = self._ignore_aware_loss(logits, targets)
            case "small_gt_weighted_soft":
                loss = self._small_gt_weighted_loss(logits, targets, batch, images)
            case "peak_ignore_aware":
                loss = self._peak_ignore_aware_loss(logits, targets)
            case "small_crop_soft":
                loss = self._small_crop_soft_loss(logits, batch, images)
            case "small_tile_soft":
                loss = self._small_tile_soft_loss(logits, batch, images)
            case unreachable:
                assert_never(unreachable)
        return loss

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

    def _small_crop_soft_loss(
        self,
        logits: torch.Tensor,
        batch: dict[str, BatchValue],
        images: torch.Tensor,
    ) -> torch.Tensor:
        targets, weights = self._small_crop_targets(
            batch=batch,
            images=images,
            target_size=tuple(logits.shape[-2:]),
        )
        if weights.sum() <= 0:
            return logits.sum() * 0.0
        return (
            weighted_soft_objectness_loss(logits, targets, weights)
            * self.objectness_settings.lambda_objectness
        )

    def _small_crop_targets(
        self,
        batch: dict[str, BatchValue],
        images: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = tensor_value(batch, "batch_idx").to(device=images.device)
        bboxes = tensor_value(batch, "bboxes").to(device=images.device)
        targets = images.new_zeros((images.shape[0], 1, target_size[0], target_size[1]))
        weights = images.new_zeros(targets.shape)
        if bboxes.numel() == 0:
            return targets, weights

        image_h, image_w = images.shape[-2:]
        crop_settings = self.aux_settings.small_crop
        crop_tensors: list[torch.Tensor] = []
        target_regions: list[tuple[int, int, int, int, int]] = []
        for image_index in range(images.shape[0]):
            indices = self._small_box_indices(batch_idx, bboxes, image_index, image_h, image_w)
            for box_index in indices[: crop_settings.max_crops_per_image]:
                crop = self._crop_bounds_for_box(bboxes[box_index], (image_h, image_w))
                left, top, right, bottom = crop
                target_left, target_top, target_right, target_bottom = (
                    self._target_bounds_for_crop(crop, (image_h, image_w), target_size)
                )
                crop_tensor = images[image_index, :, top:bottom, left:right].unsqueeze(0)
                resized = functional.interpolate(
                    crop_tensor,
                    size=(
                        self.objectness_settings.teacher_image_size,
                        self.objectness_settings.teacher_image_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                crop_tensors.append(resized)
                target_regions.append(
                    (image_index, target_left, target_top, target_right, target_bottom)
                )
        self._apply_small_crop_targets(crop_tensors, target_regions, targets, weights)
        return targets, weights

    def _small_tile_soft_loss(
        self,
        logits: torch.Tensor,
        batch: dict[str, BatchValue],
        images: torch.Tensor,
    ) -> torch.Tensor:
        targets, weights = self._small_tile_targets(
            batch=batch,
            images=images,
            target_size=tuple(logits.shape[-2:]),
        )
        if weights.sum() <= 0:
            return logits.sum() * 0.0
        return (
            weighted_soft_objectness_loss(logits, targets, weights)
            * self.objectness_settings.lambda_objectness
        )

    def _small_tile_targets(
        self,
        batch: dict[str, BatchValue],
        images: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = tensor_value(batch, "batch_idx").to(device=images.device)
        bboxes = tensor_value(batch, "bboxes").to(device=images.device)
        targets = images.new_zeros((images.shape[0], 1, target_size[0], target_size[1]))
        weights = images.new_zeros(targets.shape)
        if bboxes.numel() == 0:
            return targets, weights

        image_h, image_w = images.shape[-2:]
        tile_tensors: list[torch.Tensor] = []
        target_regions: list[tuple[int, int, int, int, int]] = []
        for image_index in range(images.shape[0]):
            tiles = self._small_dense_tiles(batch_idx, bboxes, image_index, image_h, image_w)
            for tile in tiles[: self.aux_settings.small_tile.max_tiles_per_image]:
                left, top, right, bottom = tile
                target_left, target_top, target_right, target_bottom = (
                    self._target_bounds_for_crop(tile, (image_h, image_w), target_size)
                )
                tile_tensor = images[image_index, :, top:bottom, left:right].unsqueeze(0)
                resized = functional.interpolate(
                    tile_tensor,
                    size=(
                        self.objectness_settings.teacher_image_size,
                        self.objectness_settings.teacher_image_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                tile_tensors.append(resized)
                target_regions.append(
                    (image_index, target_left, target_top, target_right, target_bottom)
                )
        self._apply_tile_targets(tile_tensors, target_regions, targets, weights)
        return targets, weights

    def _apply_small_crop_targets(
        self,
        crop_tensors: list[torch.Tensor],
        target_regions: list[tuple[int, int, int, int, int]],
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        if not crop_tensors:
            return
        teacher_batch = self.objectness_settings.teacher_batch
        for start in range(0, len(crop_tensors), teacher_batch):
            batch_crops = torch.stack(crop_tensors[start : start + teacher_batch])
            tokens = self._teacher_tokens(batch_crops)
            regions = target_regions[start : start + teacher_batch]
            for token, region in zip(tokens, regions, strict=True):
                self._apply_one_crop_target(token, region, targets, weights)

    def _apply_tile_targets(
        self,
        tile_tensors: list[torch.Tensor],
        target_regions: list[tuple[int, int, int, int, int]],
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        if not tile_tensors:
            return
        teacher_batch = self.objectness_settings.teacher_batch
        for start in range(0, len(tile_tensors), teacher_batch):
            batch_tiles = torch.stack(tile_tensors[start : start + teacher_batch])
            tokens = self._teacher_tokens(batch_tiles)
            regions = target_regions[start : start + teacher_batch]
            for token, region in zip(tokens, regions, strict=True):
                self._apply_one_tile_target(token, region, targets, weights)

    def _apply_one_crop_target(
        self,
        token: torch.Tensor,
        region: tuple[int, int, int, int, int],
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        image_index, target_left, target_top, target_right, target_bottom = region
        crop_score = objectness_score(
            token,
            self.objectness_settings.teacher_patch_grid,
            self.objectness_settings.method,
        )
        resized = functional.interpolate(
            crop_score[None, None],
            size=(target_bottom - target_top, target_right - target_left),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        current_target = targets[image_index, 0, target_top:target_bottom, target_left:target_right]
        targets[image_index, 0, target_top:target_bottom, target_left:target_right] = torch.maximum(
            current_target, resized
        )
        current_weight = weights[image_index, 0, target_top:target_bottom, target_left:target_right]
        weights[image_index, 0, target_top:target_bottom, target_left:target_right] = torch.maximum(
            current_weight,
            current_weight.new_tensor(self.aux_settings.small_crop.weight),
        )

    def _apply_one_tile_target(
        self,
        token: torch.Tensor,
        region: tuple[int, int, int, int, int],
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        image_index, target_left, target_top, target_right, target_bottom = region
        tile_score = objectness_score(
            token,
            self.objectness_settings.teacher_patch_grid,
            self.objectness_settings.method,
        )
        resized = functional.interpolate(
            tile_score[None, None],
            size=(target_bottom - target_top, target_right - target_left),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        current_target = targets[image_index, 0, target_top:target_bottom, target_left:target_right]
        targets[image_index, 0, target_top:target_bottom, target_left:target_right] = torch.maximum(
            current_target, resized
        )
        current_weight = weights[image_index, 0, target_top:target_bottom, target_left:target_right]
        weights[image_index, 0, target_top:target_bottom, target_left:target_right] = torch.maximum(
            current_weight,
            current_weight.new_tensor(self.aux_settings.small_tile.weight),
        )

    def _small_box_indices(
        self,
        batch_idx: torch.Tensor,
        bboxes: torch.Tensor,
        image_index: int,
        image_h: int,
        image_w: int,
    ) -> list[int]:
        selected: list[tuple[float, int]] = []
        for index in range(bboxes.shape[0]):
            if int(batch_idx[index].item()) != image_index:
                continue
            _x, _y, box_w, box_h = bboxes[index]
            area_px = float((box_w * image_w * box_h * image_h).item())
            if area_px <= self.aux_settings.small_crop.small_area_px:
                selected.append((area_px, index))
        return [index for _area, index in sorted(selected)]

    def _small_dense_tiles(
        self,
        batch_idx: torch.Tensor,
        bboxes: torch.Tensor,
        image_index: int,
        image_h: int,
        image_w: int,
    ) -> list[tuple[int, int, int, int]]:
        centers: list[tuple[float, float]] = []
        areas: list[float] = []
        for index in range(bboxes.shape[0]):
            if int(batch_idx[index].item()) != image_index:
                continue
            x_center, y_center, box_w, box_h = bboxes[index]
            area_px = float((box_w * image_w * box_h * image_h).item())
            if area_px > self.aux_settings.small_tile.small_area_px:
                continue
            centers.append((float((x_center * image_w).item()), float((y_center * image_h).item())))
            areas.append(area_px)
        if not centers:
            return []

        tile_settings = self.aux_settings.small_tile
        candidates: list[tuple[int, float, tuple[int, int, int, int]]] = []
        for top in _tile_starts(image_h, tile_settings.tile_size, tile_settings.tile_stride):
            bottom = min(image_h, top + tile_settings.tile_size)
            for left in _tile_starts(image_w, tile_settings.tile_size, tile_settings.tile_stride):
                right = min(image_w, left + tile_settings.tile_size)
                matched_areas = [
                    areas[index]
                    for index, (center_x, center_y) in enumerate(centers)
                    if left <= center_x < right and top <= center_y < bottom
                ]
                count = len(matched_areas)
                if count < tile_settings.min_small_boxes:
                    continue
                mean_area = sum(matched_areas) / max(count, 1)
                candidates.append((count, -mean_area, (left, top, right, bottom)))
        candidates.sort(reverse=True)
        return [tile for _count, _mean_area, tile in candidates]

    def _crop_bounds_for_box(
        self, bbox_xywhn: torch.Tensor, image_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        image_h, image_w = image_size
        x_center, y_center, box_w, box_h = bbox_xywhn
        crop_settings = self.aux_settings.small_crop
        width_px = max(
            float((box_w * image_w * crop_settings.context_scale).item()),
            float(crop_settings.min_crop_size),
        )
        height_px = max(
            float((box_h * image_h * crop_settings.context_scale).item()),
            float(crop_settings.min_crop_size),
        )
        center_x = float((x_center * image_w).item())
        center_y = float((y_center * image_h).item())
        left = round(center_x - width_px * 0.5)
        right = round(center_x + width_px * 0.5)
        top = round(center_y - height_px * 0.5)
        bottom = round(center_y + height_px * 0.5)
        return _clamp_crop((left, top, right, bottom), image_size)

    @staticmethod
    def _target_bounds_for_crop(
        crop: tuple[int, int, int, int],
        image_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = crop
        image_h, image_w = image_size
        target_h, target_w = target_size
        target_left = max(0, round(left * target_w / image_w))
        target_right = min(target_w, max(target_left + 1, round(right * target_w / image_w)))
        target_top = max(0, round(top * target_h / image_h))
        target_bottom = min(target_h, max(target_top + 1, round(bottom * target_h / image_h)))
        return target_left, target_top, target_right, target_bottom

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
        case (
            "soft"
            | "ignore_aware"
            | "small_gt_weighted_soft"
            | "peak_ignore_aware"
            | "small_crop_soft"
            | "small_tile_soft"
        ):
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
        small_crop=SmallCropTargetSettings(
            small_area_px=float(objectness.get("small_area_px", 1024.0)),
            context_scale=float(objectness.get("crop_context_scale", 6.0)),
            min_crop_size=int(objectness.get("crop_min_size", 160)),
            max_crops_per_image=int(objectness.get("max_crops_per_image", 12)),
            weight=float(objectness.get("crop_weight", 1.0)),
        ),
        small_tile=SmallTileTargetSettings(
            small_area_px=float(objectness.get("small_area_px", 1024.0)),
            tile_size=int(objectness.get("density_tile_size", objectness.get("tile_size", 480))),
            tile_stride=int(
                objectness.get("density_tile_stride", objectness.get("tile_stride", 240))
            ),
            max_tiles_per_image=int(objectness.get("max_tiles_per_image", 4)),
            min_small_boxes=int(objectness.get("min_small_boxes_per_tile", 3)),
            weight=float(objectness.get("tile_weight", 1.0)),
        ),
    )


def _clamp_crop(
    crop: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = crop
    image_h, image_w = image_size
    crop_w = min(image_w, max(1, right - left))
    crop_h = min(image_h, max(1, bottom - top))
    left = min(max(0, left), image_w - crop_w)
    top = min(max(0, top), image_h - crop_h)
    return left, top, left + crop_w, top + crop_h


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    tile_size = max(1, min(length, tile_size))
    stride = max(1, stride)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


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
