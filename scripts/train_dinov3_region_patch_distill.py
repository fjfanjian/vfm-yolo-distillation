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

import argparse
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model


@dataclass(frozen=True, slots=True)
class DistillSettings:
    teacher_repo: Path
    teacher_weights: Path
    teacher_arch: str
    teacher_image_size: int
    teacher_patch_grid: int
    student_layer: int
    student_dim: int
    teacher_dim: int
    box_expand_tokens: int
    lambda_region: float


@dataclass(frozen=True, slots=True)
class TrainSettings:
    model: str
    data: Path
    image_size: int
    epochs: int
    batch: int
    device: str
    workers: int
    seed: int
    project: Path
    name: str
    distill: DistillSettings


class ConfigError(RuntimeError):
    pass


def capture_dino_student_feature(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    module.dino_student_feature = output


class DinoRegionPatchDistillTrainer(DetectionTrainer):
    loss_names = ("box_loss", "cls_loss", "dfl_loss", "dino_region_loss")

    def __init__(self, settings: TrainSettings, overrides: dict[str, str | int | float | bool]) -> None:
        self.distill_settings = settings.distill
        self.teacher_model: nn.Module | None = None
        self.teacher_mean: torch.Tensor | None = None
        self.teacher_std: torch.Tensor | None = None
        super().__init__(overrides=overrides)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> DetectionModel:
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.dino_patch_projector = nn.Conv2d(self.distill_settings.student_dim, self.distill_settings.teacher_dim, 1)
        self._register_student_feature_hook(model)
        model.loss = types.MethodType(self._distill_loss, model)
        return model

    def _register_student_feature_hook(self, model: DetectionModel) -> None:
        model.model[self.distill_settings.student_layer].register_forward_hook(capture_dino_student_feature)

    def _load_teacher(self, device: torch.device) -> nn.Module:
        if self.teacher_model is None:
            sys.path.insert(0, str(self.distill_settings.teacher_repo))
            from dinov3.hub.backbones import dinov3_vitb16

            if self.distill_settings.teacher_arch != "dinov3_vitb16":
                raise ConfigError(f"Unsupported DINOv3 arch: {self.distill_settings.teacher_arch}")
            teacher = dinov3_vitb16(weights=str(self.distill_settings.teacher_weights))
            teacher.eval()
            teacher.requires_grad_(False)
            self.teacher_model = teacher
        return self.teacher_model.to(device)

    def _teacher_norm(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.teacher_mean is None or self.teacher_std is None:
            self.teacher_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            self.teacher_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return self.teacher_mean.to(device=device, dtype=dtype), self.teacher_std.to(device=device, dtype=dtype)

    def _teacher_tokens(self, images: torch.Tensor) -> torch.Tensor:
        teacher = self._load_teacher(images.device)
        mean, std = self._teacher_norm(images.device, images.dtype)
        resized = F.interpolate(images, size=(self.distill_settings.teacher_image_size, self.distill_settings.teacher_image_size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = teacher.forward_features((resized - mean) / std)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise ConfigError("DINOv3 teacher did not return x_norm_patchtokens")
        return features["x_norm_patchtokens"].detach()

    def _region_mask(self, batch: dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        grid = self.distill_settings.teacher_patch_grid
        mask = torch.zeros((batch_size, grid, grid), device=device)
        boxes = batch["bboxes"].detach().to(device)
        batch_idx = batch["batch_idx"].detach().to(device).long()
        expand = self.distill_settings.box_expand_tokens
        for index, box in zip(batch_idx.tolist(), boxes.tolist(), strict=True):
            cx, cy, width, height = box[:4]
            left = max(0, int((cx - width / 2.0) * grid) - expand)
            right = min(grid, int((cx + width / 2.0) * grid) + expand + 1)
            top = max(0, int((cy - height / 2.0) * grid) - expand)
            bottom = min(grid, int((cy + height / 2.0) * grid) + expand + 1)
            mask[index, top:max(top + 1, bottom), left:max(left + 1, right)] = 1.0
        return mask.flatten(1)

    def _distill_loss(self, model: DetectionModel, batch: dict[str, torch.Tensor], preds: torch.Tensor | None = None):
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        if preds is None:
            preds = model.forward(batch["img"])
        detection_loss, detection_items = model.criterion(preds, batch)
        layer = model.model[self.distill_settings.student_layer]
        student_feature = getattr(layer, "dino_student_feature", None)
        if not isinstance(student_feature, torch.Tensor):
            return detection_loss, torch.cat((detection_items, detection_loss.new_zeros(1)))
        grid = self.distill_settings.teacher_patch_grid
        resized = F.interpolate(student_feature, size=(grid, grid), mode="bilinear", align_corners=False)
        projected = model.dino_patch_projector(resized).flatten(2).transpose(1, 2)
        per_token = 1.0 - F.cosine_similarity(projected, self._teacher_tokens(batch["img"]), dim=2)
        mask = self._region_mask(batch, batch["img"].shape[0], batch["img"].device)
        region_loss = (per_token * mask).sum() / mask.sum().clamp_min(1.0)
        return detection_loss + region_loss * self.distill_settings.lambda_region, torch.cat((detection_items, region_loss.detach().reshape(1)))

    def save_model(self) -> bool:
        models = [unwrap_model(self.model)]
        if self.ema is not None:
            models.append(unwrap_model(self.ema.ema))
        saved_loss = [model.__dict__.pop("loss", None) for model in models]
        saved_hooks = []
        for model in models:
            layer = model.model[self.distill_settings.student_layer]
            saved_hooks.append(layer._forward_hooks.copy())
            layer._forward_hooks = OrderedDict()
            layer.__dict__.pop("dino_student_feature", None)
        try:
            return bool(super().save_model())
        finally:
            for model, loss_method, hooks in zip(models, saved_loss, saved_hooks, strict=True):
                model.model[self.distill_settings.student_layer]._forward_hooks = hooks
                if loss_method is not None:
                    model.loss = loss_method


def expect_mapping(value: str | int | float | bool | dict[str, str | int | float], name: str) -> dict[str, str | int | float]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def settings_from_config(path: Path, args: argparse.Namespace) -> TrainSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    dataset = expect_mapping(raw["dataset"], "dataset")
    training = expect_mapping(raw["training"], "training")
    outputs = expect_mapping(raw["outputs"], "outputs")
    student = expect_mapping(raw["student"], "student")
    distill = expect_mapping(raw["distillation"], "distillation")
    label_budget = str(dataset["label_budget"])
    data_path = Path(str(dataset["config"])).with_stem(f"{Path(str(dataset['config'])).stem}_{label_budget}")
    return TrainSettings(str(student["model"]), data_path, int(training["image_size"]), args.epochs or int(training["epochs"]), args.batch or int(training["batch"]), str(training["device"]), int(training["workers"]), int(training["seed"]), Path(str(outputs["project"])).resolve(), args.name or str(outputs["name"]), DistillSettings(Path(str(distill["teacher_repo"])), Path(str(distill["teacher_weights"])), str(distill["teacher_arch"]), int(distill["teacher_image_size"]), int(distill["teacher_patch_grid"]), int(distill["student_layer"]), int(distill["student_dim"]), int(distill["teacher_dim"]), int(distill["box_expand_tokens"]), float(distill["lambda_region"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/dinov3_region_patch_distill_visdrone_10pct.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = settings_from_config(Path(args.config), args)
    overrides = {"task": "detect", "mode": "train", "model": settings.model, "data": settings.data.as_posix(), "imgsz": settings.image_size, "epochs": settings.epochs, "batch": settings.batch, "device": settings.device, "workers": settings.workers, "seed": settings.seed, "project": settings.project.as_posix(), "name": settings.name, "exist_ok": True, "pretrained": True, "optimizer": "auto", "patience": 100, "close_mosaic": 10, "amp": True, "plots": False}
    DinoRegionPatchDistillTrainer(settings=settings, overrides=overrides).train()


if __name__ == "__main__":
    main()
