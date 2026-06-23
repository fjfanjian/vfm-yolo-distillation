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
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as torch_functional
import yaml
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model
from vfm_yolo_distillation.config import load_experiment_config, training_dataset_config_path
from vfm_yolo_distillation.relation_distillation import relation_distillation_loss

YamlScalar = str | int | float | bool
YamlSimpleMapping = dict[str, YamlScalar]
YamlMapping = dict[str, YamlScalar | YamlSimpleMapping]


@dataclass(frozen=True, slots=True)
class DistillSettings:
    teacher_repo: Path
    teacher_weights: Path
    teacher_arch: str
    teacher_image_size: int
    teacher_patch_grid: int
    teacher_dim: int
    student_layer: int
    student_dim: int
    relation_tokens: int
    lambda_relation: float


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


def capture_dino_student_feature(
    module: nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> None:
    setattr(module, "dino_student_feature", output)


class DinoRelationDistillTrainer(DetectionTrainer):
    loss_names = ("box_loss", "cls_loss", "dfl_loss", "dino_relation_loss")

    def __init__(self, settings: TrainSettings, overrides: dict[str, str | int | float | bool]) -> None:
        self.distill_settings = settings.distill
        self.teacher_model: nn.Module | None = None
        self.teacher_mean: torch.Tensor | None = None
        self.teacher_std: torch.Tensor | None = None
        super().__init__(overrides=overrides)

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> DetectionModel:
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        projector = nn.Conv2d(
            self.distill_settings.student_dim,
            self.distill_settings.teacher_dim,
            kernel_size=1,
        )
        model.add_module("dino_relation_projector", projector)
        self._register_student_feature_hook(model)
        model.loss = types.MethodType(self._distill_loss, model)
        return model

    def _register_student_feature_hook(self, model: DetectionModel) -> None:
        layer = model.model[self.distill_settings.student_layer]
        layer.register_forward_hook(capture_dino_student_feature)

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

    def _teacher_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        teacher = self._load_teacher(images.device)
        mean, std = self._teacher_norm(images.device, images.dtype)
        resized = torch_functional.interpolate(
            images,
            size=(self.distill_settings.teacher_image_size, self.distill_settings.teacher_image_size),
            mode="bilinear",
            align_corners=False,
        )
        normalized = (resized - mean) / std
        with torch.no_grad():
            features = teacher.forward_features(normalized)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise ConfigError("DINOv3 teacher did not return x_norm_patchtokens")
        patch_tokens = features["x_norm_patchtokens"]
        if not isinstance(patch_tokens, torch.Tensor):
            raise ConfigError("DINOv3 patch tokens must be a tensor")
        return patch_tokens.detach()

    def _student_patch_tokens(self, model: DetectionModel, student_feature: torch.Tensor) -> torch.Tensor:
        projector = getattr(model, "dino_relation_projector", None)
        if not isinstance(projector, nn.Conv2d):
            raise ConfigError("Missing DINO relation projector")
        grid = self.distill_settings.teacher_patch_grid
        pooled = torch_functional.adaptive_avg_pool2d(student_feature, (grid, grid))
        return projector(pooled).flatten(2).transpose(1, 2)

    def _distill_loss(
        self,
        model: DetectionModel,
        batch: dict[str, torch.Tensor],
        preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        if preds is None:
            preds = model.forward(batch["img"])
        detection_loss, detection_items = model.criterion(preds, batch)
        student_layer = model.model[self.distill_settings.student_layer]
        student_feature = getattr(student_layer, "dino_student_feature", None)
        if not isinstance(student_feature, torch.Tensor):
            zero_relation_loss = detection_loss.new_zeros(1)
            return detection_loss, torch.cat((detection_items, zero_relation_loss))
        student_tokens = self._student_patch_tokens(model, student_feature)
        teacher_tokens = self._teacher_patch_tokens(batch["img"]).to(dtype=student_tokens.dtype)
        relation_loss = relation_distillation_loss(
            student_tokens,
            teacher_tokens,
            max_tokens=self.distill_settings.relation_tokens,
        )
        total_loss = detection_loss + relation_loss * self.distill_settings.lambda_relation
        loss_items = torch.cat((detection_items, relation_loss.detach().reshape(1)))
        return total_loss, loss_items

    def save_model(self) -> bool:
        models = [unwrap_model(self.model)]
        if self.ema is not None:
            models.append(unwrap_model(self.ema.ema))
        saved_loss_methods = [model.__dict__.pop("loss", None) for model in models]
        saved_projectors = [model._modules.pop("dino_relation_projector", None) for model in models]
        for model in models:
            layer = model.model[self.distill_settings.student_layer]
            if hasattr(layer, "dino_student_feature"):
                delattr(layer, "dino_student_feature")
        try:
            return bool(super().save_model())
        finally:
            for model, loss_method, projector in zip(
                models,
                saved_loss_methods,
                saved_projectors,
                strict=True,
            ):
                if projector is not None:
                    model.add_module("dino_relation_projector", projector)
                if loss_method is not None:
                    model.loss = loss_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/dinov3_relation_distill_visdrone_10pct_imgsz960.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--name")
    return parser.parse_args()


def read_yaml(path: Path) -> YamlMapping:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    return loaded


def expect_mapping(value: YamlScalar | YamlSimpleMapping, name: str) -> YamlSimpleMapping:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def settings_from_config(path: Path, args: argparse.Namespace) -> TrainSettings:
    config = load_experiment_config(path)
    raw = read_yaml(path)
    distill = expect_mapping(raw["distillation"], "distillation")
    return TrainSettings(
        model=config.student.model,
        data=training_dataset_config_path(config),
        image_size=config.training.image_size,
        epochs=args.epochs or config.training.epochs,
        batch=args.batch or config.training.batch,
        device=str(config.training.device),
        workers=config.training.workers,
        seed=config.training.seed,
        project=config.outputs.project.resolve(),
        name=args.name or config.outputs.name,
        distill=DistillSettings(
            teacher_repo=Path(str(distill["teacher_repo"])),
            teacher_weights=Path(str(distill["teacher_weights"])),
            teacher_arch=str(distill["teacher_arch"]),
            teacher_image_size=int(distill["teacher_image_size"]),
            teacher_patch_grid=int(distill["teacher_patch_grid"]),
            teacher_dim=int(distill["teacher_dim"]),
            student_layer=int(distill["student_layer"]),
            student_dim=int(distill["student_dim"]),
            relation_tokens=int(distill["relation_tokens"]),
            lambda_relation=float(distill["lambda_relation"]),
        ),
    )


def trainer_overrides(settings: TrainSettings) -> dict[str, str | int | float | bool]:
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
        "project": settings.project.as_posix(),
        "name": settings.name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "patience": 100,
        "close_mosaic": 10,
        "amp": True,
        "plots": False,
    }


def main() -> None:
    args = parse_args()
    settings = settings_from_config(Path(args.config), args)
    trainer = DinoRelationDistillTrainer(settings=settings, overrides=trainer_overrides(settings))
    trainer.train()


if __name__ == "__main__":
    main()
