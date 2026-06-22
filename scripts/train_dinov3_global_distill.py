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


@dataclass(frozen=True, slots=True)
class DistillSettings:
    teacher_repo: Path
    teacher_weights: Path
    teacher_arch: str
    teacher_image_size: int
    teacher_dim: int
    student_layer: int
    student_dim: int
    lambda_global: float


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


class DinoGlobalDistillTrainer(DetectionTrainer):
    loss_names = ("box_loss", "cls_loss", "dfl_loss", "dino_loss")

    def __init__(self, settings: TrainSettings, overrides: dict[str, str | int | float | bool]) -> None:
        self.distill_settings = settings.distill
        self.teacher_model: nn.Module | None = None
        self.teacher_mean: torch.Tensor | None = None
        self.teacher_std: torch.Tensor | None = None
        super().__init__(overrides=overrides)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> DetectionModel:
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.dino_projector = nn.Linear(self.distill_settings.student_dim, self.distill_settings.teacher_dim)
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

    def _teacher_features(self, images: torch.Tensor) -> torch.Tensor:
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
        if not isinstance(features, dict) or "x_norm_clstoken" not in features:
            raise ConfigError("DINOv3 teacher did not return x_norm_clstoken")
        return features["x_norm_clstoken"].detach()

    def _distill_loss(self, model: DetectionModel, batch: dict[str, torch.Tensor], preds: torch.Tensor | None = None):
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        if preds is None:
            preds = model.forward(batch["img"])
        detection_loss, detection_items = model.criterion(preds, batch)
        student_layer = model.model[self.distill_settings.student_layer]
        student_feature = getattr(student_layer, "dino_student_feature", None)
        if not isinstance(student_feature, torch.Tensor):
            zero_dino_loss = detection_loss.new_zeros(1)
            return detection_loss, torch.cat((detection_items, zero_dino_loss))
        pooled = torch_functional.adaptive_avg_pool2d(student_feature, (1, 1)).flatten(1)
        student_global = model.dino_projector(pooled)
        teacher_global = self._teacher_features(batch["img"])
        dino_loss = 1.0 - torch_functional.cosine_similarity(student_global, teacher_global, dim=1).mean()
        total_loss = detection_loss + dino_loss * self.distill_settings.lambda_global
        loss_items = torch.cat((detection_items, dino_loss.detach().reshape(1)))
        return total_loss, loss_items

    def save_model(self) -> bool:
        models = [unwrap_model(self.model)]
        if self.ema is not None:
            models.append(unwrap_model(self.ema.ema))
        saved_loss_methods = [model.__dict__.pop("loss", None) for model in models]
        for model in models:
            layer = model.model[self.distill_settings.student_layer]
            if hasattr(layer, "dino_student_feature"):
                delattr(layer, "dino_student_feature")
        try:
            return bool(super().save_model())
        finally:
            for model, loss_method in zip(models, saved_loss_methods, strict=True):
                if loss_method is not None:
                    model.loss = loss_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/dinov3_global_distill_visdrone_10pct.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--name")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, str | int | float | bool | dict[str, str | int | float]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    return loaded


def settings_from_config(path: Path, args: argparse.Namespace) -> TrainSettings:
    raw = read_yaml(path)
    dataset = expect_mapping(raw["dataset"], "dataset")
    training = expect_mapping(raw["training"], "training")
    outputs = expect_mapping(raw["outputs"], "outputs")
    student = expect_mapping(raw["student"], "student")
    distill = expect_mapping(raw["distillation"], "distillation")
    label_budget = str(dataset["label_budget"])
    dataset_config = Path(str(dataset["config"]))
    data_path = dataset_config.with_stem(f"{dataset_config.stem}_{label_budget}") if label_budget != "full" else dataset_config
    return TrainSettings(
        model=str(student["model"]),
        data=data_path,
        image_size=int(training["image_size"]),
        epochs=args.epochs or int(training["epochs"]),
        batch=args.batch or int(training["batch"]),
        device=str(training["device"]),
        workers=int(training["workers"]),
        seed=int(training["seed"]),
        project=Path(str(outputs["project"])).resolve(),
        name=args.name or str(outputs["name"]),
        distill=DistillSettings(
            teacher_repo=Path(str(distill["teacher_repo"])),
            teacher_weights=Path(str(distill["teacher_weights"])),
            teacher_arch=str(distill["teacher_arch"]),
            teacher_image_size=int(distill["teacher_image_size"]),
            teacher_dim=int(distill["teacher_dim"]),
            student_layer=int(distill["student_layer"]),
            student_dim=int(distill["student_dim"]),
            lambda_global=float(distill["lambda_global"]),
        ),
    )


def expect_mapping(value: str | int | float | bool | dict[str, str | int | float], name: str) -> dict[str, str | int | float]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


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
        "plots": True,
    }


def main() -> None:
    args = parse_args()
    settings = settings_from_config(Path(args.config), args)
    trainer = DinoGlobalDistillTrainer(settings=settings, overrides=trainer_overrides(settings))
    trainer.train()


if __name__ == "__main__":
    main()
