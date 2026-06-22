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
import hashlib
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import yaml
from audit_dinov3_objectness import objectness_score, tile_positions
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model

ObjectnessMethod = Literal["border_pca", "local_contrast", "local_residual", "local_fusion"]
YamlScalar = str | int | float | bool
YamlMap = dict[str, YamlScalar | dict[str, YamlScalar]]
BatchValue = torch.Tensor | list[str]


@dataclass(frozen=True, slots=True)
class ObjectnessSettings:
    teacher_repo: Path
    teacher_weights: Path
    teacher_arch: str
    teacher_image_size: int
    teacher_patch_grid: int
    teacher_batch: int
    method: ObjectnessMethod
    tile_size: int
    tile_stride: int
    student_layer: int
    student_dim: int
    lambda_objectness: float
    cache: Path


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
    fraction: float
    project: Path
    name: str
    objectness: ObjectnessSettings


class ConfigError(RuntimeError):
    pass


def capture_student_feature(
    module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> None:
    module.dino_objectness_feature = output


def normalize_map(score: torch.Tensor) -> torch.Tensor:
    return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-6)


class DinoObjectnessPretrainTrainer(DetectionTrainer):
    loss_names = ("objectness_loss", "objectness_bce", "objectness_dice")

    def __init__(
        self, settings: TrainSettings, overrides: dict[str, str | int | float | bool]
    ) -> None:
        self.objectness_settings = settings.objectness
        self.teacher_model: nn.Module | None = None
        self.teacher_mean: torch.Tensor | None = None
        self.teacher_std: torch.Tensor | None = None
        super().__init__(overrides=overrides)

    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ) -> DetectionModel:
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.dino_objectness_head = nn.Conv2d(
            self.objectness_settings.student_dim, 1, kernel_size=1
        )
        model.criterion = None
        model.model[self.objectness_settings.student_layer].register_forward_hook(
            capture_student_feature
        )
        model.loss = types.MethodType(self._objectness_loss, model)
        return model

    def _load_teacher(self, device: torch.device) -> nn.Module:
        if self.teacher_model is None:
            sys.path.insert(0, self.objectness_settings.teacher_repo.as_posix())
            from dinov3.hub.backbones import dinov3_vitb16

            if self.objectness_settings.teacher_arch != "dinov3_vitb16":
                raise ConfigError(
                    f"Unsupported DINOv3 arch: {self.objectness_settings.teacher_arch}"
                )
            teacher = dinov3_vitb16(weights=self.objectness_settings.teacher_weights.as_posix())
            teacher.eval()
            teacher.requires_grad_(False)
            self.teacher_model = teacher
        return self.teacher_model.to(device)

    def _teacher_norm(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.teacher_mean is None or self.teacher_std is None:
            self.teacher_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            self.teacher_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return self.teacher_mean.to(device=device, dtype=dtype), self.teacher_std.to(
            device=device, dtype=dtype
        )

    def _teacher_tokens(self, crops: torch.Tensor) -> torch.Tensor:
        settings = self.objectness_settings
        mean, std = self._teacher_norm(crops.device, crops.dtype)
        resized = F.interpolate(
            crops,
            size=(settings.teacher_image_size, settings.teacher_image_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            features = self._load_teacher(crops.device).forward_features((resized - mean) / std)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise ConfigError("DINOv3 teacher did not return x_norm_patchtokens")
        tokens = features["x_norm_patchtokens"]
        if not isinstance(tokens, torch.Tensor):
            raise ConfigError("DINOv3 patch tokens must be a tensor")
        return tokens.detach()

    def _cache_path(self, image_file: str, target_size: tuple[int, int]) -> Path:
        settings = self.objectness_settings
        key = "|".join(
            (
                image_file,
                settings.method,
                str(settings.tile_size),
                str(settings.tile_stride),
                str(settings.teacher_image_size),
                f"{target_size[0]}x{target_size[1]}",
            )
        )
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
        return (
            settings.cache
            / f"{Path(image_file).stem}_{target_size[0]}x{target_size[1]}_{digest}.pt"
        )

    def _target_from_cache(
        self, image_file: str, target_size: tuple[int, int], device: torch.device
    ) -> torch.Tensor | None:
        cache_path = self._cache_path(image_file, target_size)
        if not cache_path.exists():
            return None
        cached = torch.load(cache_path, map_location=device, weights_only=True)
        if not isinstance(cached, torch.Tensor) or tuple(cached.shape) != target_size:
            return None
        return cached.to(device=device, dtype=torch.float32)

    def _save_target(self, image_file: str, target: torch.Tensor) -> None:
        cache_path = self._cache_path(image_file, tuple(target.shape))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(target.detach().cpu().to(torch.float16), cache_path)

    def _objectness_for_image(
        self, image: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        settings = self.objectness_settings
        _channels, height, width = image.shape
        stride = settings.tile_stride or max(1, settings.tile_size // 2)
        score_sum = torch.zeros(target_size, device=image.device)
        score_count = torch.zeros(target_size, device=image.device)
        windows = [
            (
                left,
                top,
                min(left + settings.tile_size, width),
                min(top + settings.tile_size, height),
            )
            for top in tile_positions(height, settings.tile_size, stride)
            for left in tile_positions(width, settings.tile_size, stride)
        ]
        for start in range(0, len(windows), settings.teacher_batch):
            batch_windows = windows[start : start + settings.teacher_batch]
            crops = torch.stack(
                [image[:, top:bottom, left:right] for left, top, right, bottom in batch_windows]
            )
            tokens = self._teacher_tokens(crops)
            for token, window in zip(tokens, batch_windows, strict=True):
                self._add_window_score(
                    token, window, (height, width), target_size, score_sum, score_count
                )
        return normalize_map(score_sum / score_count.clamp_min(1.0))

    def _add_window_score(
        self,
        token: torch.Tensor,
        window: tuple[int, int, int, int],
        image_size: tuple[int, int],
        target_size: tuple[int, int],
        score_sum: torch.Tensor,
        score_count: torch.Tensor,
    ) -> None:
        left, top, right, bottom = window
        height, width = image_size
        target_h, target_w = target_size
        target_left = max(0, round(left * target_w / width))
        target_right = min(target_w, max(target_left + 1, round(right * target_w / width)))
        target_top = max(0, round(top * target_h / height))
        target_bottom = min(target_h, max(target_top + 1, round(bottom * target_h / height)))
        tile_score = objectness_score(
            token, self.objectness_settings.teacher_patch_grid, self.objectness_settings.method
        )
        resized = (
            F.interpolate(
                tile_score[None, None],
                size=(target_bottom - target_top, target_right - target_left),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        score_sum[target_top:target_bottom, target_left:target_right] += resized
        score_count[target_top:target_bottom, target_left:target_right] += 1.0

    def _teacher_targets(
        self,
        images: torch.Tensor,
        image_files: list[str],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        targets: list[torch.Tensor] = []
        for image, image_file in zip(images, image_files, strict=True):
            cached = self._target_from_cache(image_file, target_size, images.device)
            if cached is None:
                cached = self._objectness_for_image(image, target_size)
                self._save_target(image_file, cached)
            targets.append(cached)
        return torch.stack(targets).unsqueeze(1)

    def _objectness_loss(
        self, model: DetectionModel, batch: dict[str, BatchValue], preds: torch.Tensor | None = None
    ):
        images = tensor_value(batch, "img")
        if preds is None:
            _ = model.forward(images)
        layer = model.model[self.objectness_settings.student_layer]
        feature = getattr(layer, "dino_objectness_feature", None)
        head = getattr(model, "dino_objectness_head", None)
        if not isinstance(feature, torch.Tensor) or not isinstance(head, nn.Conv2d):
            zero = images.sum() * 0.0
            return zero, torch.stack((zero.detach(), zero.detach(), zero.detach()))
        logits = head(feature)
        targets = self._teacher_targets(
            images, image_files(batch, images.shape[0]), tuple(logits.shape[-2:])
        )
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dice = dice_loss(probs, targets)
        total = (bce + dice) * 0.5 * self.objectness_settings.lambda_objectness
        return total, torch.stack((total.detach(), bce.detach(), dice.detach()))

    def save_model(self) -> bool:
        models = [unwrap_model(self.model)]
        if self.ema is not None:
            models.append(unwrap_model(self.ema.ema))
        saved_loss = [model.__dict__.pop("loss", None) for model in models]
        saved_heads: list[nn.Module | None] = []
        saved_hooks = []
        for model in models:
            layer = model.model[self.objectness_settings.student_layer]
            saved_hooks.append(layer._forward_hooks.copy())
            layer._forward_hooks = OrderedDict()
            layer.__dict__.pop("dino_objectness_feature", None)
            saved_heads.append(model._modules.pop("dino_objectness_head", None))
        try:
            return bool(super().save_model())
        finally:
            for model, loss_method, hooks, head in zip(
                models, saved_loss, saved_hooks, saved_heads, strict=True
            ):
                model.model[self.objectness_settings.student_layer]._forward_hooks = hooks
                if head is not None:
                    model.dino_objectness_head = head
                if loss_method is not None:
                    model.loss = loss_method


def dice_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def tensor_value(batch: dict[str, BatchValue], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise ConfigError(f"Batch value must be a tensor: {key}")
    return value


def image_files(batch: dict[str, BatchValue], batch_size: int) -> list[str]:
    value = batch.get("im_file")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return [f"batch_item_{index}" for index in range(batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/dinov3_objectness_pretrain_visdrone_full_imgsz960.yaml",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--name")
    parser.add_argument("--fraction", type=float)
    return parser.parse_args()


def expect_mapping(value: YamlScalar | dict[str, YamlScalar], name: str) -> dict[str, YamlScalar]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def read_yaml(path: Path) -> YamlMap:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    return loaded


def data_path(dataset: dict[str, YamlScalar]) -> Path:
    label_budget = str(dataset["label_budget"])
    dataset_config = Path(str(dataset["config"]))
    return (
        dataset_config
        if label_budget == "full"
        else dataset_config.with_stem(f"{dataset_config.stem}_{label_budget}")
    )


def objectness_method(value: str) -> ObjectnessMethod:
    match value:
        case "border_pca" | "local_contrast" | "local_residual" | "local_fusion":
            return value
        case _:
            raise ConfigError(f"Unsupported objectness method: {value}")


def settings_from_config(path: Path, args: argparse.Namespace) -> TrainSettings:
    raw = read_yaml(path)
    dataset = expect_mapping(raw["dataset"], "dataset")
    training = expect_mapping(raw["training"], "training")
    outputs = expect_mapping(raw["outputs"], "outputs")
    student = expect_mapping(raw["student"], "student")
    objectness = expect_mapping(raw["objectness"], "objectness")
    return TrainSettings(
        model=str(student["model"]),
        data=data_path(dataset),
        image_size=int(training["image_size"]),
        epochs=args.epochs or int(training["epochs"]),
        batch=args.batch or int(training["batch"]),
        device=str(training["device"]),
        workers=int(training["workers"]),
        seed=int(training["seed"]),
        fraction=args.fraction or float(training.get("fraction", 1.0)),
        project=Path(str(outputs["project"])).resolve(),
        name=args.name or str(outputs["name"]),
        objectness=ObjectnessSettings(
            teacher_repo=Path(str(objectness["teacher_repo"])),
            teacher_weights=Path(str(objectness["teacher_weights"])),
            teacher_arch=str(objectness["teacher_arch"]),
            teacher_image_size=int(objectness["teacher_image_size"]),
            teacher_patch_grid=int(objectness["teacher_patch_grid"]),
            teacher_batch=int(objectness["teacher_batch"]),
            method=objectness_method(str(objectness["method"])),
            tile_size=int(objectness["tile_size"]),
            tile_stride=int(objectness["tile_stride"]),
            student_layer=int(objectness["student_layer"]),
            student_dim=int(objectness["student_dim"]),
            lambda_objectness=float(objectness["lambda_objectness"]),
            cache=Path(str(objectness["cache"])),
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
        "fraction": settings.fraction,
        "project": settings.project.as_posix(),
        "name": settings.name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "patience": 0,
        "val": False,
        "plots": False,
        "amp": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }


def main() -> None:
    args = parse_args()
    settings = settings_from_config(Path(args.config), args)
    trainer = DinoObjectnessPretrainTrainer(
        settings=settings, overrides=trainer_overrides(settings)
    )
    trainer.train()


if __name__ == "__main__":
    main()
