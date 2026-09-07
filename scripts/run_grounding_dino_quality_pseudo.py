#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow>=10.0",
#   "pydantic>=2.7",
#   "pyyaml>=6.0",
#   "transformers>=4.45",
#   "ultralytics>=8.3",
# ]
# ///
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfm_yolo_distillation.grounding_dino_eval import (  # noqa: E402
    GroundingDinoSettings,
    collect_predictions,
)
from vfm_yolo_distillation.pseudo_label_quality import (  # noqa: E402
    DetectionBox,
    MixedDatasetRequest,
    ThresholdPolicy,
    audit_candidates,
    candidate_from_box,
    load_yolo_boxes,
    read_candidates_csv,
    read_image_source,
    write_candidates_csv,
    write_mixed_dataset,
    write_quality_summary,
)


class Phase(StrEnum):
    ALL = "all"
    CACHE = "cache"
    AUDIT = "audit"
    BUILD = "build"
    TRAIN_EVAL_EXPORT = "train_eval_export"


class GroundingDinoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    box_threshold: float
    text_threshold: float
    max_det: int
    device: str


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    epochs: int
    batch: int
    image_size: int
    workers: int
    seed: int
    device: str


class ThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: float | None = None
    small: float | None = None
    medium: float | None = None
    large: float | None = None

    def policy(self) -> ThresholdPolicy:
        if self.default is not None:
            return ThresholdPolicy.uniform(self.default)
        if self.small is None or self.medium is None or self.large is None:
            raise ValueError("Area thresholds require small, medium, and large.")
        return ThresholdPolicy.area_sensitive(self.small, self.medium, self.large)


class VariantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    run_name: str
    thresholds: ThresholdConfig


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_data: Path
    labeled_data: Path
    dataset_root_override: Path | None = None
    prediction_cache: Path
    output_root: Path
    project: Path
    report_dir: Path
    grounding_dino: GroundingDinoConfig
    training: TrainingConfig
    variants: dict[str, VariantConfig]


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    root: Path
    train_source: Path
    val_source: Path
    names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[phase.value for phase in Phase],
        default=Phase.ALL.value,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("s030", "area"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return ExperimentConfig.model_validate(cast("dict[str, object]", payload))


def load_layout(data_yaml: Path, dataset_root: Path | None) -> DatasetLayout:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset config must be a YAML mapping: {data_yaml}")
    root = dataset_root if dataset_root is not None else Path(str(payload["path"]))
    names = tuple(str(name) for name in cast("list[object]", payload["names"]))
    return DatasetLayout(
        root=root,
        train_source=root / str(payload["train"]),
        val_source=root / str(payload["val"]),
        names=names,
    )


def grounding_settings(config: ExperimentConfig) -> GroundingDinoSettings:
    settings = config.grounding_dino
    return GroundingDinoSettings(
        model_id=settings.model_id,
        box_threshold=settings.box_threshold,
        text_threshold=settings.text_threshold,
        max_det=settings.max_det,
        device=settings.device,
    )


def run_cache(config: ExperimentConfig, dataset_root: Path | None, rebuild: bool) -> None:
    if config.prediction_cache.exists() and not rebuild:
        print(f"raw_prediction_cache=exists path={config.prediction_cache}", flush=True)
        return
    layout = load_layout(config.source_data, dataset_root)
    images = read_image_source(layout.train_source, layout.root)
    print(f"grounding_dino_cache_images={len(images)}", flush=True)
    predictions = collect_predictions(grounding_settings(config), list(images))
    image_index = {image_path.stem: image_path for image_path in images}
    candidates = tuple(
        candidate_from_box(
            image_index[box.image_id],
            layout.names,
            DetectionBox(box.image_id, box.class_id, box.xyxy, box.area, box.score),
        )
        for box in predictions
        if box.image_id in image_index
    )
    write_candidates_csv(config.prediction_cache, candidates)
    print(
        f"raw_prediction_cache={config.prediction_cache} pseudo_candidates={len(candidates)}",
        flush=True,
    )


def run_audit(config: ExperimentConfig, dataset_root: Path | None) -> None:
    labeled_layout = load_layout(config.labeled_data, dataset_root)
    labeled_images = read_image_source(labeled_layout.train_source, labeled_layout.root)
    labeled_ids = {path.stem for path in labeled_images}
    ground_truths = tuple(box for image in labeled_images for box in load_yolo_boxes(image))
    candidates = tuple(
        candidate
        for candidate in read_candidates_csv(config.prediction_cache)
        if candidate.image_id in labeled_ids
    )
    config.report_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, variant in config.variants.items():
        summary = audit_candidates(
            variant=variant_name,
            candidates=candidates,
            ground_truths=ground_truths,
            threshold_policy=variant.thresholds.policy(),
            iou_threshold=0.5,
        )
        path = config.report_dir / f"{variant.run_name}_quality_summary.json"
        write_quality_summary(path, summary)
        print(
            "quality_audit="
            f"{variant_name} gt={summary.overall.gt_count} pred={summary.overall.prediction_count} "
            f"tp={summary.overall.matched_tp} fp_rate={summary.overall.fp_rate:.4f} "
            f"fn_rate={summary.overall.fn_rate:.4f}",
            flush=True,
        )


def run_build(config: ExperimentConfig, dataset_root: Path | None, variant_name: str) -> None:
    source_layout = load_layout(config.source_data, dataset_root)
    labeled_layout = load_layout(config.labeled_data, dataset_root)
    variant = config.variants[variant_name]
    request = MixedDatasetRequest(
        output_root=config.output_root / variant.dataset_name,
        variant=variant_name,
        names=source_layout.names,
        train_images=read_image_source(source_layout.train_source, source_layout.root),
        labeled_images=read_image_source(labeled_layout.train_source, labeled_layout.root),
        val_images=read_image_source(source_layout.val_source, source_layout.root),
        candidates=read_candidates_csv(config.prediction_cache),
        threshold_policy=variant.thresholds.policy(),
    )
    summary = write_mixed_dataset(request)
    print(
        f"mixed_dataset={request.output_root} train_images={summary.train_images} "
        f"pseudo_boxes={summary.pseudo_boxes} empty_label_files={summary.empty_label_files}",
        flush=True,
    )


def run_train_eval_export(
    config: ExperimentConfig,
    dataset_root: Path | None,
    variant_name: str,
) -> None:
    variant = config.variants[variant_name]
    dataset_yaml = config.output_root / variant.dataset_name / "data.yaml"
    if not dataset_yaml.exists():
        run_build(config, dataset_root, variant_name)
    train_yolo(config, variant, dataset_yaml)
    evaluate_yolo(config, variant, dataset_yaml)
    export_onnx(config, variant)


def train_yolo(config: ExperimentConfig, variant: VariantConfig, dataset_yaml: Path) -> None:
    from ultralytics import YOLO

    training = config.training
    model = YOLO(training.model)
    model.train(
        data=str(dataset_yaml),
        imgsz=training.image_size,
        epochs=training.epochs,
        batch=training.batch,
        workers=training.workers,
        seed=training.seed,
        device=training.device,
        project=str(config.project),
        name=variant.run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=100,
        close_mosaic=10,
        amp=True,
        plots=False,
    )


def evaluate_yolo(config: ExperimentConfig, variant: VariantConfig, dataset_yaml: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_weight(config, variant)))
    model.val(
        data=str(dataset_yaml),
        imgsz=config.training.image_size,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=config.training.device,
        project=str(config.report_dir / "standard_val"),
        name=variant.run_name,
        exist_ok=True,
    )
    run_area_ap(config, variant, dataset_yaml)


def run_area_ap(config: ExperimentConfig, variant: VariantConfig, dataset_yaml: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_area_ap.py"),
        "--model",
        str(best_weight(config, variant)),
        "--data",
        str(dataset_yaml),
        "--imgsz",
        str(config.training.image_size),
        "--name",
        variant.run_name,
        "--output-dir",
        str(config.report_dir),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
    ]
    subprocess.run(command, check=True)


def export_onnx(config: ExperimentConfig, variant: VariantConfig) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_weight(config, variant)))
    print("clean_yolo_load=OK", flush=True)
    model.export(
        format="onnx",
        imgsz=config.training.image_size,
        opset=12,
        simplify=False,
        dynamic=False,
    )


def best_weight(config: ExperimentConfig, variant: VariantConfig) -> Path:
    return config.project / variant.run_name / "weights" / "best.pt"


def selected_dataset_root(config: ExperimentConfig, override_text: str | None) -> Path | None:
    if override_text is not None:
        return Path(override_text)
    return config.dataset_root_override


def require_variant(config: ExperimentConfig, variant_name: str | None) -> str:
    if variant_name is None:
        raise ValueError("--variant is required for build and train_eval_export phases.")
    if variant_name not in config.variants:
        raise ValueError(f"Unknown variant: {variant_name}")
    return variant_name


def run() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    dataset_root = selected_dataset_root(config, args.dataset_root)
    phase = Phase(args.phase)
    match phase:
        case Phase.ALL:
            run_cache(config, dataset_root, bool(args.rebuild_cache))
            run_audit(config, dataset_root)
            for variant_name in config.variants:
                run_build(config, dataset_root, variant_name)
            for variant_name in config.variants:
                run_train_eval_export(config, dataset_root, variant_name)
        case Phase.CACHE:
            run_cache(config, dataset_root, bool(args.rebuild_cache))
        case Phase.AUDIT:
            run_audit(config, dataset_root)
        case Phase.BUILD:
            run_build(config, dataset_root, require_variant(config, args.variant))
        case Phase.TRAIN_EVAL_EXPORT:
            run_train_eval_export(config, dataset_root, require_variant(config, args.variant))


if __name__ == "__main__":
    run()
