#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow>=10.0",
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
from typing import assert_never

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfm_yolo_distillation.grounding_dino_eval import (  # noqa: E402
    GroundingDinoSettings,
    collect_predictions,
)
from vfm_yolo_distillation.pseudo_label_dataset import (  # noqa: E402
    PseudoDatasetSummary,
    link_images,
    link_labels,
    list_images,
    load_source_spec,
    write_dataset_yaml,
    write_pseudo_labels,
    write_summary,
)


class Phase(StrEnum):
    ALL = "all"
    BUILD = "build"
    TRAIN = "train"
    EVALUATE = "evaluate"
    EXPORT = "export"


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    model: str
    epochs: int
    batch: int
    image_size: int
    workers: int
    seed: int
    device: str


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    phase: Phase
    source_data: Path
    eval_data: Path
    pseudo_root: Path
    project: Path
    name: str
    report_dir: Path
    train_limit: int
    rebuild: bool
    grounding_dino: GroundingDinoSettings
    training: TrainingSettings


def parse_phase(text: str) -> Phase:
    try:
        return Phase(text)
    except ValueError as error:
        choices = ", ".join(item.value for item in Phase)
        message = f"phase must be one of: {choices}"
        raise argparse.ArgumentTypeError(message) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=parse_phase, default=Phase.ALL)
    parser.add_argument("--source-data", default="configs/datasets/visdrone.yaml")
    parser.add_argument("--eval-data", default="configs/datasets/visdrone.yaml")
    parser.add_argument(
        "--pseudo-root",
        default="/home/fj/datasets/visdrone/pseudo/grounding_dino_base_thr005_text020_fulltrain",
    )
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--grounding-device", default="cuda")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--yolo-model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="0")
    parser.add_argument("--project", default="runs/grounding_dino_pseudolabel")
    parser.add_argument("--name", default="yolo26n_visdrone_groundingdino_fullpseudo_seed42")
    parser.add_argument("--report-dir", default="runs/grounding_dino_pseudolabel/reports")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> ExperimentSettings:
    return ExperimentSettings(
        phase=args.phase,
        source_data=Path(args.source_data),
        eval_data=Path(args.eval_data),
        pseudo_root=Path(args.pseudo_root),
        project=Path(args.project),
        name=str(args.name),
        report_dir=Path(args.report_dir),
        train_limit=int(args.train_limit),
        rebuild=bool(args.rebuild),
        grounding_dino=GroundingDinoSettings(
            model_id=str(args.model_id),
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            max_det=int(args.max_det),
            device=str(args.grounding_device),
        ),
        training=TrainingSettings(
            model=str(args.yolo_model),
            epochs=int(args.epochs),
            batch=int(args.batch),
            image_size=int(args.imgsz),
            workers=int(args.workers),
            seed=int(args.seed),
            device=str(args.train_device),
        ),
    )


def build_dataset(settings: ExperimentSettings) -> None:
    spec = load_source_spec(settings.source_data, settings.pseudo_root)
    data_yaml = spec.output_root / "data.yaml"
    summary_path = spec.output_root / "pseudo_summary.json"
    if summary_path.exists() and data_yaml.exists() and not settings.rebuild:
        print(f"pseudo_dataset=exists summary={summary_path} data={data_yaml}", flush=True)
        return
    images = list_images(spec.train_images)
    if settings.train_limit > 0:
        images = images[: settings.train_limit]
    print(f"pseudo_build_images={len(images)}", flush=True)
    predictions = collect_predictions(settings.grounding_dino, list(images))
    link_images(images, spec.output_root / "images" / "train")
    pseudo_boxes, empty_labels = write_pseudo_labels(spec, images, predictions)
    val_images = list_images(spec.val_images)
    link_images(val_images, spec.output_root / "images" / "val")
    val_label_count = link_labels(spec.val_labels, spec.output_root / "labels" / "val")
    write_dataset_yaml(spec)
    summary = PseudoDatasetSummary(
        output_root=str(spec.output_root),
        train_images=len(images),
        train_label_files=len(images),
        pseudo_boxes=pseudo_boxes,
        empty_label_files=empty_labels,
        val_images=len(val_images),
        val_label_files=val_label_count,
    )
    write_summary(spec, summary)
    print(f"pseudo_boxes={pseudo_boxes} empty_label_files={empty_labels}", flush=True)


def train_yolo(settings: ExperimentSettings) -> None:
    from ultralytics import YOLO

    data_yaml = settings.pseudo_root / "data.yaml"
    model = YOLO(settings.training.model)
    model.train(
        data=str(data_yaml),
        imgsz=settings.training.image_size,
        epochs=settings.training.epochs,
        batch=settings.training.batch,
        workers=settings.training.workers,
        seed=settings.training.seed,
        device=settings.training.device,
        project=str(settings.project),
        name=settings.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=100,
        close_mosaic=10,
        amp=True,
        plots=False,
    )


def best_weight(settings: ExperimentSettings) -> Path:
    return settings.project / settings.name / "weights" / "best.pt"


def evaluate_yolo(settings: ExperimentSettings) -> None:
    from ultralytics import YOLO

    model_path = best_weight(settings)
    model = YOLO(str(model_path))
    model.val(
        data=str(settings.eval_data),
        imgsz=settings.training.image_size,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=settings.training.device,
        project=str(settings.report_dir / "standard_val"),
        name=settings.name,
        exist_ok=True,
    )
    run_area_ap(settings)


def run_area_ap(settings: ExperimentSettings) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_area_ap.py"),
        "--model",
        str(best_weight(settings)),
        "--data",
        str(settings.eval_data),
        "--imgsz",
        str(settings.training.image_size),
        "--name",
        settings.name,
        "--output-dir",
        str(settings.report_dir),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
    ]
    subprocess.run(command, check=True)


def export_onnx(settings: ExperimentSettings) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_weight(settings)))
    print("clean_yolo_load=OK", flush=True)
    model.export(
        format="onnx",
        imgsz=settings.training.image_size,
        opset=12,
        simplify=False,
        dynamic=False,
    )


def run(settings: ExperimentSettings) -> None:
    match settings.phase:
        case Phase.ALL:
            build_dataset(settings)
            train_yolo(settings)
            evaluate_yolo(settings)
            export_onnx(settings)
        case Phase.BUILD:
            build_dataset(settings)
        case Phase.TRAIN:
            train_yolo(settings)
        case Phase.EVALUATE:
            evaluate_yolo(settings)
        case Phase.EXPORT:
            export_onnx(settings)
        case unreachable:
            assert_never(unreachable)


def main() -> None:
    run(settings_from_args(parse_args()))


if __name__ == "__main__":
    main()
