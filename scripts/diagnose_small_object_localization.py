#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10.0",
#   "pyyaml>=6.0",
#   "ultralytics>=8.0",
# ]
# ///
# --- How to run ---
# uv run scripts/diagnose_small_object_localization.py \
#   --model baseline=/path/to/baseline.pt \
#   --model smallcenter=/path/to/smallcenter.pt
# ------------------

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import yaml
from PIL import Image
from ultralytics import YOLO

from vfm_yolo_distillation.small_object_diagnostics import (
    DetectionBox,
    DiagnosticSettings,
    DiagnosticSummary,
    summarize_small_object_diagnostics,
)

IMAGE_SUFFIXES: Final = {".jpg", ".jpeg", ".png"}


class InvalidDataConfigError(RuntimeError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Invalid dataset config: {path}")


class InvalidModelSpecError(RuntimeError):
    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__(f"Model spec must be name=path, got: {value}")


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    root: Path
    val_images: Path
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class PredictSettings:
    imgsz: int
    conf: float
    iou: float
    max_det: int
    device: str


@dataclass(frozen=True, slots=True)
class DiagnosticRow:
    model: str
    class_name: str
    small_gt_count: int
    small_prediction_count: int
    near_prediction_count: int
    near_predictions_per_gt: float
    mean_best_near_conf: float
    mean_best_iou: float
    recall50: float
    recall75: float
    near_iou50_rate: float
    near_iou75_rate: float
    near_poor_iou_rate: float
    objectness_without_iou50_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/datasets/visdrone.yaml")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--small-area-px", type=float, default=1024.0)
    parser.add_argument("--near-expand-ratio", type=float, default=2.0)
    parser.add_argument("--near-min-margin-px", type=float, default=8.0)
    parser.add_argument("--objectness-conf-threshold", type=float, default=0.10)
    parser.add_argument("--output-dir", default="runs/reports/diagnostics")
    parser.add_argument("--name", default="small_object_localization_diagnostics")
    return parser.parse_args()


def load_dataset_info(data_yaml: Path) -> DatasetInfo:
    with data_yaml.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    match raw:
        case {"path": str(root), "val": str(val), "names": list(names)}:
            return DatasetInfo(
                root=Path(root),
                val_images=Path(root) / val,
                class_names=tuple(str(name) for name in names),
            )
        case _:
            raise InvalidDataConfigError(data_yaml)


def parse_model_specs(values: list[str]) -> tuple[ModelSpec, ...]:
    return tuple(parse_model_spec(value) for value in values)


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        raise InvalidModelSpecError(value)
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise InvalidModelSpecError(value)
    return ModelSpec(name=name, path=Path(path))


def load_val_images(dataset: DatasetInfo) -> list[Path]:
    return sorted(
        path
        for path in dataset.val_images.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_ground_truths(images: list[Path]) -> list[DetectionBox]:
    boxes: list[DetectionBox] = []
    for image_path in images:
        label_path = label_path_for_image(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if not label_path.exists():
            continue
        boxes.extend(
            yolo_label_to_box(line, image_path.stem, width, height)
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return boxes


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        image_dir_index = parts.index("images")
        parts[image_dir_index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def yolo_label_to_box(line: str, image_id: str, width: int, height: int) -> DetectionBox:
    class_text, x_text, y_text, w_text, h_text = line.split()[:5]
    center_x = float(x_text) * width
    center_y = float(y_text) * height
    box_w = float(w_text) * width
    box_h = float(h_text) * height
    left = center_x - box_w * 0.5
    top = center_y - box_h * 0.5
    return DetectionBox(
        image_id=image_id,
        class_id=int(class_text),
        xyxy=(left, top, left + box_w, top + box_h),
        score=1.0,
    )


def collect_predictions(
    model_spec: ModelSpec,
    images: list[Path],
    settings: PredictSettings,
) -> list[DetectionBox]:
    model = YOLO(str(model_spec.path))
    boxes: list[DetectionBox] = []
    for image_path in images:
        results = model.predict(
            source=str(image_path),
            imgsz=settings.imgsz,
            conf=settings.conf,
            iou=settings.iou,
            max_det=settings.max_det,
            device=settings.device,
            batch=1,
            verbose=False,
            stream=True,
        )
        for result in results:
            image_id = Path(result.path).stem
            for prediction in result.boxes:
                xyxy = tuple(float(value) for value in prediction.xyxy[0].tolist())
                boxes.append(
                    DetectionBox(
                        image_id=image_id,
                        class_id=int(prediction.cls[0]),
                        xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        score=float(prediction.conf[0]),
                    )
                )
    return boxes


def build_rows(
    model_name: str,
    class_names: tuple[str, ...],
    ground_truths: list[DetectionBox],
    predictions: list[DetectionBox],
    settings: DiagnosticSettings,
) -> list[DiagnosticRow]:
    rows = [
        row_from_summary(
            model_name,
            "all_small",
            summarize_small_object_diagnostics(ground_truths, predictions, settings),
        )
    ]
    for class_id, class_name in enumerate(class_names):
        class_truths = [box for box in ground_truths if box.class_id == class_id]
        class_predictions = [box for box in predictions if box.class_id == class_id]
        rows.append(
            row_from_summary(
                model_name,
                class_name,
                summarize_small_object_diagnostics(class_truths, class_predictions, settings),
            )
        )
    return rows


def row_from_summary(
    model_name: str,
    class_name: str,
    summary: DiagnosticSummary,
) -> DiagnosticRow:
    return DiagnosticRow(model=model_name, class_name=class_name, **asdict(summary))


def write_outputs(output_dir: Path, name: str, rows: list[DiagnosticRow]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(csv_path)
    print(json_path)


def run() -> int:
    args = parse_args()
    dataset = load_dataset_info(Path(args.data))
    images = load_val_images(dataset)
    ground_truths = load_ground_truths(images)
    predict_settings = PredictSettings(
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )
    diagnostic_settings = DiagnosticSettings(
        small_area_px=args.small_area_px,
        near_expand_ratio=args.near_expand_ratio,
        near_min_margin_px=args.near_min_margin_px,
        objectness_conf_threshold=args.objectness_conf_threshold,
    )
    rows: list[DiagnosticRow] = []
    for model_spec in parse_model_specs(args.model):
        predictions = collect_predictions(model_spec, images, predict_settings)
        rows.extend(
            build_rows(
                model_spec.name,
                dataset.class_names,
                ground_truths,
                predictions,
                diagnostic_settings,
            )
        )
    write_outputs(Path(args.output_dir), args.name, rows)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
