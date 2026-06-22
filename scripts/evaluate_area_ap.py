#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10.0",
#   "pyyaml>=6.0",
#   "ultralytics>=8.0",
# ]
# ///
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


IOU_THRESHOLDS: Final = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
AREA_RANGES: Final = {
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, float("inf")),
}


@dataclass(frozen=True, slots=True)
class Box:
    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    area: float
    score: float = 1.0


@dataclass(frozen=True, slots=True)
class AreaMetrics:
    name: str
    imgsz: int
    area: str
    gt_count: int
    prediction_count: int
    ap50: float | None
    ap50_95: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="configs/datasets/visdrone.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default="runs/baselines/reports")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def load_val_images(data_yaml: Path) -> list[Path]:
    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    root = Path(data["path"])
    val_dir = root / data["val"]
    return sorted(path for path in val_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def yolo_label_to_box(line: str, image_id: str, width: int, height: int) -> Box:
    class_text, x_text, y_text, w_text, h_text = line.split()[:5]
    class_id = int(class_text)
    center_x = float(x_text) * width
    center_y = float(y_text) * height
    box_w = float(w_text) * width
    box_h = float(h_text) * height
    left = center_x - box_w / 2.0
    top = center_y - box_h / 2.0
    return Box(
        image_id=image_id,
        class_id=class_id,
        xyxy=(left, top, left + box_w, top + box_h),
        area=box_w * box_h,
    )


def load_ground_truths(images: list[Path]) -> list[Box]:
    ground_truths: list[Box] = []
    for image_path in images:
        label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
        with Image.open(image_path) as image:
            width, height = image.size
        if not label_path.exists():
            continue
        lines = label_path.read_text(encoding="utf-8").splitlines()
        ground_truths.extend(
            yolo_label_to_box(line, image_path.stem, width, height)
            for line in lines
            if line.strip()
        )
    return ground_truths


def collect_predictions(model_path: Path, images: list[Path], imgsz: int, conf: float, iou: float, max_det: int) -> list[Box]:
    model = YOLO(str(model_path))
    predictions: list[Box] = []
    stream = model.predict(
        source=[str(path) for path in images],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=0,
        verbose=False,
        stream=True,
    )
    for result in stream:
        image_id = Path(result.path).stem
        for box in result.boxes:
            xyxy_values = tuple(float(value) for value in box.xyxy[0].tolist())
            width = max(0.0, xyxy_values[2] - xyxy_values[0])
            height = max(0.0, xyxy_values[3] - xyxy_values[1])
            predictions.append(
                Box(
                    image_id=image_id,
                    class_id=int(box.cls[0]),
                    xyxy=xyxy_values,
                    area=width * height,
                    score=float(box.conf[0]),
                )
            )
    return predictions


def iou(box_a: Box, box_b: Box) -> float:
    left = max(box_a.xyxy[0], box_b.xyxy[0])
    top = max(box_a.xyxy[1], box_b.xyxy[1])
    right = min(box_a.xyxy[2], box_b.xyxy[2])
    bottom = min(box_a.xyxy[3], box_b.xyxy[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = box_a.area + box_b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def is_in_area(box: Box, area_name: str) -> bool:
    low, high = AREA_RANGES[area_name]
    return low <= box.area < high


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    total = 0.0
    for recall_threshold in (index / 100.0 for index in range(101)):
        candidates = [precision for recall, precision in zip(recalls, precisions, strict=True) if recall >= recall_threshold]
        total += max(candidates) if candidates else 0.0
    return total / 101.0


def ap_at_threshold(ground_truths: list[Box], predictions: list[Box], area_name: str, iou_threshold: float) -> float | None:
    target_gt = [box for box in ground_truths if is_in_area(box, area_name)]
    ignored_gt = [box for box in ground_truths if not is_in_area(box, area_name)]
    if not target_gt:
        return None
    target_index = index_boxes(target_gt)
    ignored_index = index_boxes(ignored_gt)
    matched: set[tuple[str, int, int]] = set()
    true_positive: list[int] = []
    false_positive: list[int] = []
    sorted_predictions = sorted(predictions, key=lambda box: box.score, reverse=True)
    for prediction in sorted_predictions:
        key = (prediction.image_id, prediction.class_id)
        best_index = -1
        best_iou = 0.0
        for index, truth in enumerate(target_index.get(key, [])):
            match_key = (prediction.image_id, prediction.class_id, index)
            if match_key in matched:
                continue
            overlap = iou(prediction, truth)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched.add((prediction.image_id, prediction.class_id, best_index))
            true_positive.append(1)
            false_positive.append(0)
            continue
        if matches_ignored(prediction, ignored_index.get(key, []), iou_threshold):
            continue
        true_positive.append(0)
        false_positive.append(1)
    return precision_recall_ap(true_positive, false_positive, len(target_gt))


def index_boxes(boxes: list[Box]) -> dict[tuple[str, int], list[Box]]:
    indexed: dict[tuple[str, int], list[Box]] = {}
    for box in boxes:
        indexed.setdefault((box.image_id, box.class_id), []).append(box)
    return indexed


def matches_ignored(prediction: Box, ignored_gt: list[Box], iou_threshold: float) -> bool:
    return any(
        iou(prediction, truth) >= iou_threshold
        for truth in ignored_gt
    )


def precision_recall_ap(true_positive: list[int], false_positive: list[int], gt_count: int) -> float:
    tp_total = 0
    fp_total = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp_value, fp_value in zip(true_positive, false_positive, strict=True):
        tp_total += tp_value
        fp_total += fp_value
        recalls.append(tp_total / gt_count)
        precisions.append(tp_total / max(tp_total + fp_total, 1))
    return average_precision(recalls, precisions)


def compute_metrics(name: str, imgsz: int, ground_truths: list[Box], predictions: list[Box]) -> list[AreaMetrics]:
    metrics: list[AreaMetrics] = []
    for area_name in AREA_RANGES:
        aps = [ap_at_threshold(ground_truths, predictions, area_name, threshold) for threshold in IOU_THRESHOLDS]
        valid_aps = [value for value in aps if value is not None]
        metrics.append(
            AreaMetrics(
                name=name,
                imgsz=imgsz,
                area=area_name,
                gt_count=sum(1 for box in ground_truths if is_in_area(box, area_name)),
                prediction_count=sum(1 for box in predictions if is_in_area(box, area_name)),
                ap50=aps[0],
                ap50_95=sum(valid_aps) / len(valid_aps) if valid_aps else None,
            )
        )
    return metrics


def write_outputs(output_dir: Path, name: str, metrics: list[AreaMetrics]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}_area_ap.csv"
    json_path = output_dir / f"{name}_area_ap.json"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in metrics)
    json_path.write_text(json.dumps([asdict(metric) for metric in metrics], indent=2), encoding="utf-8")
    print(csv_path)
    print(json_path)


def main() -> None:
    args = parse_args()
    images = load_val_images(Path(args.data))
    ground_truths = load_ground_truths(images)
    predictions = collect_predictions(Path(args.model), images, args.imgsz, args.conf, args.iou, args.max_det)
    metrics = compute_metrics(args.name, args.imgsz, ground_truths, predictions)
    write_outputs(Path(args.output_dir), args.name, metrics)


if __name__ == "__main__":
    main()
