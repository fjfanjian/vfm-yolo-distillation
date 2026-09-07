from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_area_ap import (
    Box,
    compute_metrics,
    load_ground_truths,
    load_val_images,
    write_outputs,
)
from vfm_yolo_distillation.grounding_dino_coco import CLASS_NAMES, coco_summary


WORLD_PROMPTS: Final = (
    "pedestrian",
    "person",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning tricycle",
    "bus",
    "motorcycle",
)


@dataclass(frozen=True, slots=True)
class YoloWorldSettings:
    model: str
    imgsz: int
    conf: float
    iou: float
    max_det: int
    device: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--model", default="yolov8s-worldv2.pt")
    _ = parser.add_argument("--data", default="configs/datasets/visdrone.yaml")
    _ = parser.add_argument("--name", default="yolov8s_worldv2_visdrone_val")
    _ = parser.add_argument("--output-dir", default="runs/yolo_world/reports")
    _ = parser.add_argument("--imgsz", type=int, default=960)
    _ = parser.add_argument("--conf", type=float, default=0.001)
    _ = parser.add_argument("--iou", type=float, default=0.7)
    _ = parser.add_argument("--max-det", type=int, default=300)
    _ = parser.add_argument("--device", type=int, default=0)
    _ = parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def collect_predictions(settings: YoloWorldSettings, images: list[Path]) -> list[Box]:
    from ultralytics import YOLOWorld

    model = YOLOWorld(settings.model)
    model.set_classes(list(WORLD_PROMPTS))
    predictions: list[Box] = []
    for index, image_path in enumerate(images, start=1):
        stream = model.predict(
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
        for result in stream:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                predictions.append(
                    Box(
                        image_id=image_path.stem,
                        class_id=int(box.cls[0]),
                        xyxy=(x1, y1, x2, y2),
                        area=width * height,
                        score=float(box.conf[0]),
                    )
                )
        if index % 25 == 0:
            print(f"processed={index}/{len(images)} predictions={len(predictions)}", flush=True)
    return predictions


def write_predictions(path: Path, predictions: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "class_id", "class_name", "score", "x1", "y1", "x2", "y2", "area"])
        for box in predictions:
            writer.writerow(
                [box.image_id, box.class_id, CLASS_NAMES[box.class_id], box.score, *box.xyxy, box.area]
            )


def main() -> None:
    args = parse_args()
    settings = YoloWorldSettings(
        model=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )
    images = load_val_images(Path(args.data))
    if args.limit > 0:
        images = images[: args.limit]
    ground_truths = load_ground_truths(images)
    predictions = collect_predictions(settings, images)
    area_metrics = compute_metrics(args.name, settings.imgsz, ground_truths, predictions)
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, args.name, area_metrics)
    summary = {
        "name": args.name,
        "model": settings.model,
        "data": args.data,
        "image_count": len(images),
        "ground_truth_count": len(ground_truths),
        "prediction_count": len(predictions),
        "imgsz": settings.imgsz,
        "conf": settings.conf,
        "iou": settings.iou,
        "max_det": settings.max_det,
        "prompts": list(WORLD_PROMPTS),
        "coco": asdict(coco_summary(images, ground_truths, predictions)),
        "area_metrics": [asdict(metric) for metric in area_metrics],
    }
    summary_path = output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_predictions(output_dir / f"{args.name}_predictions.csv", predictions)
    print(summary_path)


if __name__ == "__main__":
    main()
