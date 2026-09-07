from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from scripts.evaluate_area_ap import (
    Box,
    compute_metrics,
    load_ground_truths,
    load_val_images,
    write_outputs,
)
from vfm_yolo_distillation.grounding_dino_coco import CLASS_NAMES, coco_summary


PROMPTS: Final = (
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
ALIASES: Final = {
    "pedestrian": 0,
    "person": 1,
    "people": 1,
    "bicycle": 2,
    "bike": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning tricycle": 7,
    "bus": 8,
    "motor": 9,
    "motorcycle": 9,
}


@dataclass(frozen=True, slots=True)
class GroundingDinoSettings:
    model_id: str
    box_threshold: float
    text_threshold: float
    max_det: int
    device: str


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    data: Path
    output_dir: Path
    limit: int
    grounding_dino: GroundingDinoSettings


def prompt_text() -> str:
    return ". ".join(PROMPTS) + "."


def normalize_label(label: str) -> str:
    cleaned = label.lower().replace("-", " ").replace("_", " ").strip(" .")
    words = [word for word in cleaned.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def class_id_from_label(label: str) -> int | None:
    normalized = normalize_label(label)
    if normalized in ALIASES:
        return ALIASES[normalized]
    for alias, class_id in sorted(ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if f" {alias} " in f" {normalized} ":
            return class_id
    return None


def collect_predictions(settings: GroundingDinoSettings, images: list[Path]) -> list[Box]:
    processor, model, torch = load_grounding_dino(settings)
    predictions: list[Box] = []
    for index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = processor(images=image, text=prompt_text(), return_tensors="pt").to(
            settings.device
        )
        with torch.inference_mode():
            outputs = model(**inputs)
        result = post_process(processor, outputs, inputs, image, settings)[0]
        predictions.extend(image_predictions(image_path, result, settings.max_det))
        if index % 25 == 0:
            print(f"processed={index}/{len(images)} predictions={len(predictions)}", flush=True)
    return predictions


def load_grounding_dino(settings: GroundingDinoSettings):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(settings.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(settings.model_id)
    model = model.to(settings.device)
    model.eval()
    return processor, model, torch


def post_process(processor, outputs, inputs, image: Image.Image, settings: GroundingDinoSettings):
    target_sizes = [(image.height, image.width)]
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=settings.box_threshold,
            text_threshold=settings.text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs,
            threshold=settings.box_threshold,
            text_threshold=settings.text_threshold,
            target_sizes=target_sizes,
        )


def image_predictions(image_path: Path, result, max_det: int) -> list[Box]:
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = result.get("text_labels", result.get("labels", []))
    predictions: list[Box] = []
    for xyxy, score, label in zip(boxes, scores, labels, strict=True):
        class_id = class_id_from_label(str(label))
        if class_id is None:
            continue
        left, top, right, bottom = (float(value) for value in xyxy)
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        predictions.append(
            Box(
                image_id=image_path.stem,
                class_id=class_id,
                xyxy=(left, top, right, bottom),
                area=width * height,
                score=float(score),
            )
        )
    return sorted(predictions, key=lambda box: box.score, reverse=True)[:max_det]


def run(settings: RunSettings) -> None:
    images = load_val_images(settings.data)
    if settings.limit > 0:
        images = images[: settings.limit]
    ground_truths = load_ground_truths(images)
    predictions = collect_predictions(settings.grounding_dino, images)
    area_metrics = compute_metrics(settings.name, 0, ground_truths, predictions)
    write_outputs(settings.output_dir, settings.name, area_metrics)
    write_summary(settings, images, ground_truths, predictions)
    write_predictions(settings.output_dir / f"{settings.name}_predictions.csv", predictions)


def write_summary(
    settings: RunSettings,
    images: list[Path],
    ground_truths: list[Box],
    predictions: list[Box],
) -> None:
    area_metrics = compute_metrics(settings.name, 0, ground_truths, predictions)
    payload = {
        "name": settings.name,
        "model_id": settings.grounding_dino.model_id,
        "data": str(settings.data),
        "image_count": len(images),
        "ground_truth_count": len(ground_truths),
        "prediction_count": len(predictions),
        "box_threshold": settings.grounding_dino.box_threshold,
        "text_threshold": settings.grounding_dino.text_threshold,
        "max_det": settings.grounding_dino.max_det,
        "prompt": prompt_text(),
        "coco": asdict(coco_summary(images, ground_truths, predictions)),
        "area_metrics": [asdict(metric) for metric in area_metrics],
    }
    path = settings.output_dir / f"{settings.name}_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)


def write_predictions(path: Path, predictions: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["image_id", "class_id", "class_name", "score", "x1", "y1", "x2", "y2", "area"]
        )
        for box in predictions:
            writer.writerow(
                [
                    box.image_id,
                    box.class_id,
                    CLASS_NAMES[box.class_id],
                    box.score,
                    *box.xyxy,
                    box.area,
                ]
            )
