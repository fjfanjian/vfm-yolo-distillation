from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PseudoBox:
    image: str
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]
    image_width: int
    image_height: int


@dataclass(frozen=True, slots=True)
class ScoreSettings:
    small_area_px: float
    score_image_size: int
    class_diversity_weight: float
    scale_diversity_weight: float
    box_score_mode: str


@dataclass(frozen=True, slots=True)
class ImageScore:
    image: str
    score: float
    pseudo_boxes: int
    small_boxes: int
    small_classes: int
    scale_bins: int
    mean_small_confidence: float


def budget_count(total: int, budget_ratio: int) -> int:
    if budget_ratio <= 0:
        raise ValueError("budget_ratio must be positive")
    return max(1, round(total * budget_ratio / 100.0))


def normalized_area(box: PseudoBox, score_image_size: int) -> float:
    x1, y1, x2, y2 = box.xyxy
    width = max(0.0, x2 - x1) / max(1, box.image_width) * float(score_image_size)
    height = max(0.0, y2 - y1) / max(1, box.image_height) * float(score_image_size)
    return width * height


def scale_bin(area_px: float, small_area_px: float) -> str:
    if area_px <= small_area_px / 4.0:
        return "tiny"
    if area_px <= small_area_px / 2.0:
        return "small_low"
    return "small_high"


def score_image(image: str, boxes: Sequence[PseudoBox], settings: ScoreSettings) -> ImageScore:
    small_boxes = [
        box
        for box in boxes
        if normalized_area(box, settings.score_image_size) <= settings.small_area_px
    ]
    small_classes = {box.class_id for box in small_boxes}
    scale_bins = {
        scale_bin(normalized_area(box, settings.score_image_size), settings.small_area_px)
        for box in small_boxes
    }
    box_score = _box_score(small_boxes, settings.box_score_mode)
    confidence_mean = _mean(box.confidence for box in small_boxes)
    score = (
        box_score
        + settings.class_diversity_weight * len(small_classes)
        + settings.scale_diversity_weight * len(scale_bins)
    )
    return ImageScore(
        image=image,
        score=score,
        pseudo_boxes=len(boxes),
        small_boxes=len(small_boxes),
        small_classes=len(small_classes),
        scale_bins=len(scale_bins),
        mean_small_confidence=confidence_mean,
    )


def select_top(scores: Sequence[ImageScore], budget: int) -> tuple[ImageScore, ...]:
    selected = sorted(scores, key=lambda item: (item.score, item.image), reverse=True)[:budget]
    return tuple(sorted(selected, key=lambda item: item.image))


def selected_for_visualization(scores: Sequence[ImageScore], count: int) -> tuple[ImageScore, ...]:
    if count <= 0:
        return ()
    return tuple(sorted(scores, key=lambda item: (item.score, item.image), reverse=True)[:count])


def write_split(scores: Sequence[ImageScore], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(item.image for item in scores) + "\n", encoding="utf-8")


def write_image_scores(scores: Sequence[ImageScore], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(scores[0]).keys()) if scores else list(ImageScore.__dataclass_fields__))
        writer.writeheader()
        for score in scores:
            writer.writerow(asdict(score))


def write_pseudo_boxes(boxes: Sequence[PseudoBox], output_path: Path, settings: ScoreSettings) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "normalized_area_px",
        "area_group",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for box in boxes:
            area = normalized_area(box, settings.score_image_size)
            x1, y1, x2, y2 = box.xyxy
            writer.writerow(
                {
                    "image": box.image,
                    "class_id": box.class_id,
                    "confidence": f"{box.confidence:.6f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "normalized_area_px": f"{area:.4f}",
                    "area_group": "small" if area <= settings.small_area_px else "other",
                }
            )


def write_summary(output_path: Path, all_scores: Sequence[ImageScore], selected: Sequence[ImageScore], settings: ScoreSettings) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "pseudo_small_object_score_topk",
        "images": len(all_scores),
        "selected_images": len(selected),
        "small_area_px": settings.small_area_px,
        "score_image_size": settings.score_image_size,
        "box_score_mode": settings.box_score_mode,
        "class_diversity_weight": settings.class_diversity_weight,
        "scale_diversity_weight": settings.scale_diversity_weight,
        "mean_score_all": _mean(score.score for score in all_scores),
        "mean_score_selected": _mean(score.score for score in selected),
        "selected_with_small_boxes": sum(1 for score in selected if score.small_boxes > 0),
        "selected_preview": [score.image for score in selected[:10]],
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _box_score(boxes: Sequence[PseudoBox], mode: str) -> float:
    match mode:
        case "count":
            return float(len(boxes))
        case "confidence":
            return sum(box.confidence for box in boxes)
        case _:
            raise ValueError(f"unsupported box_score_mode: {mode}")


def _mean(values: Iterable[float]) -> float:
    numbers = tuple(float(value) for value in values)
    return sum(numbers) / len(numbers) if numbers else 0.0
