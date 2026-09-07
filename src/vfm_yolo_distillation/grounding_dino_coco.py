from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scripts.evaluate_area_ap import Box


CLASS_NAMES: Final = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


@dataclass(frozen=True, slots=True)
class CocoSummary:
    ap50_95: float
    ap50: float
    ap75: float
    small_ap: float
    medium_ap: float
    large_ap: float
    ar100: float
    small_ar: float
    medium_ar: float
    large_ar: float


def coco_summary(
    images: list[Path], ground_truths: list[Box], predictions: list[Box]
) -> CocoSummary:
    if not predictions:
        return CocoSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = {path.stem: index for index, path in enumerate(images, start=1)}
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "VisDrone validation"},
        "images": [{"id": image_ids[path.stem], "file_name": path.name} for path in images],
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)],
        "annotations": gt_annotations(image_ids, ground_truths),
    }
    coco_gt.createIndex()
    evaluator = COCOeval(coco_gt, coco_gt.loadRes(detections(image_ids, predictions)), "bbox")
    evaluator.params.imgIds = list(image_ids.values())
    evaluator.params.catIds = list(range(1, len(CLASS_NAMES) + 1))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return CocoSummary(
        ap50_95=float(stats[0]),
        ap50=float(stats[1]),
        ap75=float(stats[2]),
        small_ap=float(stats[3]),
        medium_ap=float(stats[4]),
        large_ap=float(stats[5]),
        ar100=float(stats[8]),
        small_ar=float(stats[9]),
        medium_ar=float(stats[10]),
        large_ar=float(stats[11]),
    )


def gt_annotations(image_ids: dict[str, int], boxes: list[Box]) -> list[dict]:
    return [
        {
            "id": index,
            "image_id": image_ids[box.image_id],
            "category_id": box.class_id + 1,
            "bbox": [
                box.xyxy[0],
                box.xyxy[1],
                box.xyxy[2] - box.xyxy[0],
                box.xyxy[3] - box.xyxy[1],
            ],
            "area": box.area,
            "iscrowd": 0,
        }
        for index, box in enumerate(boxes, start=1)
    ]


def detections(image_ids: dict[str, int], boxes: list[Box]) -> list[dict]:
    return [
        {
            "image_id": image_ids[box.image_id],
            "category_id": box.class_id + 1,
            "bbox": [
                box.xyxy[0],
                box.xyxy[1],
                box.xyxy[2] - box.xyxy[0],
                box.xyxy[3] - box.xyxy[1],
            ],
            "score": box.score,
        }
        for box in boxes
    ]
