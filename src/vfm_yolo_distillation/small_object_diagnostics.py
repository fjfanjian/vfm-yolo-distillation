"""Small-object localization diagnostics for YOLO predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectionBox:
    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    score: float

    @property
    def area(self) -> float:
        width = max(0.0, self.xyxy[2] - self.xyxy[0])
        height = max(0.0, self.xyxy[3] - self.xyxy[1])
        return width * height


@dataclass(frozen=True, slots=True)
class DiagnosticSettings:
    small_area_px: float
    near_expand_ratio: float
    near_min_margin_px: float
    objectness_conf_threshold: float


@dataclass(frozen=True, slots=True)
class DiagnosticSummary:
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


def summarize_small_object_diagnostics(
    ground_truths: list[DetectionBox],
    predictions: list[DetectionBox],
    settings: DiagnosticSettings,
) -> DiagnosticSummary:
    small_truths = [box for box in ground_truths if box.area <= settings.small_area_px]
    small_predictions = [box for box in predictions if box.area <= settings.small_area_px]
    if not small_truths:
        return DiagnosticSummary(
            small_gt_count=0,
            small_prediction_count=len(small_predictions),
            near_prediction_count=0,
            near_predictions_per_gt=0.0,
            mean_best_near_conf=0.0,
            mean_best_iou=0.0,
            recall50=0.0,
            recall75=0.0,
            near_iou50_rate=0.0,
            near_iou75_rate=0.0,
            near_poor_iou_rate=0.0,
            objectness_without_iou50_rate=0.0,
        )

    best_near_conf: list[float] = []
    best_ious: list[float] = []
    objectness_without_iou50 = 0
    predictions_by_key = _index_by_image_and_class(predictions)
    truths_by_key = _index_by_image_and_class(small_truths)
    for truth in small_truths:
        candidates = predictions_by_key.get((truth.image_id, truth.class_id), [])
        near_candidates = [box for box in candidates if is_near_truth_center(box, truth, settings)]
        near_conf = max((box.score for box in near_candidates), default=0.0)
        best_iou = max((box_iou(box, truth) for box in candidates), default=0.0)
        best_near_conf.append(near_conf)
        best_ious.append(best_iou)
        if near_conf >= settings.objectness_conf_threshold and best_iou < 0.50:
            objectness_without_iou50 += 1

    near_predictions = _near_predictions(truths_by_key, predictions, settings)
    near_prediction_ious = [
        max(
            (
                box_iou(prediction, truth)
                for truth in truths_by_key.get((prediction.image_id, prediction.class_id), [])
            ),
            default=0.0,
        )
        for prediction in near_predictions
    ]
    small_gt_count = len(small_truths)
    near_prediction_count = len(near_predictions)
    return DiagnosticSummary(
        small_gt_count=small_gt_count,
        small_prediction_count=len(small_predictions),
        near_prediction_count=near_prediction_count,
        near_predictions_per_gt=near_prediction_count / small_gt_count,
        mean_best_near_conf=_mean(best_near_conf),
        mean_best_iou=_mean(best_ious),
        recall50=_rate_at_least(best_ious, 0.50),
        recall75=_rate_at_least(best_ious, 0.75),
        near_iou50_rate=_rate_at_least(near_prediction_ious, 0.50),
        near_iou75_rate=_rate_at_least(near_prediction_ious, 0.75),
        near_poor_iou_rate=_rate_below(near_prediction_ious, 0.30),
        objectness_without_iou50_rate=objectness_without_iou50 / small_gt_count,
    )


def box_iou(first: DetectionBox, second: DetectionBox) -> float:
    left = max(first.xyxy[0], second.xyxy[0])
    top = max(first.xyxy[1], second.xyxy[1])
    right = min(first.xyxy[2], second.xyxy[2])
    bottom = min(first.xyxy[3], second.xyxy[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def is_near_truth_center(
    prediction: DetectionBox,
    truth: DetectionBox,
    settings: DiagnosticSettings,
) -> bool:
    if not _same_image_and_class(prediction, truth):
        return False
    center_x = (prediction.xyxy[0] + prediction.xyxy[2]) * 0.5
    center_y = (prediction.xyxy[1] + prediction.xyxy[3]) * 0.5
    width = truth.xyxy[2] - truth.xyxy[0]
    height = truth.xyxy[3] - truth.xyxy[1]
    margin_x = max(width * (settings.near_expand_ratio - 1.0) * 0.5, settings.near_min_margin_px)
    margin_y = max(height * (settings.near_expand_ratio - 1.0) * 0.5, settings.near_min_margin_px)
    return (
        truth.xyxy[0] - margin_x
        <= center_x
        <= truth.xyxy[2] + margin_x
        and truth.xyxy[1] - margin_y
        <= center_y
        <= truth.xyxy[3] + margin_y
    )


def _index_by_image_and_class(
    boxes: list[DetectionBox],
) -> dict[tuple[str, int], list[DetectionBox]]:
    indexed: dict[tuple[str, int], list[DetectionBox]] = {}
    for box in boxes:
        indexed.setdefault((box.image_id, box.class_id), []).append(box)
    return indexed


def _near_predictions(
    truths_by_key: dict[tuple[str, int], list[DetectionBox]],
    predictions: list[DetectionBox],
    settings: DiagnosticSettings,
) -> list[DetectionBox]:
    near_predictions: list[DetectionBox] = []
    for prediction in predictions:
        if any(
            is_near_truth_center(prediction, truth, settings)
            for truth in truths_by_key.get((prediction.image_id, prediction.class_id), [])
        ):
            near_predictions.append(prediction)
    return near_predictions


def _same_image_and_class(first: DetectionBox, second: DetectionBox) -> bool:
    return first.image_id == second.image_id and first.class_id == second.class_id


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rate_at_least(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value >= threshold) / len(values)


def _rate_below(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value < threshold) / len(values)
