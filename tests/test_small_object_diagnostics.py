# ruff: noqa: D103, INP001, S101

from pathlib import Path

from scripts.diagnose_small_object_localization import label_path_for_image
from vfm_yolo_distillation.small_object_diagnostics import (
    DetectionBox,
    DiagnosticSettings,
    summarize_small_object_diagnostics,
)

FLOAT_TOLERANCE = 1e-6


def test_summarize_small_object_diagnostics_when_near_prediction_is_poorly_localized() -> None:
    # Given
    settings = DiagnosticSettings(
        small_area_px=1024.0,
        near_expand_ratio=2.0,
        near_min_margin_px=8.0,
        objectness_conf_threshold=0.10,
    )
    ground_truths = [
        DetectionBox("image-a", 0, (10.0, 10.0, 20.0, 20.0), 1.0),
        DetectionBox("image-a", 1, (100.0, 100.0, 180.0, 180.0), 1.0),
    ]
    predictions = [
        DetectionBox("image-a", 0, (14.0, 14.0, 24.0, 24.0), 0.90),
        DetectionBox("image-a", 1, (100.0, 100.0, 180.0, 180.0), 0.95),
    ]

    # When
    summary = summarize_small_object_diagnostics(ground_truths, predictions, settings)

    # Then
    assert summary.small_gt_count == 1
    assert summary.near_prediction_count == 1
    _assert_close(summary.mean_best_near_conf, 0.90)
    _assert_close(summary.recall50, 0.0)
    _assert_close(summary.objectness_without_iou50_rate, 1.0)


def test_summarize_small_object_diagnostics_when_prediction_is_well_localized() -> None:
    # Given
    settings = DiagnosticSettings(
        small_area_px=1024.0,
        near_expand_ratio=2.0,
        near_min_margin_px=8.0,
        objectness_conf_threshold=0.10,
    )
    ground_truths = [DetectionBox("image-a", 0, (10.0, 10.0, 20.0, 20.0), 1.0)]
    predictions = [DetectionBox("image-a", 0, (10.0, 10.0, 20.0, 20.0), 0.80)]

    # When
    summary = summarize_small_object_diagnostics(ground_truths, predictions, settings)

    # Then
    _assert_close(summary.mean_best_iou, 1.0)
    _assert_close(summary.recall50, 1.0)
    _assert_close(summary.recall75, 1.0)
    _assert_close(summary.objectness_without_iou50_rate, 0.0)


def test_label_path_for_image_when_dataset_uses_images_split_layout() -> None:
    # Given
    image_path = Path("/datasets/visdrone/images/val/000001.jpg")

    # When
    label_path = label_path_for_image(image_path)

    # Then
    assert label_path == Path("/datasets/visdrone/labels/val/000001.txt")


def _assert_close(value: float, expected: float) -> None:
    assert abs(value - expected) < FLOAT_TOLERANCE
