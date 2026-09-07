# ruff: noqa: D103, INP001, S101
import torch

from vfm_yolo_distillation.background_contrast_targets import (
    _BackgroundContrastRequest,
    _BackgroundContrastSettings,
    _build_background_contrast_targets,
)
from vfm_yolo_distillation.objectness_targets import (
    AreaThresholds,
    GtAreaMasks,
    GtAreaMaskSettings,
    PeakIgnoreSettings,
    SmallCenterTargetSettings,
    SmallGtWeightMapSettings,
    SmallGtWeights,
    build_gt_area_masks,
    build_peak_ignore_targets,
    build_small_center_targets,
    build_small_gt_weight_map,
)

FLOAT_TOLERANCE = 1e-6


def test_gt_area_masks_when_boxes_have_different_pixel_areas() -> None:
    # Given
    batch_idx = torch.tensor([0, 0, 0])
    bboxes_xywhn = torch.tensor(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.50, 0.50, 0.25, 0.25],
            [0.80, 0.80, 0.60, 0.60],
        ]
    )

    # When
    masks = build_gt_area_masks(
        batch_idx=batch_idx,
        bboxes_xywhn=bboxes_xywhn,
        settings=GtAreaMaskSettings(
            batch_size=1,
            image_size_hw=(128, 128),
            target_hw=(8, 8),
            thresholds=AreaThresholds(small_area_px=256.0, medium_area_px=4096.0),
            box_expand_cells=0,
        ),
    )

    # Then
    assert isinstance(masks, GtAreaMasks)
    assert masks.small.any()
    assert masks.medium.any()
    assert masks.large.any()
    assert torch.equal(masks.any, masks.small | masks.medium | masks.large)


def test_small_gt_weight_map_prioritizes_small_boxes_and_suppresses_false_positives() -> None:
    # Given
    targets = torch.tensor(
        [
            [
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.20, 0.95, 0.25, 0.35],
                    [0.30, 0.40, 0.50, 0.99],
                    [0.10, 0.20, 0.30, 0.40],
                ]
            ]
        ]
    )
    settings = SmallGtWeightMapSettings(
        thresholds=AreaThresholds(small_area_px=1024.0, medium_area_px=9216.0),
        weights=SmallGtWeights(
            background=0.25,
            small=3.0,
            medium=1.0,
            large=0.5,
            false_positive=0.05,
        ),
        false_positive_quantile=0.85,
        box_expand_cells=0,
    )

    # When
    weights = build_small_gt_weight_map(
        targets=targets,
        batch_idx=torch.tensor([0]),
        bboxes_xywhn=torch.tensor([[0.375, 0.375, 0.125, 0.125]]),
        image_size_hw=(64, 64),
        settings=settings,
    )

    # Then
    _assert_close(weights[0, 0, 1, 1], 3.0)
    _assert_close(weights[0, 0, 2, 3], 0.05)
    _assert_close(weights[0, 0, 0, 0], 0.25)


def test_peak_ignore_targets_when_map_has_peak_and_mid_values() -> None:
    # Given
    targets = torch.tensor(
        [
            [
                [
                    [0.05, 0.30, 0.05],
                    [0.30, 0.95, 0.60],
                    [0.05, 0.55, 0.40],
                ]
            ]
        ]
    )

    # When
    labels, weights = build_peak_ignore_targets(
        targets,
        PeakIgnoreSettings(positive_quantile=0.85, negative_quantile=0.45, peak_kernel=3),
    )

    # Then
    _assert_close(labels[0, 0, 1, 1], 1.0)
    _assert_close(weights[0, 0, 1, 1], 1.0)
    _assert_close(weights[0, 0, 0, 0], 1.0)
    _assert_close(labels[0, 0, 0, 0], 0.0)
    _assert_close(weights[0, 0, 1, 2], 0.0)


def test_small_gt_weight_map_when_gt_is_empty() -> None:
    # Given
    targets = torch.tensor([[[[0.90, 0.10], [0.20, 0.30]]]])

    # When
    weights = build_small_gt_weight_map(
        targets=targets,
        batch_idx=torch.empty(0, dtype=torch.long),
        bboxes_xywhn=torch.empty((0, 4)),
        image_size_hw=(64, 64),
        settings=SmallGtWeightMapSettings(
            thresholds=AreaThresholds(small_area_px=1024.0, medium_area_px=9216.0),
            weights=SmallGtWeights(
                background=0.25,
                small=3.0,
                medium=1.0,
                large=0.5,
                false_positive=0.05,
            ),
            false_positive_quantile=0.85,
            box_expand_cells=1,
        ),
    )

    # Then
    _assert_close(weights[0, 0, 0, 0], 0.05)
    _assert_close(weights[0, 0, 1, 1], 0.25)


def test_small_center_targets_focus_on_small_box_centers() -> None:
    # Given
    batch_idx = torch.tensor([0, 0])
    bboxes_xywhn = torch.tensor(
        [
            [0.55, 0.55, 0.10, 0.10],
            [0.20, 0.20, 0.60, 0.60],
        ]
    )

    # When
    targets, weights = build_small_center_targets(
        batch_idx=batch_idx,
        bboxes_xywhn=bboxes_xywhn,
        image_size_hw=(100, 100),
        target_hw=(10, 10),
        batch_size=1,
        settings=SmallCenterTargetSettings(
            small_area_px=200.0,
            radius_cells=1.5,
            sigma_cells=1.0,
            max_centers_per_image=16,
            weight=2.0,
        ),
    )

    # Then
    assert targets.max().item() > 0.85
    _assert_close(weights.max(), 2.0)
    _assert_close(weights[0, 0, 2, 2], 0.0)


def test_small_center_targets_when_no_small_boxes() -> None:
    # Given
    batch_idx = torch.tensor([0])
    bboxes_xywhn = torch.tensor([[0.50, 0.50, 0.80, 0.80]])

    # When
    targets, weights = build_small_center_targets(
        batch_idx=batch_idx,
        bboxes_xywhn=bboxes_xywhn,
        image_size_hw=(100, 100),
        target_hw=(10, 10),
        batch_size=1,
        settings=SmallCenterTargetSettings(
            small_area_px=200.0,
            radius_cells=1.5,
            sigma_cells=1.0,
            max_centers_per_image=16,
            weight=2.0,
        ),
    )

    # Then
    _assert_close(targets.sum(), 0.0)
    _assert_close(weights.sum(), 0.0)


def test_background_contrast_targets_use_small_object_and_nearby_background() -> None:
    # Given
    teacher_targets = torch.zeros((1, 1, 8, 8))
    teacher_targets[0, 0, 3:5, 3:5] = 0.8
    request = _BackgroundContrastRequest(
        teacher_targets=teacher_targets,
        batch_idx=torch.tensor([0]),
        bboxes_xywhn=torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
        image_size_hw=(100, 100),
        settings=_BackgroundContrastSettings(
            small_area_px=500.0,
            ring_context_cells=1,
            positive_weight=2.0,
            background_weight=0.5,
            max_boxes_per_image=16,
        ),
    )

    # When
    labels, weights = _build_background_contrast_targets(request)

    # Then
    _assert_close(labels[0, 0, 3, 3], 0.8)
    _assert_close(weights[0, 0, 3, 3], 2.0)
    _assert_close(labels[0, 0, 2, 3], 0.0)
    _assert_close(weights[0, 0, 2, 3], 0.5)


def _assert_close(value: torch.Tensor, expected: float) -> None:
    assert abs(value.item() - expected) < FLOAT_TOLERANCE
