from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class _BackgroundContrastSettings:
    small_area_px: float
    ring_context_cells: int
    positive_weight: float
    background_weight: float
    max_boxes_per_image: int


@dataclass(frozen=True, slots=True)
class _BackgroundContrastRequest:
    teacher_targets: torch.Tensor
    batch_idx: torch.Tensor
    bboxes_xywhn: torch.Tensor
    image_size_hw: tuple[int, int]
    settings: _BackgroundContrastSettings


@dataclass(frozen=True, slots=True)
class _ObjectRegionWrite:
    labels: torch.Tensor
    weights: torch.Tensor
    teacher_targets: torch.Tensor
    image_index: int
    bounds: tuple[int, int, int, int]
    settings: _BackgroundContrastSettings


@dataclass(frozen=True, slots=True)
class _RingRegionWrite:
    weights: torch.Tensor
    any_gt: torch.Tensor
    image_index: int
    object_bounds: tuple[int, int, int, int]
    ring_bounds: tuple[int, int, int, int]
    settings: _BackgroundContrastSettings


def _build_background_contrast_targets(
    request: _BackgroundContrastRequest,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_targets = request.teacher_targets
    labels = torch.zeros_like(teacher_targets)
    weights = torch.zeros_like(teacher_targets)
    if request.bboxes_xywhn.numel() == 0:
        return labels, weights

    target_h, target_w = teacher_targets.shape[-2:]
    image_h, image_w = request.image_size_hw
    any_gt = _build_any_gt_mask(
        batch_size=teacher_targets.shape[0],
        target_hw=(target_h, target_w),
        batch_idx=request.batch_idx,
        bboxes_xywhn=request.bboxes_xywhn,
    )
    boxes_per_image = [0 for _ in range(teacher_targets.shape[0])]
    for index in range(request.bboxes_xywhn.shape[0]):
        image_index = int(request.batch_idx[index].item())
        if image_index < 0 or image_index >= teacher_targets.shape[0]:
            continue
        if boxes_per_image[image_index] >= request.settings.max_boxes_per_image:
            continue
        box = request.bboxes_xywhn[index]
        area_px = float((box[2] * image_w * box[3] * image_h).item())
        if area_px > request.settings.small_area_px:
            continue
        object_bounds = _box_bounds(box, target_hw=(target_h, target_w), expand_cells=0)
        ring_bounds = _box_bounds(
            box,
            target_hw=(target_h, target_w),
            expand_cells=request.settings.ring_context_cells,
        )
        _apply_object_region(
            _ObjectRegionWrite(
                labels=labels,
                weights=weights,
                teacher_targets=teacher_targets,
                image_index=image_index,
                bounds=object_bounds,
                settings=request.settings,
            ),
        )
        _apply_background_ring(
            _RingRegionWrite(
                weights=weights,
                any_gt=any_gt,
                image_index=image_index,
                object_bounds=object_bounds,
                ring_bounds=ring_bounds,
                settings=request.settings,
            ),
        )
        boxes_per_image[image_index] += 1
    return labels, weights


def _build_any_gt_mask(
    batch_size: int,
    target_hw: tuple[int, int],
    batch_idx: torch.Tensor,
    bboxes_xywhn: torch.Tensor,
) -> torch.Tensor:
    target_h, target_w = target_hw
    mask = torch.zeros(
        (batch_size, 1, target_h, target_w),
        dtype=torch.bool,
        device=bboxes_xywhn.device,
    )
    for index in range(bboxes_xywhn.shape[0]):
        image_index = int(batch_idx[index].item())
        if image_index < 0 or image_index >= batch_size:
            continue
        left, top, right, bottom = _box_bounds(
            bboxes_xywhn[index],
            target_hw=target_hw,
            expand_cells=0,
        )
        mask[image_index, 0, top:bottom, left:right] = True
    return mask


def _box_bounds(
    bbox_xywhn: torch.Tensor,
    target_hw: tuple[int, int],
    expand_cells: int,
) -> tuple[int, int, int, int]:
    target_h, target_w = target_hw
    x_center, y_center, box_w, box_h = bbox_xywhn
    left = int(torch.floor((x_center - box_w * 0.5) * target_w).item())
    right = int(torch.ceil((x_center + box_w * 0.5) * target_w).item())
    top = int(torch.floor((y_center - box_h * 0.5) * target_h).item())
    bottom = int(torch.ceil((y_center + box_h * 0.5) * target_h).item())
    return (
        max(0, left - expand_cells),
        max(0, top - expand_cells),
        min(target_w, max(left + 1, right + expand_cells)),
        min(target_h, max(top + 1, bottom + expand_cells)),
    )


def _apply_object_region(write: _ObjectRegionWrite) -> None:
    left, top, right, bottom = write.bounds
    image_index = write.image_index
    label_region = write.labels[image_index, 0, top:bottom, left:right]
    teacher_region = write.teacher_targets[image_index, 0, top:bottom, left:right]
    write.labels[image_index, 0, top:bottom, left:right] = torch.maximum(
        label_region,
        teacher_region,
    )
    weight_region = write.weights[image_index, 0, top:bottom, left:right]
    write.weights[image_index, 0, top:bottom, left:right] = torch.maximum(
        weight_region,
        weight_region.new_tensor(write.settings.positive_weight),
    )


def _apply_background_ring(write: _RingRegionWrite) -> None:
    object_left, object_top, object_right, object_bottom = write.object_bounds
    ring_left, ring_top, ring_right, ring_bottom = write.ring_bounds
    ring_mask = torch.ones(
        (ring_bottom - ring_top, ring_right - ring_left),
        dtype=torch.bool,
        device=write.weights.device,
    )
    inner_left = object_left - ring_left
    inner_right = object_right - ring_left
    inner_top = object_top - ring_top
    inner_bottom = object_bottom - ring_top
    ring_mask[inner_top:inner_bottom, inner_left:inner_right] = False
    gt_region = write.any_gt[
        write.image_index,
        0,
        ring_top:ring_bottom,
        ring_left:ring_right,
    ]
    ring_mask = ring_mask & ~gt_region
    weight_region = write.weights[
        write.image_index,
        0,
        ring_top:ring_bottom,
        ring_left:ring_right,
    ]
    write.weights[
        write.image_index,
        0,
        ring_top:ring_bottom,
        ring_left:ring_right,
    ] = torch.where(
        ring_mask,
        torch.maximum(
            weight_region,
            weight_region.new_tensor(write.settings.background_weight),
        ),
        weight_region,
    )
