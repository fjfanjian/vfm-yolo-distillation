"""Utilities for building DINOv3 objectness auxiliary targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class AreaThresholds:
    """Pixel-area thresholds used to split GT boxes by object scale."""

    small_area_px: float
    medium_area_px: float


@dataclass(frozen=True, slots=True)
class SmallGtWeights:
    """Loss weights for scale-aware objectness auxiliary supervision."""

    background: float
    small: float
    medium: float
    large: float
    false_positive: float


@dataclass(frozen=True, slots=True)
class GtAreaMaskSettings:
    """Grid and scale settings for GT area mask construction."""

    batch_size: int
    image_size_hw: tuple[int, int]
    target_hw: tuple[int, int]
    thresholds: AreaThresholds
    box_expand_cells: int


@dataclass(frozen=True, slots=True)
class SmallGtWeightMapSettings:
    """Settings for GT-aware soft objectness weighting."""

    thresholds: AreaThresholds
    weights: SmallGtWeights
    false_positive_quantile: float
    box_expand_cells: int


@dataclass(frozen=True, slots=True)
class PeakIgnoreSettings:
    """Settings for peak-only ignore-aware target construction."""

    positive_quantile: float
    negative_quantile: float
    peak_kernel: int


@dataclass(frozen=True, slots=True)
class GtAreaMasks:
    """Boolean masks for small, medium, large, and any GT areas."""

    small: torch.Tensor
    medium: torch.Tensor
    large: torch.Tensor
    any: torch.Tensor


class InvalidPeakKernelError(ValueError):
    """Raised when peak-only target construction receives an invalid kernel."""

    def __init__(self, peak_kernel: int) -> None:
        """Initialize the error with the invalid peak kernel."""
        self.peak_kernel = peak_kernel
        super().__init__(
            f"peak_kernel must be a positive odd integer, got {peak_kernel}"
        )


def build_gt_area_masks(
    batch_idx: torch.Tensor,
    bboxes_xywhn: torch.Tensor,
    settings: GtAreaMaskSettings,
) -> GtAreaMasks:
    """Build grid masks marking GT neighborhoods split by pixel area."""
    device = bboxes_xywhn.device if bboxes_xywhn.numel() else batch_idx.device
    target_h, target_w = settings.target_hw
    shape = (settings.batch_size, 1, target_h, target_w)
    small = torch.zeros(shape, dtype=torch.bool, device=device)
    medium = torch.zeros_like(small)
    large = torch.zeros_like(small)
    if bboxes_xywhn.numel() == 0:
        return GtAreaMasks(small=small, medium=medium, large=large, any=small.clone())

    image_h, image_w = settings.image_size_hw
    for index in range(bboxes_xywhn.shape[0]):
        image_index = int(batch_idx[index].item())
        if image_index < 0 or image_index >= settings.batch_size:
            continue
        x_center, y_center, box_w, box_h = bboxes_xywhn[index]
        area_px = float((box_w * image_w * box_h * image_h).item())
        x1 = int(torch.floor((x_center - box_w * 0.5) * target_w).item())
        y1 = int(torch.floor((y_center - box_h * 0.5) * target_h).item())
        x2 = int(torch.ceil((x_center + box_w * 0.5) * target_w).item())
        y2 = int(torch.ceil((y_center + box_h * 0.5) * target_h).item())
        x1, x2 = _expanded_bounds(x1, x2, target_w, settings.box_expand_cells)
        y1, y2 = _expanded_bounds(y1, y2, target_h, settings.box_expand_cells)
        mask = _mask_for_area(area_px, settings.thresholds, small, medium, large)
        mask[image_index, 0, y1:y2, x1:x2] = True

    return GtAreaMasks(small=small, medium=medium, large=large, any=small | medium | large)


def build_small_gt_weight_map(
    targets: torch.Tensor,
    batch_idx: torch.Tensor,
    bboxes_xywhn: torch.Tensor,
    image_size_hw: tuple[int, int],
    settings: SmallGtWeightMapSettings,
) -> torch.Tensor:
    """Build a soft-objectness loss weight map biased toward small GT boxes."""
    batch_idx = batch_idx.to(device=targets.device)
    bboxes_xywhn = bboxes_xywhn.to(device=targets.device)
    masks = build_gt_area_masks(
        batch_idx=batch_idx,
        bboxes_xywhn=bboxes_xywhn,
        settings=GtAreaMaskSettings(
            batch_size=targets.shape[0],
            image_size_hw=image_size_hw,
            target_hw=(targets.shape[-2], targets.shape[-1]),
            thresholds=settings.thresholds,
            box_expand_cells=settings.box_expand_cells,
        ),
    )
    weight_map = targets.new_full(targets.shape, settings.weights.background)
    weight_map = torch.where(
        masks.large.to(device=targets.device),
        targets.new_tensor(settings.weights.large),
        weight_map,
    )
    weight_map = torch.where(
        masks.medium.to(device=targets.device),
        targets.new_tensor(settings.weights.medium),
        weight_map,
    )
    weight_map = torch.where(
        masks.small.to(device=targets.device),
        targets.new_tensor(settings.weights.small),
        weight_map,
    )

    high_response = targets >= _per_image_quantile(targets, settings.false_positive_quantile)
    outside_gt = ~masks.any.to(device=targets.device)
    return torch.where(
        high_response & outside_gt,
        targets.new_tensor(settings.weights.false_positive),
        weight_map,
    )


def build_peak_ignore_targets(
    targets: torch.Tensor, settings: PeakIgnoreSettings
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build peak-only positive labels and ignore-aware weights."""
    if settings.peak_kernel < 1 or settings.peak_kernel % 2 == 0:
        raise InvalidPeakKernelError(settings.peak_kernel)

    padding = settings.peak_kernel // 2
    pooled = functional.max_pool2d(
        targets,
        kernel_size=settings.peak_kernel,
        stride=1,
        padding=padding,
    )
    peaks = targets == pooled
    positives = peaks & (targets >= _per_image_quantile(targets, settings.positive_quantile))
    negatives = targets <= _per_image_quantile(targets, settings.negative_quantile)
    labels = positives.to(dtype=targets.dtype)
    weights = (positives | negatives).to(dtype=targets.dtype)
    return labels, weights


def weighted_soft_objectness_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Combine weighted BCE and weighted soft Dice losses."""
    bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted_bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
    probs = torch.sigmoid(logits)
    weighted_dice = _weighted_soft_dice_loss(probs, targets, weights)
    return (weighted_bce + weighted_dice) * 0.5


def _expanded_bounds(start: int, stop: int, limit: int, expand: int) -> tuple[int, int]:
    expanded_start = max(0, start - expand)
    expanded_stop = min(limit, stop + expand)
    if expanded_stop <= expanded_start:
        expanded_stop = min(limit, expanded_start + 1)
    return expanded_start, expanded_stop


def _mask_for_area(
    area_px: float,
    thresholds: AreaThresholds,
    small: torch.Tensor,
    medium: torch.Tensor,
    large: torch.Tensor,
) -> torch.Tensor:
    if area_px <= thresholds.small_area_px:
        return small
    if area_px <= thresholds.medium_area_px:
        return medium
    return large


def _per_image_quantile(targets: torch.Tensor, quantile: float) -> torch.Tensor:
    flat_targets = targets.float().flatten(start_dim=1)
    thresholds = torch.quantile(flat_targets, quantile, dim=1)
    return thresholds.to(dtype=targets.dtype).view(-1, 1, 1, 1)


def _weighted_soft_dice_loss(
    probs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    dims = (1, 2, 3)
    weighted_probs = probs * weights
    weighted_targets = targets * weights
    intersection = (weighted_probs * weighted_targets).sum(dim=dims)
    denominator = weighted_probs.sum(dim=dims) + weighted_targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
