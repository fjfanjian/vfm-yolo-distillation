#!/usr/bin/env python3
"""Audit spatial alignment between DINOv3 objectness maps and VisDrone GT boxes."""
# ruff: noqa: D103, S311, T201

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_dinov3_objectness import (
    Box,
    image_records,
    image_tensor,
    load_boxes,
    load_teacher,
    normalize_score,
    objectness_score,
    tiled_objectness,
)

SMALL_AREA_PX: Final = 32 * 32
MEDIUM_AREA_PX: Final = 96 * 96
HEAT_GREEN_THRESHOLD: Final = 0.65


@dataclass(frozen=True, slots=True)
class BoxAlignmentMetric:
    """One GT box matched against one objectness map."""

    image: str
    area_group: str
    area_px: float
    center_score: float
    box_mean: float
    box_max: float
    box_p90: float
    center_percentile: float
    hit_q85: bool
    hit_q90: bool


@dataclass(frozen=True, slots=True)
class ImageAlignmentMetric:
    """Image-level high-response ownership statistics."""

    image: str
    boxes: int
    small_boxes: int
    q85_gt_overlap: float
    q85_small_overlap: float
    q85_false_positive: float
    q90_gt_overlap: float
    q90_small_overlap: float
    q90_false_positive: float
    foreground_mean: float
    background_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("configs/datasets/visdrone.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-count", type=int, default=24)
    parser.add_argument("--teacher-image-size", type=int, default=448)
    parser.add_argument("--tile-size", type=int, default=448)
    parser.add_argument("--tile-stride", type=int, default=224)
    parser.add_argument("--target-height", type=int, default=120)
    parser.add_argument("--target-width", type=int, default=120)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--method",
        choices=("border_pca", "local_contrast", "local_residual", "local_fusion"),
        default="local_contrast",
    )
    parser.add_argument("--teacher-repo", type=Path, default=Path("/home/featurize/work/dinov3"))
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=Path(
            "/home/featurize/work/weights/dinov3/"
            "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ),
    )
    return parser.parse_args()


def area_group(box: Box) -> str:
    if box.area_px <= SMALL_AREA_PX:
        return "small"
    if box.area_px <= MEDIUM_AREA_PX:
        return "medium"
    return "large"


def box_bounds(box: Box, score_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = score_shape
    left = max(0, int((box.center_x - box.width * 0.5) * width))
    right = min(width, int((box.center_x + box.width * 0.5) * width) + 1)
    top = max(0, int((box.center_y - box.height * 0.5) * height))
    bottom = min(height, int((box.center_y + box.height * 0.5) * height) + 1)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def box_mask(
    boxes: list[Box], score_shape: tuple[int, int], group: str | None = None
) -> torch.Tensor:
    mask = torch.zeros(score_shape, dtype=torch.bool)
    for box in boxes:
        if group is not None and area_group(box) != group:
            continue
        left, top, right, bottom = box_bounds(box, score_shape)
        mask[top:bottom, left:right] = True
    return mask


def center_index(box: Box, score_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = score_shape
    row = min(height - 1, max(0, int(box.center_y * height)))
    col = min(width - 1, max(0, int(box.center_x * width)))
    return row, col


def metric_for_box(image: str, box: Box, score: torch.Tensor) -> BoxAlignmentMetric:
    left, top, right, bottom = box_bounds(box, tuple(score.shape))
    region = score[top:bottom, left:right].float()
    row, col = center_index(box, tuple(score.shape))
    center_score = score[row, col].float()
    flat = score.flatten().float()
    q85 = torch.quantile(flat, 0.85)
    q90 = torch.quantile(flat, 0.90)
    return BoxAlignmentMetric(
        image=image,
        area_group=area_group(box),
        area_px=box.area_px,
        center_score=float(center_score.item()),
        box_mean=float(region.mean().item()),
        box_max=float(region.max().item()),
        box_p90=float(torch.quantile(region.flatten(), 0.90).item()),
        center_percentile=float((flat <= center_score).float().mean().item()),
        hit_q85=bool(region.max() >= q85),
        hit_q90=bool(region.max() >= q90),
    )


def metric_for_image(image: str, boxes: list[Box], score: torch.Tensor) -> ImageAlignmentMetric:
    score = score.detach().cpu().float()
    gt_mask = box_mask(boxes, tuple(score.shape))
    small_mask = box_mask(boxes, tuple(score.shape), group="small")
    flat = score.flatten()
    high85 = score >= torch.quantile(flat, 0.85)
    high90 = score >= torch.quantile(flat, 0.90)
    return ImageAlignmentMetric(
        image=image,
        boxes=len(boxes),
        small_boxes=sum(area_group(box) == "small" for box in boxes),
        q85_gt_overlap=mask_overlap(high85, gt_mask),
        q85_small_overlap=mask_overlap(high85, small_mask),
        q85_false_positive=1.0 - mask_overlap(high85, gt_mask),
        q90_gt_overlap=mask_overlap(high90, gt_mask),
        q90_small_overlap=mask_overlap(high90, small_mask),
        q90_false_positive=1.0 - mask_overlap(high90, gt_mask),
        foreground_mean=masked_mean(score, gt_mask),
        background_mean=masked_mean(score, ~gt_mask),
    )


def mask_overlap(source: torch.Tensor, target: torch.Tensor) -> float:
    total = int(source.sum().item())
    if total == 0:
        return 0.0
    return float((source & target).sum().item() / total)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(values[mask].mean().item()) if mask.any() else 0.0


def summarize(
    box_metrics: list[BoxAlignmentMetric],
    image_metrics: list[ImageAlignmentMetric],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows.append(("images", str(len(image_metrics))))
    rows.append(("boxes", str(len(box_metrics))))
    for group in ("small", "medium", "large"):
        subset = [item for item in box_metrics if item.area_group == group]
        rows.extend(summarize_box_group(group, subset))
    rows.extend(summarize_image_metrics(image_metrics))
    return rows


def summarize_box_group(
    group: str, metrics: list[BoxAlignmentMetric]
) -> list[tuple[str, str]]:
    if not metrics:
        return [(f"{group}_boxes", "0")]
    return [
        (f"{group}_boxes", str(len(metrics))),
        (f"{group}_center_mean", f"{mean(item.center_score for item in metrics):.6f}"),
        (f"{group}_box_mean", f"{mean(item.box_mean for item in metrics):.6f}"),
        (f"{group}_box_max_mean", f"{mean(item.box_max for item in metrics):.6f}"),
        (
            f"{group}_center_percentile_mean",
            f"{mean(item.center_percentile for item in metrics):.6f}",
        ),
        (f"{group}_hit_q85", f"{mean(float(item.hit_q85) for item in metrics):.6f}"),
        (f"{group}_hit_q90", f"{mean(float(item.hit_q90) for item in metrics):.6f}"),
    ]


def summarize_image_metrics(metrics: list[ImageAlignmentMetric]) -> list[tuple[str, str]]:
    return [
        ("q85_gt_overlap", f"{mean(item.q85_gt_overlap for item in metrics):.6f}"),
        ("q85_small_overlap", f"{mean(item.q85_small_overlap for item in metrics):.6f}"),
        ("q85_false_positive", f"{mean(item.q85_false_positive for item in metrics):.6f}"),
        ("q90_gt_overlap", f"{mean(item.q90_gt_overlap for item in metrics):.6f}"),
        ("q90_small_overlap", f"{mean(item.q90_small_overlap for item in metrics):.6f}"),
        ("q90_false_positive", f"{mean(item.q90_false_positive for item in metrics):.6f}"),
        ("foreground_mean", f"{mean(item.foreground_mean for item in metrics):.6f}"),
        ("background_mean", f"{mean(item.background_mean for item in metrics):.6f}"),
    ]


def mean(values: object) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def save_overlay(
    image_path: Path,
    boxes: list[Box],
    score: torch.Tensor,
    output_path: Path,
) -> None:
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    heat = heat_image(score, base.size)
    blended = Image.blend(base, heat, alpha=0.45)
    draw = ImageDraw.Draw(blended)
    width, height = base.size
    for box in boxes:
        color = {"small": "red", "medium": "yellow", "large": "cyan"}[area_group(box)]
        left = int((box.center_x - box.width * 0.5) * width)
        right = int((box.center_x + box.width * 0.5) * width)
        top = int((box.center_y - box.height * 0.5) * height)
        bottom = int((box.center_y + box.height * 0.5) * height)
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)


def heat_image(score: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    score = normalize_score(score.detach().cpu()).numpy()
    red = (score * 255.0).astype("uint8")
    green = ((score > HEAT_GREEN_THRESHOLD) * 180).astype("uint8")
    blue = ((1.0 - score) * 50.0).astype("uint8")
    array = np.stack((red, green, blue), axis=2)
    return Image.fromarray(array).resize(size, Image.BILINEAR).convert("RGB")


def write_csv(path: Path, header: tuple[str, ...], rows: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def run() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = image_records(args.dataset, args.split)
    if args.cache:
        target_size = (args.target_height, args.target_width)
        records = [
            record for record in records if cache_path_for(record[0], target_size, args).exists()
        ]
        print(f"cached records: {len(records)}", flush=True)
        if not records:
            message = f"no cached records found under {args.cache}"
            raise FileNotFoundError(message)
    sample_count = len(records) if args.samples <= 0 else min(args.samples, len(records))
    sampled = (
        records
        if args.samples <= 0
        else random.Random(args.seed).sample(records, sample_count)
    )
    teacher = None if args.cache else load_teacher(args.teacher_repo, args.teacher_weights, device)
    grid = args.teacher_image_size // 16

    box_metrics: list[BoxAlignmentMetric] = []
    image_metrics: list[ImageAlignmentMetric] = []
    for index, record in enumerate(sampled, start=1):
        if args.cache:
            score, image_size = cached_objectness(record[0], args)
        elif args.tile_size > 0:
            score, image_size = tiled_objectness(record, teacher, args, device, grid)
        else:
            tensor, image_size = image_tensor(record[0], args.teacher_image_size)
            with torch.inference_mode():
                features = teacher.forward_features(tensor.unsqueeze(0).to(device))
            score = objectness_score(features["x_norm_patchtokens"][0], grid, args.method)
        boxes = load_boxes(record, image_size)
        score_cpu = score.detach().cpu()
        image_metrics.append(metric_for_image(record[0].as_posix(), boxes, score_cpu))
        box_metrics.extend(
            metric_for_box(record[0].as_posix(), box, score_cpu) for box in boxes
        )
        if index <= args.overlay_count:
            save_overlay(
                record[0],
                boxes,
                score_cpu,
                args.output / "overlays" / f"{record[0].stem}_alignment.jpg",
            )
        print(f"processed {index}/{len(sampled)}", flush=True)

    write_outputs(args.output, box_metrics, image_metrics)
    return 0


def cached_objectness(
    image_path: Path, args: argparse.Namespace
) -> tuple[torch.Tensor, tuple[int, int]]:
    target_size = (args.target_height, args.target_width)
    cache_path = cache_path_for(image_path, target_size, args)
    if not cache_path.exists():
        message = f"missing cached target: {cache_path}"
        raise FileNotFoundError(message)
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cached, torch.Tensor) or tuple(cached.shape) != target_size:
        message = f"invalid cached target shape in {cache_path}: {getattr(cached, 'shape', None)}"
        raise ValueError(message)
    with Image.open(image_path) as image:
        image_size = image.size
    return cached.float(), image_size


def cache_path_for(
    image_path: Path, target_size: tuple[int, int], args: argparse.Namespace
) -> Path:
    key = "|".join(
        (
            image_path.as_posix(),
            args.method,
            str(args.tile_size),
            str(args.tile_stride),
            str(args.teacher_image_size),
            f"{target_size[0]}x{target_size[1]}",
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
    exact = args.cache / f"{image_path.stem}_{target_size[0]}x{target_size[1]}_{digest}.pt"
    if exact.exists():
        return exact
    matches = sorted(args.cache.glob(f"{image_path.stem}_{target_size[0]}x{target_size[1]}_*.pt"))
    return matches[0] if matches else exact


def write_outputs(
    output: Path,
    box_metrics: list[BoxAlignmentMetric],
    image_metrics: list[ImageAlignmentMetric],
) -> None:
    write_csv(
        output / "box_alignment.csv",
        tuple(BoxAlignmentMetric.__dataclass_fields__),
        (
            (
                item.image,
                item.area_group,
                f"{item.area_px:.3f}",
                f"{item.center_score:.6f}",
                f"{item.box_mean:.6f}",
                f"{item.box_max:.6f}",
                f"{item.box_p90:.6f}",
                f"{item.center_percentile:.6f}",
                int(item.hit_q85),
                int(item.hit_q90),
            )
            for item in box_metrics
        ),
    )
    write_csv(
        output / "image_alignment.csv",
        tuple(ImageAlignmentMetric.__dataclass_fields__),
        (
            (
                item.image,
                item.boxes,
                item.small_boxes,
                f"{item.q85_gt_overlap:.6f}",
                f"{item.q85_small_overlap:.6f}",
                f"{item.q85_false_positive:.6f}",
                f"{item.q90_gt_overlap:.6f}",
                f"{item.q90_small_overlap:.6f}",
                f"{item.q90_false_positive:.6f}",
                f"{item.foreground_mean:.6f}",
                f"{item.background_mean:.6f}",
            )
            for item in image_metrics
        ),
    )
    write_csv(output / "summary.csv", ("metric", "value"), summarize(box_metrics, image_metrics))


if __name__ == "__main__":
    raise SystemExit(run())
