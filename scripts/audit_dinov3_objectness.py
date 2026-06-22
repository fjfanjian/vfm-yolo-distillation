# /// script
# requires-python = ">=3.10"
# ///
# --- How to run ---
# python scripts/audit_dinov3_objectness.py --samples 300 --output runs/dinov3_objectness/audit

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


NORMALIZATION: Final = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
SMALL_AREA: Final = 32 * 32


@dataclass(frozen=True, slots=True)
class Box:
    center_x: float
    center_y: float
    width: float
    height: float
    image_width: int
    image_height: int

    @property
    def area_px(self) -> float:
        return self.width * self.image_width * self.height * self.image_height


@dataclass(frozen=True, slots=True)
class ImageMetric:
    image: str
    boxes: int
    small_boxes: int
    top10_hits: int
    top20_hits: int
    small_top10_hits: int
    small_top20_hits: int
    foreground_score: float
    background_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("configs/datasets/visdrone.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("runs/dinov3_objectness/audit"))
    parser.add_argument("--overlay-count", type=int, default=24)
    parser.add_argument("--teacher-image-size", type=int, default=448)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-stride", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--method",
        choices=("border_pca", "local_contrast", "local_residual", "local_fusion"),
        default="border_pca",
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


def read_dataset_paths(config_path: Path, split: str) -> tuple[Path, Path]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    root = Path(config["path"])
    split_value = Path(config[split])
    split_path = split_value if split_value.is_absolute() else root / split_value
    return root, split_path


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_index = len(parts) - 1 - parts[::-1].index("images")
    label_parts = parts[:image_index] + ["labels"] + parts[image_index + 1 :]
    return Path(*label_parts).with_suffix(".txt")


def image_records(config_path: Path, split: str) -> list[tuple[Path, Path]]:
    split_path = read_dataset_paths(config_path, split)[1]
    if split_path.is_file():
        paths = [Path(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines()]
    else:
        paths = sorted(split_path.glob("*.jpg"))
    return [(path, label_path_for(path)) for path in paths]


def load_boxes(record: tuple[Path, Path], image_size: tuple[int, int]) -> list[Box]:
    _image_path, label_path = record
    if not label_path.exists():
        return []
    image_width, image_height = image_size
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        center_x, center_y, width, height = (float(value) for value in parts[1:])
        boxes.append(Box(center_x, center_y, width, height, image_width, image_height))
    return boxes


def load_teacher(repo: Path, weights: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, repo.as_posix())
    from dinov3.hub.backbones import dinov3_vitb16

    teacher = dinov3_vitb16(weights=weights.as_posix())
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher.to(device)


def normalize_image(rgb: Image.Image, image_size: int) -> torch.Tensor:
    resized = rgb.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(NORMALIZATION[0], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION[1], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def image_tensor(image_path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        return normalize_image(rgb, image_size), rgb.size


def normalize_score(score: torch.Tensor) -> torch.Tensor:
    score = score.float()
    return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-6)


def border_contrast(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    token_grid = tokens.reshape(grid, grid, -1)
    border_mask = torch.zeros((grid, grid), dtype=torch.bool, device=tokens.device)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    normalized = F.normalize(token_grid.flatten(0, 1), dim=1)
    background = F.normalize(normalized[border_mask.flatten()].mean(dim=0, keepdim=True), dim=1)
    contrast = 1.0 - F.cosine_similarity(normalized, background, dim=1)
    return normalize_score(contrast.reshape(grid, grid))


def pca_foreground(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    _u, _s, vectors = torch.pca_lowrank(centered.float(), q=1)
    score = (centered.float() @ vectors[:, 0]).reshape(grid, grid)
    border = torch.cat((score[0, :], score[-1, :], score[:, 0], score[:, -1]))
    inner = score[1:-1, 1:-1].flatten()
    if border.mean() > inner.mean():
        score = -score
    return normalize_score(score)


def token_feature_map(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    return tokens.reshape(grid, grid, -1).permute(2, 0, 1).unsqueeze(0).float()


def local_contrast(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    features = F.normalize(token_feature_map(tokens, grid), dim=1)
    smooth = F.normalize(F.avg_pool2d(features, kernel_size=3, stride=1, padding=1, count_include_pad=False), dim=1)
    return normalize_score((1.0 - F.cosine_similarity(features, smooth, dim=1)).squeeze(0))


def local_residual(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    features = token_feature_map(tokens, grid)
    smooth = F.avg_pool2d(features, kernel_size=5, stride=1, padding=2, count_include_pad=False)
    return normalize_score((features - smooth).pow(2).mean(dim=1).sqrt().squeeze(0))


def objectness_score(tokens: torch.Tensor, grid: int, method: str) -> torch.Tensor:
    match method:
        case "border_pca":
            return normalize_score((border_contrast(tokens, grid) + pca_foreground(tokens, grid)) / 2.0)
        case "local_contrast":
            return local_contrast(tokens, grid)
        case "local_residual":
            return local_residual(tokens, grid)
        case "local_fusion":
            return normalize_score((local_contrast(tokens, grid) + local_residual(tokens, grid)) / 2.0)
        case _:
            raise ValueError(f"Unsupported objectness method: {method}")


def box_slice(box: Box, score_shape: tuple[int, int]) -> tuple[slice, slice]:
    grid_h, grid_w = score_shape
    left = max(0, int((box.center_x - box.width / 2.0) * grid_w))
    right = min(grid_w, int((box.center_x + box.width / 2.0) * grid_w) + 1)
    top = max(0, int((box.center_y - box.height / 2.0) * grid_h))
    bottom = min(grid_h, int((box.center_y + box.height / 2.0) * grid_h) + 1)
    return slice(top, max(top + 1, bottom)), slice(left, max(left + 1, right))


def metric_for_image(record: tuple[Path, Path], score: torch.Tensor, image_size: tuple[int, int]) -> ImageMetric:
    boxes = load_boxes(record, image_size)
    mask = torch.zeros(score.shape, dtype=torch.bool, device=score.device)
    top10 = torch.quantile(score.flatten(), 0.90)
    top20 = torch.quantile(score.flatten(), 0.80)
    top10_hits = top20_hits = small_top10_hits = small_top20_hits = 0
    small_boxes = 0
    for box in boxes:
        rows, cols = box_slice(box, score.shape)
        mask[rows, cols] = True
        region_max = score[rows, cols].max()
        is_small = box.area_px < SMALL_AREA
        top10_hit = bool(region_max >= top10)
        top20_hit = bool(region_max >= top20)
        top10_hits += int(top10_hit)
        top20_hits += int(top20_hit)
        small_boxes += int(is_small)
        small_top10_hits += int(is_small and top10_hit)
        small_top20_hits += int(is_small and top20_hit)
    foreground = float(score[mask].mean().item()) if mask.any() else 0.0
    background = float(score[~mask].mean().item()) if (~mask).any() else 0.0
    return ImageMetric(
        image=record[0].as_posix(),
        boxes=len(boxes),
        small_boxes=small_boxes,
        top10_hits=top10_hits,
        top20_hits=top20_hits,
        small_top10_hits=small_top10_hits,
        small_top20_hits=small_top20_hits,
        foreground_score=foreground,
        background_score=background,
    )


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    return positions if positions[-1] == last else [*positions, last]


def tiled_objectness(
    record: tuple[Path, Path],
    teacher: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    grid: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(record[0]) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size
    stride = args.tile_stride or max(1, args.tile_size // 2)
    score_sum = torch.zeros((height, width), device=device)
    score_count = torch.zeros((height, width), device=device)
    windows = [
        (left, top, min(left + args.tile_size, width), min(top + args.tile_size, height))
        for top in tile_positions(height, args.tile_size, stride)
        for left in tile_positions(width, args.tile_size, stride)
    ]
    for start in range(0, len(windows), args.batch):
        batch_windows = windows[start : start + args.batch]
        crops = [normalize_image(rgb.crop(window), args.teacher_image_size) for window in batch_windows]
        with torch.inference_mode():
            tokens = teacher.forward_features(torch.stack(crops).to(device))["x_norm_patchtokens"].detach()
        for token, (left, top, right, bottom) in zip(tokens, batch_windows, strict=True):
            tile_score = objectness_score(token, grid, args.method)
            resized = F.interpolate(
                tile_score[None, None],
                size=(bottom - top, right - left),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            score_sum[top:bottom, left:right] += resized
            score_count[top:bottom, left:right] += 1.0
    return normalize_score(score_sum / score_count.clamp_min(1.0)), rgb.size


def save_overlay(record: tuple[Path, Path], score: torch.Tensor, output_path: Path) -> None:
    score_array = (score.detach().cpu().numpy() * 255.0).astype(np.uint8)
    heat_array = np.stack((score_array, np.clip((score_array.astype(np.int16) - 80) * 2, 0, 255).astype(np.uint8), np.zeros_like(score_array)), axis=2)
    with Image.open(record[0]) as image:
        base = image.convert("RGB")
    heat = np.asarray(Image.fromarray(heat_array).resize(base.size, Image.BILINEAR), dtype=np.float32)
    alpha = np.clip((heat[:, :, :1] - 128.0) / 127.0, 0.0, 1.0) * 0.75
    base_array = np.asarray(base, dtype=np.float32)
    blended = Image.fromarray((base_array * (1.0 - alpha) + heat * alpha).astype(np.uint8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)


def write_metrics(metrics: list[ImageMetric], output: Path, args: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "objectness_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(ImageMetric.__dataclass_fields__.keys())
        writer.writerows(
            (
                item.image,
                item.boxes,
                item.small_boxes,
                item.top10_hits,
                item.top20_hits,
                item.small_top10_hits,
                item.small_top20_hits,
                f"{item.foreground_score:.6f}",
                f"{item.background_score:.6f}",
            )
            for item in metrics
        )
    boxes = sum(item.boxes for item in metrics)
    small_boxes = sum(item.small_boxes for item in metrics)
    summary = (
        ("method", args.method),
        ("tile_size", args.tile_size),
        ("tile_stride", args.tile_stride or max(0, args.tile_size // 2)),
        ("images", len(metrics)),
        ("boxes", boxes),
        ("small_boxes", small_boxes),
        ("top10_box_recall", sum(item.top10_hits for item in metrics) / max(1, boxes)),
        ("top20_box_recall", sum(item.top20_hits for item in metrics) / max(1, boxes)),
        ("small_top10_recall", sum(item.small_top10_hits for item in metrics) / max(1, small_boxes)),
        ("small_top20_recall", sum(item.small_top20_hits for item in metrics) / max(1, small_boxes)),
        ("mean_foreground_score", sum(item.foreground_score for item in metrics) / max(1, len(metrics))),
        ("mean_background_score", sum(item.background_score for item in metrics) / max(1, len(metrics))),
    )
    with (output / "objectness_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "value"))
        writer.writerows(summary)


def run() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = image_records(args.dataset, args.split)
    sample_count = min(args.samples, len(records))
    sampled = records if args.samples <= 0 else random.Random(args.seed).sample(records, sample_count)
    teacher = load_teacher(args.teacher_repo, args.teacher_weights, device)
    output = args.output
    overlays: list[Path] = []
    metrics: list[ImageMetric] = []
    grid = args.teacher_image_size // 16
    if args.tile_size > 0:
        for index, record in enumerate(sampled, start=1):
            score, size = tiled_objectness(record, teacher, args, device, grid)
            metrics.append(metric_for_image(record, score, size))
            if len(overlays) < args.overlay_count:
                path = output / "overlays" / f"{record[0].stem}_objectness.jpg"
                save_overlay(record, score, path)
                overlays.append(path)
            print(f"processed {index}/{len(sampled)}", flush=True)
        write_metrics(metrics, output, args)
        return 0
    for start in range(0, len(sampled), args.batch):
        batch_records = sampled[start : start + args.batch]
        tensors: list[torch.Tensor] = []
        sizes: list[tuple[int, int]] = []
        for record in batch_records:
            tensor, size = image_tensor(record[0], args.teacher_image_size)
            tensors.append(tensor)
            sizes.append(size)
        images = torch.stack(tensors).to(device)
        with torch.inference_mode():
            features = teacher.forward_features(images)
        tokens = features["x_norm_patchtokens"].detach()
        for index, record in enumerate(batch_records):
            token = tokens[index]
            score = objectness_score(token, grid, args.method)
            metrics.append(metric_for_image(record, score, sizes[index]))
            if len(overlays) < args.overlay_count:
                path = output / "overlays" / f"{record[0].stem}_objectness.jpg"
                save_overlay(record, score, path)
                overlays.append(path)
        print(f"processed {min(start + len(batch_records), len(sampled))}/{len(sampled)}", flush=True)
    write_metrics(metrics, output, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
