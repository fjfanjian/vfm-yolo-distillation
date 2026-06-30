from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as vision_functional

from vfm_yolo_distillation.dino_coverage_subset import (
    _coverage_budget_count,
    _CoverageSelection,
    _EmbeddingRecord,
    _HybridSelectionSettings,
    _select_coverage_subset,
    _write_split_file,
)


class _DinoEmbeddingError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--teacher-repo", required=True)
    parser.add_argument("--teacher-weights", required=True)
    parser.add_argument("--teacher-arch", default="dinov3_vitb16")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--budget-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-mode",
        choices=("coverage", "hybrid_small"),
        default="coverage",
    )
    parser.add_argument("--hybrid-weight", type=float, default=0.35)
    parser.add_argument("--small-area-px", type=float, default=1024.0)
    parser.add_argument("--score-image-size", type=int, default=960)
    parser.add_argument("--class-diversity-weight", type=float, default=1.0)
    parser.add_argument("--scale-diversity-weight", type=float, default=0.5)
    parser.add_argument("--summary-output")
    return parser.parse_args()


def _collect_train_images(dataset_root: Path) -> tuple[Path, ...]:
    image_dir = dataset_root / "VisDrone2019-DET-train" / "images"
    return tuple(sorted(image_dir.glob("*.jpg")))


def _image_label_path(dataset_root: Path, image_path: Path) -> Path:
    label_dir = dataset_root / "VisDrone2019-DET-train" / "labels"
    return label_dir / f"{image_path.stem}.txt"


def _small_object_scores(
    dataset_root: Path,
    image_paths: tuple[Path, ...],
    small_area_px: float,
    score_image_size: int,
    class_diversity_weight: float,
    scale_diversity_weight: float,
) -> dict[Path, float]:
    scores: dict[Path, float] = {}
    for image_path in image_paths:
        label_path = _image_label_path(dataset_root, image_path)
        if not label_path.exists():
            scores[image_path] = 0.0
            continue
        small_classes: set[int] = set()
        scale_bins: set[str] = set()
        small_count = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_yolo_label(line)
            if parsed is None:
                continue
            class_id, box_width, box_height = parsed
            area_px = box_width * score_image_size * box_height * score_image_size
            if area_px > small_area_px:
                continue
            small_count += 1
            small_classes.add(class_id)
            scale_bins.add(_scale_bin(area_px, small_area_px))
        scores[image_path] = (
            float(small_count)
            + class_diversity_weight * float(len(small_classes))
            + scale_diversity_weight * float(len(scale_bins))
        )
    return scores


def _parse_yolo_label(line: str) -> tuple[int, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    return int(float(parts[0])), float(parts[3]), float(parts[4])


def _scale_bin(area_px: float, small_area_px: float) -> str:
    if area_px <= small_area_px / 4:
        return "tiny"
    if area_px <= small_area_px / 2:
        return "small_low"
    return "small_high"


def _extract_dino_embeddings(
    args: argparse.Namespace,
    image_paths: tuple[Path, ...],
) -> tuple[_EmbeddingRecord, ...]:
    sys.path.insert(0, str(Path(args.teacher_repo)))
    from dinov3.hub.backbones import dinov3_vitb16  # noqa: PLC0415

    if args.teacher_arch != "dinov3_vitb16":
        message = f"Unsupported DINOv3 arch: {args.teacher_arch}"
        raise _DinoEmbeddingError(message)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = dinov3_vitb16(weights=str(args.teacher_weights)).to(device)
    teacher.eval()
    teacher.requires_grad_(requires_grad=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    records: list[_EmbeddingRecord] = []

    for start in range(0, len(image_paths), args.batch):
        batch_paths = image_paths[start : start + args.batch]
        images = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB").resize((args.image_size, args.image_size))
                images.append(vision_functional.pil_to_tensor(rgb).float() / 255.0)
        tensor = torch.stack(images, dim=0).to(device)
        normalized = (tensor - mean) / std
        with torch.no_grad():
            features = teacher.forward_features(normalized)
        cls_tokens = features.get("x_norm_clstoken")
        if not isinstance(cls_tokens, torch.Tensor):
            patch_tokens = features.get("x_norm_patchtokens")
            if not isinstance(patch_tokens, torch.Tensor):
                message = "DINOv3 teacher did not return usable embeddings"
                raise TypeError(message)
            cls_tokens = patch_tokens.mean(dim=1)
        for image_path, embedding in zip(batch_paths, cls_tokens.detach().cpu(), strict=True):
            records.append(
                _EmbeddingRecord(
                    image_path=image_path,
                    vector=tuple(float(value) for value in embedding.tolist()),
                )
            )
    return tuple(records)


def _main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    image_paths = _collect_train_images(dataset_root)
    records = _extract_dino_embeddings(args, image_paths)
    scores = (
        _small_object_scores(
            dataset_root=dataset_root,
            image_paths=image_paths,
            small_area_px=args.small_area_px,
            score_image_size=args.score_image_size,
            class_diversity_weight=args.class_diversity_weight,
            scale_diversity_weight=args.scale_diversity_weight,
        )
        if args.selection_mode == "hybrid_small"
        else None
    )
    selection = _select_subset_fast(
        records=records,
        count=_coverage_budget_count(
            total_count=len(records),
            budget_ratio=args.budget_ratio,
        ),
        seed=args.seed,
        hybrid_weight=args.hybrid_weight,
        candidate_scores=scores,
    )
    _write_split_file(selection, Path(args.output_split))
    if args.summary_output:
        _write_summary(
            output_path=Path(args.summary_output),
            image_paths=image_paths,
            selection=selection.selected_paths,
            scores=scores,
            args=args,
        )
    sys.stdout.write(
        f"Selected {len(selection.selected_paths)}/{len(records)} images -> {args.output_split}\n"
    )
    return 0


def _write_summary(
    output_path: Path,
    image_paths: tuple[Path, ...],
    selection: tuple[Path, ...],
    scores: dict[Path, float] | None,
    args: argparse.Namespace,
) -> None:
    score_values = scores or {}
    summary = {
        "selection_mode": args.selection_mode,
        "budget_ratio": args.budget_ratio,
        "seed": args.seed,
        "total_images": len(image_paths),
        "selected_images": len(selection),
        "hybrid_weight": args.hybrid_weight,
        "small_area_px": args.small_area_px,
        "score_image_size": args.score_image_size,
        "mean_score_all": _mean(score_values.get(path, 0.0) for path in image_paths),
        "mean_score_selected": _mean(score_values.get(path, 0.0) for path in selection),
        "selected_with_small_score": sum(
            1 for path in selection if score_values.get(path, 0.0) > 0
        ),
        "selected_preview": [path.as_posix() for path in selection[:10]],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean(values: object) -> float:
    numbers = tuple(float(value) for value in values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _select_subset_fast(
    records: tuple[_EmbeddingRecord, ...],
    count: int,
    seed: int,
    hybrid_weight: float,
    candidate_scores: dict[Path, float] | None,
) -> _CoverageSelection:
    if len(records) <= 2048:
        return _select_coverage_subset(
            records=records,
            count=count,
            seed=seed,
            hybrid=(
                _HybridSelectionSettings(
                    candidate_scores=candidate_scores,
                    score_weight=hybrid_weight,
                )
                if candidate_scores is not None
                else None
            ),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = torch.tensor(
        [record.vector for record in records],
        dtype=torch.float32,
        device=device,
    )
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    scores = _candidate_score_tensor(
        records=records,
        candidate_scores=candidate_scores,
        hybrid_weight=hybrid_weight,
        device=device,
    )
    first_index = _first_index_from_scores(scores=scores, seed=seed)
    selected_indices = [first_index]
    selected_mask = torch.zeros(len(records), dtype=torch.bool, device=device)
    selected_mask[first_index] = True
    min_distances = ((vectors - vectors[first_index]) ** 2).sum(dim=1)
    for _ in range(1, count):
        candidate_values = min_distances + scores
        candidate_values = candidate_values.masked_fill(selected_mask, float("-inf"))
        next_index = int(torch.argmax(candidate_values).item())
        selected_indices.append(next_index)
        selected_mask[next_index] = True
        distances = ((vectors - vectors[next_index]) ** 2).sum(dim=1)
        min_distances = torch.minimum(min_distances, distances)
    selected_paths = tuple(sorted(records[index].image_path for index in selected_indices))
    return _CoverageSelection(selected_paths=selected_paths, budget_count=count)


def _candidate_score_tensor(
    records: tuple[_EmbeddingRecord, ...],
    candidate_scores: dict[Path, float] | None,
    hybrid_weight: float,
    device: torch.device,
) -> torch.Tensor:
    if candidate_scores is None or hybrid_weight <= 0:
        return torch.zeros(len(records), dtype=torch.float32, device=device)
    raw_scores = torch.tensor(
        [float(candidate_scores.get(record.image_path, 0.0)) for record in records],
        dtype=torch.float32,
        device=device,
    )
    max_score = torch.max(raw_scores)
    if float(max_score.item()) <= 0:
        return torch.zeros(len(records), dtype=torch.float32, device=device)
    return raw_scores / max_score * hybrid_weight


def _first_index_from_scores(scores: torch.Tensor, seed: int) -> int:
    if scores.numel() == 0 or float(torch.max(scores).item()) <= 0:
        return seed % int(scores.numel())
    return int(torch.argmax(scores).item())


if __name__ == "__main__":
    raise SystemExit(_main())
