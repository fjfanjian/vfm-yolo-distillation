from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--budget-ratio", type=int, default=10)
    parser.add_argument("--small-area-px", type=float, default=1024.0)
    parser.add_argument("--score-image-size", type=int, default=960)
    parser.add_argument("--class-diversity-weight", type=float, default=1.0)
    parser.add_argument("--scale-diversity-weight", type=float, default=0.5)
    return parser.parse_args()


def _collect_train_images(dataset_root: Path) -> tuple[Path, ...]:
    image_dir = dataset_root / "VisDrone2019-DET-train" / "images"
    return tuple(sorted(image_dir.glob("*.jpg")))


def _label_path(dataset_root: Path, image_path: Path) -> Path:
    return dataset_root / "VisDrone2019-DET-train" / "labels" / f"{image_path.stem}.txt"


def _small_object_scores(
    dataset_root: Path,
    image_paths: tuple[Path, ...],
    args: argparse.Namespace,
) -> dict[Path, float]:
    scores: dict[Path, float] = {}
    for image_path in image_paths:
        label_path = _label_path(dataset_root, image_path)
        small_count = 0
        small_classes: set[int] = set()
        scale_bins: set[str] = set()
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_yolo_label(line)
                if parsed is None:
                    continue
                class_id, box_width, box_height = parsed
                area_px = (
                    box_width
                    * float(args.score_image_size)
                    * box_height
                    * float(args.score_image_size)
                )
                if area_px > float(args.small_area_px):
                    continue
                small_count += 1
                small_classes.add(class_id)
                scale_bins.add(_scale_bin(area_px, float(args.small_area_px)))
        scores[image_path] = (
            float(small_count)
            + float(args.class_diversity_weight) * float(len(small_classes))
            + float(args.scale_diversity_weight) * float(len(scale_bins))
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


def _budget_count(total: int, budget_ratio: int) -> int:
    if budget_ratio <= 0:
        raise ValueError("budget-ratio must be positive")
    return max(1, round(total * budget_ratio / 100.0))


def _select_top_small(
    image_paths: tuple[Path, ...],
    scores: dict[Path, float],
    budget: int,
) -> tuple[Path, ...]:
    selected = sorted(
        image_paths,
        key=lambda path: (scores.get(path, 0.0), path.as_posix()),
        reverse=True,
    )[:budget]
    return tuple(sorted(selected))


def _write_split(paths: tuple[Path, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(path.as_posix() for path in paths) + "\n",
        encoding="utf-8",
    )


def _mean(values: Iterable[float]) -> float:
    numbers = tuple(float(value) for value in values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _write_summary(
    output_path: Path,
    all_images: tuple[Path, ...],
    selected: tuple[Path, ...],
    scores: dict[Path, float],
    args: argparse.Namespace,
) -> None:
    summary = {
        "method": "small_object_score_topk",
        "total_images": len(all_images),
        "selected_images": len(selected),
        "budget_ratio": args.budget_ratio,
        "small_area_px": args.small_area_px,
        "score_image_size": args.score_image_size,
        "mean_score_all": _mean(scores.get(path, 0.0) for path in all_images),
        "mean_score_selected": _mean(scores.get(path, 0.0) for path in selected),
        "selected_with_small_score": sum(1 for path in selected if scores.get(path, 0.0) > 0),
        "selected_preview": [path.as_posix() for path in selected[:10]],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    all_images = _collect_train_images(dataset_root)
    scores = _small_object_scores(dataset_root, all_images, args)
    selected = _select_top_small(
        image_paths=all_images,
        scores=scores,
        budget=_budget_count(len(all_images), int(args.budget_ratio)),
    )
    _write_split(selected, Path(args.output_split))
    _write_summary(Path(args.summary_output), all_images, selected, scores, args)
    sys.stdout.write(f"Selected {len(selected)}/{len(all_images)} images -> {args.output_split}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
