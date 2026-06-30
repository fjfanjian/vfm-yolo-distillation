from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-split", required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--replace-ratio", type=float, default=0.30)
    parser.add_argument("--small-area-px", type=float, default=1024.0)
    parser.add_argument("--score-image-size", type=int, default=960)
    parser.add_argument("--class-diversity-weight", type=float, default=1.0)
    parser.add_argument("--scale-diversity-weight", type=float, default=0.5)
    return parser.parse_args()


def _collect_train_images(dataset_root: Path) -> tuple[Path, ...]:
    image_dir = dataset_root / "VisDrone2019-DET-train" / "images"
    return tuple(sorted(image_dir.glob("*.jpg")))


def _read_split(path: Path) -> tuple[Path, ...]:
    return tuple(
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


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


def _select_hybrid(
    all_images: tuple[Path, ...],
    base_selection: tuple[Path, ...],
    scores: dict[Path, float],
    replace_ratio: float,
) -> tuple[Path, ...]:
    replace_count = round(len(base_selection) * replace_ratio)
    base_set = set(base_selection)
    removed = tuple(
        sorted(base_selection, key=lambda path: (scores.get(path, 0.0), path.as_posix()))[
            :replace_count
        ]
    )
    kept = base_set.difference(removed)
    candidates = (path for path in all_images if path not in base_set)
    added = tuple(
        sorted(
            candidates,
            key=lambda path: (scores.get(path, 0.0), path.as_posix()),
            reverse=True,
        )[:replace_count]
    )
    return tuple(sorted((*kept, *added)))


def _write_split(paths: tuple[Path, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(path.as_posix() for path in paths) + "\n",
        encoding="utf-8",
    )


def _write_summary(
    output_path: Path,
    all_images: tuple[Path, ...],
    base_selection: tuple[Path, ...],
    hybrid_selection: tuple[Path, ...],
    scores: dict[Path, float],
    args: argparse.Namespace,
) -> None:
    base_set = set(base_selection)
    hybrid_set = set(hybrid_selection)
    summary = {
        "method": "dinov3_coverage_split_plus_small_object_replacement",
        "total_images": len(all_images),
        "base_selected_images": len(base_selection),
        "hybrid_selected_images": len(hybrid_selection),
        "replace_ratio": args.replace_ratio,
        "replaced_images": len(base_set.difference(hybrid_set)),
        "added_images": len(hybrid_set.difference(base_set)),
        "small_area_px": args.small_area_px,
        "score_image_size": args.score_image_size,
        "mean_score_all": _mean(scores.get(path, 0.0) for path in all_images),
        "mean_score_base": _mean(scores.get(path, 0.0) for path in base_selection),
        "mean_score_hybrid": _mean(scores.get(path, 0.0) for path in hybrid_selection),
        "hybrid_with_small_score": sum(
            1 for path in hybrid_selection if scores.get(path, 0.0) > 0
        ),
        "added_preview": [
            path.as_posix() for path in sorted(hybrid_set.difference(base_set))[:10]
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean(values: object) -> float:
    numbers = tuple(float(value) for value in values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    all_images = _collect_train_images(dataset_root)
    base_selection = _read_split(Path(args.base_split).expanduser().resolve())
    scores = _small_object_scores(dataset_root=dataset_root, image_paths=all_images, args=args)
    hybrid_selection = _select_hybrid(
        all_images=all_images,
        base_selection=base_selection,
        scores=scores,
        replace_ratio=float(args.replace_ratio),
    )
    _write_split(hybrid_selection, Path(args.output_split))
    _write_summary(
        output_path=Path(args.summary_output),
        all_images=all_images,
        base_selection=base_selection,
        hybrid_selection=hybrid_selection,
        scores=scores,
        args=args,
    )
    sys.stdout.write(
        f"Selected {len(hybrid_selection)}/{len(all_images)} images -> {args.output_split}\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
