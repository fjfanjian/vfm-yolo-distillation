from __future__ import annotations

import argparse
import html
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from vfm_yolo_distillation.pseudo_small_selection import (
    ImageScore,
    PseudoBox,
    ScoreSettings,
    budget_count,
    normalized_area,
    score_image,
    select_top,
    selected_for_visualization,
    write_image_scores,
    write_pseudo_boxes,
    write_split,
    write_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--image-list")
    parser.add_argument("--budget-ratio", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--small-area-px", type=float, default=1024.0)
    parser.add_argument("--score-image-size", type=int, default=960)
    parser.add_argument("--class-diversity-weight", type=float, default=1.0)
    parser.add_argument("--scale-diversity-weight", type=float, default=0.5)
    parser.add_argument("--box-score-mode", choices=("count", "confidence"), default="count")
    parser.add_argument("--visualize-count", type=int, default=24)
    parser.add_argument("--max-boxes-per-image", type=int, default=160)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _collect_images(dataset_root: Path, image_list: str | None, limit: int) -> tuple[Path, ...]:
    if image_list:
        paths = tuple(
            Path(line.strip()).resolve()
            for line in Path(image_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        paths = tuple(sorted(path.resolve() for path in (dataset_root / "VisDrone2019-DET-train" / "images").glob("*.jpg")))
    return paths[:limit] if limit > 0 else paths


def _predict_boxes(args: argparse.Namespace, images: tuple[Path, ...]) -> tuple[PseudoBox, ...]:
    from ultralytics import YOLO

    model = YOLO(str(args.teacher))
    boxes: list[PseudoBox] = []
    batch_size = max(1, int(args.predict_batch_size))
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        stream = model.predict(
            source=[path.as_posix() for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            stream=True,
            verbose=False,
        )
        for result in stream:
            height, width = (int(value) for value in result.orig_shape)
            image_path = Path(str(result.path)).resolve().as_posix()
            if result.boxes is None:
                continue
            for prediction in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in prediction.xyxy[0].tolist())
                boxes.append(
                    PseudoBox(
                        image=image_path,
                        class_id=int(prediction.cls[0]),
                        confidence=float(prediction.conf[0]),
                        xyxy=(x1, y1, x2, y2),
                        image_width=width,
                        image_height=height,
                    )
                )
        sys.stdout.write(f"predicted {min(start + batch_size, len(images))}/{len(images)} images\n")
        sys.stdout.flush()
    return tuple(boxes)


def _score_images(images: tuple[Path, ...], boxes: tuple[PseudoBox, ...], settings: ScoreSettings) -> tuple[ImageScore, ...]:
    boxes_by_image: dict[str, list[PseudoBox]] = defaultdict(list)
    for box in boxes:
        boxes_by_image[Path(box.image).resolve().as_posix()].append(box)
    return tuple(
        score_image(image.as_posix(), boxes_by_image.get(image.resolve().as_posix(), []), settings)
        for image in images
    )


def _draw_visualizations(
    scores: tuple[ImageScore, ...],
    boxes: tuple[PseudoBox, ...],
    output_dir: Path,
    settings: ScoreSettings,
    max_boxes_per_image: int,
) -> None:
    boxes_by_image: dict[str, list[PseudoBox]] = defaultdict(list)
    for box in boxes:
        boxes_by_image[Path(box.image).resolve().as_posix()].append(box)
    image_items: list[tuple[str, str, str]] = []
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for rank, score in enumerate(scores, start=1):
        image_path = Path(score.image)
        output_path = viz_dir / f"{rank:03d}_{image_path.stem}.jpg"
        _draw_one(
            image_path,
            boxes_by_image.get(image_path.resolve().as_posix(), []),
            output_path,
            settings,
            max_boxes_per_image,
        )
        image_items.append((output_path.name, image_path.name, f"score={score.score:.2f}, small={score.small_boxes}, boxes={score.pseudo_boxes}"))
    _write_gallery(output_dir / "index.html", image_items)


def _draw_one(
    image_path: Path,
    boxes: list[PseudoBox],
    output_path: Path,
    settings: ScoreSettings,
    max_boxes_per_image: int,
) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    visible = sorted(boxes, key=lambda box: box.confidence, reverse=True)[:max_boxes_per_image]
    for box in visible:
        area = normalized_area(box, settings.score_image_size)
        color = (255, 45, 45) if area <= settings.small_area_px else (50, 180, 255)
        x1, y1, x2, y2 = box.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = f"{box.class_id}:{box.confidence:.2f}"
        draw.text((x1, max(0.0, y1 - 10.0)), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _write_gallery(output_path: Path, image_items: list[tuple[str, str, str]]) -> None:
    cards = []
    for filename, source_name, caption in image_items:
        cards.append(
            "<figure>"
            f"<img src='visualizations/{html.escape(filename)}' alt='{html.escape(source_name)}'>"
            f"<figcaption>{html.escape(source_name)}<br>{html.escape(caption)}</figcaption>"
            "</figure>"
        )
    output_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Pseudo small-object boxes</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}"
        "figure{margin:0;background:white;padding:10px;border:1px solid #ddd}"
        "img{width:100%;height:auto;display:block}figcaption{font-size:13px;margin-top:8px;color:#333}</style>"
        "<h1>Pseudo small-object boxes</h1><p>Red boxes are small by normalized area threshold; blue boxes are other pseudo boxes.</p>"
        f"<main>{''.join(cards)}</main>",
        encoding="utf-8",
    )


def _main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    images = _collect_images(Path(args.dataset_root), args.image_list, int(args.limit))
    settings = ScoreSettings(
        small_area_px=float(args.small_area_px),
        score_image_size=int(args.score_image_size),
        class_diversity_weight=float(args.class_diversity_weight),
        scale_diversity_weight=float(args.scale_diversity_weight),
        box_score_mode=str(args.box_score_mode),
    )
    boxes = _predict_boxes(args, images)
    scores = _score_images(images, boxes, settings)
    selected = select_top(scores, budget_count(len(scores), int(args.budget_ratio)))
    ranked = tuple(sorted(scores, key=lambda item: (item.score, item.image), reverse=True))
    write_split(selected, Path(args.output_split))
    write_pseudo_boxes(boxes, output_dir / "pseudo_boxes.csv", settings)
    write_image_scores(ranked, output_dir / "pseudo_image_scores.csv")
    write_summary(output_dir / "pseudo_selection_summary.json", ranked, selected, settings)
    _draw_visualizations(
        selected_for_visualization(ranked, int(args.visualize_count)),
        boxes,
        output_dir,
        settings,
        int(args.max_boxes_per_image),
    )
    sys.stdout.write(f"Selected {len(selected)}/{len(scores)} images -> {args.output_split}\n")
    sys.stdout.write(f"Visualization gallery -> {output_dir / 'index.html'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
