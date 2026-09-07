from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


COLORS = (
    (72, 190, 120),
    (120, 170, 255),
    (235, 190, 70),
    (240, 110, 90),
    (190, 130, 240),
    (80, 210, 220),
    (245, 150, 60),
    (180, 180, 80),
    (255, 120, 180),
    (130, 220, 130),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side ground-truth/prediction visualizations.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--max-boxes", type=int, default=80)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--title", default="Detection Visualizations")
    return parser.parse_args()


def load_names(data_yaml: Path) -> list[str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def load_predictions(path: Path, conf: float) -> dict[str, list[tuple[float, int, float, float, float, float]]]:
    predictions: dict[str, list[tuple[float, int, float, float, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            if score < conf:
                continue
            predictions[row["image_id"]].append(
                (
                    score,
                    int(row["class_id"]),
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                )
            )
    return predictions


def load_ground_truth(label_path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, box_w, box_h = map(float, parts[1:5])
        boxes.append(
            (
                class_id,
                (cx - box_w / 2) * width,
                (cy - box_h / 2) * height,
                (cx + box_w / 2) * width,
                (cy + box_h / 2) * height,
            )
        )
    return boxes


def select_images(
    image_dir: Path,
    label_dir: Path,
    predictions: dict[str, list[tuple[float, int, float, float, float, float]]],
    count: int,
    max_boxes: int,
) -> list[str]:
    stats: list[tuple[int, int, float, str]] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        with Image.open(image_path) as image:
            width, height = image.size
        gt_count = len(load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height))
        preds = predictions.get(image_path.stem, [])
        if gt_count and preds:
            stats.append((min(len(preds), max_boxes), gt_count, max(score for score, *_ in preds), image_path.stem))
    return [image_id for *_, image_id in sorted(stats, reverse=True)[:count]]


def draw_boxes(
    image: Image.Image,
    boxes: list[tuple],
    names: list[str],
    prediction: bool,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if prediction:
            score, class_id, x1, y1, x2, y2 = box
            label = f"{names[class_id] if 0 <= class_id < len(names) else class_id} {score:.2f}"
            color = (240, 70, 60)
        else:
            class_id, x1, y1, x2, y2 = box
            label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            color = COLORS[class_id % len(COLORS)]
        coords = [int(round(value)) for value in (x1, y1, x2, y2)]
        draw.rectangle(coords, outline=color, width=2)
        text_w = max(36, 7 * len(label))
        text_h = 13
        y0 = max(0, coords[1] - text_h)
        draw.rectangle([coords[0], y0, coords[0] + text_w, y0 + text_h], fill=color)
        draw.text((coords[0] + 2, y0), label, fill=(0, 0, 0))
    return image


def write_visualizations(args: argparse.Namespace) -> list[dict[str, object]]:
    names = load_names(args.data)
    predictions = load_predictions(args.predictions, args.conf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_images(args.images_dir, args.labels_dir, predictions, args.count, args.max_boxes)
    items: list[dict[str, object]] = []
    for image_id in selected:
        image = Image.open(args.images_dir / f"{image_id}.jpg").convert("RGB")
        width, height = image.size
        gt_boxes = load_ground_truth(args.labels_dir / f"{image_id}.txt", width, height)
        pred_boxes = sorted(predictions.get(image_id, []), reverse=True)[: args.max_boxes]
        left = draw_boxes(image.copy(), gt_boxes, names, prediction=False)
        right = draw_boxes(image.copy(), pred_boxes, names, prediction=True)
        banner_h = 34
        canvas = Image.new("RGB", (width * 2, height + banner_h), (245, 245, 245))
        canvas.paste(left, (0, banner_h))
        canvas.paste(right, (width, banner_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"GT labels: {len(gt_boxes)} boxes", fill=(20, 80, 40))
        draw.text(
            (width + 12, 10),
            f"Predictions: conf>={args.conf:.2f}, shown={len(pred_boxes)}",
            fill=(130, 30, 25),
        )
        file_name = f"{image_id}_gt_vs_pred.jpg"
        canvas.save(args.output_dir / file_name, quality=92)
        items.append({"image_id": image_id, "file": file_name, "gt": len(gt_boxes), "pred_shown": len(pred_boxes)})
    return items


def write_index(output_dir: Path, title: str, items: list[dict[str, object]], conf: float) -> None:
    cards = []
    for item in items:
        image_id = html.escape(str(item["image_id"]))
        file_name = html.escape(str(item["file"]))
        cards.append(
            "<div class='card'>"
            f"<h2>{image_id}</h2>"
            f"<div class='meta'>GT={item['gt']} | predictions shown={item['pred_shown']} | conf>={conf:.2f}</div>"
            f"<img src='{file_name}' alt='{image_id}'>"
            "</div>"
        )
    page = "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{html.escape(title)}</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#222}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}",
            ".card{background:white;border:1px solid #ddd;padding:12px}",
            "img{max-width:100%;height:auto;display:block}",
            ".meta{font-size:13px;color:#555;margin:8px 0}",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
            "<p>左侧为标注文件中的 GT，右侧为预测框。</p>",
            '<div class="grid">',
            "\n".join(cards),
            "</div>",
            "</body>",
            "</html>",
        ]
    )
    output_dir.joinpath("index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    items = write_visualizations(args)
    write_index(args.output_dir, args.title, items, args.conf)
    args.output_dir.joinpath("selection.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
