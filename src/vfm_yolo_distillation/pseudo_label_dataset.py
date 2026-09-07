from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml
from PIL import Image

from vfm_yolo_distillation.grounding_dino_coco import CLASS_NAMES

if TYPE_CHECKING:
    from scripts.evaluate_area_ap import Box


IMAGE_SUFFIXES: Final = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class PseudoDatasetSpec:
    source_data: Path
    output_root: Path
    train_images: Path
    val_images: Path
    val_labels: Path
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PseudoDatasetSummary:
    output_root: str
    train_images: int
    train_label_files: int
    pseudo_boxes: int
    empty_label_files: int
    val_images: int
    val_label_files: int


def label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = image_dir.parts
    if "images" not in parts:
        raise DatasetPathError(image_dir)
    image_index = parts.index("images")
    return Path(*parts[:image_index], "labels", *parts[image_index + 1 :])


@dataclass(frozen=True, slots=True)
class DatasetPathError(RuntimeError):
    image_dir: Path

    def __str__(self) -> str:
        return f"Expected image directory path to contain an 'images' segment: {self.image_dir}"


def load_source_spec(data_yaml: Path, output_root: Path) -> PseudoDatasetSpec:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    train_images = root / payload["train"]
    val_images = root / payload["val"]
    names = payload["names"]
    return PseudoDatasetSpec(
        source_data=data_yaml,
        output_root=output_root,
        train_images=train_images,
        val_images=val_images,
        val_labels=label_dir_from_image_dir(val_images),
        names=tuple(names),
    )


def list_images(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    )


def box_to_yolo_line(box: Box, width: int, height: int) -> str | None:
    left = max(0.0, min(float(width), box.xyxy[0]))
    top = max(0.0, min(float(height), box.xyxy[1]))
    right = max(0.0, min(float(width), box.xyxy[2]))
    bottom = max(0.0, min(float(height), box.xyxy[3]))
    if right <= left or bottom <= top:
        return None
    box_width = right - left
    box_height = bottom - top
    center_x = left + box_width / 2.0
    center_y = top + box_height / 2.0
    return (
        f"{box.class_id} "
        f"{center_x / width:.6f} {center_y / height:.6f} "
        f"{box_width / width:.6f} {box_height / height:.6f}"
    )


def write_pseudo_labels(
    spec: PseudoDatasetSpec,
    images: tuple[Path, ...],
    predictions: list[Box],
) -> tuple[int, int]:
    label_dir = spec.output_root / "labels" / "train"
    label_dir.mkdir(parents=True, exist_ok=True)
    indexed: dict[str, list[Box]] = {}
    for prediction in predictions:
        indexed.setdefault(prediction.image_id, []).append(prediction)
    box_count = 0
    empty_count = 0
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        lines = [
            line
            for box in indexed.get(image_path.stem, [])
            if (line := box_to_yolo_line(box, width, height)) is not None
        ]
        if not lines:
            empty_count += 1
        box_count += len(lines)
        label_text = "\n".join(lines) + ("\n" if lines else "")
        (label_dir / f"{image_path.stem}.txt").write_text(label_text, encoding="utf-8")
    return box_count, empty_count


def link_images(images: tuple[Path, ...], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        target = output_dir / image_path.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(image_path)


def link_labels(label_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        target = output_dir / label_path.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(label_path)
        count += 1
    return count


def write_dataset_yaml(spec: PseudoDatasetSpec) -> Path:
    path = spec.output_root / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(spec.output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": list(spec.names),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_summary(spec: PseudoDatasetSpec, summary: PseudoDatasetSummary) -> Path:
    path = spec.output_root / "pseudo_summary.json"
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def class_names() -> tuple[str, ...]:
    return tuple(CLASS_NAMES)
