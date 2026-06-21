from dataclasses import dataclass
from pathlib import Path
import random
import struct
import sys


CLASS_COUNT = 10
SPLIT_RATIOS = (5, 10, 25, 50)


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Box:
    left: float
    top: float
    width: float
    height: float
    category: int


def jpeg_size(path: Path) -> ImageSize:
    with path.open("rb") as file:
        if file.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG image: {path}")
        while marker := file.read(2):
            while marker[:1] != b"\xff":
                marker = marker[1:] + file.read(1)
            marker_code = marker[1]
            if marker_code in {0xD8, 0xD9}:
                continue
            length = struct.unpack(">H", file.read(2))[0]
            if marker_code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                file.read(1)
                height, width = struct.unpack(">HH", file.read(4))
                return ImageSize(width=width, height=height)
            file.seek(length - 2, 1)
    raise ValueError(f"Cannot read JPEG size: {path}")


def parse_annotation_line(line: str) -> Box | None:
    parts = line.strip().split(",")
    if len(parts) < 6:
        return None
    left, top, width, height = (float(value) for value in parts[:4])
    category = int(parts[5])
    if category < 1 or category > CLASS_COUNT or width <= 0 or height <= 0:
        return None
    return Box(left=left, top=top, width=width, height=height, category=category)


def yolo_line(box: Box, size: ImageSize) -> str:
    center_x = (box.left + box.width / 2) / size.width
    center_y = (box.top + box.height / 2) / size.height
    width = box.width / size.width
    height = box.height / size.height
    return f"{box.category - 1} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def convert_split(dataset_root: Path, split_dir: str) -> int:
    image_dir = dataset_root / split_dir / "images"
    annotation_dir = dataset_root / split_dir / "annotations"
    label_dir = dataset_root / split_dir / "labels"
    if not annotation_dir.exists():
        return 0
    label_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for image_path in sorted(image_dir.glob("*.jpg")):
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        label_path = label_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            label_path.write_text("", encoding="utf-8")
            continue
        size = jpeg_size(image_path)
        labels = [
            yolo_line(box, size)
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if (box := parse_annotation_line(line)) is not None
        ]
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        converted += 1
    return converted


def write_train_splits(dataset_root: Path, seed: int) -> None:
    image_paths = sorted((dataset_root / "VisDrone2019-DET-train" / "images").glob("*.jpg"))
    rng = random.Random(seed)
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)
    split_dir = dataset_root / "splits" / "visdrone"
    split_dir.mkdir(parents=True, exist_ok=True)
    for ratio in SPLIT_RATIOS:
        count = max(1, round(len(shuffled) * ratio / 100))
        selected = sorted(shuffled[:count])
        content = "\n".join(path.as_posix() for path in selected)
        (split_dir / f"train_{ratio}pct.txt").write_text(f"{content}\n", encoding="utf-8")


def run(argv: list[str]) -> int:
    if len(argv) not in {2, 3}:
        sys.stderr.write("Usage: python scripts/prepare_visdrone.py <dataset-root> [seed]\n")
        return 2
    dataset_root = Path(argv[1]).expanduser().resolve()
    seed = int(argv[2]) if len(argv) == 3 else 42
    train_count = convert_split(dataset_root, "VisDrone2019-DET-train")
    val_count = convert_split(dataset_root, "VisDrone2019-DET-val")
    write_train_splits(dataset_root, seed)
    sys.stdout.write(f"Converted train images: {train_count}\n")
    sys.stdout.write(f"Converted val images: {val_count}\n")
    sys.stdout.write(f"Wrote splits under: {dataset_root / 'splits' / 'visdrone'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv))
