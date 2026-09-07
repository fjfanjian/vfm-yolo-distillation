from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml
from PIL import Image

IMAGE_SUFFIXES: Final = {".jpg", ".jpeg", ".png"}


class AreaBucket(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True, slots=True)
class DetectionBox:
    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    area: float
    score: float


@dataclass(frozen=True, slots=True)
class PseudoCandidate:
    image_id: str
    image_path: str
    class_id: int
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    area: float
    area_bucket: AreaBucket

    def as_box(self) -> DetectionBox:
        return DetectionBox(
            image_id=self.image_id,
            class_id=self.class_id,
            xyxy=(self.x1, self.y1, self.x2, self.y2),
            area=self.area,
            score=self.score,
        )


@dataclass(frozen=True, slots=True)
class ThresholdRecord:
    area: AreaBucket | None
    score: float


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    thresholds: tuple[ThresholdRecord, ...]

    @classmethod
    def uniform(cls, score: float) -> ThresholdPolicy:
        return cls((ThresholdRecord(None, score),))

    @classmethod
    def area_sensitive(cls, small: float, medium: float, large: float) -> ThresholdPolicy:
        return cls(
            (
                ThresholdRecord(AreaBucket.SMALL, small),
                ThresholdRecord(AreaBucket.MEDIUM, medium),
                ThresholdRecord(AreaBucket.LARGE, large),
            )
        )

    def threshold_for(self, area_bucket: AreaBucket) -> float:
        fallback = 1.0
        for threshold in self.thresholds:
            if threshold.area is None:
                fallback = threshold.score
            if threshold.area is area_bucket:
                return threshold.score
        return fallback


@dataclass(frozen=True, slots=True)
class CountRecord:
    name: str
    count: int


@dataclass(frozen=True, slots=True)
class MixedDatasetRequest:
    output_root: Path
    variant: str
    names: tuple[str, ...]
    train_images: tuple[Path, ...]
    labeled_images: tuple[Path, ...]
    val_images: tuple[Path, ...]
    candidates: tuple[PseudoCandidate, ...]
    threshold_policy: ThresholdPolicy


@dataclass(frozen=True, slots=True)
class MixedDatasetSummary:
    output_root: str
    variant: str
    train_images: int
    labeled_gt_images: int
    unlabeled_pseudo_images: int
    train_label_files: int
    pseudo_boxes: int
    empty_label_files: int
    pseudo_empty_label_files: int
    val_images: int
    val_label_files: int
    per_area_pseudo_count: tuple[CountRecord, ...]
    per_class_pseudo_count: tuple[CountRecord, ...]


@dataclass(frozen=True, slots=True)
class AuditStats:
    gt_count: int
    prediction_count: int
    matched_tp: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    fp_rate: float
    fn_rate: float


@dataclass(frozen=True, slots=True)
class AreaAuditStats:
    area: AreaBucket
    stats: AuditStats


@dataclass(frozen=True, slots=True)
class QualityAuditSummary:
    variant: str
    iou_threshold: float
    thresholds: tuple[ThresholdRecord, ...]
    overall: AuditStats
    by_area: tuple[AreaAuditStats, ...]


def area_bucket_for_area(area: float) -> AreaBucket:
    if area < 32.0 * 32.0:
        return AreaBucket.SMALL
    if area < 96.0 * 96.0:
        return AreaBucket.MEDIUM
    return AreaBucket.LARGE


def candidate_from_box(
    image_path: Path,
    class_names: tuple[str, ...],
    box: DetectionBox,
) -> PseudoCandidate:
    return PseudoCandidate(
        image_id=box.image_id,
        image_path=str(image_path),
        class_id=box.class_id,
        class_name=class_names[box.class_id],
        score=box.score,
        x1=box.xyxy[0],
        y1=box.xyxy[1],
        x2=box.xyxy[2],
        y2=box.xyxy[3],
        area=box.area,
        area_bucket=area_bucket_for_area(box.area),
    )


def write_candidates_csv(path: Path, candidates: tuple[PseudoCandidate, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id",
                "image_path",
                "class_id",
                "class_name",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
                "area",
                "area_bucket",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.image_id,
                    candidate.image_path,
                    candidate.class_id,
                    candidate.class_name,
                    candidate.score,
                    candidate.x1,
                    candidate.y1,
                    candidate.x2,
                    candidate.y2,
                    candidate.area,
                    candidate.area_bucket.value,
                ]
            )


def read_candidates_csv(path: Path) -> tuple[PseudoCandidate, ...]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return tuple(_candidate_from_row(row) for row in reader)


def read_image_source(source: Path, root: Path) -> tuple[Path, ...]:
    if source.is_file():
        return tuple(_resolve_split_line(line, root) for line in _read_nonempty_lines(source))
    if source.is_dir():
        return tuple(
            sorted(
                path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
            )
        )
    raise FileNotFoundError(source)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Image path must contain an images segment: {image_path}")
    index = parts.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_yolo_boxes(image_path: Path) -> tuple[DetectionBox, ...]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return ()
    with Image.open(image_path) as image:
        width, height = image.size
    return tuple(
        _yolo_line_to_box(line, image_path.stem, width, height)
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def filter_candidates(
    candidates: tuple[PseudoCandidate, ...],
    threshold_policy: ThresholdPolicy,
) -> tuple[PseudoCandidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if candidate.score >= threshold_policy.threshold_for(candidate.area_bucket)
    )


def write_mixed_dataset(request: MixedDatasetRequest) -> MixedDatasetSummary:
    _reset_dataset_dirs(request.output_root)
    train_image_dir = request.output_root / "images" / "train"
    val_image_dir = request.output_root / "images" / "val"
    train_label_dir = request.output_root / "labels" / "train"
    val_label_dir = request.output_root / "labels" / "val"
    _mkdirs(train_image_dir, val_image_dir, train_label_dir, val_label_dir)
    labeled_ids = {path.stem for path in request.labeled_images}
    filtered = filter_candidates(request.candidates, request.threshold_policy)
    indexed = _index_candidates(filtered)
    pseudo_boxes = 0
    empty_labels = 0
    pseudo_empty_labels = 0
    for image_path in request.train_images:
        _link_or_copy(image_path, train_image_dir / image_path.name)
        if image_path.stem in labeled_ids:
            _copy_gt_label(image_path, train_label_dir / f"{image_path.stem}.txt")
            if not (train_label_dir / f"{image_path.stem}.txt").read_text(encoding="utf-8").strip():
                empty_labels += 1
            continue
        line_count = _write_candidate_label(
            image_path,
            indexed.get(image_path.stem, ()),
            train_label_dir / f"{image_path.stem}.txt",
        )
        pseudo_boxes += line_count
        if line_count == 0:
            empty_labels += 1
            pseudo_empty_labels += 1
    val_label_count = 0
    for image_path in request.val_images:
        _link_or_copy(image_path, val_image_dir / image_path.name)
        _copy_gt_label(image_path, val_label_dir / f"{image_path.stem}.txt")
        val_label_count += 1
    _write_dataset_yaml(request.output_root, request.names)
    summary = MixedDatasetSummary(
        output_root=str(request.output_root),
        variant=request.variant,
        train_images=len(request.train_images),
        labeled_gt_images=len(labeled_ids),
        unlabeled_pseudo_images=len(request.train_images) - len(labeled_ids),
        train_label_files=len(request.train_images),
        pseudo_boxes=pseudo_boxes,
        empty_label_files=empty_labels,
        pseudo_empty_label_files=pseudo_empty_labels,
        val_images=len(request.val_images),
        val_label_files=val_label_count,
        per_area_pseudo_count=_count_by_area(filtered),
        per_class_pseudo_count=_count_by_class(filtered, request.names),
    )
    (request.output_root / "pseudo_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def audit_candidates(
    variant: str,
    candidates: tuple[PseudoCandidate, ...],
    ground_truths: tuple[DetectionBox, ...],
    threshold_policy: ThresholdPolicy,
    iou_threshold: float,
) -> QualityAuditSummary:
    predictions = tuple(
        candidate.as_box()
        for candidate in filter_candidates(candidates, threshold_policy)
    )
    matched_gt, matched_pred = _greedy_match(ground_truths, predictions, iou_threshold)
    overall = _stats(
        gt_count=len(ground_truths),
        prediction_count=len(predictions),
        true_positive=len(matched_gt),
        false_positive=len(predictions) - len(matched_pred),
        false_negative=len(ground_truths) - len(matched_gt),
    )
    by_area = tuple(
        _area_stats(bucket, ground_truths, predictions, matched_gt, matched_pred)
        for bucket in AreaBucket
    )
    return QualityAuditSummary(
        variant=variant,
        iou_threshold=iou_threshold,
        thresholds=threshold_policy.thresholds,
        overall=overall,
        by_area=by_area,
    )


def write_quality_summary(path: Path, summary: QualityAuditSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_from_row(row: dict[str, str]) -> PseudoCandidate:
    return PseudoCandidate(
        image_id=row["image_id"],
        image_path=row["image_path"],
        class_id=int(row["class_id"]),
        class_name=row["class_name"],
        score=float(row["score"]),
        x1=float(row["x1"]),
        y1=float(row["y1"]),
        x2=float(row["x2"]),
        y2=float(row["y2"]),
        area=float(row["area"]),
        area_bucket=AreaBucket(row["area_bucket"]),
    )


def _read_nonempty_lines(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _resolve_split_line(line: str, root: Path) -> Path:
    candidate = Path(line.replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    rooted = root / candidate
    if rooted.exists():
        return rooted
    return root / "VisDrone2019-DET-train" / "images" / candidate.name


def _yolo_line_to_box(line: str, image_id: str, width: int, height: int) -> DetectionBox:
    class_text, x_text, y_text, w_text, h_text = line.split()[:5]
    box_width = float(w_text) * width
    box_height = float(h_text) * height
    center_x = float(x_text) * width
    center_y = float(y_text) * height
    left = center_x - box_width / 2.0
    top = center_y - box_height / 2.0
    return DetectionBox(
        image_id=image_id,
        class_id=int(class_text),
        xyxy=(left, top, left + box_width, top + box_height),
        area=box_width * box_height,
        score=1.0,
    )


def _reset_dataset_dirs(output_root: Path) -> None:
    for child in ("images", "labels"):
        path = output_root / child
        if path.exists():
            shutil.rmtree(path)
    for file_name in ("data.yaml", "pseudo_summary.json"):
        path = output_root / file_name
        if path.exists():
            path.unlink()


def _mkdirs(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _copy_gt_label(image_path: Path, target: Path) -> None:
    source = label_path_for_image(image_path)
    if source.exists():
        _link_or_copy(source, target)
        return
    target.write_text("", encoding="utf-8")


def _write_candidate_label(
    image_path: Path,
    candidates: tuple[PseudoCandidate, ...],
    target: Path,
) -> int:
    with Image.open(image_path) as image:
        width, height = image.size
    lines = [
        line
        for candidate in candidates
        if (line := _candidate_to_yolo_line(candidate, width, height)) is not None
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _candidate_to_yolo_line(candidate: PseudoCandidate, width: int, height: int) -> str | None:
    left = max(0.0, min(float(width), candidate.x1))
    top = max(0.0, min(float(height), candidate.y1))
    right = max(0.0, min(float(width), candidate.x2))
    bottom = max(0.0, min(float(height), candidate.y2))
    if right <= left or bottom <= top:
        return None
    box_width = right - left
    box_height = bottom - top
    center_x = left + box_width / 2.0
    center_y = top + box_height / 2.0
    return (
        f"{candidate.class_id} "
        f"{center_x / width:.6f} {center_y / height:.6f} "
        f"{box_width / width:.6f} {box_height / height:.6f}"
    )


def _write_dataset_yaml(output_root: Path, names: tuple[str, ...]) -> None:
    payload = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": list(names),
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _index_candidates(
    candidates: tuple[PseudoCandidate, ...],
) -> dict[str, tuple[PseudoCandidate, ...]]:
    indexed: dict[str, list[PseudoCandidate]] = {}
    for candidate in candidates:
        indexed.setdefault(candidate.image_id, []).append(candidate)
    return {image_id: tuple(items) for image_id, items in indexed.items()}


def _count_by_area(candidates: tuple[PseudoCandidate, ...]) -> tuple[CountRecord, ...]:
    return tuple(
        CountRecord(
            bucket.value,
            sum(1 for candidate in candidates if candidate.area_bucket is bucket),
        )
        for bucket in AreaBucket
    )


def _count_by_class(
    candidates: tuple[PseudoCandidate, ...],
    names: tuple[str, ...],
) -> tuple[CountRecord, ...]:
    return tuple(
        CountRecord(name, sum(1 for candidate in candidates if candidate.class_id == class_id))
        for class_id, name in enumerate(names)
    )


def _greedy_match(
    ground_truths: tuple[DetectionBox, ...],
    predictions: tuple[DetectionBox, ...],
    iou_threshold: float,
) -> tuple[set[int], set[int]]:
    pairs = [
        (overlap, gt_index, pred_index)
        for gt_index, gt_box in enumerate(ground_truths)
        for pred_index, pred_box in enumerate(predictions)
        if gt_box.image_id == pred_box.image_id and gt_box.class_id == pred_box.class_id
        if (overlap := _iou(gt_box, pred_box)) >= iou_threshold
    ]
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in sorted(pairs, reverse=True):
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
    return matched_gt, matched_pred


def _iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
    left = max(box_a.xyxy[0], box_b.xyxy[0])
    top = max(box_a.xyxy[1], box_b.xyxy[1])
    right = min(box_a.xyxy[2], box_b.xyxy[2])
    bottom = min(box_a.xyxy[3], box_b.xyxy[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = box_a.area + box_b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _area_stats(
    bucket: AreaBucket,
    ground_truths: tuple[DetectionBox, ...],
    predictions: tuple[DetectionBox, ...],
    matched_gt: set[int],
    matched_pred: set[int],
) -> AreaAuditStats:
    gt_indices = {
        index
        for index, box in enumerate(ground_truths)
        if area_bucket_for_area(box.area) is bucket
    }
    pred_indices = {
        index
        for index, box in enumerate(predictions)
        if area_bucket_for_area(box.area) is bucket
    }
    true_positive = len(gt_indices & matched_gt)
    false_positive = len(pred_indices - matched_pred)
    false_negative = len(gt_indices - matched_gt)
    return AreaAuditStats(
        area=bucket,
        stats=_stats(
            gt_count=len(gt_indices),
            prediction_count=len(pred_indices),
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        ),
    )


def _stats(
    gt_count: int,
    prediction_count: int,
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> AuditStats:
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / gt_count if gt_count else 0.0
    fp_rate = false_positive / prediction_count if prediction_count else 0.0
    fn_rate = false_negative / gt_count if gt_count else 0.0
    return AuditStats(
        gt_count=gt_count,
        prediction_count=prediction_count,
        matched_tp=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        fp_rate=fp_rate,
        fn_rate=fn_rate,
    )
