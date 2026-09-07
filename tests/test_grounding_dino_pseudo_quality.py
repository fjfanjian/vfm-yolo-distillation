from __future__ import annotations

from pathlib import Path

from PIL import Image

from vfm_yolo_distillation.pseudo_label_quality import (
    AreaBucket,
    DetectionBox,
    MixedDatasetRequest,
    PseudoCandidate,
    ThresholdPolicy,
    area_bucket_for_area,
    audit_candidates,
    read_candidates_csv,
    read_image_source,
    write_candidates_csv,
    write_mixed_dataset,
)


def test_area_bucket_for_area_uses_coco_thresholds() -> None:
    assert area_bucket_for_area(1023.0) is AreaBucket.SMALL
    assert area_bucket_for_area(1024.0) is AreaBucket.MEDIUM
    assert area_bucket_for_area(9216.0) is AreaBucket.LARGE


def test_threshold_policy_when_variant_is_area_sensitive() -> None:
    policy = ThresholdPolicy.area_sensitive(small=0.25, medium=0.30, large=0.40)

    assert policy.threshold_for(AreaBucket.SMALL) == 0.25
    assert policy.threshold_for(AreaBucket.MEDIUM) == 0.30
    assert policy.threshold_for(AreaBucket.LARGE) == 0.40


def test_candidates_csv_preserves_score_and_area_bucket(tmp_path: Path) -> None:
    candidate = PseudoCandidate(
        image_id="000001",
        image_path="/data/images/000001.jpg",
        class_id=3,
        class_name="car",
        score=0.42,
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=60.0,
        area=800.0,
        area_bucket=AreaBucket.SMALL,
    )

    csv_path = tmp_path / "raw_predictions.csv"
    write_candidates_csv(csv_path, (candidate,))

    assert read_candidates_csv(csv_path) == (candidate,)


def test_read_image_source_supports_split_files_and_directories(tmp_path: Path) -> None:
    root = tmp_path / "visdrone"
    train_dir = root / "VisDrone2019-DET-train" / "images"
    train_dir.mkdir(parents=True)
    image_a = train_dir / "000001.jpg"
    image_b = train_dir / "000002.jpg"
    _write_image(image_a)
    _write_image(image_b)
    split = root / "splits" / "visdrone" / "train_10pct.txt"
    split.parent.mkdir(parents=True)
    split.write_text("VisDrone2019-DET-train/images/000002.jpg\n", encoding="utf-8")

    assert read_image_source(split, root) == (image_b,)
    assert read_image_source(train_dir, root) == (image_a, image_b)


def test_write_mixed_dataset_keeps_labeled_gt_and_uses_unlabeled_pseudo(tmp_path: Path) -> None:
    root = tmp_path / "visdrone"
    train_images = root / "VisDrone2019-DET-train" / "images"
    train_labels = root / "VisDrone2019-DET-train" / "labels"
    val_images = root / "VisDrone2019-DET-val" / "images"
    val_labels = root / "VisDrone2019-DET-val" / "labels"
    for directory in (train_images, train_labels, val_images, val_labels):
        directory.mkdir(parents=True)
    labeled = train_images / "labeled.jpg"
    unlabeled = train_images / "unlabeled.jpg"
    no_pseudo = train_images / "no_pseudo.jpg"
    val = val_images / "val.jpg"
    for image_path in (labeled, unlabeled, no_pseudo, val):
        _write_image(image_path)
    gt_text = "3 0.500000 0.500000 0.200000 0.200000\n"
    (train_labels / "labeled.txt").write_text(gt_text, encoding="utf-8")
    (val_labels / "val.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    request = MixedDatasetRequest(
        output_root=tmp_path / "pseudo" / "gdino_quality_mix10_s030_seed42",
        variant="s030",
        names=("pedestrian", "people", "bicycle", "car"),
        train_images=(labeled, unlabeled, no_pseudo),
        labeled_images=(labeled,),
        val_images=(val,),
        candidates=(
            PseudoCandidate(
                image_id="unlabeled",
                image_path=str(unlabeled),
                class_id=3,
                class_name="car",
                score=0.31,
                x1=10.0,
                y1=20.0,
                x2=30.0,
                y2=60.0,
                area=800.0,
                area_bucket=AreaBucket.SMALL,
            ),
        ),
        threshold_policy=ThresholdPolicy.uniform(0.30),
    )

    summary = write_mixed_dataset(request)

    label_root = request.output_root / "labels" / "train"
    assert summary.train_images == 3
    assert summary.labeled_gt_images == 1
    assert summary.unlabeled_pseudo_images == 2
    assert summary.pseudo_boxes == 1
    assert (label_root / "labeled.txt").read_text(encoding="utf-8") == gt_text
    assert (label_root / "unlabeled.txt").read_text(
        encoding="utf-8"
    ) == "3 0.200000 0.400000 0.200000 0.400000\n"
    assert (label_root / "no_pseudo.txt").read_text(encoding="utf-8") == ""


def test_audit_candidates_reports_tp_fp_and_fn_by_threshold() -> None:
    ground_truths = (
        DetectionBox("image", 3, (10.0, 10.0, 30.0, 30.0), 400.0, 1.0),
        DetectionBox("image", 0, (60.0, 60.0, 80.0, 80.0), 400.0, 1.0),
    )
    candidates = (
        PseudoCandidate(
            image_id="image",
            image_path="/data/image.jpg",
            class_id=3,
            class_name="car",
            score=0.90,
            x1=11.0,
            y1=11.0,
            x2=29.0,
            y2=29.0,
            area=324.0,
            area_bucket=AreaBucket.SMALL,
        ),
        PseudoCandidate(
            image_id="image",
            image_path="/data/image.jpg",
            class_id=3,
            class_name="car",
            score=0.80,
            x1=70.0,
            y1=10.0,
            x2=90.0,
            y2=30.0,
            area=400.0,
            area_bucket=AreaBucket.SMALL,
        ),
    )

    summary = audit_candidates(
        variant="s030",
        candidates=candidates,
        ground_truths=ground_truths,
        threshold_policy=ThresholdPolicy.uniform(0.30),
        iou_threshold=0.5,
    )

    assert summary.overall.gt_count == 2
    assert summary.overall.prediction_count == 2
    assert summary.overall.matched_tp == 1
    assert summary.overall.false_positive == 1
    assert summary.overall.false_negative == 1
    assert summary.overall.fp_rate == 0.5
    assert summary.overall.fn_rate == 0.5


def _write_image(path: Path) -> None:
    Image.new("RGB", (100, 100)).save(path)
