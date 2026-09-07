from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.evaluate_area_ap import Box
from vfm_yolo_distillation.pseudo_label_dataset import (
    DatasetPathError,
    PseudoDatasetSpec,
    box_to_yolo_line,
    label_dir_from_image_dir,
    load_source_spec,
    write_dataset_yaml,
    write_pseudo_labels,
)


def test_label_dir_from_image_dir_replaces_images_segment() -> None:
    label_dir = label_dir_from_image_dir(Path("/data/VisDrone2019-DET-val/images"))

    assert label_dir == Path("/data/VisDrone2019-DET-val/labels")


def test_label_dir_from_image_dir_rejects_non_image_path() -> None:
    with pytest.raises(DatasetPathError, match="images"):
        label_dir_from_image_dir(Path("/data/VisDrone2019-DET-val/JPEGImages"))


def test_load_source_spec_uses_train_and_val_paths(tmp_path: Path) -> None:
    data_yaml = tmp_path / "visdrone.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path / "dataset"),
                "train": "VisDrone2019-DET-train/images",
                "val": "VisDrone2019-DET-val/images",
                "names": ["pedestrian", "car"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    spec = load_source_spec(data_yaml, tmp_path / "pseudo")

    assert spec.train_images == tmp_path / "dataset" / "VisDrone2019-DET-train" / "images"
    assert spec.val_labels == tmp_path / "dataset" / "VisDrone2019-DET-val" / "labels"
    assert spec.names == ("pedestrian", "car")


def test_box_to_yolo_line_clips_to_image_bounds() -> None:
    box = Box(
        image_id="image",
        class_id=3,
        xyxy=(-10.0, 10.0, 110.0, 60.0),
        area=6000.0,
        score=0.8,
    )

    line = box_to_yolo_line(box, width=100, height=100)

    assert line == "3 0.500000 0.350000 1.000000 0.500000"


def test_write_pseudo_labels_creates_empty_files_for_images_without_boxes(tmp_path: Path) -> None:
    image_dir = tmp_path / "source" / "images" / "train"
    image_dir.mkdir(parents=True)
    image_with_box = image_dir / "with_box.jpg"
    image_without_box = image_dir / "without_box.jpg"
    Image.new("RGB", (100, 100)).save(image_with_box)
    Image.new("RGB", (100, 100)).save(image_without_box)
    spec = PseudoDatasetSpec(
        source_data=tmp_path / "source.yaml",
        output_root=tmp_path / "pseudo",
        train_images=image_dir,
        val_images=tmp_path / "source" / "images" / "val",
        val_labels=tmp_path / "source" / "labels" / "val",
        names=("pedestrian", "car"),
    )

    pseudo_boxes, empty_labels = write_pseudo_labels(
        spec,
        (image_with_box, image_without_box),
        [
            Box(
                image_id="with_box",
                class_id=1,
                xyxy=(10.0, 20.0, 30.0, 60.0),
                area=800.0,
                score=0.9,
            )
        ],
    )

    assert pseudo_boxes == 1
    assert empty_labels == 1
    assert (tmp_path / "pseudo" / "labels" / "train" / "with_box.txt").read_text(
        encoding="utf-8"
    ) == "1 0.200000 0.400000 0.200000 0.400000\n"
    assert (tmp_path / "pseudo" / "labels" / "train" / "without_box.txt").read_text(
        encoding="utf-8"
    ) == ""


def test_write_dataset_yaml_points_to_pseudo_root(tmp_path: Path) -> None:
    spec = PseudoDatasetSpec(
        source_data=tmp_path / "source.yaml",
        output_root=tmp_path / "pseudo",
        train_images=tmp_path / "source" / "images" / "train",
        val_images=tmp_path / "source" / "images" / "val",
        val_labels=tmp_path / "source" / "labels" / "val",
        names=("pedestrian", "car"),
    )

    dataset_yaml = write_dataset_yaml(spec)

    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert payload["path"] == str(tmp_path / "pseudo")
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["names"] == ["pedestrian", "car"]
