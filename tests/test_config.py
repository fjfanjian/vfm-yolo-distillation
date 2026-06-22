import subprocess
import sys
from pathlib import Path

from scripts.prepare_visdrone import convert_split, write_train_splits
from vfm_yolo_distillation.config import (
    load_experiment_config,
    training_dataset_config_path,
    ultralytics_train_command,
)


def test_load_experiment_config_when_valid_baseline() -> None:
    # Given
    config_path = Path("configs/experiments/yolo_baseline_visdrone.yaml")

    # When
    config = load_experiment_config(config_path)

    # Then
    assert config.name == "yolo26n_baseline_visdrone"
    assert config.student.model == "yolo26n.pt"


def test_ultralytics_train_command_when_valid_baseline() -> None:
    # Given
    config = load_experiment_config(Path("configs/experiments/yolo_baseline_visdrone.yaml"))

    # When
    command = ultralytics_train_command(config)

    # Then
    assert command[:3] == ("yolo", "detect", "train")
    assert "model=yolo26n.pt" in command
    assert "epochs=120" in command


def test_training_dataset_config_path_when_budget_is_full() -> None:
    # Given
    config = load_experiment_config(Path("configs/experiments/yolo_baseline_visdrone.yaml"))

    # When
    dataset_path = training_dataset_config_path(config)

    # Then
    assert dataset_path == Path("configs/datasets/visdrone.yaml")


def test_load_experiment_config_when_stage_is_objectness_pretrain() -> None:
    # Given
    config_path = Path("configs/experiments/dinov3_objectness_pretrain_visdrone_full_imgsz960.yaml")

    # When
    config = load_experiment_config(config_path)

    # Then
    assert config.stage == "objectness_pretrain"
    assert config.training.image_size == 960


def test_ultralytics_train_command_when_budget_is_partial() -> None:
    # Given
    config = load_experiment_config(Path("configs/experiments/yolo_baseline_visdrone_25pct.yaml"))

    # When
    command = ultralytics_train_command(config)

    # Then
    assert "data=configs/datasets/visdrone_25pct.yaml" in command
    assert "name=yolo26n_visdrone_25pct" in command


def test_show_experiment_cli_when_budget_is_partial() -> None:
    # Given
    config_path = "configs/experiments/yolo_baseline_visdrone_25pct.yaml"

    # When
    result = subprocess.run(
        [sys.executable, "scripts/show_experiment.py", config_path],
        check=True,
        capture_output=True,
        text=True,
    )

    # Then
    assert "yolo26n_baseline_visdrone_25pct" in result.stdout
    assert "data=configs/datasets/visdrone_25pct.yaml" in result.stdout
    assert "\n+  " not in result.stdout


def test_prepare_visdrone_when_annotations_exist(tmp_path: Path) -> None:
    # Given
    dataset_root = tmp_path / "visdrone"
    split_root = dataset_root / "VisDrone2019-DET-train"
    image_dir = split_root / "images"
    annotation_dir = split_root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    _write_jpeg(image_dir / "000001.jpg", width=100, height=50)
    (annotation_dir / "000001.txt").write_text(
        "10,5,20,10,1,4,0,0\n0,0,5,5,1,0,0,0\n", encoding="utf-8"
    )

    # When
    converted = convert_split(dataset_root, "VisDrone2019-DET-train")

    # Then
    label_path = split_root / "labels" / "000001.txt"
    assert converted == 1
    assert label_path.read_text(encoding="utf-8") == "3 0.200000 0.200000 0.200000 0.200000\n"


def test_write_train_splits_when_images_exist(tmp_path: Path) -> None:
    # Given
    dataset_root = tmp_path / "visdrone"
    image_dir = dataset_root / "VisDrone2019-DET-train" / "images"
    image_dir.mkdir(parents=True)
    for index in range(4):
        _write_jpeg(image_dir / f"{index:06}.jpg", width=100, height=50)

    # When
    write_train_splits(dataset_root, seed=42)

    # Then
    split_path = dataset_root / "splits" / "visdrone" / "train_50pct.txt"
    assert len(split_path.read_text(encoding="utf-8").splitlines()) == 2


def _write_jpeg(path: Path, width: int, height: int) -> None:
    path.write_bytes(
        b"\xff\xd8"
        b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
        + b"\xff\xd9"
    )
