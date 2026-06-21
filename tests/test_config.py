from pathlib import Path

from vfm_yolo_distillation.config import load_experiment_config, ultralytics_train_command


def test_load_experiment_config_when_valid_baseline() -> None:
    # Given
    config_path = Path("configs/experiments/yolo_baseline_visdrone.yaml")

    # When
    config = load_experiment_config(config_path)

    # Then
    assert config.name == "yolo_baseline_visdrone"
    assert config.student.model == "yolo11s.pt"


def test_ultralytics_train_command_when_valid_baseline() -> None:
    # Given
    config = load_experiment_config(Path("configs/experiments/yolo_baseline_visdrone.yaml"))

    # When
    command = ultralytics_train_command(config)

    # Then
    assert command[:3] == ("yolo", "detect", "train")
    assert "model=yolo11s.pt" in command
    assert "epochs=120" in command
