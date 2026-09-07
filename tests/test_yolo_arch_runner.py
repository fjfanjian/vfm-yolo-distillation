from pathlib import Path

from scripts.run_yolo_arch_experiment import RunSettings, training_overrides
from vfm_yolo_distillation.config import load_experiment_config


def test_training_overrides_when_p2_experiment_uses_pretrained_yolo26n() -> None:
    # Given
    config = load_experiment_config(
        Path("configs/experiments/yolo_arch_p2_visdrone_10pct_imgsz960_seed42.yaml"),
    )
    settings = RunSettings(pretrained="yolo26n.pt")

    # When
    overrides = training_overrides(config, settings)

    # Then
    assert overrides["model"] == "configs/models/yolo26n-p2.yaml"
    assert overrides["pretrained"] == "yolo26n.pt"
    assert overrides["data"] == "configs/datasets/visdrone_10pct.yaml"
    assert overrides["project"] == "/home/fj/vfm-yolo-distillation/runs/yolo_arch"
    assert overrides["name"] == "yolo26n_p2_visdrone_10pct_imgsz960_seed42"
