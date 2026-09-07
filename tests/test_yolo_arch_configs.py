from pathlib import Path

import yaml


def test_yolo26n_p2_config_when_loaded_has_four_detect_scales() -> None:
    # Given
    config_path = Path("configs/models/yolo26n-p2.yaml")

    # When
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Then
    assert isinstance(payload, dict)
    head = payload["head"]
    assert isinstance(head, list)
    detect_layer = head[-1]
    assert isinstance(detect_layer, list)
    assert detect_layer[2] == "Detect"
    assert detect_layer[0] == [19, 22, 25, 28]


def test_yolo26n_p2_experiment_when_loaded_points_to_p2_model() -> None:
    # Given
    config_path = Path("configs/experiments/yolo_arch_p2_visdrone_10pct_imgsz960_seed42.yaml")

    # When
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Then
    assert isinstance(payload, dict)
    assert payload["stage"] == "supervised_baseline"
    assert payload["student"]["model"] == "configs/models/yolo26n-p2.yaml"
    assert payload["training"]["image_size"] == 960
    assert payload["outputs"]["name"] == "yolo26n_p2_visdrone_10pct_imgsz960_seed42"
