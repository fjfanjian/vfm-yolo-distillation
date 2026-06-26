# ruff: noqa: D103, INP001, S101
from pathlib import Path

import yaml

from vfm_yolo_distillation.objectness_multilayer import (
    MultiLayerConfigError,
    parse_multilayer_settings,
)


def test_parse_multilayer_settings_when_layers_and_dims_match() -> None:
    # Given
    objectness = {
        "student_layer": 16,
        "student_dim": 64,
        "student_layers": [16, 19],
        "student_dims": [64, 128],
    }

    # When
    settings = parse_multilayer_settings(objectness)

    # Then
    assert tuple((spec.layer, spec.dim) for spec in settings.layers) == ((16, 64), (19, 128))


def test_parse_multilayer_settings_when_only_single_layer_exists() -> None:
    # Given
    objectness = {"student_layer": 19, "student_dim": 128}

    # When
    settings = parse_multilayer_settings(objectness)

    # Then
    assert tuple((spec.layer, spec.dim) for spec in settings.layers) == ((19, 128),)


def test_parse_multilayer_settings_when_layer_dim_counts_differ() -> None:
    # Given
    objectness = {
        "student_layer": 16,
        "student_dim": 64,
        "student_layers": [16, 19],
        "student_dims": [64],
    }

    # When
    try:
        parse_multilayer_settings(objectness)
    except MultiLayerConfigError as exc:
        message = str(exc)
    else:
        message = ""

    # Then
    assert "student_layers and student_dims must have the same length" in message


def test_parse_multilayer_settings_when_using_l16_l19_config() -> None:
    # Given
    config_path = Path(
        "configs/experiments/"
        "dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_l16_l19_seed42.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # When
    settings = parse_multilayer_settings(raw["objectness"])

    # Then
    assert tuple((spec.layer, spec.dim) for spec in settings.layers) == ((16, 64), (19, 128))
    assert raw["outputs"]["name"].endswith("smalltile_l16_l19_lam002_seed42")
