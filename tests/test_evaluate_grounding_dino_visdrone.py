# ruff: noqa: D103, INP001, S101
from vfm_yolo_distillation import grounding_dino_eval as gdino


def test_class_id_from_label_maps_prompt_aliases() -> None:
    # Given / When / Then
    assert gdino.class_id_from_label("a pedestrian") == 0
    assert gdino.class_id_from_label("person") == 1
    assert gdino.class_id_from_label("awning tricycle") == 7
    assert gdino.class_id_from_label("motorcycle") == 9


def test_class_id_from_label_returns_none_for_unknown_label() -> None:
    # Given / When / Then
    assert gdino.class_id_from_label("traffic light") is None


def test_prompt_text_uses_period_separated_classes() -> None:
    # Given / When
    text = gdino.prompt_text()

    # Then
    assert text.endswith(".")
    assert "car. van. truck" in text
