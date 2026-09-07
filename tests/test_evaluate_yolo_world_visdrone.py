# ruff: noqa: D103, INP001, S101
from scripts import evaluate_yolo_world_visdrone as yolo_world


def test_world_prompts_keep_visdrone_class_order() -> None:
    assert yolo_world.WORLD_PROMPTS[0] == "pedestrian"
    assert yolo_world.WORLD_PROMPTS[1] == "person"
    assert yolo_world.WORLD_PROMPTS[7] == "awning tricycle"
    assert yolo_world.WORLD_PROMPTS[9] == "motorcycle"


def test_world_prompts_match_visdrone_class_count() -> None:
    assert len(yolo_world.WORLD_PROMPTS) == len(yolo_world.CLASS_NAMES)
