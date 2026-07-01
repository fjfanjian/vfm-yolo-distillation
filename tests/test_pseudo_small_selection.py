from __future__ import annotations

from pathlib import Path

import pytest

from vfm_yolo_distillation.pseudo_small_selection import (
    PseudoBox,
    ScoreSettings,
    budget_count,
    normalized_area,
    score_image,
    select_top,
)


def _box(image: str, class_id: int, confidence: float, xyxy: tuple[float, float, float, float]) -> PseudoBox:
    return PseudoBox(
        image=image,
        class_id=class_id,
        confidence=confidence,
        xyxy=xyxy,
        image_width=960,
        image_height=960,
    )


def _settings(mode: str = "count") -> ScoreSettings:
    return ScoreSettings(
        small_area_px=1024.0,
        score_image_size=960,
        class_diversity_weight=1.0,
        scale_diversity_weight=0.5,
        box_score_mode=mode,
    )


def test_score_image_when_count_mode_uses_small_box_count_and_diversity() -> None:
    boxes = [
        _box("a.jpg", 0, 0.9, (0.0, 0.0, 16.0, 16.0)),
        _box("a.jpg", 1, 0.8, (20.0, 20.0, 48.0, 48.0)),
        _box("a.jpg", 2, 0.7, (0.0, 0.0, 200.0, 200.0)),
    ]

    score = score_image("a.jpg", boxes, _settings())

    assert score.small_boxes == 2
    assert score.small_classes == 2
    assert score.scale_bins == 2
    assert score.score == pytest.approx(5.0)


def test_score_image_when_confidence_mode_uses_small_box_confidence_sum() -> None:
    boxes = [
        _box("a.jpg", 0, 0.9, (0.0, 0.0, 16.0, 16.0)),
        _box("a.jpg", 1, 0.8, (20.0, 20.0, 48.0, 48.0)),
    ]

    score = score_image("a.jpg", boxes, _settings("confidence"))

    assert score.score == pytest.approx(4.7)
    assert score.mean_small_confidence == pytest.approx(0.85)


def test_select_top_returns_selected_paths_in_stable_split_order() -> None:
    scores = [
        score_image("b.jpg", [_box("b.jpg", 0, 0.9, (0.0, 0.0, 16.0, 16.0))], _settings()),
        score_image("a.jpg", [_box("a.jpg", 0, 0.9, (0.0, 0.0, 16.0, 16.0))], _settings()),
        score_image("c.jpg", [], _settings()),
    ]

    selected = select_top(scores, 2)

    assert [Path(item.image).name for item in selected] == ["a.jpg", "b.jpg"]


def test_normalized_area_uses_score_image_size_not_original_size() -> None:
    box = PseudoBox("wide.jpg", 0, 0.9, (0.0, 0.0, 100.0, 50.0), 2000, 1000)

    assert normalized_area(box, 1000) == pytest.approx(2500.0)


def test_budget_count_rejects_non_positive_ratio() -> None:
    with pytest.raises(ValueError, match="budget_ratio"):
        budget_count(100, 0)
