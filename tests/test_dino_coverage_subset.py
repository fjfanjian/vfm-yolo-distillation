from pathlib import Path

import pytest

from vfm_yolo_distillation.dino_coverage_subset import (
    _coverage_budget_count,
    _CoverageSelectionError,
    _EmbeddingRecord,
    _HybridSelectionSettings,
    _records_from_embeddings,
    _select_coverage_subset,
    _write_split_file,
)


def test_budget_count_when_ratio_is_partial() -> None:
    # Given
    total_count = 6471

    # When
    count = _coverage_budget_count(total_count=total_count, budget_ratio=10)

    # Then
    assert count == 647


def test_select_farthest_first_when_embeddings_cover_extremes() -> None:
    # Given
    records = (
        _EmbeddingRecord(Path("/data/a.jpg"), (1.0, 0.0)),
        _EmbeddingRecord(Path("/data/b.jpg"), (0.9, 0.1)),
        _EmbeddingRecord(Path("/data/c.jpg"), (0.0, 1.0)),
        _EmbeddingRecord(Path("/data/d.jpg"), (0.1, 0.9)),
    )

    # When
    selection = _select_coverage_subset(records=records, count=2, seed=0)

    # Then
    assert selection.budget_count == 2
    assert selection.selected_paths == (Path("/data/a.jpg"), Path("/data/c.jpg"))


def test_select_hybrid_when_small_object_score_breaks_coverage_tie() -> None:
    # Given
    records = (
        _EmbeddingRecord(Path("/data/a.jpg"), (1.0, 0.0)),
        _EmbeddingRecord(Path("/data/b.jpg"), (0.95, 0.05)),
        _EmbeddingRecord(Path("/data/c.jpg"), (0.0, 1.0)),
    )

    # When
    selection = _select_coverage_subset(
        records=records,
        count=2,
        seed=0,
        hybrid=_HybridSelectionSettings(
            candidate_scores={Path("/data/b.jpg"): 10.0},
            score_weight=1.0,
        ),
    )

    # Then
    assert Path("/data/b.jpg") in selection.selected_paths


def test_records_from_embeddings_when_image_is_missing_embedding() -> None:
    # Given
    image_paths = (Path("/data/a.jpg"), Path("/data/b.jpg"))
    embeddings = {Path("/data/a.jpg"): (1.0, 0.0)}

    # When / Then
    with pytest.raises(_CoverageSelectionError):
        _records_from_embeddings(image_paths=image_paths, embeddings=embeddings)


def test_write_split_file_when_selection_exists(tmp_path: Path) -> None:
    # Given
    output_path = tmp_path / "splits" / "train_10pct_dinov3_coverage_seed42.txt"
    selection = _select_coverage_subset(
        records=(
            _EmbeddingRecord(Path("/data/a.jpg"), (1.0, 0.0)),
            _EmbeddingRecord(Path("/data/b.jpg"), (0.0, 1.0)),
        ),
        count=1,
        seed=42,
    )

    # When
    _write_split_file(selection=selection, output_path=output_path)

    # Then
    assert output_path.read_text(encoding="utf-8").endswith(".jpg\n")
