from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

MAX_PERCENT = 100


class _CoverageSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _EmbeddingRecord:
    image_path: Path
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _CoverageSelection:
    selected_paths: tuple[Path, ...]
    budget_count: int


@dataclass(frozen=True, slots=True)
class _HybridSelectionSettings:
    candidate_scores: Mapping[Path, float]
    score_weight: float


def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in vector))
    if norm <= 0:
        message = "Embedding vector norm must be positive"
        raise _CoverageSelectionError(message)
    return tuple(value / norm for value in vector)


def _coverage_budget_count(total_count: int, budget_ratio: int) -> int:
    if total_count <= 0:
        message = "Image collection must not be empty"
        raise _CoverageSelectionError(message)
    if budget_ratio <= 0 or budget_ratio > MAX_PERCENT:
        message = "Budget ratio must be in the range 1..100"
        raise _CoverageSelectionError(message)
    return max(1, round(total_count * budget_ratio / MAX_PERCENT))


def _select_coverage_subset(
    records: Sequence[_EmbeddingRecord],
    count: int,
    seed: int,
    hybrid: _HybridSelectionSettings | None = None,
) -> _CoverageSelection:
    if not records:
        message = "At least one embedding is required"
        raise _CoverageSelectionError(message)
    if count <= 0 or count > len(records):
        message = "Selection count must fit the embedding collection"
        raise _CoverageSelectionError(message)

    normalized = tuple(
        _EmbeddingRecord(image_path=record.image_path, vector=_normalize_vector(record.vector))
        for record in records
    )
    candidate_scores = _normalized_candidate_scores(normalized, hybrid)
    first_index = _first_selected_index(candidate_scores, seed)
    selected_indices = [first_index]
    selected = {first_index}
    min_distances = [
        _squared_distance(record.vector, normalized[first_index].vector) for record in normalized
    ]

    while len(selected_indices) < count:
        next_index = _next_farthest_index(
            distances=min_distances,
            selected=selected,
            candidate_scores=candidate_scores,
        )
        selected_indices.append(next_index)
        selected.add(next_index)
        next_vector = normalized[next_index].vector
        min_distances = [
            min(distance, _squared_distance(record.vector, next_vector))
            for distance, record in zip(min_distances, normalized, strict=True)
        ]

    selected_paths = tuple(sorted(normalized[index].image_path for index in selected_indices))
    return _CoverageSelection(selected_paths=selected_paths, budget_count=count)


def _normalized_candidate_scores(
    records: Sequence[_EmbeddingRecord],
    hybrid: _HybridSelectionSettings | None,
) -> tuple[float, ...]:
    if hybrid is None or hybrid.score_weight <= 0:
        return tuple(0.0 for _ in records)
    raw_scores = tuple(
        float(hybrid.candidate_scores.get(record.image_path, 0.0))
        for record in records
    )
    max_score = max(raw_scores, default=0.0)
    if max_score <= 0:
        return tuple(0.0 for _ in records)
    return tuple(hybrid.score_weight * score / max_score for score in raw_scores)


def _first_selected_index(candidate_scores: Sequence[float], seed: int) -> int:
    if not candidate_scores or max(candidate_scores) <= 0:
        return seed % len(candidate_scores)
    seed_index = seed % len(candidate_scores)
    candidates = (
        (score, -abs(index - seed_index), -index, index)
        for index, score in enumerate(candidate_scores)
    )
    return max(candidates)[3]


def _records_from_embeddings(
    image_paths: Sequence[Path],
    embeddings: Mapping[Path, Sequence[float]],
) -> tuple[_EmbeddingRecord, ...]:
    records: list[_EmbeddingRecord] = []
    for image_path in image_paths:
        if image_path not in embeddings:
            message = f"Missing embedding for image: {image_path}"
            raise _CoverageSelectionError(message)
        records.append(
            _EmbeddingRecord(image_path=image_path, vector=tuple(embeddings[image_path])),
        )
    return tuple(records)


def _write_split_file(selection: _CoverageSelection, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(path.as_posix() for path in selection.selected_paths)
    output_path.write_text(f"{content}\n", encoding="utf-8")


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        message = "Embedding dimensions must match"
        raise _CoverageSelectionError(message)
    return sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    )


def _next_farthest_index(
    distances: Sequence[float],
    selected: set[int],
    candidate_scores: Sequence[float] | None = None,
) -> int:
    scores = candidate_scores or tuple(0.0 for _ in distances)
    candidates = (
        (distance + scores[index], distance, -index, index)
        for index, distance in enumerate(distances)
        if index not in selected
    )
    try:
        return max(candidates)[3]
    except ValueError as exc:
        message = "No remaining candidate images"
        raise _CoverageSelectionError(message) from exc
