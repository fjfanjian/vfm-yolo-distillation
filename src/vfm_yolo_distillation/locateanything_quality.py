from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from vfm_yolo_distillation.pseudo_label_quality import (
    DetectionBox,
    PseudoCandidate,
    area_bucket_for_area,
)

BOX_PATTERN: Final = re.compile(
    r"(?:<ref>(?P<label>.*?)</ref>\s*)?<box>(?P<body>(?:<\d+>\s*){4})</box>",
    re.IGNORECASE | re.DOTALL,
)
COORD_PATTERN: Final = re.compile(r"<(\d+)>")
TRIM_CHARS: Final = " \t\r\n.,;:()[]{}\"'"
UNKNOWN_LABEL: Final = "__unknown__"


class LocateAnythingRawResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    image: str | None = None
    image_path: str | None = None
    query: str | None = None
    prompt_mode: str | None = None
    answer: str | None = None
    raw_response: str | None = None
    response: str | None = None
    output: str | None = None

    @property
    def resolved_image_path(self) -> Path:
        image_text = self.image_path or self.image
        if image_text is None:
            message = "LocateAnything result has no image path."
            raise LocateAnythingParseError(message)
        return Path(image_text)

    @property
    def resolved_answer(self) -> str:
        answer = self.answer or self.raw_response or self.response or self.output
        if answer is None:
            message = "LocateAnything result has no answer text."
            raise LocateAnythingParseError(message)
        return answer


@dataclass(frozen=True, slots=True)
class LocateAnythingParseStats:
    records: int
    records_with_boxes: int
    parsed_boxes: int
    skipped_unknown_label: int
    skipped_invalid_box: int


class LocateAnythingParseError(RuntimeError):
    pass


def class_aliases(class_names: tuple[str, ...]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for class_id, class_name in enumerate(class_names):
        aliases[_canonical_label(class_name)] = class_id
    aliases.update(
        {
            "person": aliases.get("pedestrian", 0),
            "human": aliases.get("pedestrian", 0),
            "motorbike": aliases.get("motor", len(class_names) - 1),
            "motorcycle": aliases.get("motor", len(class_names) - 1),
            "awning tricycle": aliases.get("awning tricycle", 7),
            "covered tricycle": aliases.get("awning tricycle", 7),
        }
    )
    return aliases


def parse_answer_candidates(
    answer: str,
    image_path: Path,
    image_size: tuple[int, int],
    class_names: tuple[str, ...],
    fallback_class: str | None = None,
) -> tuple[PseudoCandidate, ...]:
    aliases = class_aliases(class_names)
    image_width, image_height = image_size
    candidates: list[PseudoCandidate] = []
    current_label = fallback_class
    for match in BOX_PATTERN.finditer(answer):
        matched_label = match.group("label")
        if matched_label is not None and matched_label.strip(TRIM_CHARS):
            current_label = matched_label
        label = current_label or UNKNOWN_LABEL
        class_id = aliases.get(_canonical_label(label))
        if class_id is None:
            continue
        coords = tuple(int(value) for value in COORD_PATTERN.findall(match.group("body")))
        box = _candidate_from_coords(
            coords,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            class_id=class_id,
            class_name=class_names[class_id],
        )
        if box is not None:
            candidates.append(box)
    return tuple(candidates)


def parse_raw_results(
    records: tuple[LocateAnythingRawResult, ...],
    class_names: tuple[str, ...],
    fallback_class: str | None = None,
) -> tuple[tuple[PseudoCandidate, ...], LocateAnythingParseStats]:
    parsed: list[PseudoCandidate] = []
    records_with_boxes = 0
    for record in records:
        image_path = record.resolved_image_path
        from PIL import Image

        with Image.open(image_path) as image:
            candidates = parse_answer_candidates(
                record.resolved_answer,
                image_path=image_path,
                image_size=image.size,
                class_names=class_names,
                fallback_class=fallback_class,
            )
        records_with_boxes += int(bool(candidates))
        parsed.extend(candidates)
    return (
        tuple(parsed),
        LocateAnythingParseStats(
            records=len(records),
            records_with_boxes=records_with_boxes,
            parsed_boxes=len(parsed),
            skipped_unknown_label=0,
            skipped_invalid_box=0,
        ),
    )


def class_agnostic_boxes(boxes: tuple[DetectionBox, ...]) -> tuple[DetectionBox, ...]:
    return tuple(DetectionBox(box.image_id, 0, box.xyxy, box.area, box.score) for box in boxes)


def class_agnostic_candidates(
    candidates: tuple[PseudoCandidate, ...],
) -> tuple[PseudoCandidate, ...]:
    return tuple(
        PseudoCandidate(
            image_id=candidate.image_id,
            image_path=candidate.image_path,
            class_id=0,
            class_name="object",
            score=candidate.score,
            x1=candidate.x1,
            y1=candidate.y1,
            x2=candidate.x2,
            y2=candidate.y2,
            area=candidate.area,
            area_bucket=candidate.area_bucket,
        )
        for candidate in candidates
    )


def _candidate_from_coords(
    coords: tuple[int, ...],
    *,
    image_path: Path,
    image_width: int,
    image_height: int,
    class_id: int,
    class_name: str,
) -> PseudoCandidate | None:
    if len(coords) != 4:
        return None
    x1_norm, y1_norm, x2_norm, y2_norm = (max(0, min(1000, coord)) / 1000.0 for coord in coords)
    if x2_norm <= x1_norm or y2_norm <= y1_norm:
        return None
    x1 = x1_norm * image_width
    y1 = y1_norm * image_height
    x2 = x2_norm * image_width
    y2 = y2_norm * image_height
    area = (x2 - x1) * (y2 - y1)
    return PseudoCandidate(
        image_id=image_path.stem,
        image_path=str(image_path),
        class_id=class_id,
        class_name=class_name,
        score=1.0,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        area=area,
        area_bucket=area_bucket_for_area(area),
    )


def _canonical_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("-", " ").replace("_", " ").lower().strip(TRIM_CHARS))
