from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from PIL import Image

from vfm_yolo_distillation.locateanything_model_loading import (
    patch_dynamic_cache_legacy_compat,
    patch_missing_all_tied_weights_keys,
    patch_remote_attn_compat,
    patch_tied_weights_compat,
)
from vfm_yolo_distillation.locateanything_quality import (
    LocateAnythingRawResult,
    class_agnostic_boxes,
    class_agnostic_candidates,
    parse_answer_candidates,
    parse_raw_results,
)
from vfm_yolo_distillation.locateanything_runner import (
    read_completed_raw_keys,
    resize_image_for_locateanything,
)
from vfm_yolo_distillation.pseudo_label_quality import DetectionBox

NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def test_parse_answer_candidates_when_standard_ref_box_output() -> None:
    answer = "<ref>car</ref><box><100><200><300><500></box>"

    candidates = parse_answer_candidates(answer, Path("/data/000001.jpg"), (1000, 500), NAMES)

    assert len(candidates) == 1
    assert candidates[0].class_id == 3
    assert candidates[0].class_name == "car"
    assert candidates[0].x1 == 100.0
    assert candidates[0].y1 == 100.0
    assert candidates[0].x2 == 300.0
    assert candidates[0].y2 == 250.0


def test_parse_answer_candidates_when_label_uses_supported_alias() -> None:
    answer = "<ref>motorcycle</ref><box><100><100><200><200></box>"

    candidates = parse_answer_candidates(answer, Path("/data/000001.jpg"), (100, 100), NAMES)

    assert len(candidates) == 1
    assert candidates[0].class_id == 9
    assert candidates[0].class_name == "motor"


def test_parse_answer_candidates_inherits_label_for_consecutive_boxes() -> None:
    answer = (
        "<ref>car</ref>"
        "<box><100><100><200><200></box>"
        "<box><300><300><400><400></box>"
        "<ref>motorcycle</ref>"
        "<box><500><500><600><600></box>"
    )

    candidates = parse_answer_candidates(answer, Path("/data/000001.jpg"), (100, 100), NAMES)

    assert [candidate.class_name for candidate in candidates] == ["car", "car", "motor"]
    assert candidates[1].x1 == 30.0
    assert candidates[1].y1 == 30.0


def test_parse_answer_candidates_skips_points_and_unknown_labels() -> None:
    answer = (
        "<ref>dog</ref><box><100><100><200><200></box>"
        "<ref>car</ref><box><100><100></box>"
    )

    candidates = parse_answer_candidates(answer, Path("/data/000001.jpg"), (100, 100), NAMES)

    assert candidates == ()


def test_parse_raw_results_reads_supported_answer_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "000001.jpg"
    Image.new("RGB", (100, 100)).save(image_path)
    record = LocateAnythingRawResult(
        image=str(image_path),
        raw_response="<ref>car</ref><box><0><0><500><500></box>",
    )

    candidates, stats = parse_raw_results((record,), NAMES)

    assert stats.records == 1
    assert stats.records_with_boxes == 1
    assert stats.parsed_boxes == 1
    assert candidates[0].area == 2500.0


def test_class_agnostic_helpers_zero_class_ids() -> None:
    box = DetectionBox("image", 3, (1.0, 2.0, 3.0, 4.0), 4.0, 0.5)
    candidate = parse_answer_candidates(
        "<ref>car</ref><box><0><0><500><500></box>",
        Path("/data/image.jpg"),
        (100, 100),
        NAMES,
    )[0]

    assert class_agnostic_boxes((box,))[0].class_id == 0
    assert class_agnostic_candidates((candidate,))[0].class_name == "object"


def test_patch_remote_attn_compat_drops_new_transformers_keyword() -> None:
    class RemoteBase:
        def _check_and_adjust_attn_implementation(self, attn: str, is_init_check: bool) -> str:
            return f"{attn}:{is_init_check}"

    class RemoteModel(RemoteBase):
        pass

    patch_remote_attn_compat(RemoteModel)

    assert (
        RemoteModel()._check_and_adjust_attn_implementation(
            "eager",
            is_init_check=True,
            allow_all_kernels=True,
        )
        == "eager:True"
    )


def test_patch_tied_weights_compat_keeps_remote_list_keys() -> None:
    class RemoteModel:
        _tied_weights_keys: ClassVar[list[str]] = ["lm_head.weight"]

    patch_tied_weights_compat(RemoteModel)

    assert RemoteModel().get_expanded_tied_weights_keys(all_submodels=False) == [
        "lm_head.weight"
    ]


def test_patch_missing_all_tied_weights_keys_adds_empty_mapping() -> None:
    class RemoteModel:
        def __init__(self) -> None:
            self.ready = True

    patch_missing_all_tied_weights_keys(RemoteModel)
    model = RemoteModel()

    assert model.ready is True
    assert model.all_tied_weights_keys == {}


def test_patch_dynamic_cache_legacy_compat_returns_self_when_missing_method() -> None:
    class Cache:
        def __init__(self, legacy: tuple[object, ...] | None = None) -> None:
            self.legacy = legacy

    patch_dynamic_cache_legacy_compat(Cache)
    cache = Cache()
    legacy = (object(),)
    converted = Cache.from_legacy_cache(legacy)

    assert cache.to_legacy_cache() is cache
    assert Cache.from_legacy_cache(cache) is cache
    assert isinstance(Cache.from_legacy_cache(), Cache)
    assert isinstance(converted, Cache)
    assert converted.legacy is legacy


def test_read_completed_raw_keys_reads_image_and_prompt_mode(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        '{"image": "/data/a.jpg", "prompt_mode": "all_classes", "answer": "ok"}\n'
        '{"image_path": "/data/b.jpg", "prompt_mode": "small", "answer": "ok"}\n',
        encoding="utf-8",
    )

    assert read_completed_raw_keys(raw) == {
        ("/data/a.jpg", "all_classes"),
        ("/data/b.jpg", "small"),
    }


def test_resize_image_for_locateanything_preserves_aspect_without_upscale() -> None:
    large = Image.new("RGB", (1920, 1080))
    small = Image.new("RGB", (960, 540))

    resized = resize_image_for_locateanything(large, 1360)

    assert resized.size == (1360, 765)
    assert resize_image_for_locateanything(small, 1360) is small
