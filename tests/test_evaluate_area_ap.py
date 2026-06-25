# ruff: noqa: ANN001, D103, INP001, PLR0913, S101
from pathlib import Path

from scripts import evaluate_area_ap


class _FakeBoxes:
    def __iter__(self) -> iter(()):
        return iter(())


class _FakeResult:
    path = "sample.jpg"
    boxes = _FakeBoxes()


class _FakeYOLO:
    last_batch: int | None = None

    def __init__(self, _model_path: str) -> None:
        pass

    def predict(
        self,
        *,
        source: list[str],
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: int,
        verbose: bool,
        stream: bool,
        batch: int | None = None,
    ) -> list[_FakeResult]:
        _ = (source, imgsz, conf, iou, max_det, device, verbose, stream)
        type(self).last_batch = batch
        return [_FakeResult()]


def test_collect_predictions_uses_batch_one_to_bound_gpu_memory(monkeypatch) -> None:
    # Given
    monkeypatch.setattr(evaluate_area_ap, "YOLO", _FakeYOLO)

    # When
    predictions = evaluate_area_ap.collect_predictions(
        model_path=Path("best.pt"),
        images=[Path("sample.jpg")],
        imgsz=960,
        conf=0.001,
        iou=0.7,
        max_det=300,
    )

    # Then
    assert predictions == []
    assert _FakeYOLO.last_batch == 1
