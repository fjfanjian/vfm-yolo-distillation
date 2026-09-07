#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2.7",
#   "pyyaml>=6.0",
#   "ultralytics>=8.3",
# ]
# ///
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict, assert_never

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfm_yolo_distillation.config import (  # noqa: E402
    ExperimentConfig,
    load_experiment_config,
    training_dataset_config_path,
)


class _Phase(StrEnum):
    ALL = "all"
    TRAIN = "train"
    EVALUATE = "evaluate"
    EXPORT = "export"


class _TrainingOverrides(TypedDict):
    model: str
    data: str
    imgsz: int
    epochs: int
    batch: int
    workers: int
    seed: int
    device: str | int
    project: str
    name: str
    pretrained: str
    exist_ok: bool
    optimizer: str
    patience: int
    close_mosaic: int
    amp: bool
    plots: bool


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Training-time initialization settings for architecture experiments."""

    pretrained: str = "yolo26n.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[phase.value for phase in _Phase],
        default=_Phase.ALL.value,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", default="yolo26n.pt")
    return parser.parse_args()


def training_overrides(config: ExperimentConfig, settings: RunSettings) -> _TrainingOverrides:
    """Build Ultralytics train overrides while preserving partial pretrained loading."""
    training = config.training
    return {
        "model": config.student.model,
        "data": training_dataset_config_path(config).as_posix(),
        "imgsz": training.image_size,
        "epochs": training.epochs,
        "batch": training.batch,
        "workers": training.workers,
        "seed": training.seed,
        "device": training.device,
        "project": config.outputs.project.as_posix(),
        "name": config.outputs.name,
        "pretrained": settings.pretrained,
        "exist_ok": True,
        "optimizer": "auto",
        "patience": 100,
        "close_mosaic": 10,
        "amp": True,
        "plots": False,
    }


def _best_weight(config: ExperimentConfig) -> Path:
    return config.outputs.project / config.outputs.name / "weights" / "best.pt"


def _run_train(config: ExperimentConfig, settings: RunSettings) -> None:
    model = YOLO(config.student.model)
    model.train(**training_overrides(config, settings))


def _run_evaluate(config: ExperimentConfig) -> None:
    data_path = training_dataset_config_path(config)
    model = YOLO(str(_best_weight(config)))
    model.val(
        data=str(data_path),
        imgsz=config.training.image_size,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=config.training.device,
        project=str(config.outputs.project / "reports" / "standard_val"),
        name=config.outputs.name,
        exist_ok=True,
    )
    subprocess.run(_area_ap_command(config, data_path), check=True)  # noqa: S603


def _area_ap_command(config: ExperimentConfig, data_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_area_ap.py"),
        "--model",
        str(_best_weight(config)),
        "--data",
        str(data_path),
        "--imgsz",
        str(config.training.image_size),
        "--name",
        config.outputs.name,
        "--output-dir",
        str(config.outputs.project / "reports"),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
    ]


def _run_export(config: ExperimentConfig) -> None:
    model = YOLO(str(_best_weight(config)))
    sys.stdout.write("clean_yolo_load=OK\n")
    sys.stdout.flush()
    model.export(
        format="onnx",
        imgsz=config.training.image_size,
        opset=12,
        simplify=False,
        dynamic=False,
    )


def _run_phase(config: ExperimentConfig, settings: RunSettings, phase: _Phase) -> None:
    match phase:
        case _Phase.ALL:
            _run_train(config, settings)
            _run_evaluate(config)
            _run_export(config)
        case _Phase.TRAIN:
            _run_train(config, settings)
        case _Phase.EVALUATE:
            _run_evaluate(config)
        case _Phase.EXPORT:
            _run_export(config)
        case unreachable:
            assert_never(unreachable)


def _run() -> None:
    args = _parse_args()
    config = load_experiment_config(Path(args.config))
    settings = RunSettings(pretrained=str(args.pretrained))
    _run_phase(config, settings, _Phase(str(args.phase)))


if __name__ == "__main__":
    _run()
