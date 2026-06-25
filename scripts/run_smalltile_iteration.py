#!/usr/bin/env -S uv run --script
# ruff: noqa: D101, D103, EM102, TRY003
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2.7",
#   "pyyaml>=6.0",
# ]
# ///
# --- How to run ---
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Preview commands:
#      uv run scripts/run_smalltile_iteration.py --phase commands
# 3. Run the full train/eval/export/gate workflow on the experiment machine:
#      uv run scripts/run_smalltile_iteration.py --phase all --python python3
# ------------------

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from vfm_yolo_distillation.config import load_experiment_config  # noqa: E402

DEFAULT_CONFIG: Final = Path(
    "configs/experiments/"
    "dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_seed42.yaml"
)
BASELINE_OVERALL_MAP50_95: Final = 0.14439
BASELINE_SMALL_AP50_95: Final = 0.11601
SMALL_AP_GATE: Final = 0.1170
EXPAND_SEEDS: Final = (2026, 3407)


class WorkflowError(RuntimeError):
    pass


class GateOutcome(StrEnum):
    EXPAND_MULTI_SEED = "expand_multi_seed"
    KEEP_TARGET_CANDIDATE = "keep_target_candidate"
    RUN_LAYER_ABLATION = "run_layer_ablation"


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    best_epoch: int
    best_precision: float
    best_recall: float
    best_map50: float
    best_map50_95: float
    last_epoch: int
    last_map50: float
    last_map50_95: float


@dataclass(frozen=True, slots=True)
class AreaMetric:
    area: str
    gt_count: int
    prediction_count: int
    ap50: float
    ap50_95: float


@dataclass(frozen=True, slots=True)
class GateDecision:
    outcome: GateOutcome
    message: str
    extra_seeds: tuple[int, ...]


def build_train_command(python_executable: str, config_path: Path) -> tuple[str, ...]:
    return (
        python_executable,
        "scripts/train_dinov3_objectness_aux.py",
        "--config",
        config_path.as_posix(),
    )


def build_area_ap_command(
    python_executable: str,
    model_path: Path,
    name: str,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        python_executable,
        "scripts/evaluate_area_ap.py",
        "--model",
        model_path.as_posix(),
        "--data",
        "configs/datasets/visdrone.yaml",
        "--imgsz",
        "960",
        "--name",
        name,
        "--output-dir",
        output_dir.as_posix(),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
    )


def build_export_command(python_executable: str, model_path: Path) -> tuple[str, ...]:
    code = "\n".join(
        (
            "from ultralytics import YOLO",
            f"model = YOLO({model_path.as_posix()!r})",
            "print('clean_yolo_load=OK')",
            "print(type(model.model).__name__)",
            "model.export(format='onnx', imgsz=960, opset=12, simplify=False, dynamic=False)",
        )
    )
    return (python_executable, "-c", code)


def read_training_metrics(results_csv: Path) -> TrainingMetrics:
    rows = _read_csv_rows(results_csv)
    if not rows:
        raise WorkflowError(f"results.csv has no metric rows: {results_csv}")
    best = max(rows, key=lambda row: _float_field(row, "metrics/mAP50-95(B)", results_csv))
    last = rows[-1]
    return TrainingMetrics(
        best_epoch=_int_field(best, "epoch", results_csv),
        best_precision=_float_field(best, "metrics/precision(B)", results_csv),
        best_recall=_float_field(best, "metrics/recall(B)", results_csv),
        best_map50=_float_field(best, "metrics/mAP50(B)", results_csv),
        best_map50_95=_float_field(best, "metrics/mAP50-95(B)", results_csv),
        last_epoch=_int_field(last, "epoch", results_csv),
        last_map50=_float_field(last, "metrics/mAP50(B)", results_csv),
        last_map50_95=_float_field(last, "metrics/mAP50-95(B)", results_csv),
    )


def read_area_metrics(area_csv: Path) -> Mapping[str, AreaMetric]:
    metrics: dict[str, AreaMetric] = {}
    for row in _read_csv_rows(area_csv):
        area = _field(row, "area", area_csv)
        metrics[area] = AreaMetric(
            area=area,
            gt_count=_int_field(row, "gt_count", area_csv),
            prediction_count=_int_field(row, "prediction_count", area_csv),
            ap50=_float_field(row, "ap50", area_csv),
            ap50_95=_float_field(row, "ap50_95", area_csv),
        )
    missing = {"small", "medium", "large"} - set(metrics)
    if missing:
        raise WorkflowError(f"area AP output is missing rows {sorted(missing)}: {area_csv}")
    return metrics


def decide_gate(small_ap50_95: float, overall_map50_95: float) -> GateDecision:
    if small_ap50_95 >= SMALL_AP_GATE and overall_map50_95 >= BASELINE_OVERALL_MAP50_95:
        return GateDecision(
            outcome=GateOutcome.EXPAND_MULTI_SEED,
            message="seed42 passed small AP and overall mAP gates; run seed=2026/3407 next.",
            extra_seeds=EXPAND_SEEDS,
        )
    if small_ap50_95 >= SMALL_AP_GATE:
        return GateDecision(
            outcome=GateOutcome.KEEP_TARGET_CANDIDATE,
            message=(
                "small AP passed but overall mAP fell below baseline; "
                "keep as target candidate."
            ),
            extra_seeds=(),
        )
    return GateDecision(
        outcome=GateOutcome.RUN_LAYER_ABLATION,
        message="small AP missed the gate; stop small-tile multi-seed and run layer ablation.",
        extra_seeds=(),
    )


def run_command(command: tuple[str, ...], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, env=_pythonpath_env(cwd), check=True)  # noqa: S603


def write_summary(
    summary_json: Path,
    training: TrainingMetrics,
    area_metrics: Mapping[str, AreaMetric],
    decision: GateDecision,
) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": {
            "overall_mAP50_95": BASELINE_OVERALL_MAP50_95,
            "small_AP50_95": BASELINE_SMALL_AP50_95,
            "small_gate_AP50_95": SMALL_AP_GATE,
        },
        "training": asdict(training),
        "area_ap": {area: asdict(metric) for area, metric in area_metrics.items()},
        "decision": {
            "outcome": decision.outcome.value,
            "message": decision.message,
            "extra_seeds": decision.extra_seeds,
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("commands", "train", "evaluate", "export", "decide", "all"),
        default="commands",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--area-output-dir", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    run_dir = args.run_dir or Path(config.outputs.project) / config.outputs.name
    area_output_dir = args.area_output_dir or run_dir.parent / "reports"
    summary_json = args.summary_json or area_output_dir / f"{config.outputs.name}_gate_summary.json"
    best_model = run_dir / "weights" / "best.pt"
    area_csv = area_output_dir / f"{config.outputs.name}_area_ap.csv"

    train_command = build_train_command(args.python, args.config)
    area_command = build_area_ap_command(
        args.python,
        best_model,
        config.outputs.name,
        area_output_dir,
    )
    export_command = build_export_command(args.python, best_model)

    phase = args.phase
    if phase == "commands":
        _print_commands(train_command, area_command, export_command)
        return 0
    if phase in {"train", "all"}:
        run_command(train_command, ROOT)
    if phase in {"evaluate", "all"}:
        run_command(area_command, ROOT)
    if phase in {"export", "all"}:
        run_command(export_command, ROOT)
    if phase in {"decide", "all"}:
        training = read_training_metrics(run_dir / "results.csv")
        area_metrics = read_area_metrics(area_csv)
        decision = decide_gate(area_metrics["small"].ap50_95, training.best_map50_95)
        write_summary(summary_json, training, area_metrics, decision)
        sys.stdout.write(f"{decision.outcome.value}: {decision.message}\n{summary_json}\n")
    return 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [{key.strip(): value.strip() for key, value in row.items()} for row in reader]


def _field(row: Mapping[str, str], key: str, source: Path) -> str:
    try:
        return row[key]
    except KeyError as exc:
        raise WorkflowError(f"Missing column {key!r} in {source}") from exc


def _float_field(row: Mapping[str, str], key: str, source: Path) -> float:
    return float(_field(row, key, source))


def _int_field(row: Mapping[str, str], key: str, source: Path) -> int:
    return int(float(_field(row, key, source)))


def _pythonpath_env(cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    parts = [(cwd / "src").as_posix(), cwd.as_posix(), (cwd / "scripts").as_posix()]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _print_commands(*commands: tuple[str, ...]) -> None:
    for command in commands:
        sys.stdout.write(f"{shlex.join(command)}\n")


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
