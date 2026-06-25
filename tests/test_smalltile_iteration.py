# ruff: noqa: D103, FLY002, INP001, PLR2004, S101
from pathlib import Path

from scripts.run_smalltile_iteration import (
    AreaMetric,
    GateOutcome,
    TrainingMetrics,
    build_area_ap_command,
    decide_gate,
    read_area_metrics,
    read_training_metrics,
    write_summary,
)


def test_read_training_metrics_when_results_csv_has_best_and_last_rows(tmp_path: Path) -> None:
    # Given
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "\n".join(
            (
                "epoch,metrics/precision(B),metrics/recall(B),"
                "metrics/mAP50(B),metrics/mAP50-95(B)",
                "1,0.1,0.2,0.3,0.04",
                "2,0.4,0.5,0.6,0.12",
                "3,0.7,0.8,0.9,0.10",
            )
        ),
        encoding="utf-8",
    )

    # When
    metrics = read_training_metrics(results_csv)

    # Then
    assert metrics.best_epoch == 2
    assert metrics.best_precision == 0.4
    assert metrics.best_recall == 0.5
    assert metrics.best_map50 == 0.6
    assert metrics.best_map50_95 == 0.12
    assert metrics.last_epoch == 3
    assert metrics.last_map50 == 0.9
    assert metrics.last_map50_95 == 0.10


def test_read_area_metrics_when_small_medium_large_rows_exist(tmp_path: Path) -> None:
    # Given
    area_csv = tmp_path / "smalltile_area_ap.csv"
    area_csv.write_text(
        "\n".join(
            (
                "name,imgsz,area,gt_count,prediction_count,ap50,ap50_95",
                "smalltile,960,small,26586,122000,0.27,0.1172",
                "smalltile,960,medium,11105,34000,0.58,0.395",
                "smalltile,960,large,1068,2600,0.62,0.49",
            )
        ),
        encoding="utf-8",
    )

    # When
    metrics = read_area_metrics(area_csv)

    # Then
    assert metrics["small"].ap50_95 == 0.1172
    assert metrics["small"].prediction_count == 122000
    assert metrics["medium"].gt_count == 11105
    assert metrics["large"].ap50 == 0.62


def test_decide_gate_when_small_and_overall_pass() -> None:
    # Given
    small_ap50_95 = 0.1171
    overall_map50_95 = 0.1444

    # When
    decision = decide_gate(small_ap50_95, overall_map50_95)

    # Then
    assert decision.outcome is GateOutcome.EXPAND_MULTI_SEED


def test_decide_gate_when_small_fails() -> None:
    # Given
    small_ap50_95 = 0.1169
    overall_map50_95 = 0.1500

    # When
    decision = decide_gate(small_ap50_95, overall_map50_95)

    # Then
    assert decision.outcome is GateOutcome.RUN_LAYER_ABLATION


def test_build_area_ap_command_uses_plan_evaluation_contract() -> None:
    # Given
    command = build_area_ap_command(
        python_executable="python3",
        model_path=Path("runs/dinov3_objectness/example/weights/best.pt"),
        name="example",
        output_dir=Path("runs/dinov3_objectness/reports"),
    )

    # Then
    assert command == (
        "python3",
        "scripts/evaluate_area_ap.py",
        "--model",
        "runs/dinov3_objectness/example/weights/best.pt",
        "--data",
        "configs/datasets/visdrone.yaml",
        "--imgsz",
        "960",
        "--name",
        "example",
        "--output-dir",
        "runs/dinov3_objectness/reports",
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
    )


def test_write_summary_when_metrics_are_slotted_dataclasses(tmp_path: Path) -> None:
    # Given
    summary_json = tmp_path / "summary.json"
    training = TrainingMetrics(
        best_epoch=10,
        best_precision=0.3,
        best_recall=0.4,
        best_map50=0.5,
        best_map50_95=0.145,
        last_epoch=12,
        last_map50=0.49,
        last_map50_95=0.14,
    )
    area_metrics = {
        "small": AreaMetric(
            area="small",
            gt_count=1,
            prediction_count=2,
            ap50=0.2,
            ap50_95=0.1171,
        ),
    }
    decision = decide_gate(small_ap50_95=0.1171, overall_map50_95=0.145)

    # When
    write_summary(summary_json, training, area_metrics, decision)

    # Then
    text = summary_json.read_text(encoding="utf-8")
    assert '"best_epoch": 10' in text
    assert '"outcome": "expand_multi_seed"' in text
