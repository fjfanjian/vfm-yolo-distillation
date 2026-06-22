from __future__ import annotations

import base64
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT: Final = Path(__file__).resolve().parents[1]
BASE: Final = ROOT / "runs" / "baselines"
REPORT_DATA: Final = BASE / "report_data"
REPORTS: Final = BASE / "reports"
TEMPLATE: Final = BASE / "report_tools" / "report_template.html"
NAMES: Final = (
    "yolo26n_visdrone_10pct",
    "yolo26n_visdrone_25pct",
    "yolo26n_visdrone_full",
)
LABELS: Final = {
    "yolo26n_visdrone_10pct": "10% labeled",
    "yolo26n_visdrone_25pct": "25% labeled",
    "yolo26n_visdrone_full": "100% labeled",
}
BUDGETS: Final = {
    "yolo26n_visdrone_10pct": 10,
    "yolo26n_visdrone_25pct": 25,
    "yolo26n_visdrone_full": 100,
}


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    precision: float
    recall: float
    map50: float
    map50_95: float
    preprocess_ms: float | None
    inference_ms: float | None
    postprocess_ms: float | None


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    name: str
    label: str
    budget_pct: int
    best_epoch: int
    train_final_map50: float
    train_final_map50_95: float
    train_best_map50_95: float
    pytorch: EvalMetrics
    onnx: EvalMetrics
    params: str
    flops: str
    best_pt: str
    best_onnx: str
    results_csv: str


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def parse_eval_log(path: Path) -> EvalMetrics:
    content = text(path)
    all_lines = [line for line in content.splitlines() if re.match(r"\s*all\s+\d+\s+\d+", line)]
    if not all_lines:
        msg = f"No aggregate metrics line found in {path}"
        raise RuntimeError(msg)
    parts = all_lines[-1].split()
    speed = re.search(
        r"Speed:\s*([0-9.]+)ms preprocess,\s*([0-9.]+)ms inference,"
        r"\s*[0-9.]+ms loss,\s*([0-9.]+)ms postprocess per image",
        content,
    )
    return EvalMetrics(
        precision=float(parts[3]),
        recall=float(parts[4]),
        map50=float(parts[5]),
        map50_95=float(parts[6]),
        preprocess_ms=float(speed.group(1)) if speed else None,
        inference_ms=float(speed.group(2)) if speed else None,
        postprocess_ms=float(speed.group(3)) if speed else None,
    )


def parse_model_info(path: Path) -> tuple[str, str]:
    found = re.search(
        r"YOLO26n summary \(fused\): .*?([0-9,]+) parameters.*?([0-9.]+ GFLOPs)",
        text(path),
    )
    if found:
        return found.group(1), found.group(2)
    return "2,376,786", "5.2 GFLOPs"


def load_result(name: str) -> ExperimentResult:
    with (REPORT_DATA / f"{name}_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    data = summary["data"]
    final_metrics = data["final_metrics"]
    best_metrics = data["best_metrics"]
    pt_log = BASE / "eval_logs" / f"{name}_val.log"
    params, flops = parse_model_info(pt_log)
    return ExperimentResult(
        name=name,
        label=LABELS[name],
        budget_pct=BUDGETS[name],
        best_epoch=int(data["best_epoch"]),
        train_final_map50=float(final_metrics["metrics/mAP50(B)"]),
        train_final_map50_95=float(final_metrics["metrics/mAP50-95(B)"]),
        train_best_map50_95=float(best_metrics["metrics/mAP50-95(B)"]),
        pytorch=parse_eval_log(pt_log),
        onnx=parse_eval_log(BASE / "eval_logs" / f"{name}_onnx_val.log"),
        params=params,
        flops=flops,
        best_pt=str(BASE / name / "weights" / "best.pt"),
        best_onnx=str(BASE / name / "weights" / "best.onnx"),
        results_csv=str(BASE / name / "results.csv"),
    )


def write_summaries(results: list[ExperimentResult]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "baseline_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "experiment",
                "label_budget_pct",
                "best_epoch",
                "pytorch_precision",
                "pytorch_recall",
                "pytorch_mAP50",
                "pytorch_mAP50_95",
                "pytorch_inference_ms",
                "onnx_precision",
                "onnx_recall",
                "onnx_mAP50",
                "onnx_mAP50_95",
                "onnx_cpu_inference_ms",
                "params",
                "flops",
                "best_pt",
                "best_onnx",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.budget_pct,
                    result.best_epoch,
                    result.pytorch.precision,
                    result.pytorch.recall,
                    result.pytorch.map50,
                    result.pytorch.map50_95,
                    result.pytorch.inference_ms,
                    result.onnx.precision,
                    result.onnx.recall,
                    result.onnx.map50,
                    result.onnx.map50_95,
                    result.onnx.inference_ms,
                    result.params,
                    result.flops,
                    result.best_pt,
                    result.best_onnx,
                ]
            )
    payload = [asdict(result) for result in results]
    (REPORTS / "baseline_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_comparison(results: list[ExperimentResult]) -> Path:
    budgets = [result.budget_pct for result in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(budgets, [r.pytorch.map50 for r in results], marker="o", label="PyTorch mAP50")
    axes[0].plot(budgets, [r.pytorch.map50_95 for r in results], marker="o", label="PyTorch mAP50-95")
    axes[0].plot(budgets, [r.onnx.map50_95 for r in results], marker="s", linestyle="--", label="ONNX mAP50-95")
    axes[0].set_xlabel("Label budget (%)")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Data efficiency")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    speeds = [result.pytorch.inference_ms or math.nan for result in results]
    axes[1].bar([str(budget) for budget in budgets], speeds)
    axes[1].set_xlabel("Label budget (%)")
    axes[1].set_ylabel("ms/image")
    axes[1].set_title("PyTorch GPU validation speed")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output = REPORT_DATA / "baseline_comparison_curves.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output
