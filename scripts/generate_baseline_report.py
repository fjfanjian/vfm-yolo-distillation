#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.8",
#   "pyyaml>=6.0",
# ]
# ///
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from baseline_report_data import BASE, NAMES, REPORTS, TEMPLATE, ExperimentResult
from baseline_report_data import data_uri, load_result, plot_comparison, text, write_summaries


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def row(key: str, value: float | int | str | None) -> str:
    return f"<tr><td>{html.escape(key)}</td><td>{html.escape(fmt(value))}</td></tr>"


def comparison_table(results: list[ExperimentResult]) -> str:
    body = []
    for result in results:
        body.append(
            "<tr>"
            f"<td>{html.escape(result.label)}</td><td>{result.best_epoch}</td>"
            f"<td>{fmt(result.pytorch.precision)}</td><td>{fmt(result.pytorch.recall)}</td>"
            f"<td class=\"metric-neutral\">{fmt(result.pytorch.map50)}</td>"
            f"<td class=\"metric-good\">{fmt(result.pytorch.map50_95)}</td>"
            f"<td>{fmt(result.pytorch.inference_ms)} ms</td>"
            f"<td>{fmt(result.onnx.map50)}</td><td>{fmt(result.onnx.map50_95)}</td>"
            f"<td>{fmt(result.onnx.inference_ms)} ms</td></tr>"
        )
    return (
        "<table><tr><th>实验</th><th>最佳 epoch</th><th>P</th><th>R</th>"
        "<th>PyTorch mAP50</th><th>PyTorch mAP50-95</th><th>PyTorch GPU 推理</th>"
        "<th>ONNX mAP50</th><th>ONNX mAP50-95</th><th>ONNX CPU 推理</th></tr>"
        + "\n".join(body)
        + "</table>"
    )


def report_replacements(results: list[ExperimentResult], chart: Path) -> dict[str, str]:
    full = results[-1]
    return {
        "__EXPERIMENT_NAME__": "YOLO26n VisDrone 监督 baseline 实验报告",
        "__SUBTITLE__": "VisDrone · YOLO26n · 10%/25%/100% 标注量 · 120 epochs · RTX 4090",
        "__BASIC_INFO_ROWS__": basic_info_rows(),
        "__EXPERIMENT_PURPOSE__": "建立纯监督 YOLO-only student 的少标注基线，为后续 DINOv3 特征蒸馏与开放词表伪标签实验提供可复现对照。",
        "__TRAIN_CONFIG_ROWS__": train_config_rows(),
        "__HYPERPARAM_ROWS__": "\n".join([row("box", 7.5), row("cls", 0.5), row("dfl", 1.5), row("weight_decay", 0.0005), row("patience", 100)]),
        "__AUGMENT_ROWS__": "\n".join([row("mosaic", 1.0), row("scale", 0.5), row("fliplr", 0.5), row("auto_augment", "randaugment")]),
        "__TRAINING_CURVES_SRC__": data_uri(chart),
        "__FINAL_METRICS_ROWS__": final_metrics_rows(results),
        "__BEST_MODEL_ROWS__": best_model_rows(results),
        "__LOSS_CONVERGENCE_ROWS__": "<tr><td>训练曲线</td><td>见 report_data/*_curves.png</td><td>见 results.csv</td><td>三组均完成 120 epoch</td></tr>",
        "__RESOURCE_STATS__": resource_stats(full),
        "__TRAINING_ANALYSIS__": "<div class=\"analysis-block\">三组实验均完成 120 epoch，统一验证未发现 corrupt label 或运行错误。标注比例从 10% 到 25% 再到 100% 时，mAP50-95 从 0.101 提升到 0.128 和 0.176。</div>",
        "__KEY_FINDINGS__": "<div class=\"analysis-block\">全量监督相对 10% baseline 的 mAP50-95 绝对提升约 0.075，说明当前 student 主要受标注量限制；car、bus、van 等类别收益明显，小目标和细粒度类别仍是短板。</div>",
        "__PROBLEMS__": "<div class=\"analysis-block\">ONNX 导出可用，但当前 ONNX 验证使用 CPUExecutionProvider；真正部署前还需要 TensorRT FP16 benchmark。导出时 Ultralytics 自动升级 protobuf/ml-dtypes，可能影响同环境 TensorFlow，但不影响本次 PyTorch/Ultralytics 结果。</div>",
        "__COMPARISON_TABLE__": comparison_table(results),
        "__COMPARISON_ANALYSIS__": "<div class=\"analysis-block\">PyTorch 与 ONNX 指标基本一致，说明导出模型保持检测行为。PyTorch GPU 验证推理约 0.8ms/图，ONNX CPU 约 38-39ms/图；二者不是同后端速度对比，只作为格式验证和粗略 benchmark。</div>",
        "__CONCLUSION__": "<div class=\"analysis-block\">YOLO26n 纯监督 baseline 已形成可复现锚点：三种标注预算均有 best.pt、best.onnx、统一 val 日志、训练曲线和汇总表。后续 DINOv3 蒸馏应优先在 10%/25% 上比较 AP_small 与 mAP50-95 增益。</div>",
        "__NEXT_STEPS__": next_steps(),
        "__GENERATED_TIME__": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def basic_info_rows() -> str:
    return "\n".join(
        [
            row("项目", "VFM-assisted YOLO Distillation"),
            row("数据集", "VisDrone2019-DET train/val"),
            row("模型", "Ultralytics YOLO26n"),
            row("评估集", "548 张图，38759 个实例"),
            row("运行目录", str(BASE)),
        ]
    )


def train_config_rows() -> str:
    return "\n".join(
        [
            row("imgsz", 640),
            row("epochs", 120),
            row("batch", 16),
            row("device", "RTX 4090 / CUDA:0"),
            row("workers", 8),
            row("seed", 42),
        ]
    )


def final_metrics_rows(results: list[ExperimentResult]) -> str:
    full = results[-1]
    return "\n".join(
        [
            row("10% PyTorch mAP50-95", results[0].pytorch.map50_95),
            row("25% PyTorch mAP50-95", results[1].pytorch.map50_95),
            row("100% PyTorch mAP50-95", full.pytorch.map50_95),
            row("100% PyTorch mAP50", full.pytorch.map50),
            row("100% ONNX CPU mAP50-95", full.onnx.map50_95),
            row("100% ONNX CPU 推理 ms/图", full.onnx.inference_ms),
        ]
    )


def best_model_rows(results: list[ExperimentResult]) -> str:
    full = results[-1]
    return "\n".join(
        [
            row("10% best.pt", results[0].best_pt),
            row("25% best.pt", results[1].best_pt),
            row("100% best.pt", full.best_pt),
            row("100% best.onnx", full.best_onnx),
            row("参数量", full.params),
            row("FLOPs", full.flops),
        ]
    )


def resource_stats(full: ExperimentResult) -> str:
    return "\n".join(
        [
            f"<div class=\"stat-card\"><div class=\"stat-value\">{fmt(full.params)}</div><div class=\"stat-label\">参数量</div></div>",
            f"<div class=\"stat-card\"><div class=\"stat-value\">{html.escape(full.flops)}</div><div class=\"stat-label\">FLOPs</div></div>",
            f"<div class=\"stat-card\"><div class=\"stat-value\">{fmt(full.pytorch.inference_ms)}ms</div><div class=\"stat-label\">PyTorch GPU 推理</div></div>",
            f"<div class=\"stat-card\"><div class=\"stat-value\">{fmt(full.onnx.inference_ms)}ms</div><div class=\"stat-label\">ONNX CPU 推理</div></div>",
        ]
    )


def next_steps() -> str:
    return "\n".join(
        [
            "<li><span class=\"checkbox\"></span>补充 COCO-style AP_small / AR_small 评估脚本。</li>",
            "<li><span class=\"checkbox\"></span>实现 DINOv3 dense feature 缓存，并优先做 10%/25% 蒸馏实验。</li>",
            "<li><span class=\"checkbox\"></span>增加 TensorRT FP16 导出与测速。</li>",
        ]
    )


def fill_report(results: list[ExperimentResult], chart: Path) -> str:
    template = text(TEMPLATE)
    for key, value in report_replacements(results, chart).items():
        template = template.replace(key, value)
    return template


def main() -> None:
    results = [load_result(name) for name in NAMES]
    write_summaries(results)
    chart = plot_comparison(results)
    (REPORTS / "baseline_report.html").write_text(fill_report(results, chart), encoding="utf-8")
    print(REPORTS / "baseline_summary.csv")
    print(REPORTS / "baseline_summary.json")
    print(REPORTS / "baseline_report.html")


if __name__ == "__main__":
    main()
