#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path("runs/dinov3_distill")
RUN = ROOT / "yolo26n_visdrone_10pct_dinov3_patch_b32"
ARCHIVE = ROOT / "archives" / "yolo26n_visdrone_10pct_dinov3_patch_b32"


def rows_from(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def best_row(path: Path) -> dict[str, str]:
    rows = rows_from(path)
    return max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def last_row(path: Path) -> dict[str, str]:
    return rows_from(path)[-1]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def collect_comparison() -> list[dict[str, str]]:
    experiments = [
        ("baseline_10pct_640", Path("runs/baselines/yolo26n_visdrone_10pct/results.csv"), "runs/baselines/yolo26n_visdrone_10pct/weights/best.pt"),
        ("dinov3_global_10pct_640", ROOT / "yolo26n_visdrone_10pct_dinov3_global/results.csv", "runs/dinov3_distill/yolo26n_visdrone_10pct_dinov3_global/weights/best.pt"),
        ("dinov3_patch_b16_partial", ROOT / "yolo26n_visdrone_10pct_dinov3_patch/results.csv", "runs/dinov3_distill/yolo26n_visdrone_10pct_dinov3_patch/weights/best.pt"),
        ("dinov3_patch_b32_10pct_640", RUN / "results.csv", str(RUN / "weights/best_clean.pt")),
    ]
    rows: list[dict[str, str]] = []
    for name, path, weights in experiments:
        if not path.exists():
            continue
        best = best_row(path)
        last = last_row(path)
        rows.append({
            "experiment": name,
            "epochs_recorded": str(sum(1 for _ in path.open()) - 1),
            "best_epoch": best["epoch"],
            "best_precision": best["metrics/precision(B)"],
            "best_recall": best["metrics/recall(B)"],
            "best_mAP50": best["metrics/mAP50(B)"],
            "best_mAP50_95": best["metrics/mAP50-95(B)"],
            "last_epoch": last["epoch"],
            "last_mAP50": last["metrics/mAP50(B)"],
            "last_mAP50_95": last["metrics/mAP50-95(B)"],
            "weights": weights,
        })
    return rows


def collect_area_rows() -> list[dict[str, str]]:
    sources = [
        ("baseline_10pct_640", Path("runs/baselines/reports/yolo26n_visdrone_10pct_imgsz640_area_ap.csv")),
        ("dinov3_patch_b32_10pct_640", ROOT / "reports/yolo26n_visdrone_10pct_dinov3_patch_b32_area_ap.csv"),
    ]
    rows: list[dict[str, str]] = []
    for experiment, path in sources:
        if not path.exists():
            continue
        for row in rows_from(path):
            row["experiment"] = experiment
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def archive_assets() -> None:
    for subdir in ("weights", "val", "reports", "logs"):
        (ARCHIVE / subdir).mkdir(parents=True, exist_ok=True)
    for name in ("results.csv", "args.yaml"):
        copy_if_exists(RUN / name, ARCHIVE / name)
    for name in ("best.pt", "last.pt", "best_clean.pt"):
        copy_if_exists(RUN / "weights" / name, ARCHIVE / "weights" / name)
    copy_if_exists(ROOT / "logs/yolo26n_visdrone_10pct_dinov3_patch_b32.log", ARCHIVE / "logs/yolo26n_visdrone_10pct_dinov3_patch_b32.log")
    for suffix in ("csv", "json"):
        name = f"yolo26n_visdrone_10pct_dinov3_patch_b32_area_ap.{suffix}"
        copy_if_exists(ROOT / "reports" / name, ARCHIVE / "reports" / name)
    val_dir = ROOT / "val/yolo26n_visdrone_10pct_dinov3_patch_b32_best_clean"
    if val_dir.exists():
        target = ARCHIVE / "val" / val_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(val_dir, target)


def write_readme(generated_at: str, comparison: list[dict[str, str]], area_rows: list[dict[str, str]]) -> None:
    lines = [
        "# DINOv3 Patch Distillation 10% VisDrone 归档摘要",
        "",
        f"生成时间: {generated_at}",
        "",
        "## 主实验",
        f"- 目录: `{RUN}`",
        f"- 归档: `{ARCHIVE}`",
        f"- 可加载权重: `{RUN / 'weights/best_clean.pt'}`",
        "- 原始权重问题: `best.pt/last.pt` 含训练 hook pickle 引用，常规 `yolo val` 会加载失败；已生成 `best_clean.pt`。",
        "",
        "## 总体指标对比",
        "",
        "| 实验 | best epoch | P | R | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(f"| {row['experiment']} | {row['best_epoch']} | {float(row['best_precision']):.4f} | {float(row['best_recall']):.4f} | {float(row['best_mAP50']):.4f} | {float(row['best_mAP50_95']):.4f} |")
    lines.extend(["", "## 面积段 AP 对比", "", "| 实验 | area | AP50 | AP50-95 | GT | Pred |", "|---|---|---:|---:|---:|---:|"])
    for row in area_rows:
        lines.append(f"| {row['experiment']} | {row['area']} | {float(row['ap50']):.4f} | {float(row['ap50_95']):.4f} | {row['gt_count']} | {row['prediction_count']} |")
    lines.extend(["", "## 初步结论", "- patch b32 提高了 GPU 吞吐，但最终精度没有超过 baseline/global。", "- small AP50-95 低于 baseline，说明当前 patch-token 全图平均对齐没有给小目标带来有效增益。", "- 主要问题不是训练没收敛，而是蒸馏信号和检测目标未充分对齐。", ""])
    (ARCHIVE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    archive_assets()
    comparison = collect_comparison()
    area_rows = collect_area_rows()
    write_csv(ARCHIVE / "dinov3_distill_comparison.csv", comparison, list(comparison[0].keys()))
    write_csv(ARCHIVE / "area_ap_comparison.csv", area_rows, ["experiment", "name", "imgsz", "area", "gt_count", "prediction_count", "ap50", "ap50_95"])
    generated_at = datetime.now(UTC).isoformat()
    summary = {
        "generated_at": generated_at,
        "archive": str(ARCHIVE),
        "primary_run": str(RUN),
        "primary_weights": str(RUN / "weights/best_clean.pt"),
        "comparison": comparison,
        "area_ap_comparison": area_rows,
        "known_issue": "original best.pt/last.pt contain a pickled training hook reference; best_clean.pt strips hooks and is the deploy/eval-safe checkpoint.",
    }
    (ARCHIVE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(generated_at, comparison, area_rows)
    for path in (ARCHIVE, ARCHIVE / "dinov3_distill_comparison.csv", ARCHIVE / "area_ap_comparison.csv", ARCHIVE / "summary.json", ARCHIVE / "README.md"):
        print(path)


if __name__ == "__main__":
    main()
