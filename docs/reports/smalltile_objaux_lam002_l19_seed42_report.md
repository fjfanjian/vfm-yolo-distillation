# Small-tile objectness auxiliary layer19 seed42 实验报告

## 1. 摘要

本实验是 high-density small-object tile objectness auxiliary 的 student layer 消融，固定 VisDrone 10% / imgsz=960 / YOLO26n / small_tile_soft / lambda_objectness=0.02 / seed=42，仅将辅助层从 layer16 改为 layer19。

结论：layer19 的 best overall mAP50-95 为 0.16423，相比 baseline 0.14439 提升 +0.01984；但 small AP50-95 为 0.11527，低于 baseline small 0.11601，也低于进入多 seed 的 0.11700 门槛。layer16 和 layer19 都呈现 overall 明显提升、small AP 不升反降的模式，因此单层 small-tile 辅助不再补 seed，下一轮进入 multi-layer 16+19。

## 2. 实验配置

| 字段 | 内容 |
| --- | --- |
| 实验名称 | yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42 |
| 数据集 | VisDrone 10% label budget |
| 学生模型 | YOLO26n |
| 图像尺寸 | 960 |
| 训练轮数 | 120 |
| 配置文件 | configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_l19_seed42.yaml |
| 服务器运行目录 | /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42 |
| 生成时间 | 2026-06-26 Asia/Shanghai |

### 2.1 训练参数

| 参数 | 值 |
| --- | --- |
| model | yolo26n.pt |
| data | configs/datasets/visdrone_10pct.yaml |
| epochs | 120 |
| batch | 16 |
| imgsz | 960 |
| device | 0 |
| workers | 8 |
| seed | 42 |
| optimizer | auto |
| amp | True |
| project | /home/fj/vfm-yolo-distillation/runs/dinov3_objectness |
| name | yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42 |

### 2.2 objectness / small-tile 参数

| 参数 | 值 |
| --- | --- |
| target_mode | small_tile_soft |
| student_layer | 19 |
| student_dim | 128 |
| lambda_objectness | 0.02 |
| teacher_arch | dinov3_vitb16 |
| teacher_image_size | 448 |
| teacher_patch_grid | 28 |
| method | local_contrast |
| tile_size / stride | 480 / 240 |
| max_tiles_per_image | 4 |
| min_small_boxes_per_tile | 3 |
| small_area_px | 1024 |
| teacher_batch | 16 |

## 3. 训练结果

| 指标 | 值 |
| --- | --- |
| best_epoch | 89 |
| best Precision | 0.37152 |
| best Recall | 0.30709 |
| best mAP50 | 0.28456 |
| best mAP50-95 | 0.16423 |
| last_epoch | 120 |
| last mAP50 | 0.27790 |
| last mAP50-95 | 0.15946 |
| best 到 last 的 mAP50-95 回落 | 0.00477 |
| 训练耗时 | 3.59 h |

### 3.1 area AP

| area | GT 数量 | Pred 数量 | AP50 | AP50-95 |
| --- | --- | --- | --- | --- |
| small | 26586 | 120665 | 0.26949 | 0.11527 |
| medium | 11105 | 36669 | 0.58095 | 0.39385 |
| large | 1068 | 2568 | 0.57615 | 0.43847 |

### 3.2 baseline 与 layer16 对比

| 实验 | overall mAP50-95 | small AP50-95 | 备注 |
| --- | --- | --- | --- |
| baseline_10pct_960 | 0.14439 | 0.11601 | 主对照 |
| small_tile_soft layer16 lam0.02 seed42 | 0.16477 | 0.11519 | overall 提升，small 未过门槛 |
| small_tile_soft layer19 lam0.02 seed42 | 0.16423 | 0.11527 | overall 提升，small 未过门槛 |

## 4. 可复现步骤

```bash
cd /home/fj/vfm-yolo-distillation
PYTHONUNBUFFERED=1 /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python scripts/run_smalltile_iteration.py --phase all --config configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_l19_seed42.yaml --python /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python
```

关键产物：

- results.csv: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42/results.csv
- best.pt: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42/weights/best.pt
- best.onnx: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42/weights/best.onnx
- area AP: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/reports/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42_area_ap.csv
- gate summary: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/reports/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l19_lam002_seed42_gate_summary.json

## 5. 分析

- layer19 没有解决小目标目标：small AP50-95=0.11527，相对 baseline 变化 -0.00074，距离 gate 仍差 0.00173。这说明单纯把辅助层后移到更语义化的层级，并没有让 small-tile target 更好地转化为小目标检测收益。
- overall 仍然稳定受益：mAP50-95=0.16423，相对 baseline 提升 +0.01984。layer16 best=0.16477、layer19 best=0.16423 非常接近，表明该辅助更像整体 objectness/regularization，而不是小目标专向增强。
- area 分布继续偏向中大目标：medium AP50-95=0.39385，large AP50-95=0.43847，small 明显落后。layer19 的 large AP 低于 layer16，提示更深层可能损失部分定位细节。
- best epoch=89，last mAP50-95 回落 0.00477，120 epoch 后没有刷新 best；后续仍用 best.pt 做 gate。

## 6. 可靠性与部署验收

- 训练完成 120/120，results.csv、best.pt、last.pt 完整落盘。
- 日志扫描未发现 OOM、out of memory、Traceback、RuntimeError、ConnectionResetError、FileNotFoundError。
- clean Ultralytics YOLO() 加载成功，best.pt 可导出 ONNX，best.onnx 已生成。
- teacher 仅训练期使用；评估和导出产物保持 YOLO-only。

## 7. 决策与下一步

当前 gate summary 决策：`run_layer_ablation`，原因是 small AP missed the gate。

下一步执行 multi-layer 16+19：

1. 新增 multi-layer small_tile_soft 配置，固定 lambda_objectness=0.02、seed=42、tile 设置不变。
2. 训练时同时挂 layer16 和 layer19 的 objectness head，将两层辅助 loss 做均值汇聚，保持总 lambda 量级不翻倍。
3. 完成后按同样标准执行 area AP、clean YOLO() 加载、ONNX 导出和 gate summary。
4. 只有 small AP50-95 >= 0.1170 且 overall mAP50-95 >= 0.14439 时，才补 seed 2026/3407。
