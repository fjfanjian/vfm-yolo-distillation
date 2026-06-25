# Small-tile objectness auxiliary seed42 实验报告

## 1. 摘要

本实验验证 high-density-small-object tile objectness auxiliary 在 VisDrone 10% / imgsz=960 / YOLO26n 设置下是否能优先突破 small AP。实验使用 small_tile_soft target、student_layer=16、lambda_objectness=0.02、seed=42，训练 120 epoch。

结论：overall mAP50-95 达到 0.16477，相比 baseline 0.14439 提升 +0.02038；但 small AP50-95 为 0.11519，低于 baseline small 0.11601，也未达到进入多 seed 的 0.11700 门槛。因此本分支不补多 seed，下一步转入 student_layer=19 消融。

## 2. 实验配置

| 字段 | 内容 |
| --- | --- |
| 实验名称 | yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42 |
| 数据集 | VisDrone 10% label budget |
| 学生模型 | YOLO26n |
| 图像尺寸 | 960 |
| 训练轮数 | 120 |
| 运行目录 | /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42 |
| 配置文件 | configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_seed42.yaml |
| 生成时间 | 2026-06-26 01:50:31 Asia/Shanghai |

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
| name | yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42 |

### 2.2 objectness / small-tile 参数

| 参数 | 值 |
| --- | --- |
| student_layer | 16 |
| student_dim | 64 |
| lambda_objectness | 0.02 |
| target_mode | small_tile_soft |
| teacher_arch | dinov3_vitb16 |
| teacher_image_size | 448 |
| teacher_patch_grid | 28 |
| tile_size / stride | 480 / 240 |
| max_tiles_per_image | 4 |
| min_small_boxes_per_tile | 3 |
| small_area_px | 1024 |
| teacher_batch | 16 |

### 2.3 数据增强

| 参数 | 值 |
| --- | --- |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |
| degrees | 0.0 |
| translate | 0.1 |
| scale | 0.5 |
| shear | 0.0 |
| perspective | 0.0 |
| flipud | 0.0 |
| fliplr | 0.5 |
| mosaic | 1.0 |
| mixup | 0.0 |
| copy_paste | 0.0 |
| auto_augment | randaugment |
| erasing | 0.4 |

## 3. 训练结果

| 指标 | 值 |
| --- | --- |
| best_epoch | 82 |
| best Precision | 0.37804 |
| best Recall | 0.30317 |
| best mAP50 | 0.28412 |
| best mAP50-95 | 0.16477 |
| last_epoch | 120 |
| last mAP50 | 0.28110 |
| last mAP50-95 | 0.16024 |
| best 到 last 的 mAP50-95 回落 | 0.00453 |
| 训练耗时 | 3.63 h |

### 3.1 area AP

| area | GT 数量 | Pred 数量 | AP50 | AP50-95 |
| --- | --- | --- | --- | --- |
| small | 26586 | 121389 | 0.27225 | 0.11519 |
| medium | 11105 | 36288 | 0.57342 | 0.38861 |
| large | 1068 | 2493 | 0.61494 | 0.48458 |

### 3.2 baseline 对比

| 实验 | overall mAP50-95 | small AP50-95 | 备注 |
| --- | --- | --- | --- |
| baseline_10pct_960 | 0.14439 | 0.11601 | 主对照 |
| small_tile_soft layer16 lam0.02 seed42 | 0.16477 | 0.11519 | overall 明显提升，small 未过门槛 |

## 4. 过程与可复现步骤

`ash
cd /home/fj/vfm-yolo-distillation
PYTHONUNBUFFERED=1 /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python scripts/run_smalltile_iteration.py --phase all --config configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_seed42.yaml --python /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python
/home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python scripts/run_smalltile_iteration.py --phase evaluate --config configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_seed42.yaml --python /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python
/home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python scripts/run_smalltile_iteration.py --phase export --config configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_seed42.yaml --python /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python
/home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python scripts/run_smalltile_iteration.py --phase decide --config configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_seed42.yaml --python /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python
`

关键产物：

- 
esults.csv: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42/results.csv
- est.pt: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42/weights/best.pt
- est.onnx: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42/weights/best.onnx
- area AP: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/reports/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42_area_ap.csv
- gate summary: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/reports/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_lam002_seed42_gate_summary.json

## 5. 分析

- overall 有明确收益：best mAP50-95 从 0.14439 提升到 0.16477，绝对提升 +0.02038，说明 small-tile objectness auxiliary 没有破坏主检测能力，甚至对中大目标或整体定位分类有帮助。
- small AP 没有达成主线目标：small AP50-95 为 0.11519，相对 baseline small 0.11601 下降 -0.00082，距离进入多 seed 的 0.11700 门槛还差 0.00181。
- area AP 呈现偏向中大目标的收益分布：medium AP50-95 0.38861、large AP50-95 0.48458，明显高于 small 的 0.11519。这提示 layer16 辅助可能更像整体 objectness 正则，而不是足够聚焦小目标密集区域。
- best epoch 在 82，last mAP50-95 回落 0.00453，120 epoch 后没有继续刷新 best；后续消融仍保留 120 epoch，但决策指标应继续使用 best.pt。

## 6. 问题与处理

- 训练本身完成，est.pt 和 last.pt 正常落盘。
- 初次 area AP 后处理曾因一次性送入全量图片触发 GPU OOM；后续已将 area AP 推理改为 batch=1 并逐图 streaming，补跑成功。
- ONNX 导出前服务器环境缺少 onnx，安装后 clean YOLO() load 和 ONNX export 均通过；部署期产物保持 YOLO-only。
- 日志中残留历史 Traceback/OOM 记录，但最终 eval/export/decide 都已成功，不能视为当前失败。

## 7. 决策与下一步

当前 gate summary 给出的决策是 run_layer_ablation：small AP missed the gate; stop small-tile multi-seed and run layer ablation.

下一步执行：

1. 固定 small_tile_soft、lambda_objectness=0.02、seed=42，启动 student_layer=19 消融。
2. 若 layer19 的 small AP50-95 达到 >=0.1170 且 overall mAP50-95 不低于 .14439，再补 seed 2026/3407。
3. 若 layer19 仍未突破 small AP，再准备 multi-layer 16+19 辅助方案；只在 seed42 达标后进入多 seed。
