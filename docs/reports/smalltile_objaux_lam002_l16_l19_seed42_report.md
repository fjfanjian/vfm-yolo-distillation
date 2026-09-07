# Small-tile objectness auxiliary multi-layer 16+19 seed42 实验报告

## 1. 摘要

本实验是 high-density small-object tile objectness auxiliary 的 multi-layer 消融，固定 VisDrone 10% / imgsz=960 / YOLO26n / small_tile_soft / lambda_objectness=0.02 / seed=42，同时在 student layer 16 与 19 上挂 objectness auxiliary，并保持总 lambda 量级不翻倍。

结论：multi-layer 16+19 的 best overall mAP50-95 为 0.16524，相比 baseline 0.14439 提升 +0.02085，是 small-tile 系列当前 overall 最强结果；但 small AP50-95 为 0.11375，低于 baseline small 0.11601，也低于进入多 seed 的 0.11700 门槛。因此本分支不补 seed 2026/3407，small_tile_soft 的 layer16、layer19、16+19 三组 seed42 都应记录为“小目标定向失败但整体收益明显”的反例。

## 2. 实验配置

| 字段 | 内容 |
| --- | --- |
| 实验名称 | yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42 |
| 数据集 | VisDrone 10% label budget |
| 学生模型 | YOLO26n |
| 图像尺寸 | 960 |
| 训练轮数 | 120 |
| 辅助层 | student_layers=[16, 19], dims=[64, 128] |
| 配置文件 | configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_l16_l19_seed42.yaml |
| 服务器运行目录 | /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42 |
| 生成时间 | 2026-06-27 11:20:00 Asia/Shanghai |

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
| name | yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42 |

### 2.2 objectness / small-tile 参数

| 参数 | 值 |
| --- | --- |
| target_mode | small_tile_soft |
| student_layers | [16, 19] |
| student_dims | [64, 128] |
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
| cutmix | 0.0 |
| copy_paste | 0.0 |
| auto_augment | randaugment |
| erasing | 0.4 |

## 3. 训练结果

| 指标 | 值 |
| --- | --- |
| best_epoch | 82 |
| best Precision | 0.37507 |
| best Recall | 0.30295 |
| best mAP50 | 0.28484 |
| best mAP50-95 | 0.16524 |
| last_epoch | 120 |
| last mAP50 | 0.28178 |
| last mAP50-95 | 0.16096 |
| best 到 last 的 mAP50-95 回落 | 0.00428 |
| 最后 10 epoch mAP50-95 均值 | 0.15900 |
| 训练耗时 | 6.73 h |

### 3.1 area AP

| area | GT 数量 | Pred 数量 | AP50 | AP50-95 |
| --- | --- | --- | --- | --- |
| small | 26586 | 121545 | 0.26811 | 0.11375 |
| medium | 11105 | 36422 | 0.58112 | 0.39711 |
| large | 1068 | 2467 | 0.61066 | 0.48625 |

### 3.2 baseline / layer16 / layer19 对比

| 实验 | overall mAP50-95 | small AP50-95 | medium AP50-95 | large AP50-95 | 备注 |
| --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | 0.14439 | 0.11601 | N/A | N/A | 主对照 |
| small_tile_soft layer16 lam0.02 seed42 | 0.16477 | 0.11519 | 0.38861 | 0.48458 | overall 提升，small 未过门槛 |
| small_tile_soft layer19 lam0.02 seed42 | 0.16423 | 0.11527 | 0.39385 | 0.43847 | overall 提升，small 未过门槛 |
| small_tile_soft layer16+19 lam0.02 seed42 | 0.16524 | 0.11375 | 0.39711 | 0.48625 | overall 最强，small 进一步下降 |

## 4. 可复现步骤

```bash
cd /home/fj/vfm-yolo-distillation
PYTHONUNBUFFERED=1 /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python scripts/run_smalltile_iteration.py --phase all --config configs/experiments/dinov3_objectness_aux_smalltile_visdrone_10pct_imgsz960_lam002_l16_l19_seed42.yaml --python /home/fj/anaconda3/envs/vfm-yolo-distillation/bin/python
```

关键产物：

- results.csv: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42/results.csv
- best.pt: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42/weights/best.pt
- best.onnx: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42/weights/best.onnx
- area AP: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/reports/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42_area_ap.csv
- gate summary: /home/fj/vfm-yolo-distillation/runs/dinov3_objectness/reports/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smalltile_l16_l19_lam002_seed42_gate_summary.json

## 5. 分析

- overall 达到 small-tile 系列最高：best mAP50-95=0.16524，相比 baseline 提升 +0.02085，略高于 layer16 的 0.16477 与 layer19 的 0.16423。说明 multi-layer auxiliary 对整体 objectness/regularization 有收益。
- small AP 没有达成主线目标：small AP50-95=0.11375，相比 baseline small 变化 -0.00226，距离 gate 还差 0.00325。这比 layer16 的 0.11519、layer19 的 0.11527 更低，表明叠加层级没有增强小目标 AP。
- area 收益继续偏向中大目标：medium AP50-95=0.39711，large AP50-95=0.48625；small 的 prediction_count 达到 121545，但 AP50-95 下降，说明更多候选没有转化为更高质量的小目标检测。
- 收敛上 best epoch 在 82，last mAP50-95=0.16096，回落 0.00428；最后 10 个 epoch 均值 0.15900，后段没有继续刷新 best。close_mosaic 后的训练更像稳定收尾，而不是进一步提升。
- 训练耗时 6.73 h，明显高于单层 layer16/layer19 的约 3.6 h；在 small gate 未达标的前提下，不值得继续为该配置补多 seed。

## 6. 可靠性与部署验收

- 训练完成 120/120，results.csv、best.pt、last.pt 完整落盘。
- 日志扫描未发现 OOM、out of memory、Traceback、RuntimeError、ConnectionResetError、FileNotFoundError。
- clean Ultralytics YOLO() 加载成功，best.pt 可导出 ONNX，best.onnx 已生成。
- teacher 仅训练期使用；评估和导出产物保持 YOLO-only。

## 7. 决策与下一步

当前 gate summary 决策：`run_layer_ablation`，原因是 `small AP missed the gate; stop small-tile multi-seed and run layer ablation.`。

结合 layer16、layer19、multi-layer 三组结果，建议：

1. 停止 small_tile_soft lam0.02 的多 seed 扩展，避免在 small AP 未达标分支继续消耗预算。
2. multi-layer 16+19 可作为 overall 候选保留，但不作为小目标主结果。
3. 下一轮优先改 target，而不是继续改 layer：尝试 small-box foreground mask、center/scale-aware objectness 或 ignore-aware soft target，让辅助信号直接约束小框区域。
4. 复查 small prediction_count 与低 AP 的关系，抽样可视化小目标误检/低 IoU 案例，判断问题来自过多候选还是定位偏移。
