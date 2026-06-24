# VisDrone YOLO26n 实验主报告（详细版）

生成时间：2026-06-22T23:47:48+08:00

## 1. 报告范围

本报告是当前统一主报告，覆盖 baseline、分辨率实验、DINOv3 global/patch/region-aware 蒸馏、DINOv3 类别无关 objectness 审计、tiled objectness 预训练，以及 objectness 负迁移定位消融。各实验目录保留原始产物，本报告汇总配置、步骤、结果、分析与复现路径。

## 2. 总览指标

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | last mAP50 | last mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | 99 | 0.3105 | 0.2392 | 0.1901 | 0.1007 | 0.1808 | 0.0942 |
| baseline_25pct_640 | 105 | 0.3607 | 0.2834 | 0.2386 | 0.1282 | 0.2323 | 0.1248 |
| baseline_full_640 | 101 | 0.4514 | 0.3379 | 0.3168 | 0.1762 | 0.3111 | 0.1716 |
| baseline_10pct_960 | 110 | 0.3779 | 0.3003 | 0.2572 | 0.1444 | 0.2542 | 0.1420 |
| baseline_full_train960 | 82 | 0.5392 | 0.4102 | 0.4110 | 0.2424 | 0.4078 | 0.2402 |
| objectness_lam100_ep20_ft | 100 | 0.3587 | 0.2840 | 0.2366 | 0.1295 | 0.2321 | 0.1262 |
| objectness_lam010_ep5_ft | 100 | 0.3537 | 0.2904 | 0.2415 | 0.1329 | 0.2368 | 0.1302 |
| objectness_lam010_ep10_ft | 107 | 0.3545 | 0.2824 | 0.2371 | 0.1295 | 0.2325 | 0.1271 |
| objectness_lam025_ep5_ft | 110 | 0.3423 | 0.2973 | 0.2425 | 0.1338 | 0.2380 | 0.1317 |
| dinov3_global_10pct_640 | 81 | 0.3052 | 0.2374 | 0.1885 | 0.1015 | 0.1830 | 0.0973 |
| dinov3_patch_b16_partial | 45 | 0.2763 | 0.2412 | 0.1762 | 0.0920 | 0.1725 | 0.0901 |
| dinov3_patch_b32_10pct_640 | 99 | 0.3135 | 0.2449 | 0.1924 | 0.1004 | 0.1850 | 0.0973 |
| dinov3_region_patch_10pct_640 | 107 | 0.3050 | 0.2458 | 0.1920 | 0.1018 | 0.1865 | 0.0982 |
| objaux_smallgt_lam002_seed42 | 85 | 0.3703 | 0.3024 | 0.2614 | 0.1455 | 0.2557 | 0.1420 |
| objaux_peak_lam002_seed42 | 81 | 0.3692 | 0.2995 | 0.2588 | 0.1442 | 0.2535 | 0.1400 |

## 3. 标签预算 baseline：10% / 25% / full，imgsz=640

### 配置

- 模型：YOLO26n
- 训练输入：640
- epoch：120
- 数据：VisDrone train 按标签预算抽样，val 使用完整验证集
- 优化器：Ultralytics auto AdamW

### 步骤

1. 分别训练 10%、25%、full 三个标签预算 baseline。
2. 读取各自 results.csv，按 best mAP50-95 选取最佳 epoch。
3. 对 10%/25%/full 关键模型执行 small/medium/large AP 统计。

### 结果

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | last mAP50 | last mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | 99 | 0.3105 | 0.2392 | 0.1901 | 0.1007 | 0.1808 | 0.0942 |
| baseline_25pct_640 | 105 | 0.3607 | 0.2834 | 0.2386 | 0.1282 | 0.2323 | 0.1248 |
| baseline_full_640 | 101 | 0.4514 | 0.3379 | 0.3168 | 0.1762 | 0.3111 | 0.1716 |

### 分析

- 标签预算提升带来稳定收益：10% mAP50-95=0.1007，25%=0.1282，full=0.1762。
- 10% baseline 是后续 DINOv3 蒸馏的主要对照组。
- full baseline 与 10% baseline 差距明显，说明当前问题首先是低标签数据效率，而不是单纯模型容量。


## 4. 分辨率实验：full-label 640/960/1280 验证与 960 训练

### 配置

- 模型：YOLO26n full-label baseline
- 对比：640 训练模型在 640/960/1280 下验证，以及 imgsz=960 训练模型
- 指标：整体 mAP 与 small/medium/large AP
- 权重：full baseline best.pt 与 full train960 best.pt

### 步骤

1. 先对 full-label 640 训练模型做 640、960、1280 多分辨率验证。
2. 再补充 full-label imgsz=960 训练。
3. 对 960 训练模型执行 area AP 统计并归档。

### 结果

| 实验 | area | AP50 | AP50-95 | GT | Pred |
| --- | --- | --- | --- | --- | --- |
| baseline_full_640_val640 | small | 0.2982 | 0.1242 | 26586 | 118787 |
| baseline_full_640_val640 | medium | 0.6483 | 0.4380 | 11105 | 37383 |
| baseline_full_640_val640 | large | 0.7382 | 0.6134 | 1068 | 2222 |
| baseline_full_640_val960 | small | 0.3834 | 0.1753 | 26586 | 124502 |
| baseline_full_640_val960 | medium | 0.6631 | 0.4693 | 11105 | 32200 |
| baseline_full_640_val960 | large | 0.7078 | 0.5885 | 1068 | 2445 |
| baseline_full_640_val1280 | small | 0.4171 | 0.1971 | 26586 | 126759 |
| baseline_full_640_val1280 | medium | 0.6590 | 0.4695 | 11105 | 31117 |
| baseline_full_640_val1280 | large | 0.6122 | 0.4921 | 1068 | 2617 |
| baseline_full_train960_val960 | small | 0.4057 | 0.1870 | 26586 | 122781 |
| baseline_full_train960_val960 | medium | 0.7112 | 0.5040 | 11105 | 32898 |
| baseline_full_train960_val960 | large | 0.7546 | 0.6358 | 1068 | 2580 |

补充 10% 标签主对照：`baseline_10pct_960` 已完成 120 epoch，best epoch=110，best mAP50=0.2572、best mAP50-95=0.1444。该结果后续作为 960 分辨率下 objectness 预训练/联合训练的主要对照，详见第 5 节。

### 分析

- full train960 的整体 mAP50-95=0.2424，显著高于 full train640 的 0.1762。
- 小目标 AP 随验证分辨率升高明显改善，说明 VisDrone 的低 mAP50 与小目标尺度强相关。
- 1280 验证提升 small AP，但 large AP 下滑，后续部署需要在小目标收益和整体稳定性之间折中。
- 10% 标签在 960 下的 mAP50-95=0.144，高于 10%/640 baseline 的 0.1007，说明少标签场景同样显著受益于更高输入分辨率。


## 5. 10% 标签 baseline：imgsz=960 主对照

### 配置

- 模型：YOLO26n
- 数据：VisDrone train 10% 标签子集，val 使用完整验证集
- 训练输入：960
- epoch：120
- batch：16
- 优化器：Ultralytics auto AdamW
- 输出目录：`runs/baselines/yolo26n_visdrone_10pct_imgsz960`

### 步骤

1. 使用 `configs/datasets/visdrone_10pct.yaml` 作为训练集配置，保持与 10%/640 baseline 相同的数据划分。
2. 将训练和验证输入分辨率从 640 提升到 960，其他训练轮数、seed、优化器策略保持一致。
3. 训练完成后读取 `results.csv`，按 best mAP50-95 选择最佳 epoch。
4. 同步非权重产物到本地，包括 `results.csv`、`args.yaml`、`results.png`、PR/F1 曲线和可视化 batch 图；权重文件不下载到本地仓库。

### 结果

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | last mAP50 | last mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | 99 | 0.3105 | 0.2392 | 0.1901 | 0.1007 | 0.1808 | 0.0942 |
| baseline_10pct_960 | 110 | 0.3779 | 0.3003 | 0.2572 | 0.1444 | 0.2542 | 0.1420 |

| 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| all | 0.376 | 0.301 | 0.257 | 0.144 |
| pedestrian | 0.395 | 0.437 | 0.366 | 0.160 |
| people | 0.394 | 0.280 | 0.229 | 0.0801 |
| bicycle | 0.183 | 0.0847 | 0.0572 | 0.0242 |
| car | 0.618 | 0.726 | 0.704 | 0.464 |
| van | 0.369 | 0.326 | 0.266 | 0.179 |
| truck | 0.367 | 0.233 | 0.181 | 0.115 |
| tricycle | 0.297 | 0.183 | 0.124 | 0.0663 |
| awning-tricycle | 0.184 | 0.107 | 0.0664 | 0.0428 |
| bus | 0.488 | 0.307 | 0.277 | 0.193 |
| motor | 0.468 | 0.323 | 0.301 | 0.118 |

### 分析

- 10% 标签从 640 提升到 960 后，mAP50-95 从 0.1007 提升到 0.1444，绝对提升 +0.0437；mAP50 从 0.1901 提升到 0.2572，绝对提升 +0.0671。
- 该提升幅度明显大于当前 DINOv3 feature distillation 的边缘收益，进一步说明 VisDrone 少标签小目标检测首先受输入尺度限制。
- 类别层面，`car` 的 mAP50=0.704、mAP50-95=0.464，显著高于其他类别；`bicycle`、`awning-tricycle`、`tricycle` 仍然较弱，说明高分辨率不能完全解决稀有/细粒度小目标问题。
- 后续 objectness pretrain 或 tiled DINO 审计应以该实验作为 960 主对照，而不是继续只和 10%/640 baseline 比较。


## 6. DINOv3 global feature distillation

### 配置

- 学生：YOLO26n
- 教师：DINOv3 ViT-B/16
- 标签预算：10%
- 输入：YOLO imgsz=640，teacher input=224
- 蒸馏：全局 pooled feature / cosine loss，lambda=0.1
- epoch：120，batch=16

### 步骤

1. 加载冻结 DINOv3 teacher。
2. 捕获 YOLO 高层特征并映射到 teacher global embedding。
3. 检测 loss 与 global feature distill loss 联合优化。
4. 完成训练后与 10% baseline 对比。

### 结果

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | last mAP50 | last mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | 99 | 0.3105 | 0.2392 | 0.1901 | 0.1007 | 0.1808 | 0.0942 |
| dinov3_global_10pct_640 | 81 | 0.3052 | 0.2374 | 0.1885 | 0.1015 | 0.1830 | 0.0973 |

### 分析

- global 蒸馏 mAP50-95=0.1015，相对 baseline 变化 +0.0008。
- mAP50 低于 baseline，mAP50-95 略高，整体收益很小。
- 全局语义特征对密集小目标检测不够精细，不能直接提供定位友好的监督。


## 7. DINOv3 全图 patch-token distillation：batch=16 partial 与 batch=32

### 配置

- 学生：YOLO26n
- 教师：DINOv3 ViT-B/16
- 标签预算：10%
- teacher input=224，patch grid=14x14
- 蒸馏：全图 patch-token cosine loss，lambda=0.05
- batch=16 版本提前停止在 47 epoch；batch=32 完成 120 epoch

### 步骤

1. 先运行 batch=16 patch 蒸馏，观察 GPU 利用率和早期指标。
2. 发现 GPU 利用率偏低后改为 batch=32。
3. batch=32 完成训练后执行 standard val、small/medium/large AP、checkpoint clean 验证。

### 结果

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | last mAP50 | last mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | 99 | 0.3105 | 0.2392 | 0.1901 | 0.1007 | 0.1808 | 0.0942 |
| dinov3_patch_b16_partial | 45 | 0.2763 | 0.2412 | 0.1762 | 0.0920 | 0.1725 | 0.0901 |
| dinov3_patch_b32_10pct_640 | 99 | 0.3135 | 0.2449 | 0.1924 | 0.1004 | 0.1850 | 0.0973 |

| 实验 | area | AP50 | AP50-95 | GT | Pred |
| --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | small | 0.1754 | 0.0640 | 26586 | 114472 |
| baseline_10pct_640 | medium | 0.5127 | 0.3283 | 11105 | 42984 |
| baseline_10pct_640 | large | 0.6004 | 0.4714 | 1068 | 2565 |
| dinov3_patch_b32_10pct_640 | small | 0.1752 | 0.0631 | 26586 | 114831 |
| dinov3_patch_b32_10pct_640 | medium | 0.5176 | 0.3293 | 11105 | 42250 |
| dinov3_patch_b32_10pct_640 | large | 0.5949 | 0.4671 | 1068 | 2565 |
| dinov3_region_patch_10pct_640 | small | 0.1794 | 0.0659 | 26586 | 116965 |
| dinov3_region_patch_10pct_640 | medium | 0.5168 | 0.3295 | 11105 | 40762 |
| dinov3_region_patch_10pct_640 | large | 0.6095 | 0.4755 | 1068 | 2573 |

### 分析

- patch b32 的 mAP50=0.1924 略高于 baseline，但 mAP50-95=0.1004 低于 baseline。
- small AP50-95 从 baseline 0.0640 降到 patch b32 0.0631，说明全图 patch 对齐没有改善小目标。
- 主要问题是背景 token 占比高，小目标区域信号被稀释；此外原始 patch b32 checkpoint 携带 hook pickle 引用，后续以 best_clean.pt 作为可加载权重。


## 8. DINOv3 region-aware patch distillation

### 配置

- 学生：YOLO26n
- 教师：DINOv3 ViT-B/16
- 标签预算：10%
- YOLO imgsz=640，teacher input=448，patch grid=28x28
- 学生层：YOLO layer 16
- 蒸馏：仅在 GT box 扩张区域计算 patch-token cosine loss
- lambda_region=0.05，box_expand_tokens=1，batch=16，epoch=120

### 步骤

1. 先做 1 epoch smoke test，确认训练、验证和权重保存流程。
2. 验证 best.pt 可被普通 YOLO 直接加载，避免 patch b32 的 hook pickle 问题。
3. 启动正式 120 epoch 训练。
4. 训练完成后执行 small/medium/large AP 统计并更新主报告。

### 结果

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | last mAP50 | last mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | 99 | 0.3105 | 0.2392 | 0.1901 | 0.1007 | 0.1808 | 0.0942 |
| dinov3_patch_b32_10pct_640 | 99 | 0.3135 | 0.2449 | 0.1924 | 0.1004 | 0.1850 | 0.0973 |
| dinov3_region_patch_10pct_640 | 107 | 0.3050 | 0.2458 | 0.1920 | 0.1018 | 0.1865 | 0.0982 |
| objaux_smallgt_lam002_seed42 | 85 | 0.3703 | 0.3024 | 0.2614 | 0.1455 | 0.2557 | 0.1420 |
| objaux_peak_lam002_seed42 | 81 | 0.3692 | 0.2995 | 0.2588 | 0.1442 | 0.2535 | 0.1400 |

| 实验 | area | AP50 | AP50-95 | GT | Pred |
| --- | --- | --- | --- | --- | --- |
| baseline_10pct_640 | small | 0.1754 | 0.0640 | 26586 | 114472 |
| baseline_10pct_640 | medium | 0.5127 | 0.3283 | 11105 | 42984 |
| baseline_10pct_640 | large | 0.6004 | 0.4714 | 1068 | 2565 |
| dinov3_patch_b32_10pct_640 | small | 0.1752 | 0.0631 | 26586 | 114831 |
| dinov3_patch_b32_10pct_640 | medium | 0.5176 | 0.3293 | 11105 | 42250 |
| dinov3_patch_b32_10pct_640 | large | 0.5949 | 0.4671 | 1068 | 2565 |
| dinov3_region_patch_10pct_640 | small | 0.1794 | 0.0659 | 26586 | 116965 |
| dinov3_region_patch_10pct_640 | medium | 0.5168 | 0.3295 | 11105 | 40762 |
| dinov3_region_patch_10pct_640 | large | 0.6095 | 0.4755 | 1068 | 2573 |

### 分析

- region-aware mAP50-95=0.1018，相对 baseline 变化 +0.0011。
- small AP50-95=0.0659，相对 baseline 变化 +0.0019。
- 这是当前 DINOv3 系列中最好的设置，说明区域化蒸馏方向优于全图 patch 蒸馏。
- 但提升仍是边缘级别；日志尾部的 ConnectionResetError 出现在训练完成后的 dataloader 收尾阶段，权重和 results.csv 已完整落盘。


## 9. DINOv3 类别无关 objectness 审计：第一轮

### 配置

- 目的：验证 DINOv3 是否能在无标注 VisDrone 图像中提供“哪里像物体”的类别无关 objectness 信号。
- 数据：从 VisDrone train 随机抽样 300 张图像，seed=42。
- 教师：DINOv3 ViT-B/16，冻结权重。
- 输入：每张图 resize 到 448，patch size=16，得到 28x28 patch token。
- 方法：`border_contrast + PCA foregroundness` 融合。GT 框只用于离线评估覆盖率，不参与生成 objectness map。
- 输出：`runs/dinov3_objectness/audit_train300_448/objectness_summary.csv`、`objectness_metrics.csv` 和 24 张 overlay。

### 步骤

1. 对每张图提取 DINOv3 patch tokens。
2. 用图像边缘 patch 估计背景 token，计算每个 patch 与背景均值的 cosine 差异。
3. 对 patch tokens 做 PCA，用第一主成分估计前景/背景分离方向，并用边缘区域校正方向。
4. 将 border contrast 与 PCA foregroundness 归一化后平均，得到 28x28 objectness map。
5. 把 GT box 投影到 28x28 patch grid，仅用于统计 top10/top20 高响应区域是否覆盖真实目标。
6. 生成高响应红黄高亮 overlay，用人工观察判断高响应区域是否真正在物体上。

### 结果

| 指标 | 数值 |
| --- | --- |
| images | 300 |
| boxes | 16909 |
| small_boxes | 10204 |
| top10_box_recall | 0.1855 |
| top20_box_recall | 0.3330 |
| small_top10_recall | 0.1643 |
| small_top20_recall | 0.3023 |
| mean_foreground_score | 0.6042 |
| mean_background_score | 0.4942 |

### 分析

- 数值上，GT 区域平均分 0.6042，高于背景区域 0.4942，说明 DINOv3 patch token 中确实存在一定前景/背景差异。
- 但 overlay 显示，高响应区域主要覆盖道路、中心透视区域或大块场景结构，而不是稳定突出小车、行人等实例目标。
- small top20 recall=0.3023 的含义需要谨慎解释：很多小目标位于道路区域内，因此被道路显著性“间接覆盖”，不代表 DINO objectness 已经准确定位小目标实例。
- 该轮审计结论是负向但有价值：全图级 `border_contrast + PCA` 更像 scene/layout saliency，不适合作为后续 YOLO objectness pretrain 的直接监督信号。
- 下一步应转向局部实例显著性信号，包括 local contrast、local residual 以及后续 tiled DINO 审计。

## 10. DINOv3 类别无关 objectness 审计：第二轮局部方法

### 配置

- 目的：针对第一轮高亮道路/场景结构的问题，改用局部邻域差异来估计实例级 objectness。
- 数据：与第一轮一致，VisDrone train 随机抽样 300 张，seed=42。
- 教师：DINOv3 ViT-B/16，输入 448，patch grid=28x28。
- 方法：`local_contrast`、`local_residual`、`local_fusion`。
- 输出目录：`runs/dinov3_objectness/audit_train300_448_local_contrast`、`audit_train300_448_local_residual`、`audit_train300_448_local_fusion`。

### 步骤

1. `local_contrast`：对每个 DINO patch token 与 3x3 邻域平均 token 做 cosine 差异，高分表示局部语义突变。
2. `local_residual`：对原始 token feature 做 5x5 局部平滑，取残差强度，高分表示局部高频结构。
3. `local_fusion`：将 `local_contrast` 与 `local_residual` 归一化后平均。
4. 对三种方法分别统计 top10/top20 box recall 和 small object recall，并生成 24 张 overlay。

### 结果

| 方法 | top10_box_recall | top20_box_recall | small_top10_recall | small_top20_recall | foreground_score | background_score |
| --- | --- | --- | --- | --- | --- | --- |
| border_pca_first_round | 0.1855 | 0.3330 | 0.1643 | 0.3023 | 0.6042 | 0.4942 |
| local_contrast | 0.4706 | 0.6530 | 0.3141 | 0.5112 | 0.2919 | 0.1662 |
| local_residual | 0.4295 | 0.6220 | 0.2768 | 0.4822 | 0.4681 | 0.3256 |
| local_fusion | 0.4579 | 0.6463 | 0.3033 | 0.5066 | 0.3855 | 0.2478 |

### 分析

- 局部方法显著提高 GT box 覆盖率：`local_contrast` 的 top20_box_recall=0.6530，高于第一轮 0.3330。
- 小目标覆盖也明显提升：`local_contrast` 的 small_top20_recall=0.5112，高于第一轮 0.3023。
- 可视化上，局部方法不再把整条道路均匀染黄，高响应更偏局部突起、车辆边缘、行人/车流密集位置，但仍存在只突出少数实例、对密集小目标响应不均匀的问题。
- 当前最优候选是 `local_contrast`，它在整体 top10/top20 和 small top10/top20 四个覆盖率指标上均为最高。
- 第二轮结果支持继续推进，但还不建议直接进入最终训练；下一步应做 tiled DINO 或 crop-level local contrast，验证小目标在更高局部尺度下是否能被更稳定地突出。

## 11. DINOv3 类别无关 objectness 审计：第三轮 tiled local contrast

### 配置

- 目的：验证切图后 DINOv3 是否能更稳定地看到 VisDrone 小目标实例。
- 数据：与前两轮一致，VisDrone train 随机抽样 300 张，seed=42。
- 教师：DINOv3 ViT-B/16，teacher input=448。
- 方法：`local_contrast`。
- 切图：原图坐标下 `tile_size=480`，`tile_stride=240`，每个 tile resize 到 448 后提取 DINO patch tokens。
- 合并：每个 tile 的 28x28 objectness map 插值回 tile 原始区域，重叠区域取平均，形成原图坐标下的 objectness map。
- 输出目录：`runs/dinov3_objectness/audit_train300_448_tiled_local_contrast_t480_s240`。

### 步骤

1. 对每张图按 480x480 窗口、240 stride 切图。
2. 对每个 tile 运行 DINOv3，计算 3x3 邻域 `local_contrast`。
3. 将 tile-level objectness 重新映射回原图坐标并融合。
4. 用 GT box 只做离线覆盖率评估，不参与 objectness 生成。
5. 生成 24 张 overlay，人工检查高响应区域是否更接近车辆/行人等实例目标。

### 结果

| 方法 | tile_size | stride | top10_box_recall | top20_box_recall | small_top10_recall | small_top20_recall | foreground_score | background_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| border_pca_first_round | N/A | N/A | 0.1855 | 0.3330 | 0.1643 | 0.3023 | 0.6042 | 0.4942 |
| local_contrast_full_image | N/A | N/A | 0.4706 | 0.6530 | 0.3141 | 0.5112 | 0.2919 | 0.1662 |
| tiled_local_contrast | 480 | 240 | 0.7946 | 0.8964 | 0.6866 | 0.8352 | 0.3192 | 0.1742 |

### 分析

- tiled local contrast 显著提升 GT 覆盖率：top20_box_recall 从 full-image local_contrast 的 0.6530 提升到 0.8964。
- 小目标覆盖提升更明显：small_top20_recall 从 0.5112 提升到 0.8352，说明切图确实缓解了整图 resize 下小目标过小的问题。
- 可视化上，高响应不再整片覆盖道路，更多落在局部实例、目标边缘和交通参与者附近；这比第一轮全图显著性更接近“类别无关目标性”。
- 仍需注意噪声：local contrast 也会响应灯杆、广告牌、斑马线、建筑边缘、施工结构等高频局部纹理。它适合作为 soft objectness 或候选区域信号，不适合直接当作硬伪框。
- 该结果已经达到进入短程 objectness pretrain smoke 的条件，但训练时应使用软标签、低权重和强对照，避免学生学成“纹理/边缘检测器”。

## 12. DINOv3 tiled objectness pretrain + 10%/960 fine-tune

### 配置

- 目的：验证第三轮审计得到的 `tiled local_contrast` soft objectness map，是否能作为无标注预训练信号提升 10% 标签 YOLO26n 检测。
- 学生：YOLO26n，部署期仍为 YOLO-only。
- 无标注预训练数据：VisDrone full train 图像；GT 标签在 objectness map 生成中不参与，只由 Ultralytics dataloader 提供图像。
- objectness teacher：DINOv3 ViT-B/16，冻结权重，teacher input=448，patch grid=28x28。
- objectness 生成：原图按 `tile_size=480`、`tile_stride=240` 切图，每个 tile resize 到 448 后计算 DINO patch-token `local_contrast`，再插值回 YOLO layer 16 特征网格。
- 学生监督位置：YOLO layer 16，student_dim=64，临时 `1x1 objectness head`；保存 checkpoint 前剥离该 head，因此 fine-tune 和部署权重仍是普通 YOLO 权重。
- 预训练主设置：imgsz=960，epoch=20，batch=8，`lambda_objectness=1.0`，loss=BCE 与 Dice 的均值。
- 微调设置：使用预训练 `last.pt` 初始化，`configs/datasets/visdrone_10pct.yaml`，imgsz=960，epoch=120，batch=16，seed=42，optimizer=auto，close_mosaic=10。
- 输出目录：`runs/dinov3_objectness/yolo26n_visdrone_full_imgsz960_dinov3_objectness_pretrain` 与 `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objectness_pretrain_ft`。

### 步骤

1. 先运行 1 epoch smoke，确认 DINO teacher、YOLO hook、soft-map cache、临时 head 剥离和普通 YOLO 验证流程可用。
2. 启动 full train 图像上的 objectness-only pretrain，训练过程中即时生成或读取 tiled local_contrast soft map cache。
3. 预训练结束后检查 `weights/best.pt` 与 `weights/last.pt` 均可被普通 YOLO 加载。
4. 使用预训练 `last.pt` 进行 10%/960 标注微调 120 epoch。
5. 用 `results.csv` 按 best mAP50-95 选取最佳 epoch，并与 `baseline_10pct_960` 对比。

### 结果

| 实验 | pretrain lambda | pretrain epoch | fine-tune best epoch | P | R | mAP50 | mAP50-95 | 相对 10%/960 baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | N/A | N/A | 110 | 0.3779 | 0.3003 | 0.2572 | 0.1444 | 0.0000 |
| objectness_lam100_ep20_ft | 1.0 | 20 | 100 | 0.3587 | 0.2840 | 0.2366 | 0.1295 | -0.0149 |

预训练最后一轮的 objectness 记录如下。注意：Ultralytics CSV 列名仍显示为 `train/box_loss`、`train/cls_loss`、`train/dfl_loss`，但在该自定义 trainer 中分别对应 `objectness_loss`、`objectness_bce`、`objectness_dice`。

| 实验 | epoch | objectness_loss | BCE | Dice | 预训练耗时秒 |
| --- | --- | --- | --- | --- | --- |
| objectness_lam100_ep20 | 20 | 0.5275 | 0.3971 | 0.6580 | 1110.6 |

### 分析

- 主实验 best mAP50-95=0.1295，低于 `baseline_10pct_960` 的 0.1444，绝对下降 -0.0149；mAP50 也从 0.2572 降到 0.2366。
- 这说明 tiled local_contrast 虽然在审计阶段有较高 GT 覆盖率，但作为纯预训练密集监督时没有转化为检测收益。
- 可能原因是 soft objectness map 仍包含道路边缘、斑马线、灯杆、广告牌、阴影等局部高频结构；YOLO 预训练学到的是“局部突变/纹理显著性”，而不是检测任务需要的类别可分、框可回归特征。
- 纯 objectness pretrain 不包含 box/class/dfl 检测目标，可能改变了 layer 16 小目标检测特征的分布，导致后续 10% 标注微调无法完全拉回。
- 日志中的 `ConnectionResetError` 出现在预训练结束后的 dataloader/pin_memory 收尾阶段；之后 fine-tune 正常启动并完整落盘，因此不作为训练失败处理。

### 可复现信息

| 项目 | 路径 / 命令 |
| --- | --- |
| 主训练脚本 | `scripts/train_dinov3_objectness_pretrain.py` |
| 主实验配置 | `configs/experiments/dinov3_objectness_pretrain_visdrone_full_imgsz960.yaml` |
| soft map cache | `/home/featurize/vfm-yolo-distillation/runs/dinov3_objectness/cache/tiled_local_contrast_t480_s240_l16_imgsz960` |
| 训练队列日志 | `runs/dinov3_objectness/logs/yolo26n_objectness_pretrain_ft_10pct960.log` |
| 结构化汇总 | `runs/reports/tables/objectness_pretrain_ablation_summary.csv`、`runs/reports/tables/objectness_reproducibility_index.csv` |

主实验复现命令：

```bash
python scripts/train_dinov3_objectness_pretrain.py \
  --config configs/experiments/dinov3_objectness_pretrain_visdrone_full_imgsz960.yaml \
  --name yolo26n_visdrone_full_imgsz960_dinov3_objectness_pretrain

yolo detect train \
  model=runs/dinov3_objectness/yolo26n_visdrone_full_imgsz960_dinov3_objectness_pretrain/weights/last.pt \
  data=configs/datasets/visdrone_10pct.yaml \
  imgsz=960 epochs=120 batch=16 device=0 workers=8 seed=42 \
  project=/home/featurize/vfm-yolo-distillation/runs/dinov3_objectness \
  name=yolo26n_visdrone_10pct_imgsz960_dinov3_objectness_pretrain_ft \
  exist_ok=True optimizer=auto patience=100 close_mosaic=10 amp=True plots=True
```

## 13. Objectness 预训练定位消融：lambda 与预训练时长

### 配置

- 目的：定位第 12 节负迁移是否主要来自 objectness 约束过强或预训练过久。
- 固定项：YOLO26n、VisDrone full train objectness pretrain、DINOv3 ViT-B/16、teacher input=448、`local_contrast`、`tile_size=480`、`tile_stride=240`、student layer 16、imgsz=960、pretrain batch=8、fine-tune 数据与第 12 节相同。
- 变量：`lambda_objectness` 与预训练 epoch。
- 三组设置：`lam010_ep5`、`lam010_ep10`、`lam025_ep5`。
- 队列日志：`runs/dinov3_objectness/logs/objectness_ablation_queue.log`。

### 步骤

1. 复用第 12 节已经生成的 tiled local_contrast soft-map cache，避免重新计算 teacher map 造成额外变量。
2. 依次运行三组 objectness pretrain，每组完成后立即使用其 `last.pt` 做 10%/960 fine-tune。
3. 每组 fine-tune 均使用相同的 120 epoch、batch=16、seed=42、optimizer=auto、close_mosaic=10。
4. 训练完成后从各自 `results.csv` 中抽取 best mAP50-95、last mAP50-95 与预训练最后一轮 objectness loss。
5. 与 `baseline_10pct_960` 和第 12 节 `lambda=1.0, ep20` 主实验做对比。

### 结果

| 实验 | lambda | pretrain epoch | pretrain objectness_loss | pretrain BCE | pretrain Dice | best epoch | P | R | mAP50 | mAP50-95 | 相对 baseline | 相对 lam100_ep20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lam010_ep5 | 0.1 | 5 | 0.0535 | 0.4015 | 0.6679 | 100 | 0.3537 | 0.2904 | 0.2415 | 0.1329 | -0.0115 | 0.0034 |
| lam010_ep10 | 0.1 | 10 | 0.0530 | 0.3984 | 0.6613 | 107 | 0.3545 | 0.2824 | 0.2371 | 0.1295 | -0.0149 | 0.0000 |
| lam025_ep5 | 0.25 | 5 | 0.1337 | 0.4015 | 0.6678 | 110 | 0.3423 | 0.2973 | 0.2425 | 0.1338 | -0.0106 | 0.0043 |

权重路径：

| 实验 | best.pt |
| --- | --- |
| lam010_ep5 | `/home/featurize/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objpre_lam010_ep5_ft/weights/best.pt` |
| lam010_ep10 | `/home/featurize/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objpre_lam010_ep10_ft/weights/best.pt` |
| lam025_ep5 | `/home/featurize/vfm-yolo-distillation/runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objpre_lam025_ep5_ft/weights/best.pt` |

### 分析

- 最好的消融是 `lam025_ep5`，best mAP50-95=0.1338；其次是 `lam010_ep5`，best mAP50-95=0.1329。二者均高于第 12 节 `lambda=1.0, ep20` 的 0.1295，但仍低于 `baseline_10pct_960` 的 0.1444。
- `lam010_ep10` 退回到 0.1295，几乎等同主实验，说明即使低权重，预训练时间加长也没有带来正收益。
- 降低权重和缩短预训练只能缓解一小部分负迁移，不能消除负迁移；因此问题不只是“约束过强/过久”，更可能是纯 objectness pretrain 目标本身与 YOLO 检测目标不对齐。
- 预训练阶段 objectness loss 能下降，但 fine-tune mAP 没有超过 baseline，说明“拟合 DINO objectness map”不是检测性能提升的充分条件。
- 当前证据支持放弃继续扩展纯 objectness pretrain，转向检测 loss 主导的联合训练式弱辅助，或者只在高置信 top-k 区域做 ignore-aware 辅助，而不是对整图密集回归 objectness map。

### 可复现信息

消融配置均已同步到 `configs/experiments/`，队列顺序为：`lam010_ep5 -> lam010_ep10 -> lam025_ep5`。每组预训练命令形如：

```bash
python scripts/train_dinov3_objectness_pretrain.py \
  --config configs/experiments/dinov3_objectness_pretrain_visdrone_full_imgsz960_lam010_ep5.yaml \
  --name yolo26n_visdrone_full_imgsz960_dinov3_objpre_lam010_ep5
```

每组 fine-tune 命令形如：

```bash
yolo detect train \
  model=runs/dinov3_objectness/yolo26n_visdrone_full_imgsz960_dinov3_objpre_lam010_ep5/weights/last.pt \
  data=configs/datasets/visdrone_10pct.yaml \
  imgsz=960 epochs=120 batch=16 device=0 workers=8 seed=42 \
  project=/home/featurize/vfm-yolo-distillation/runs/dinov3_objectness \
  name=yolo26n_visdrone_10pct_imgsz960_dinov3_objpre_lam010_ep5_ft \
  exist_ok=True optimizer=auto patience=100 close_mosaic=10 amp=True plots=True
```

## 14. DINOv3 relation distillation：10%/960

### 实验目的

前几轮 global、patch 与 region-aware patch 蒸馏主要约束单点特征或区域特征，收益较弱。relation distillation 改为约束 patch token 之间的相似度结构：不要求 YOLO 学生逐点复刻 DINOv3 的特征值，而是让学生特征在局部 token 之间形成与 DINOv3 老师相近的关系矩阵。该实验用于验证“让学生学习 DINO 如何组织视觉区域关系”是否比直接特征模仿更适合少标签 VisDrone 检测。

### 配置

- 学生模型：`YOLO26n`，部署阶段仍为 YOLO-only。
- 训练数据：`configs/datasets/visdrone_10pct.yaml`，VisDrone train 10% 标注子集，验证使用完整 val。
- 训练参数：`imgsz=960`、`epochs=120`、`batch=16`、`workers=8`、`seed=42`、`optimizer=auto`、`patience=100`、`close_mosaic=10`、`amp=True`。
- DINOv3 老师：`dinov3_vitb16`，权重 `/home/featurize/work/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`。
- 特征对齐：老师输入 `448`，patch grid `28x28`，teacher dim `768`；学生取 YOLO layer `16`，student dim `64`，经 `1x1 Conv` 投影到 768 维。
- 关系蒸馏：每 batch 从 token 中采样最多 `256` 个 token，计算学生/老师 token 的归一化关系矩阵并做 MSE；损失权重 `lambda_relation=0.05`。
- 配置文件：`configs/experiments/dinov3_relation_distill_visdrone_10pct_imgsz960.yaml`。

### 可复现步骤

1. 在本地实现 `src/vfm_yolo_distillation/relation_distillation.py` 与 `scripts/train_dinov3_relation_distill.py`，并提交到 GitHub。
2. 远程服务器拉取对应提交；由于远程 GitHub TLS 拉取异常，实际使用本地 git bundle 同步到服务器，远程仓库最终位于提交 `f695ad6`。
3. 修复远程运行环境：安装 `opencv-python-headless==4.11.0.86`，创建 `/home/featurize/datasets/visdrone -> /home/featurize/visdrone` 数据软链，并执行 `scripts/prepare_visdrone.py` 生成 10% split。
4. 启动训练命令：

```bash
cd /home/featurize/vfm-yolo-distillation
PYTHONPATH=src:. nohup python3 scripts/train_dinov3_relation_distill.py \
  --config configs/experiments/dinov3_relation_distill_visdrone_10pct_imgsz960.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_relation \
  > runs/dinov3_distill/logs/yolo26n_visdrone_10pct_imgsz960_dinov3_relation.log 2>&1 < /dev/null &
```

5. 训练完成后读取 `results.csv`，按 best mAP50-95 选取最佳 epoch，并与同分辨率 `baseline_10pct_960` 对比。
6. 从本地 `runs/baselines/yolo26n_visdrone_10pct_imgsz960/weights/best.pt` 同步 baseline 权重到远程同名目录，补跑同口径 small/medium/large AP。
7. 使用 `scripts/evaluate_area_ap.py` 对 baseline 与 relation best 权重执行面积分组 AP 统计，参数均为 `--data configs/datasets/visdrone.yaml --imgsz 960 --conf 0.001 --iou 0.7 --max-det 300`。

### 整体指标

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | 相对 baseline | best 权重 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | 110 | 0.3779 | 0.3003 | 0.2572 | 0.1444 | 0.0000 | `runs/baselines/yolo26n_visdrone_10pct_imgsz960/weights/best.pt` |
| dinov3_relation_10pct_960 | 105 | 0.3794 | 0.3055 | 0.2612 | 0.1448 | +0.0004 | `runs/dinov3_distill/yolo26n_visdrone_10pct_imgsz960_dinov3_relation/weights/best.pt` |

### small / medium / large AP

| 实验 | area | AP50 | AP50-95 | GT | Pred | AP50-95 差值 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | small | 0.2733 | 0.1160 | 26586 | 122198 | 0.0000 |
| dinov3_relation_10pct_960 | small | 0.2739 | 0.1155 | 26586 | 122603 | -0.0005 |
| baseline_10pct_960 | medium | 0.5745 | 0.3884 | 11105 | 34764 | 0.0000 |
| dinov3_relation_10pct_960 | medium | 0.5831 | 0.3949 | 11105 | 34374 | +0.0065 |
| baseline_10pct_960 | large | 0.6196 | 0.4885 | 1068 | 2545 | 0.0000 |
| dinov3_relation_10pct_960 | large | 0.6154 | 0.4760 | 1068 | 2414 | -0.0126 |

### 分析

- relation distillation 的总体 best mAP50-95=0.1448，仅比 `baseline_10pct_960` 的 0.1444 高 +0.0004，属于边缘级提升，不能视为稳定有效收益。
- 分面积指标显示，小目标 AP50-95 从 0.1160 微降到 0.1155；这说明 relation distillation 没有解决本项目最核心的 VisDrone 小目标检测瓶颈。
- 中目标 AP50-95 从 0.3884 提升到 0.3949，是本轮总体 mAP 微弱提升的主要来源；大目标 AP50-95 从 0.4885 降到 0.4760。
- 关系蒸馏没有像 objectness pretrain 那样造成明显负迁移，说明“弱辅助 + 检测 loss 主导”的方向比纯预训练更安全；但当前关系约束对小目标没有正向作用，可能因为 28x28 teacher token grid 对 VisDrone 极小目标仍然过粗。
- 训练结束后 `best.pt` 中残留 forward hook 引用，直接评估会触发反序列化问题；本轮 area AP 使用只读兼容 wrapper 完成统计。后续应修复保存逻辑，导出 clean checkpoint 后再做 ONNX 与部署链路验证。

## 15. DINOv3 weak objectness auxiliary：lambda=0.05

### 实验目的

前面的 tiled local contrast objectness pretrain 证明：DINOv3 局部显著性可以覆盖大量小目标，但如果让 YOLO 先单独拟合 dense objectness map，再用 10% 标注微调，会产生负迁移。本轮改成检测任务主导的联合训练：保留标准 YOLO detection loss，只额外加入低权重 DINOv3 tiled local-contrast objectness auxiliary loss，验证 DINO 信号是否能作为弱正则提升少标签检测。

### 配置

- 学生模型：`YOLO26n`。
- 数据：`configs/datasets/visdrone_10pct.yaml` 训练，完整 VisDrone val 验证。
- 训练参数：`imgsz=960`、`epochs=120`、`batch=16`、`workers=8`、`seed=42`、`optimizer=auto`、`patience=100`、`close_mosaic=10`、`amp=True`。
- DINOv3 objectness：复用 tiled local contrast，`tile_size=480`、`tile_stride=240`、teacher input `448`、patch grid `28x28`。
- 辅助约束位置：YOLO layer `16`，使用 `1x1 Conv` objectness head 输出 dense logits。
- 损失：`total_loss = detection_loss + 0.05 * 0.5 * (BCEWithLogits + Dice)`。
- 配置文件：`configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam005.yaml`。
- 训练脚本：`scripts/train_dinov3_objectness_aux.py`。

### 可复现步骤

```bash
cd /home/featurize/vfm-yolo-distillation
PYTHONPATH=src:. nohup python3 scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam005.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005 \
  > runs/dinov3_objectness/logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005.log 2>&1 < /dev/null &
```

训练完成后执行：

```bash
PYTHONPATH=src:. python3 scripts/evaluate_area_ap.py \
  --model runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005/weights/best.pt \
  --data configs/datasets/visdrone.yaml \
  --imgsz 960 \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005_best \
  --output-dir runs/dinov3_objectness/reports/area_ap
```

ONNX 导出验证：

```bash
python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005/weights/best.pt")
model.export(format="onnx", imgsz=960, simplify=True, opset=12)
PY
```

### 整体指标

| 实验 | best epoch | P | R | mAP50 | mAP50-95 | 相对 10%/960 baseline | 相对 relation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | 110 | 0.3779 | 0.3003 | 0.2572 | 0.1444 | 0.0000 | -0.0004 |
| dinov3_relation_10pct_960 | 105 | 0.3794 | 0.3055 | 0.2612 | 0.1448 | +0.0004 | 0.0000 |
| dinov3_objectness_aux_lam005_10pct_960 | 93 | 0.3688 | 0.3117 | 0.2629 | 0.1471 | +0.0027 | +0.0023 |

### small / medium / large AP

| 实验 | area | AP50 | AP50-95 | GT | Pred | 相对 baseline AP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | small | 0.2733 | 0.1160 | 26586 | 122198 | 0.0000 |
| dinov3_objectness_aux_lam005_10pct_960 | small | 0.2754 | 0.1171 | 26586 | 123166 | +0.0011 |
| baseline_10pct_960 | medium | 0.5745 | 0.3884 | 11105 | 34764 | 0.0000 |
| dinov3_objectness_aux_lam005_10pct_960 | medium | 0.5717 | 0.3861 | 11105 | 33995 | -0.0023 |
| baseline_10pct_960 | large | 0.6196 | 0.4885 | 1068 | 2545 | 0.0000 |
| dinov3_objectness_aux_lam005_10pct_960 | large | 0.6314 | 0.4965 | 1068 | 2633 | +0.0079 |

### 导出与可加载性

- `best.pt` 可被普通 Ultralytics `YOLO()` 加载，类型为 `DetectionModel`，没有 hook pickle 问题。
- ONNX 导出成功：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005/weights/best.onnx`，大小约 9.6 MB。
- 导出日志：`runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005_onnx.log`。

### 分析

- 这是当前 10%/960 设置下的最好结果，best mAP50-95=0.1471，相对 baseline +0.0027，相对 relation distillation +0.0023。
- small AP50-95 从 0.1160 提升到 0.1171，说明低权重 objectness auxiliary 对小目标有轻微正向作用，但提升幅度仍小。
- medium AP50-95 从 0.3884 降到 0.3861，说明本轮总体提升不是来自中目标；large AP50-95 从 0.4885 提升到 0.4965，是更明显的收益来源。
- 相比纯 objectness pretrain，本轮没有负迁移，支持“检测 loss 主导、DINO objectness 低权重辅助”的路线；但 small AP 提升仍不足以证明问题已解决。
- 后续应围绕 lambda 与 ignore-aware 做小网格，而不是再回到纯预训练：优先尝试 `lambda=0.02`、`lambda=0.10` 和 top-k/ignore objectness auxiliary。

## 16. DINOv3 objectness auxiliary：lambda 扫描

### 实验目的

第 15 节的 `lambda=0.05` 单次实验显示 weak objectness auxiliary 有轻微正向收益，但提升幅度很小，且尚不清楚辅助损失权重是否过强或过弱。本轮固定模型、数据、输入分辨率、teacher signal 与训练轮次，只扫描 `lambda_objectness` 和 target mode，判断 DINOv3 objectness 约束的有效权重区间，并为后续多 seed 稳定性验证选择候选配置。

### 配置

- 学生模型：`YOLO26n`。
- 数据：`configs/datasets/visdrone_10pct.yaml` 训练，完整 VisDrone val 验证。
- 训练参数：`imgsz=960`、`epochs=120`、`batch=16`、`workers=8`、`seed=42`、`optimizer=auto`、`patience=100`、`close_mosaic=10`、`amp=True`。
- DINOv3 objectness：tiled local contrast，`tile_size=480`、`tile_stride=240`、teacher input `448`、patch grid `28x28`。
- 辅助约束位置：YOLO layer `16`，`1x1 Conv` objectness head。
- 扫描变量：
  - soft target：`lambda_objectness=0.02`、`0.10`。
  - ignore-aware target：`lambda_objectness=0.02`、`0.10`。
- 对照基线：`baseline_10pct_960` mAP50-95=0.14439，普通 aux `lambda=0.05` 三 seed mean=0.14539，ignore-aware `lambda=0.05` 三 seed mean=0.14622。

### 可复现步骤

四组实验使用同一训练脚本，仅替换配置文件：

```bash
cd /home/featurize/vfm-yolo-distillation

PYTHONPATH=src:. python3 scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam002_seed42.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42

PYTHONPATH=src:. python3 scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam010_seed42.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam010_seed42

PYTHONPATH=src:. python3 scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_ignore_aware_visdrone_10pct_imgsz960_lam002_seed42.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_ignore_lam002_seed42

PYTHONPATH=src:. python3 scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_ignore_aware_visdrone_10pct_imgsz960_lam010_seed42.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_ignore_lam010_seed42
```

实际运行时通过顺序队列执行，队列日志为 `runs/dinov3_objectness/logs/objaux_lambda_scan_seed42_queue.log`，队列 PID 为 `runs/dinov3_objectness/objaux_lambda_scan_seed42_queue.pid`。

`soft lambda=0.02` 在 seed42 上取得本轮扫描最优后，继续补充 `seed=2026` 与 `seed=3407`，验证候选配置的随机种子稳定性：

```bash
cd /home/featurize/vfm-yolo-distillation

PYTHONPATH=src:. python3 scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam002_seed2026.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed2026

PYTHONPATH=src:. python3 scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam002_seed3407.yaml \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed3407
```

补充队列日志为 `runs/dinov3_objectness/logs/objaux_lam002_multiseed_queue.log`，队列 PID 为 `runs/dinov3_objectness/objaux_lam002_multiseed_queue.pid`。

### 整体指标

| 实验 | target mode | lambda | best epoch | P | R | best mAP50 | best mAP50-95 | last mAP50 | last mAP50-95 | 相对 baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | none | 0 | 110 | 0.37793 | 0.30026 | 0.25717 | 0.14439 | 0.25420 | 0.14201 | 0.00000 |
| objaux_lam002_seed42 | soft | 0.02 | 106 | N/A | N/A | 0.26488 | 0.14792 | 0.25866 | 0.14325 | +0.00353 |
| objaux_lam005_seed42 | soft | 0.05 | 93 | 0.36878 | 0.31167 | 0.26292 | 0.14707 | 0.25487 | 0.14127 | +0.00268 |
| objaux_lam010_seed42 | soft | 0.10 | 100 | N/A | N/A | 0.25510 | 0.14243 | 0.25079 | 0.13869 | -0.00196 |
| objaux_ignore_lam002_seed42 | ignore-aware | 0.02 | 81 | N/A | N/A | 0.25792 | 0.14529 | 0.25154 | 0.13846 | +0.00090 |
| objaux_ignore_lam005_seed42 | ignore-aware | 0.05 | N/A | N/A | N/A | N/A | 0.14568 | N/A | N/A | +0.00129 |
| objaux_ignore_lam010_seed42 | ignore-aware | 0.10 | 106 | N/A | N/A | 0.26139 | 0.14518 | 0.25458 | 0.14096 | +0.00079 |

### `soft lambda=0.02` 多 seed 稳定性

| seed | best epoch | best P | best R | best mAP50 | best mAP50-95 | last P | last R | last mAP50 | last mAP50-95 | best 权重 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 106 | N/A | N/A | 0.26488 | 0.14792 | 0.38064 | 0.30728 | 0.25866 | 0.14325 | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42/weights/best.pt` |
| 2026 | 81 | N/A | N/A | 0.26290 | 0.14604 | 0.38882 | 0.30425 | 0.25744 | 0.14161 | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed2026/weights/best.pt` |
| 3407 | 92 | N/A | N/A | 0.26157 | 0.14626 | 0.38366 | 0.30474 | 0.25706 | 0.14307 | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed3407/weights/best.pt` |

| 统计项 | mAP50 | mAP50-95 |
| --- | --- | --- |
| mean | 0.26312 | 0.14674 |
| std | 0.00136 | 0.00084 |
| 相对 baseline_10pct_960 mean 差值 | +0.00595 | +0.00235 |
| 相对普通 aux lambda=0.05 三 seed mean 差值 | N/A | +0.00135 |
| 相对 ignore-aware lambda=0.05 三 seed mean 差值 | N/A | +0.00052 |

### `soft lambda=0.02` small / medium / large AP 与导出验证

多 seed 训练完成后，对三个 seed 的 `best.pt` 执行普通 Ultralytics `YOLO()` 加载、small/medium/large AP 统计与 ONNX 导出。三组 `best.pt` 均可作为 clean `DetectionModel` 加载，未发现训练时 hook 残留；三组 ONNX 均导出成功，单个 ONNX 文件大小约 10.0 MB。

复现命令如下：

```bash
cd /home/featurize/vfm-yolo-distillation

PYTHONPATH=src:. python3 scripts/evaluate_area_ap.py \
  --model runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42/weights/best.pt \
  --data configs/datasets/visdrone.yaml \
  --imgsz 960 \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42_best \
  --output-dir runs/dinov3_objectness/reports/area_ap

python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42/weights/best.pt")
model.export(format="onnx", imgsz=960, simplify=True, opset=12)
PY
```

`seed=2026` 与 `seed=3407` 使用相同命令，仅替换模型目录和 `--name`。实际执行队列日志为 `runs/dinov3_objectness/logs/objaux_lam002_eval_export_queue.log`。

| seed | area | AP50 | AP50-95 | GT | Pred |
| --- | --- | --- | --- | --- | --- |
| 42 | small | 0.27576 | 0.11725 | 26586 | 121818 |
| 42 | medium | 0.58402 | 0.39560 | 11105 | 34934 |
| 42 | large | 0.64101 | 0.51093 | 1068 | 2617 |
| 2026 | small | 0.27263 | 0.11449 | 26586 | 118387 |
| 2026 | medium | 0.58604 | 0.39420 | 11105 | 38503 |
| 2026 | large | 0.62169 | 0.47867 | 1068 | 2699 |
| 3407 | small | 0.27754 | 0.11573 | 26586 | 119858 |
| 3407 | medium | 0.58888 | 0.39713 | 11105 | 36759 |
| 3407 | large | 0.62338 | 0.49889 | 1068 | 2627 |

| area | AP50 mean | AP50 std | AP50-95 mean | AP50-95 std | 相对 baseline AP50-95 | 相对普通 aux lambda=0.05 seed42 AP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| small | 0.27531 | 0.00203 | 0.11582 | 0.00113 | -0.00019 | -0.00127 |
| medium | 0.58631 | 0.00200 | 0.39565 | 0.00120 | +0.00723 | +0.00950 |
| large | 0.62870 | 0.00874 | 0.49616 | 0.01331 | +0.00762 | -0.00030 |

| seed | ONNX 路径 | 大小 |
| --- | --- | --- |
| 42 | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42/weights/best.onnx` | 10021889 bytes |
| 2026 | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed2026/weights/best.onnx` | 10021889 bytes |
| 3407 | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed3407/weights/best.onnx` | 10021889 bytes |

### 结果与分析

- `soft lambda=0.02` 是本轮最强单次结果，seed42 best mAP50-95=0.14792，相对 10%/960 baseline 提升 +0.00353，也超过普通 aux `lambda=0.05` seed42 的 0.14707。
- `soft lambda=0.02` 三 seed mean mAP50-95=0.14674，std=0.00084；相比 baseline_10pct_960 的 0.14439 提升 +0.00235，相比普通 aux `lambda=0.05` 三 seed mean=0.14539 提升 +0.00135，相比 ignore-aware `lambda=0.05` 三 seed mean=0.14622 提升 +0.00052。
- area AP 显示总体提升并不来自 small object：small AP50-95 mean=0.11582，略低于 baseline_10pct_960 的 0.11601，也低于普通 aux `lambda=0.05` seed42 的 0.11709。
- medium/large 是 `soft lambda=0.02` 主要收益来源：medium AP50-95 mean=0.39565，相对 baseline +0.00723；large AP50-95 mean=0.49616，相对 baseline +0.00762，但 large 相对普通 aux `lambda=0.05` seed42 基本持平略低。
- 三组 `best.pt` 均可被普通 Ultralytics `YOLO()` 加载并成功导出 ONNX，说明当前 auxiliary 训练脚本的保存产物满足 YOLO-only 部署链路要求。
- `soft lambda=0.10` 明显下降到 0.14243，低于 baseline，说明 objectness 约束过强时会压制检测主任务，形成负迁移。
- ignore-aware 三个权重下都没有明显崩溃，但 `lambda=0.02/0.10` 单次都低于 ignore-aware `lambda=0.05` 三 seed mean=0.14622，当前证据不支持继续优先扩大 ignore-aware lambda 扫描。
- `soft lambda=0.02` 的 best epoch=106，仍在 120 epoch 内完成主要收敛；结合 200 epoch 测试没有显著突破上限，后续优先补多 seed，而不是先延长训练轮次。
- 多 seed 结果显示 `soft lambda=0.02` 的总体 mAP 提升方向较稳，但和项目目标存在偏差：它提升了 overall mAP50-95，却没有稳定提升 small AP50-95。因此它可以作为“安全 weak auxiliary”候选，但不能证明 DINO objectness 已经把小目标能力有效迁移给 YOLO。

### 产物路径

- `soft lambda=0.02` best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42/weights/best.pt`
- `soft lambda=0.02 seed2026` best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed2026/weights/best.pt`
- `soft lambda=0.02 seed3407` best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed3407/weights/best.pt`
- `soft lambda=0.02` area AP 目录：`runs/dinov3_objectness/reports/area_ap/*lam002*_best_area_ap.csv`
- `soft lambda=0.02` ONNX 导出日志：`runs/dinov3_objectness/export_logs/*lam002*_onnx.log`
- `soft lambda=0.10` best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam010_seed42/weights/best.pt`
- `ignore-aware lambda=0.02` best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_ignore_lam002_seed42/weights/best.pt`
- `ignore-aware lambda=0.10` best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_ignore_lam010_seed42/weights/best.pt`

## 17. 小目标定向 DINOv3 objectness auxiliary：target-focus seed42 screening

### 实验目的

第 16 节的 `soft lambda=0.02` 三 seed 结果显示 overall mAP50-95 有稳定小幅提升，但 small AP50-95 mean=0.11582，略低于 baseline_10pct_960 的 0.11601。说明继续扩大 lambda 网格不是主要矛盾。本轮改造 teacher target，验证两类更偏小目标/实例中心的 objectness 目标是否能真正提升 small AP。

### 配置

- 学生模型：`YOLO26n`。
- 数据：`configs/datasets/visdrone_10pct.yaml` 训练，完整 VisDrone val 验证。
- 训练参数：`imgsz=960`、`epochs=120`、`batch=16`、`workers=8`、`seed=42`、`optimizer=auto`、`patience=100`、`close_mosaic=10`、`amp=True`。
- DINOv3 objectness：tiled local contrast，`tile_size=480`、`tile_stride=240`、teacher input `448`、patch grid `28x28`。
- 辅助约束位置：YOLO layer `16`，`1x1 Conv` objectness head。
- 共同权重：`lambda_objectness=0.02`。
- 新增 target 模块：`src/vfm_yolo_distillation/objectness_targets.py`。

### Arm A：Small-GT-Aware Soft Objectness

- `target_mode=small_gt_weighted_soft`。
- 训练名：`yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42`。
- 核心策略：仍使用 DINO tiled local-contrast soft target，但从 10% 标注训练 batch 中读取 GT box，只用于 auxiliary loss 权重图。
- 权重设计：背景 `0.25`，small GT 邻域 `3.0`，medium GT 邻域 `1.0`，large GT 邻域 `0.5`。
- false positive 抑制：若 DINO target 高于 batch 内 `0.85` 分位且不落在任意 GT 邻域，权重降为 `0.05`。
- 面积阈值：`small_area_px=1024`，`medium_area_px=9216`，`box_expand_cells=1`。

### Arm B：Peak-Only Ignore-Aware Objectness

- `target_mode=peak_ignore_aware`。
- 训练名：`yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42`。
- 核心策略：不使用 GT 权重，仅处理 DINO target。
- 局部峰值：`peak_kernel=3`，使用 max-pool 找局部峰值。
- 正样本：局部峰值且 target ≥ `positive_quantile=0.85`。
- 负样本：target ≤ `negative_quantile=0.45`。
- 中间区域 ignore，目标是去掉 dense soft map 的大面积平滑响应，只保留更像实例中心的 objectness 信号。

### 可复现步骤

```bash
cd /home/featurize/vfm-yolo-distillation

PYTHONPATH=src:.:scripts python3 scripts/train_dinov3_objectness_aux.py   --config configs/experiments/dinov3_objectness_aux_smallgt_visdrone_10pct_imgsz960_lam002_seed42.yaml

PYTHONPATH=src:.:scripts python3 scripts/train_dinov3_objectness_aux.py   --config configs/experiments/dinov3_objectness_aux_peak_visdrone_10pct_imgsz960_lam002_seed42.yaml
```

实际运行使用顺序队列执行，队列日志为 `runs/dinov3_objectness/logs/objaux_target_focus_screen_seed42_queue.log`，队列 PID 为 `runs/dinov3_objectness/objaux_target_focus_screen_seed42_queue.pid`。

训练后验收命令：

```bash
cd /home/featurize/vfm-yolo-distillation

PYTHONPATH=src:. python3 scripts/evaluate_area_ap.py   --model runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42/weights/best.pt   --data configs/datasets/visdrone_10pct.yaml   --imgsz 960   --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42   --output-dir runs/dinov3_objectness/area_ap

python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42/weights/best.pt")
model.export(format="onnx", imgsz=960, opset=12, simplify=False)
PY
```

Arm B 使用相同命令，仅替换模型目录和 `--name`。

### 整体指标

| 实验 | target mode | lambda | best epoch | P | R | best mAP50 | best mAP50-95 | last mAP50 | last mAP50-95 | 相对 baseline mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | none | 0 | 110 | 0.3779 | 0.3003 | 0.2572 | 0.14439 | 0.2542 | 0.1420 | 0.00000 |
| objaux_soft_lam002_seed42 | soft | 0.02 | 106 | N/A | N/A | 0.26488 | 0.14792 | 0.25866 | 0.14325 | +0.00353 |
| objaux_smallgt_lam002_seed42 | small_gt_weighted_soft | 0.02 | 85 | 0.3703 | 0.3024 | 0.26137 | 0.14549 | 0.25569 | 0.14203 | +0.00110 |
| objaux_peak_lam002_seed42 | peak_ignore_aware | 0.02 | 81 | 0.3692 | 0.2995 | 0.25875 | 0.14423 | 0.25352 | 0.13995 | -0.00016 |

### small / medium / large AP

| 实验 | area | AP50 | AP50-95 | GT | Pred |
| --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | small | N/A | 0.11601 | 26586 | N/A |
| objaux_lam002_seed42 | small | 0.27576 | 0.11725 | 26586 | 121818 |
| objaux_smallgt_lam002_seed42 | small | 0.27278 | 0.11568 | 26586 | 121992 |
| objaux_peak_lam002_seed42 | small | 0.26752 | 0.11214 | 26586 | 118209 |
| objaux_smallgt_lam002_seed42 | medium | 0.57387 | 0.38734 | 11105 | 35300 |
| objaux_peak_lam002_seed42 | medium | 0.57423 | 0.38786 | 11105 | 38963 |
| objaux_smallgt_lam002_seed42 | large | 0.58702 | 0.45599 | 1068 | 2445 |
| objaux_peak_lam002_seed42 | large | 0.60701 | 0.48081 | 1068 | 2575 |

### clean YOLO 加载与 ONNX 导出

| 实验 | clean `YOLO()` | ONNX 路径 | 导出日志 |
| --- | --- | --- | --- |
| objaux_smallgt_lam002_seed42 | 通过，`task=detect`，`names=10` | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42/weights/best.onnx` | `runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42_clean_load_onnx.log` |
| objaux_peak_lam002_seed42 | 通过，`task=detect`，`names=10` | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42/weights/best.onnx` | `runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42_clean_load_onnx.log` |

### 结果与分析

- Arm A `small_gt_weighted_soft` 的 best mAP50-95=0.14549，高于 baseline_10pct_960 的 0.14439，但低于 `soft lambda=0.02 seed42` 的 0.14792，也低于 `soft lambda=0.02` 三 seed mean=0.14674。
- Arm A 的 small AP50-95=0.11568，低于 baseline_10pct_960 的 0.11601，也低于 `soft lambda=0.02 seed42` 的 0.11725。因此 GT-size-aware weighting 没有达成“小目标定向提升”的筛选标准。
- Arm B `peak_ignore_aware` 的 best mAP50-95=0.14423，略低于 baseline；small AP50-95=0.11214，明显低于 baseline。说明只保留 DINO 局部峰值会丢失过多可用监督，尤其对密集小目标不利。
- 两组都通过 clean `YOLO()` 加载和 ONNX 导出，部署链路仍然是 YOLO-only，没有训练时 teacher/head 依赖残留。
- 严格按预设判定规则：进入多 seed 的条件是 small AP50-95 ≥ 0.1170 且 overall mAP50-95 ≥ 0.14439。Arm A overall 达标但 small AP 不达标；Arm B 两项都不达标。因此本轮不应补多 seed，应停止训练扩展，转入 target 审计。
- 关键负向结论：简单地用 GT 小目标邻域加权 DINO dense soft objectness，不足以把 DINO signal 转化为 small AP 收益。问题可能不在 lambda，而在 target 空间与检测头学习目标之间的对齐：DINO local contrast 仍更容易响应边缘/局部纹理/中大目标结构，小目标中心的稳定性不够。

### 产物路径

- Arm A 配置：`configs/experiments/dinov3_objectness_aux_smallgt_visdrone_10pct_imgsz960_lam002_seed42.yaml`
- Arm B 配置：`configs/experiments/dinov3_objectness_aux_peak_visdrone_10pct_imgsz960_lam002_seed42.yaml`
- Arm A best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42/weights/best.pt`
- Arm B best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42/weights/best.pt`
- Arm A area AP：`runs/dinov3_objectness/area_ap/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42_area_ap.csv`
- Arm B area AP：`runs/dinov3_objectness/area_ap/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42_area_ap.csv`
- Arm A ONNX：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42/weights/best.onnx`
- Arm B ONNX：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42/weights/best.onnx`

## 18. DINOv3 target alignment audit：缓存 target 与 GT 空间对齐

### 实验目的

第 17 节 target-focus screening 显示，`small_gt_weighted_soft` 和 `peak_ignore_aware` 都没有提升 small AP。为了判断问题是否来自 teacher objectness target 本身，本轮不训练模型，而是审计训练缓存中的 tiled local-contrast target 与 GT box 的空间对齐关系。

核心问题：DINOv3 tiled local-contrast target 的高响应区域，是否真的落在小目标附近；小目标中心和框内响应，是否足够稳定地高于背景。

### 配置

- 数据：VisDrone train split 中已存在训练缓存的图片，随机采样 300 张。
- teacher target：DINOv3 ViT-B/16 tiled local-contrast objectness。
- cache：`runs/dinov3_objectness/cache/tiled_local_contrast_t480_s240_l16_imgsz960`。
- target 尺寸：`120x120`，对应 960 输入下 student layer 16 的辅助监督分辨率。
- tile 参数：`tile_size=480`、`tile_stride=240`、`teacher_image_size=448`。
- 分组阈值：small `area <= 32^2`，medium `32^2 < area <= 96^2`，large `area > 96^2`。

### 可复现步骤

```bash
python scripts/audit_dinov3_target_alignment.py \
  --dataset configs/datasets/visdrone.yaml \
  --split train \
  --samples 300 \
  --seed 42 \
  --method local_contrast \
  --teacher-image-size 448 \
  --tile-size 480 \
  --tile-stride 240 \
  --target-height 120 \
  --target-width 120 \
  --cache runs/dinov3_objectness/cache/tiled_local_contrast_t480_s240_l16_imgsz960 \
  --overlay-count 32 \
  --output runs/dinov3_objectness/audit_target_alignment_cache_t480_s240_300
```

输出：

- `summary.csv`：总体统计。
- `box_alignment.csv`：逐 GT box 的 center/mean/max/p90 响应与 top quantile 命中。
- `image_alignment.csv`：逐图 top-q 高响应区域落在 GT、小目标 GT、背景的比例。
- `overlays/*.jpg`：heatmap 与 GT box overlay，红色为 small，黄色为 medium，青色为 large。

### 结果

| 指标 | 值 |
| --- | --- |
| images | 300 |
| boxes | 15725 |
| small_boxes | 9721 |
| small_center_mean | 0.189195 |
| small_box_mean | 0.189172 |
| small_box_max_mean | 0.258480 |
| small_center_percentile_mean | 0.517120 |
| small_hit_q85 | 0.305730 |
| small_hit_q90 | 0.232281 |
| medium_boxes | 5088 |
| medium_center_mean | 0.198804 |
| medium_box_max_mean | 0.369515 |
| medium_hit_q85 | 0.552476 |
| medium_hit_q90 | 0.459906 |
| large_boxes | 916 |
| large_center_mean | 0.201824 |
| large_box_max_mean | 0.532031 |
| large_hit_q85 | 0.818777 |
| large_hit_q90 | 0.760917 |
| q85_gt_overlap | 0.100151 |
| q85_small_overlap | 0.016612 |
| q85_false_positive | 0.899849 |
| q90_gt_overlap | 0.099932 |
| q90_small_overlap | 0.016340 |
| q90_false_positive | 0.900068 |
| foreground_mean | 0.195104 |
| background_mean | 0.184686 |

### 分析

- 高响应区域和 GT 的对齐很弱：top 15% 高响应中只有约 10.0% 落在任意 GT 内，约 90.0% 落在 GT 外；top 10% 高响应也几乎相同。
- 对小目标尤其弱：top 15% 高响应中只有约 1.66% 落在 small GT 区域；top 10% 中只有约 1.63%。这解释了为什么 `small_gt_weighted_soft` 没有带来 small AP 提升：teacher target 的高响应主体并不在小目标上。
- 响应随目标面积显著增大：small box 的 `hit_q85=0.3057`，medium 为 `0.5525`，large 为 `0.8188`。DINO local-contrast target 更容易覆盖中大目标结构，而不是密集小目标。
- 小目标中心响应接近随机中位：`small_center_percentile_mean=0.5171`，说明小目标中心并没有稳定处于高响应区。直接让 YOLO 学这个 dense map，会把大量梯度分配给背景纹理/道路边缘/大结构。
- foreground/background 均值差很小：GT 内均值 `0.1951`，背景均值 `0.1847`，差值只有约 `0.0104`。这说明当前 target 在 dense 层面不是强 foreground/background 分离信号。

### 结论

本轮审计支持第 17 节的负向结论：问题不主要是 `lambda`，也不只是是否给 small GT 更高 loss weight，而是 teacher objectness target 自身对小目标的空间对齐不足。下一步不应继续训练 `small_gt_weighted_soft` 或 `peak_ignore_aware` 多 seed；应先改 target 生成方式。

优先方向：从 full-image dense local-contrast 改为 small-object crop/tile mining。具体做法是在训练 batch 中围绕 small GT 或高密度小目标区域裁 crop，让 DINO 在局部上下文内产生 objectness target，再把该 target 投回 student feature map。这样目标不是“全图哪里局部纹理显著”，而是“局部小目标上下文中哪些 patch 更像实例区域”。

### 产物路径

- 审计脚本：`scripts/audit_dinov3_target_alignment.py`
- 远程/本地输出目录：`runs/dinov3_objectness/audit_target_alignment_cache_t480_s240_300`
- summary：`runs/dinov3_objectness/audit_target_alignment_cache_t480_s240_300/summary.csv`
- 逐框统计：`runs/dinov3_objectness/audit_target_alignment_cache_t480_s240_300/box_alignment.csv`
- 逐图统计：`runs/dinov3_objectness/audit_target_alignment_cache_t480_s240_300/image_alignment.csv`
- overlay：`runs/dinov3_objectness/audit_target_alignment_cache_t480_s240_300/overlays/`
- 日志：`runs/dinov3_objectness/logs/target_alignment_cache_t480_s240_300.log`

## 19. DINOv3 small-object crop objectness auxiliary：seed42 screening（已完成）

### 实验目的

第 18 节 target alignment audit 显示，full-image tiled local-contrast target 的高响应区域约 90% 落在 GT 外，且 top-q 高响应落在 small GT 的比例只有约 1.6%。因此本轮不再继续调大 `lambda` 或给 full-image target 加权，而是把 teacher target 生成限制到小目标局部上下文。

核心假设：如果 DINOv3 在 small GT 周围的局部 crop 内生成 objectness target，再将该 target 投回 YOLO student feature map，对 small AP 的帮助应强于 full-image dense target。

### 配置

- 实验名：`yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42`
- 训练脚本：`scripts/train_dinov3_objectness_aux.py`
- 配置文件：`configs/experiments/dinov3_objectness_aux_smallcrop_visdrone_10pct_imgsz960_lam002_seed42.yaml`
- 数据：VisDrone train 10% label budget，val 使用完整验证集。
- 模型：YOLO26n，部署期仍为 YOLO-only。
- 输入分辨率：`imgsz=960`
- epoch：`120`
- batch：`16`
- seed：`42`
- student layer：`16`
- `lambda_objectness=0.02`
- `target_mode=small_crop_soft`
- small 定义：`area <= 1024 px`
- crop 参数：`crop_context_scale=8.0`、`crop_min_size=192`、`max_crops_per_image=8`、`crop_weight=1.0`
- DINOv3 teacher：ViT-B/16，`teacher_image_size=448`，`method=local_contrast`

### 核心逻辑

1. 在每个训练 batch 中读取 small GT box。
2. 对每张图最多选取 8 个 small box，按目标框中心裁局部 crop。
3. 将 crop resize 到 DINOv3 输入尺寸，在局部上下文内生成 local-contrast objectness target。
4. 将 crop target resize 并投回 student layer 16 对应的 feature map 区域。
5. 只在这些小目标局部区域计算 weighted soft objectness auxiliary loss；全图其他区域权重为 0，避免道路、纹理、大背景区域主导辅助梯度。
6. 检测主 loss 仍正常使用 10% 标注数据，DINOv3 只作为训练时 teacher target，不进入推理部署。

### 启动命令

```bash
PYTHONPATH=src:scripts python scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_smallcrop_visdrone_10pct_imgsz960_lam002_seed42.yaml
```

远程后台启动命令：

```bash
nohup env PYTHONPATH=src:scripts python scripts/train_dinov3_objectness_aux.py \
  --config configs/experiments/dinov3_objectness_aux_smallcrop_visdrone_10pct_imgsz960_lam002_seed42.yaml   > runs/dinov3_objectness/logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42.log 2>&1 &
```

### 训练完成状态

- 启动时间：2026-06-24
- 远程 PID：`107438`
- 完成状态：120 epochs 正常完成，用时约 `2.201 hours`。
- 运行检查：训练期间未发现 OOM、Traceback 或 RuntimeError；结束后 GPU 显存释放。
- 训练日志：`runs/dinov3_objectness/logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42.log`
- 结果目录：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42`
- best 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42/weights/best.pt`
- last 权重：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42/weights/last.pt`

### 整体指标

| 实验 | target mode | lambda | best epoch | P | R | best mAP50 | best mAP50-95 | last mAP50 | last mAP50-95 | 相对 baseline mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | none | 0 | 110 | 0.3779 | 0.3003 | 0.2572 | 0.14439 | 0.2542 | 0.1420 | 0.00000 |
| objaux_soft_lam002_seed42 | soft | 0.02 | 106 | N/A | N/A | 0.26488 | 0.14792 | 0.25866 | 0.14325 | +0.00353 |
| objaux_smallgt_lam002_seed42 | small_gt_weighted_soft | 0.02 | 85 | 0.3703 | 0.3024 | 0.26137 | 0.14549 | 0.25569 | 0.14203 | +0.00110 |
| objaux_peak_lam002_seed42 | peak_ignore_aware | 0.02 | 81 | 0.3692 | 0.2995 | 0.25875 | 0.14423 | 0.25352 | 0.13995 | -0.00016 |
| objaux_smallcrop_lam002_seed42 | small_crop_soft | 0.02 | 111 | 0.3710 | 0.3086 | 0.26000 | 0.14453 | 0.25193 | 0.13862 | +0.00014 |

训练后段出现一个需要记录的现象：epoch 111 达到 best 后，Ultralytics 关闭 mosaic，后续 last 指标回落到 mAP50-95=0.13862。因此本实验必须使用 best.pt，而不能使用 last.pt 作为候选权重。

### small / medium / large AP

评估命令：

```bash
python scripts/evaluate_area_ap.py \
  --model runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42/weights/best.pt \
  --data configs/datasets/visdrone.yaml \
  --imgsz 960 \
  --name yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42 \
  --output-dir runs/dinov3_objectness/reports \
  --conf 0.001 \
  --iou 0.7 \
  --max-det 300
```

| 实验 | area | AP50 | AP50-95 | GT | Pred |
| --- | --- | --- | --- | --- | --- |
| baseline_10pct_960 | small | N/A | 0.11601 | 26586 | N/A |
| objaux_lam002_seed42 | small | 0.27576 | 0.11725 | 26586 | 121818 |
| objaux_smallgt_lam002_seed42 | small | 0.27278 | 0.11568 | 26586 | 121992 |
| objaux_peak_lam002_seed42 | small | 0.26752 | 0.11214 | 26586 | 118209 |
| objaux_smallcrop_lam002_seed42 | small | 0.27540 | 0.11685 | 26586 | 122952 |
| objaux_smallcrop_lam002_seed42 | medium | 0.57528 | 0.38992 | 11105 | 34311 |
| objaux_smallcrop_lam002_seed42 | large | 0.60398 | 0.47567 | 1068 | 2564 |

### clean `YOLO()` 加载与 ONNX 导出

clean 加载命令：

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO("runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42/weights/best.pt")
print("clean_yolo_load=OK")
print(type(model.model).__name__)
PY
```

ONNX 导出命令：

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO("runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42/weights/best.pt")
model.export(format="onnx", imgsz=960, opset=12, simplify=False, dynamic=False)
PY
```

| 实验 | clean `YOLO()` | ONNX 路径 | ONNX size | 导出日志 |
| --- | --- | --- | --- | --- |
| objaux_smallcrop_lam002_seed42 | 通过，`DetectionModel` | `runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42/weights/best.onnx` | 9,974,110 bytes | `runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42_onnx.log` |

### 判定与分析

- 预设进入多 seed 的筛选标准是 small AP50-95 >= 0.1170 且 overall mAP50-95 >= 0.14439。
- 本实验 overall mAP50-95=0.14453，仅比 baseline_10pct_960 的 0.14439 高 +0.00014，属于非常边缘的正反馈。
- 本实验 small AP50-95=0.11685，比 baseline_10pct_960 的 0.11601 高 +0.00084，但低于阈值 0.1170，也低于 `soft lambda=0.02 seed42` 的 0.11725。
- medium AP50-95=0.38992，高于 `smallgt` 的 0.38734 与 `peak` 的 0.38786，但仍低于 `soft lambda=0.02` 三 seed 均值附近的 medium 收益；large AP50-95=0.47567，介于 `smallgt` 与 `peak` 之间。
- 小目标 crop target 的确把 small AP 从 target-focus 两个 arm 的失败区间拉回 baseline 以上，说明“局部化 teacher target”方向比简单 small GT 加权和 peak-only ignore 更接近目标。
- 但提升幅度仍不够显著，而且 overall 只勉强不退化，说明当前 crop 方案可能只提供了弱正则，而没有稳定改变 YOLO 对密集小目标的召回/定位能力。
- 一个值得注意的训练行为是 epoch 111 之后指标明显回落，说明关闭 mosaic 后训练分布变化对该辅助目标较敏感。后续可以尝试更早停止、调整 close_mosaic、或用 EMA/best-only 策略，但这属于训练调度问题，不应掩盖 target 仍偏弱的事实。

### 结论

`small_crop_soft` 达到了“比 baseline 略好”的最低要求，但没有达到“显著有效”的目标，也没有达到预设的 small AP50-95 >= 0.1170 筛选线。因此本轮不应直接补多 seed。下一轮更合理的方向是继续沿着 crop/tile mining 做更强的小目标密度约束，而不是回退到全图 dense target 或继续扫 lambda。

具体建议是实现 high-density-small-object tile objectness auxiliary：用训练集 10% 标注仅选择小目标密集 tile，让 DINO 在这些 tile 内生成 local-contrast target，并把 loss 作用在 tile 对应区域；与 smallcrop 不同，它不围绕单个小框，而是围绕“小目标密集上下文”，更贴近 VisDrone 中行人/车辆密集小实例场景。

## 20. 总体结论

- DINOv3 target alignment audit 进一步证明，当前 tiled local-contrast target 的高响应区域约 90% 落在 GT 外，top-q 高响应落在 small GT 的比例只有约 1.6%；small box 的 q85 命中率仅 0.3057，明显低于 medium 的 0.5525 和 large 的 0.8188。因此 small AP 无法提升的关键原因是 teacher target 对小目标空间对齐不足，而不是单纯 loss 权重不足。
- 小目标定向 target-focus screening 显示，`small_gt_weighted_soft` 的 overall mAP50-95=0.1455 略高于 baseline，但 small AP50-95=0.1157，低于 baseline；`peak_ignore_aware` 的 overall 和 small AP 均未达标。因此本轮不应补多 seed，应转入 target 审计，重点检查 DINO local-contrast target 对小目标中心、边缘纹理和道路背景的响应分布。
- 提高标签预算和提高输入分辨率都能稳定提升 VisDrone 检测，尤其对小目标更明显。
- DINOv3 global 与全图 patch 蒸馏收益不稳定，全图 patch 甚至降低 small AP。
- region-aware patch 蒸馏小幅提升 overall mAP50-95 与 small AP50-95，是当前直接特征蒸馏中最好的设置，但提升仍是边缘级别。
- DINOv3 relation distillation 在 10%/960 上只带来 +0.0004 mAP50-95 的边缘提升；分面积统计显示收益来自 medium object，小目标 AP50-95 微降，因此暂不能证明其实现了“小目标看世界”能力迁移。
- DINOv3 weak objectness auxiliary lambda=0.05 在 10%/960 上达到 mAP50-95=0.1471，是当前少标签 960 设置下最好结果；small AP50-95 小幅提升 +0.0011，但主要收益仍来自 large object。
- DINOv3 weak objectness auxiliary lambda 扫描显示，普通 soft target 的 `lambda=0.02` 在 seed42 上达到 mAP50-95=0.1479，是当前最强单次结果；补充三 seed后 mean mAP50-95=0.1467，std=0.0008，高于 baseline、普通 aux `lambda=0.05` 与 ignore-aware `lambda=0.05` 三 seed 均值。
- 但 `soft lambda=0.02` 的 small AP50-95 mean=0.1158，略低于 baseline_10pct_960 的 0.1160；其总体收益主要来自 medium/large，因此它不是“小目标能力迁移”问题的充分答案。
- DINOv3 small-object crop objectness auxiliary 在 seed42 上取得 mAP50-95=0.14453、small AP50-95=0.11685，相比 10%/960 baseline 分别提升 +0.00014 和 +0.00084；这是小目标方向的轻微正反馈，但没有达到 small AP50-95 >= 0.1170 的预设筛选线，也明显低于 `soft lambda=0.02 seed42` 的 overall mAP50-95=0.14792。
- 第一轮 DINOv3 objectness 审计显示，全图 patch 显著性更偏道路/场景结构，不足以直接作为小目标 objectness 监督。
- 第二轮局部 objectness 审计显示，local contrast/residual 明显提高 GT 与小目标覆盖率，说明 DINOv3 patch token 的局部差异比全局显著性更接近实例级目标性。
- 第三轮 tiled local contrast 进一步将 small_top20_recall 提升到 0.8352，是当前最强的无标注 objectness teacher signal。
- 但是，第 12-13 节训练实验表明：把 tiled local_contrast 作为纯 objectness pretrain 密集监督后，所有 10%/960 fine-tune 结果均低于 baseline；最佳消融 mAP50-95=0.1338，仍低于 baseline=0.1444。
- 因此当前结论是：DINOv3 确实能提供类别无关局部显著性，但“纯 objectness pretrain -> 监督微调”的桥接方式会产生负迁移；检测任务主导、低权重的 weak objectness auxiliary 是目前更可靠的桥接方式，其中 `soft lambda=0.02` 是当前最稳的 overall mAP 候选，但还没有解决 small object AP。small-crop 结果说明局部化 target 是对的方向之一，但单框 crop 仍太弱；下一步应转向小目标密集 tile 级 target，让 teacher signal 更贴近 VisDrone 的密集小实例场景。

## 21. 下一步计划

1. 停止继续扩大纯 objectness pretrain 网格，不再单独尝试更长 epoch 或更大 lambda。
2. 暂停继续扩大 lambda 网格；`soft lambda=0.02` 可保留为 overall mAP 主候选，但不能作为小目标路线的最终证据。
3. `small_crop_soft` seed42 不进入多 seed；它的 small AP 有轻微正反馈，但未达到预设筛选线，不能作为显著有效策略。
4. 下一轮优先实现 high-density-small-object tile objectness auxiliary：用 10% 训练标注只选择小目标密集 tile 作为 auxiliary target 生成区域，仍不引入教师检测模型，不使用验证集标签参与训练。
5. high-density tile seed42 继续保持 `lambda=0.02`、`imgsz=960`、`epochs=120`、`student_layer=16`；若 small AP50-95 >= 0.1170 且 overall mAP50-95 >= 0.14439，再补 `seed=2026`、`seed=3407`。
6. 若 high-density tile 仍只带来边缘提升或失败，再做 layer 16/19/multi-layer 消融，确认辅助层级是否不匹配，而不是继续扩大 lambda 网格。
7. 继续保持 clean-weight 与 ONNX 导出作为每个候选配置的必检项，确保部署链路始终是 YOLO-only。

## 22. 文件索引

- DINOv3 small-object crop auxiliary 配置：`configs/experiments/dinov3_objectness_aux_smallcrop_visdrone_10pct_imgsz960_lam002_seed42.yaml`
- DINOv3 small-object crop auxiliary 日志：`runs/dinov3_objectness/logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42.log`
- DINOv3 small-object crop auxiliary 结果目录：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42`
- DINOv3 small-object crop auxiliary area AP：`runs/dinov3_objectness/reports/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42_area_ap.csv`
- DINOv3 small-object crop auxiliary ONNX：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42/weights/best.onnx`
- DINOv3 small-object crop auxiliary ONNX 日志：`runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallcrop_lam002_seed42_onnx.log`
- DINOv3 target alignment audit 脚本：`scripts/audit_dinov3_target_alignment.py`
- DINOv3 target alignment audit 输出：`runs/dinov3_objectness/audit_target_alignment_cache_t480_s240_300`
- DINOv3 target alignment audit 日志：`runs/dinov3_objectness/logs/target_alignment_cache_t480_s240_300.log`
- 主报告 Markdown：`runs/reports/visdrone_yolo26n_experiment_report.md`
- 主报告 HTML：`runs/reports/visdrone_yolo26n_experiment_report.html`
- 全部实验指标：`runs/reports/tables/all_experiments_summary.csv`
- 全部面积 AP：`runs/reports/tables/all_area_ap_summary.csv`
- 分辨率对比：`runs/reports/tables/imgsz_comparison.csv`
- objectness 预训练消融汇总：`runs/reports/tables/objectness_pretrain_ablation_summary.csv`
- objectness 可复现索引：`runs/reports/tables/objectness_reproducibility_index.csv`
- 10%/960 baseline 本地产物：`runs/baselines/yolo26n_visdrone_10pct_imgsz960`
- 10%/960 baseline 面积分组 AP：`runs/baselines/reports/area_ap/yolo26n_visdrone_10pct_imgsz960_best_area_ap.csv`
- DINOv3 relation 配置：`configs/experiments/dinov3_relation_distill_visdrone_10pct_imgsz960.yaml`
- DINOv3 relation 训练脚本：`scripts/train_dinov3_relation_distill.py`
- DINOv3 relation 结果目录：`runs/dinov3_distill/yolo26n_visdrone_10pct_imgsz960_dinov3_relation`
- DINOv3 relation 面积分组 AP：`runs/dinov3_distill/reports/area_ap/yolo26n_visdrone_10pct_imgsz960_dinov3_relation_best_area_ap.csv`
- DINOv3 weak objectness auxiliary 配置：`configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam005.yaml`
- DINOv3 weak objectness auxiliary 训练脚本：`scripts/train_dinov3_objectness_aux.py`
- DINOv3 weak objectness auxiliary 结果目录：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005`
- DINOv3 weak objectness auxiliary 面积分组 AP：`runs/dinov3_objectness/reports/area_ap/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005_best_area_ap.csv`
- DINOv3 weak objectness auxiliary ONNX 导出日志：`runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam005_onnx.log`
- DINOv3 objectness auxiliary lambda 扫描配置：`configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam002_seed42.yaml`、`configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam010_seed42.yaml`、`configs/experiments/dinov3_objectness_aux_ignore_aware_visdrone_10pct_imgsz960_lam002_seed42.yaml`、`configs/experiments/dinov3_objectness_aux_ignore_aware_visdrone_10pct_imgsz960_lam010_seed42.yaml`
- DINOv3 objectness auxiliary lambda 扫描日志：`runs/dinov3_objectness/logs/objaux_lambda_scan_seed42_queue.log`
- DINOv3 objectness auxiliary lambda=0.02 多 seed 配置：`configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam002_seed2026.yaml`、`configs/experiments/dinov3_objectness_aux_visdrone_10pct_imgsz960_lam002_seed3407.yaml`
- DINOv3 objectness auxiliary lambda=0.02 多 seed 日志：`runs/dinov3_objectness/logs/objaux_lam002_multiseed_queue.log`
- DINOv3 objectness auxiliary lambda=0.02 area AP：`runs/dinov3_objectness/reports/area_ap/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42_best_area_ap.csv`、`...seed2026_best_area_ap.csv`、`...seed3407_best_area_ap.csv`
- DINOv3 objectness auxiliary lambda=0.02 ONNX：`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_lam002_seed42/weights/best.onnx`、`...seed2026/weights/best.onnx`、`...seed3407/weights/best.onnx`
- DINOv3 objectness auxiliary lambda=0.02 评估导出队列日志：`runs/dinov3_objectness/logs/objaux_lam002_eval_export_queue.log`
- DINOv3 objectness target-focus 配置：`configs/experiments/dinov3_objectness_aux_smallgt_visdrone_10pct_imgsz960_lam002_seed42.yaml`、`configs/experiments/dinov3_objectness_aux_peak_visdrone_10pct_imgsz960_lam002_seed42.yaml`
- DINOv3 objectness target-focus area AP：`runs/dinov3_objectness/area_ap/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42_area_ap.csv`、`runs/dinov3_objectness/area_ap/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42_area_ap.csv`
- DINOv3 objectness target-focus ONNX 导出日志：`runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_smallgt_lam002_seed42_clean_load_onnx.log`、`runs/dinov3_objectness/export_logs/yolo26n_visdrone_10pct_imgsz960_dinov3_objaux_peak_lam002_seed42_clean_load_onnx.log`
- objectness 主实验：`runs/dinov3_objectness/yolo26n_visdrone_full_imgsz960_dinov3_objectness_pretrain`、`runs/dinov3_objectness/yolo26n_visdrone_10pct_imgsz960_dinov3_objectness_pretrain_ft`
- objectness 定位消融：`runs/dinov3_objectness/yolo26n_visdrone_full_imgsz960_dinov3_objpre_lam010_ep5`、`..._lam010_ep10`、`..._lam025_ep5` 及对应 `_ft` 微调目录
- objectness 日志：`runs/dinov3_objectness/logs/yolo26n_objectness_pretrain_ft_10pct960.log`、`runs/dinov3_objectness/logs/objectness_ablation_queue.log`
- 第一轮 objectness 审计：`runs/dinov3_objectness/audit_train300_448`
- 第二轮 objectness 审计：`runs/dinov3_objectness/audit_train300_448_local_contrast`、`audit_train300_448_local_residual`、`audit_train300_448_local_fusion`
- 第三轮 tiled objectness 审计：`runs/dinov3_objectness/audit_train300_448_tiled_local_contrast_t480_s240`
