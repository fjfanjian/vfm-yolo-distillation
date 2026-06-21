# TODO

本文件用于持续跟踪 VFM-assisted YOLO distillation 实验进度。状态约定：

- `[x]` 已完成并可在仓库中找到对应配置、脚本或验证结果。
- `[ ]` 未完成。
- 每个实验完成后，在条目后补充配置路径、运行命令、关键指标和结果位置。

## P0：必须先做

- [x] 明确研究主线：DINOv3、Grounding DINO、YOLO-World 默认只作为训练期教师或伪标签来源，部署期 student 保持 YOLO-only。
- [x] 初始化项目骨架：README、`configs/`、`scripts/`、`src/`、`tests/`。
- [x] 确定 YOLO baseline 模型族：使用 Ultralytics YOLO26 `n/s/m/l/x`。
- [x] 建立 VisDrone 全量 YOLO baseline 配置：`configs/experiments/yolo_baseline_visdrone.yaml`。
- [x] 建立 VisDrone 少标注数据集模板：`5pct`、`10pct`、`25pct`、`50pct`。
- [x] 建立 VisDrone 少标注 baseline 配置：`10pct`、`25pct`。
- [x] 添加 VisDrone 转换与 split 生成脚本：`scripts/prepare_visdrone.py`。
- [ ] 在远程服务器生成真实 VisDrone YOLO labels 和 split 清单：`splits/visdrone/train_5pct.txt`、`train_10pct.txt`、`train_25pct.txt`、`train_50pct.txt`。这些文件应放在数据集目录，不提交仓库。
- [ ] 跑通 YOLO baseline：100%、25%、10%。
- [ ] 记录统一指标：mAP50-95、mAP50、mAP75、AP_small、AR_small、FPS、params、FLOPs。
- [ ] 整理统一评估脚本，保证所有实验输出可横向比较。
- [ ] 验证 ONNX 导出，固定部署期不依赖 DINOv3/Grounding DINO/YOLO-World。

## P1：核心研究实验

- [ ] 实现 DINOv3 feature extractor：冻结 teacher，输入图像输出 patch-level dense feature。
- [ ] 支持 DINOv3 batch 推理缓存，避免训练期重复计算过慢。
- [ ] 实现 YOLO-DINO 特征对齐模块：从 YOLO neck 提取 P3/P4/P5，并对齐 DINOv3 feature 尺度。
- [ ] 做全局特征蒸馏实验：YOLO baseline vs YOLO + DINOv3 global distill。
- [ ] 做 25% 标注下的全局特征蒸馏实验。
- [ ] 做 10% 标注下的全局特征蒸馏实验。
- [ ] 实现小目标区域 mask：基于 GT box 或高质量伪标签生成 small-object mask。
- [ ] 做小目标区域加权蒸馏实验，重点观察 AP_small 和 AR_small。
- [ ] 做蒸馏权重消融：`0.1`、`0.5`、`1.0`、`2.0`。

## P2：伪标签与开放词表辅助

- [ ] 配置 Grounding DINO 伪标签生成流程。
- [ ] 配置 YOLO-World 伪标签生成流程。
- [ ] 固定 VisDrone 类别文本提示：`pedestrian`、`people`、`bicycle`、`car`、`van`、`truck`、`tricycle`、`awning-tricycle`、`bus`、`motor`。
- [ ] 实现伪标签后处理：confidence filter、class-wise NMS 或 WBF、size filter、aspect-ratio filter。
- [ ] 做置信度阈值实验：`0.2`、`0.3`、`0.4`、`0.5`。
- [ ] 做 10% labeled + 90% pseudo 半监督训练实验。
- [ ] 做 25% labeled + 75% pseudo 半监督训练实验。
- [ ] 对比 Grounding DINO 与 YOLO-World 在航拍小目标伪标签上的质量差异。
- [ ] 做组合实验：YOLO + pseudo labels + DINOv3 small-object distill。

## P3：增强论文说服力

- [ ] 做跨域泛化实验：VisDrone -> UAVDT。
- [ ] 如有自有无人机数据，补 VisDrone -> 自有数据泛化实验。
- [ ] 画数据效率曲线：5%、10%、25%、50%、100%。
- [ ] 做小目标专项分析：AP_small、AR_small、漏检案例、遮挡和低对比度案例。
- [ ] 做 TensorRT FP16 部署验证，对比 baseline YOLO 和 distill YOLO 的 FPS、显存、参数量。
- [ ] 汇总最终实验矩阵：baseline、pseudo、global distill、small-object distill、pseudo + distill。

## P4：可选创新增强

- [ ] 实现 DINO token similarity 或 Gram-style 关系蒸馏。
- [ ] 研究 DINOv3 高响应区域作为 YOLO objectness prior。
- [ ] 研究 teacher/student 不一致性驱动的主动学习采样。
- [ ] 研究类别文本扩展，支持新类别快速冷启动。

## 当前最小闭环

优先完成以下闭环，再扩展 P2-P4：

1. 跑通 VisDrone YOLO baseline：100%、25%、10%。
2. 实现 DINOv3 global feature distillation。
3. 实现 small-object weighted distillation。
4. 对比 25% 和 10% 标注下 AP_small / AR_small 的提升。
5. 验证推理期模型仍为 YOLO-only，并完成 ONNX 或 TensorRT 部署检查。
