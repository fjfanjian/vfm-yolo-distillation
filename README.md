# VFM YOLO Distillation

面向少标注航拍小目标检测的视觉基础模型知识迁移研究。

本项目的核心目标不是把 DINOv3 直接放进最终检测网络，而是在训练期利用 DINOv3、Grounding DINO、YOLO-World 等视觉基础模型提供的 dense representation、开放词表伪标签和区域先验，训练一个部署阶段仍然轻量的 YOLO student。

## 研究假设

大模型具备更强的通用视觉表征和开放词表能力，但推理成本高；YOLO 部署成熟、速度快，但在少标注、小目标、跨域航拍场景下容易泛化不足。本项目研究如何把大模型知识迁移到 YOLO，使推理阶段不依赖大模型。

## 初始实验阶段

### 阶段 1：纯监督 YOLO 基线

先在 VisDrone 上建立 YOLO student 的可比基线：

| 实验 | 数据量 | 目的 |
| --- | ---: | --- |
| `yolo_baseline_full` | 100% | 生产模型上限 |
| `yolo_baseline_25pct` | 25% | 少标注性能 |
| `yolo_baseline_10pct` | 10% | 极少标注性能 |
| `yolo_cross_domain` | VisDrone -> UAVDT/自有航拍数据 | 泛化测试 |

### 阶段 2：开放词表伪标签

使用 Grounding DINO 或 YOLO-World 根据类别文本生成候选框，再经置信度过滤、NMS/WBF 和可选人工校正，扩充少标注训练集。

### 阶段 3：DINOv3 dense feature 蒸馏

训练期冻结 DINOv3 teacher，将其 dense feature、区域响应或 token 相似性蒸馏给 YOLO neck/head，重点提升 AP_small 和少标注泛化。

## 推荐目录

```text
configs/
  datasets/       # 数据集配置
  experiments/    # 实验配置
  teachers/       # teacher 模型配置
scripts/          # 可执行入口
src/              # 可复用库代码
tests/            # 轻量测试
```

## 快速开始

安装依赖：

```bash
uv sync
```

如果要立即运行 Ultralytics 训练，再安装 YOLO 额外依赖：

```bash
uv sync --extra yolo
```

查看一个实验配置对应的训练命令：

```bash
uv run python scripts/show_experiment.py configs/experiments/yolo_baseline_visdrone.yaml
```

后续真正训练时，优先调用 Ultralytics 官方训练入口；只有蒸馏需要深度改 loss/trainer 时，再考虑 fork 或 vendor Ultralytics。
