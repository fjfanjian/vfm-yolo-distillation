# ruff: noqa: D103, E402, INP001, S101, SLF001
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if SCRIPTS.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPTS.as_posix())

from scripts.train_dinov3_objectness_pretrain import (
    DinoObjectnessPretrainTrainer,
    ObjectnessSettings,
)


class _RecordingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dtype: torch.dtype | None = None

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.input_dtype = inputs.dtype
        return {"x_norm_patchtokens": torch.ones((inputs.shape[0], 4, 2), dtype=inputs.dtype)}


class _TrainerHarness(DinoObjectnessPretrainTrainer):
    def __init__(self) -> None:
        self.objectness_settings = ObjectnessSettings(
            teacher_repo=Path("unused"),
            teacher_weights=Path("unused"),
            teacher_arch="dinov3_vitb16",
            teacher_image_size=16,
            teacher_patch_grid=2,
            teacher_batch=1,
            method="local_contrast",
            tile_size=8,
            tile_stride=4,
            student_layer=16,
            student_dim=64,
            lambda_objectness=0.02,
            cache=Path("unused"),
        )
        self.teacher_model = None
        self.teacher_mean = None
        self.teacher_std = None
        self.teacher = _RecordingTeacher()

    def _load_teacher(self, device: torch.device) -> nn.Module:
        return self.teacher.to(device)


def test_teacher_tokens_casts_half_crops_to_float32_before_teacher_forward() -> None:
    # Given
    trainer = _TrainerHarness()
    crops = torch.rand((1, 3, 8, 8), dtype=torch.float16)

    # When
    tokens = trainer._teacher_tokens(crops)

    # Then
    assert trainer.teacher.input_dtype is torch.float32
    assert tokens.dtype is torch.float32
