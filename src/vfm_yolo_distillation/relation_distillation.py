from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional


@dataclass(frozen=True, slots=True)
class RelationShapeError(RuntimeError):
    student_token_count: int
    teacher_token_count: int

    def __str__(self) -> str:
        return (
            "Student and teacher token grids must have the same token count: "
            f"{self.student_token_count} != {self.teacher_token_count}"
        )


@dataclass(frozen=True, slots=True)
class RelationSampleError(RuntimeError):
    token_count: int
    max_tokens: int

    def __str__(self) -> str:
        return f"Relation sampling needs positive counts: {self.token_count=}, {self.max_tokens=}"


def sample_token_indices(token_count: int, max_tokens: int, device: torch.device) -> torch.Tensor:
    if token_count <= 0 or max_tokens <= 0:
        raise RelationSampleError(token_count=token_count, max_tokens=max_tokens)
    sample_count = min(token_count, max_tokens)
    return torch.linspace(0, token_count - 1, steps=sample_count, device=device).round().long()


def relation_matrix(tokens: torch.Tensor) -> torch.Tensor:
    normalized = torch_functional.normalize(tokens, dim=2)
    return torch.bmm(normalized, normalized.transpose(1, 2))


def relation_distillation_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    if student_tokens.shape[1] != teacher_tokens.shape[1]:
        raise RelationShapeError(
            student_token_count=student_tokens.shape[1],
            teacher_token_count=teacher_tokens.shape[1],
        )
    indices = sample_token_indices(student_tokens.shape[1], max_tokens, student_tokens.device)
    student_relations = relation_matrix(student_tokens.index_select(1, indices))
    teacher_relations = relation_matrix(teacher_tokens.index_select(1, indices))
    return torch_functional.mse_loss(student_relations, teacher_relations)
