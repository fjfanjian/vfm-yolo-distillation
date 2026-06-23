import torch

from vfm_yolo_distillation.relation_distillation import (
    relation_distillation_loss,
    sample_token_indices,
)


def test_sample_token_indices_when_budget_is_smaller_than_token_count() -> None:
    # Given
    token_count = 10
    max_tokens = 4

    # When
    indices = sample_token_indices(token_count, max_tokens, torch.device("cpu"))

    # Then
    assert indices.tolist() == [0, 3, 6, 9]


def test_relation_distillation_loss_when_relations_match_under_rescaling() -> None:
    # Given
    teacher_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    student_tokens = teacher_tokens * 3.0

    # When
    loss = relation_distillation_loss(student_tokens, teacher_tokens, max_tokens=4)

    # Then
    assert loss.item() < 1e-8
