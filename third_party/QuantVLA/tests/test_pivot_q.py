import pytest
import torch

from gr00t.experiment.pivot_q import (
    PIVOTQConfig,
    action_disagreement,
    build_pivot_q_targets,
    temporal_quantization_risk,
    weighted_distillation_loss,
)


def test_action_disagreement_uses_executed_action_and_requested_dimensions():
    teacher = torch.zeros(2, 3, 9)
    student = teacher.clone()
    student[0, 0, :7] = 1.0
    student[0, 1, :7] = 100.0
    student[1, 0, 7:] = 100.0

    disagreement = action_disagreement(student, teacher, action_dims=7)

    torch.testing.assert_close(disagreement, torch.tensor([1.0, 0.0]))


def test_temporal_risk_is_discounted_and_end_normalized():
    q_values = torch.tensor([1.0, 2.0, 4.0])

    risks = temporal_quantization_risk(q_values, horizon=2, discount=0.5)

    expected = torch.tensor(
        [
            (1.0 + 0.5 * 2.0 + 0.25 * 4.0) / 1.75,
            (2.0 + 0.5 * 4.0) / 1.5,
            4.0,
        ]
    )
    torch.testing.assert_close(risks, expected)


def test_horizon_zero_reduces_temporal_risk_to_instantaneous_disagreement():
    q_values = torch.tensor([0.1, 0.2, 0.3])
    risks = temporal_quantization_risk(q_values, horizon=0, discount=0.9)
    torch.testing.assert_close(risks, q_values)


def test_critical_weights_are_clipped_and_detached():
    student = torch.zeros(3, 2, 7, requires_grad=True)
    teacher = torch.ones_like(student)
    config = PIVOTQConfig(alpha_q=10.0, beta_r=10.0, weight_max=3.0)

    q_values, r_values, weights = build_pivot_q_targets(student, teacher, config)

    assert not q_values.requires_grad
    assert not r_values.requires_grad
    assert not weights.requires_grad
    assert torch.all(weights >= 1.0)
    assert torch.all(weights <= 3.0)


def test_weighted_loss_matches_manual_reduction():
    losses = torch.tensor([1.0, 3.0])
    weights = torch.tensor([1.0, 2.0])
    loss = weighted_distillation_loss(losses, weights)
    torch.testing.assert_close(loss, torch.tensor(7.0 / 3.0))


@pytest.mark.parametrize("horizon", [-1, -4])
def test_negative_temporal_horizon_is_rejected(horizon):
    with pytest.raises(ValueError):
        temporal_quantization_risk(torch.ones(2), horizon=horizon, discount=0.9)
