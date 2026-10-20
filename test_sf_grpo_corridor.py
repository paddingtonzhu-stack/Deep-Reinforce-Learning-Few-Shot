import torch

from sf_grpo_corridor import clipped_grpo_objective, group_advantages


def test_group_advantages_are_centered_and_scaled():
    advantages = group_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_equal_group_returns_produce_no_update_signal():
    assert torch.equal(group_advantages(torch.ones(4)), torch.zeros(4))


def test_clipped_objective_respects_positive_advantage_clip():
    result = clipped_grpo_objective(
        torch.log(torch.tensor(2.0)),
        torch.tensor(0.0),
        torch.log(torch.tensor(2.0)),
        torch.tensor(1.0),
        clip_ratio=0.1,
        beta=0.0,
    )
    assert torch.allclose(result, torch.tensor(1.1))
