"""Unit tests for deterministic multi-seed policy voting."""

import pytest
import torch

from sf_evaluate_corridor import majority_vote_actions


def test_majority_vote_is_per_action_branch():
    actions = [
        torch.tensor([[0, 1, 2, 0]]),
        torch.tensor([[0, 2, 2, 1]]),
        torch.tensor([[1, 1, 2, 1]]),
    ]
    assert torch.equal(
        majority_vote_actions(actions),
        torch.tensor([[0, 1, 2, 1]]),
    )


def test_three_way_tie_uses_fixed_member():
    actions = [
        torch.tensor([[0, 0]]),
        torch.tensor([[1, 0]]),
        torch.tensor([[2, 1]]),
    ]
    assert torch.equal(
        majority_vote_actions(actions, tie_break_index=1),
        torch.tensor([[1, 0]]),
    )


def test_vote_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        majority_vote_actions([])
    with pytest.raises(IndexError):
        majority_vote_actions([torch.tensor([[0]])], tie_break_index=1)
