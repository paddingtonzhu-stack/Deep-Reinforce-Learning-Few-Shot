"""Tests for the binary completion objective used by PBT."""
from types import SimpleNamespace

import gymnasium as gym

from sf_corridor_objective import CorridorCompletionObjective


class FakeCorridor(gym.Env):
    def __init__(self, *, terminated, truncated, dead):
        self.terminated = terminated
        self.truncated = truncated
        self.game = SimpleNamespace(is_player_dead=lambda: dead)

    def step(self, action):
        return "obs", 3.5, self.terminated, self.truncated, {"kept": True}


def test_completed_episode_has_objective_one():
    result = CorridorCompletionObjective(FakeCorridor(terminated=True, truncated=False, dead=False)).step(0)
    assert result[-1] == {"kept": True, "true_objective": 1.0}


def test_death_has_objective_zero():
    result = CorridorCompletionObjective(FakeCorridor(terminated=True, truncated=False, dead=True)).step(0)
    assert result[-1]["true_objective"] == 0.0


def test_truncation_and_nonterminal_steps_are_not_successes():
    truncated = CorridorCompletionObjective(FakeCorridor(terminated=False, truncated=True, dead=False)).step(0)
    running = CorridorCompletionObjective(FakeCorridor(terminated=False, truncated=False, dead=False)).step(0)
    assert truncated[-1]["true_objective"] == 0.0
    assert "true_objective" not in running[-1]
