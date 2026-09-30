"""Deterministic observation and dynamics shifts for ViZDoom OOD evaluation."""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from vizdoom_env import VizDoomBasicEnv


@dataclass(frozen=True)
class OODCondition:
    name: str
    brightness: float = 1.0
    noise_std: float = 0.0
    flicker_probability: float = 0.0
    frame_skip: int = 4


class ObservationShift(gym.ObservationWrapper):
    """Apply a reproducible visual shift without exposing privileged state."""

    def __init__(self, env, condition: OODCondition):
        super().__init__(env)
        self.condition = condition
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Separate transform randomness from ViZDoom's episode randomness.
            self._rng = np.random.default_rng(seed + 1_000_003)
        return super().reset(seed=seed, options=options)

    def observation(self, observation):
        image = observation.astype(np.float32)
        if self.condition.brightness != 1.0:
            image *= self.condition.brightness
        if self.condition.noise_std > 0:
            image += self._rng.normal(0.0, self.condition.noise_std, image.shape)
        if self.condition.flicker_probability > 0 and self._rng.random() < self.condition.flicker_probability:
            image.fill(0.0)
        return np.clip(image, 0, 255).astype(np.uint8)


def make_ood_env(condition: OODCondition):
    base = VizDoomBasicEnv(frame_skip=condition.frame_skip, showcase_actions=False)
    return ObservationShift(base, condition)
