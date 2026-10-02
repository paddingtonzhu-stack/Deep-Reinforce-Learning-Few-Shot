"""Completion objective for Population-Based Training in Deadly Corridor."""
import functools

import gymnasium as gym


COMPLETION_ENV = "doom_deadly_corridor_completion_pbt"
BASE_ENV = "doom_deadly_corridor"


class CorridorCompletionObjective(gym.Wrapper):
    """Report a binary terminal objective without changing observations or rewards."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            dead = bool(self.env.unwrapped.game.is_player_dead())
            # A time-limit truncation is not a successful corridor completion.
            info["true_objective"] = float(bool(terminated and not dead))
        return obs, reward, terminated, truncated, info


def make_completion_env(spec, env_name, cfg, env_config, render_mode=None, **kwargs):
    from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec

    env = make_doom_env_from_spec(
        spec, env_name, cfg, env_config, render_mode=render_mode, **kwargs
    )
    return CorridorCompletionObjective(env)


def register_corridor_completion_env():
    """Register an opt-in Deadly Corridor variant whose PBT objective is completion."""
    from sample_factory.envs.env_utils import register_env
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS

    spec = next(spec for spec in DOOM_ENVS if spec.name == BASE_ENV)
    register_env(COMPLETION_ENV, functools.partial(make_completion_env, spec))
