"""Completion objective for Population-Based Training in Deadly Corridor."""
import functools

import gymnasium as gym


COMPLETION_ENV = "doom_deadly_corridor_completion_pbt"
SHAPED_COMPLETION_ENV = "doom_deadly_corridor_completion_bonus"
SKILL_CURRICULUM_ENV = "doom_deadly_corridor_skill_curriculum"
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


class CorridorCompletionBonus(gym.Wrapper):
    """Add an opt-in terminal bonus without changing evaluation semantics."""

    def __init__(self, env, completion_bonus):
        super().__init__(env)
        self.completion_bonus = float(completion_bonus)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and not bool(self.env.unwrapped.game.is_player_dead()):
            reward = reward + self.completion_bonus
        return obs, reward, terminated, truncated, info


def make_completion_env(spec, env_name, cfg, env_config, render_mode=None, **kwargs):
    from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec

    env = make_doom_env_from_spec(
        spec, env_name, cfg, env_config, render_mode=render_mode, **kwargs
    )
    return CorridorCompletionObjective(env)


def make_shaped_completion_env(spec, env_name, cfg, env_config, render_mode=None, **kwargs):
    from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec

    env = make_doom_env_from_spec(
        spec, env_name, cfg, env_config, render_mode=render_mode, **kwargs
    )
    return CorridorCompletionBonus(env, cfg.completion_bonus)


def configure_doom_skill(env, skill):
    """Set scenario skill after config loading but before VizDoom initialization."""
    base = env.unwrapped
    original_create = base._create_doom_game

    def create_with_skill(mode):
        original_create(mode)
        base.game.set_doom_skill(int(skill))

    base._create_doom_game = create_with_skill
    return env


def make_skill_curriculum_env(spec, env_name, cfg, env_config, render_mode=None, **kwargs):
    from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec

    env = make_doom_env_from_spec(
        spec, env_name, cfg, env_config, render_mode=render_mode, **kwargs
    )
    return configure_doom_skill(env, cfg.corridor_skill)


def register_corridor_completion_env():
    """Register an opt-in Deadly Corridor variant whose PBT objective is completion."""
    from sample_factory.envs.env_utils import register_env
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS

    spec = next(spec for spec in DOOM_ENVS if spec.name == BASE_ENV)
    register_env(COMPLETION_ENV, functools.partial(make_completion_env, spec))
    register_env(
        SHAPED_COMPLETION_ENV,
        functools.partial(make_shaped_completion_env, spec),
    )
    register_env(
        SKILL_CURRICULUM_ENV,
        functools.partial(make_skill_curriculum_env, spec),
    )
