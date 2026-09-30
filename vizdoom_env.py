"""Gymnasium wrapper for the visual-only ViZDoom Basic scenario."""
from pathlib import Path
import os
import time
import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd

SCENARIO_ACTIONS = {
    "basic": {
        "buttons": ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
        "actions": {
            "MOVE_LEFT": (1, 0, 0),
            "MOVE_RIGHT": (0, 1, 0),
            "ATTACK": (0, 0, 1),
        },
        "timeout": 300,
    },
    "deadly_corridor": {
        "buttons": ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"),
        "actions": {
            "MOVE_FORWARD": (0, 0, 0, 1, 0, 0, 0),
            "MOVE_BACKWARD": (0, 0, 0, 0, 1, 0, 0),
            "STRAFE_LEFT": (1, 0, 0, 0, 0, 0, 0),
            "STRAFE_RIGHT": (0, 1, 0, 0, 0, 0, 0),
            "TURN_LEFT": (0, 0, 0, 0, 0, 1, 0),
            "TURN_RIGHT": (0, 0, 0, 0, 0, 0, 1),
            "ATTACK": (0, 0, 1, 0, 0, 0, 0),
            "FORWARD_ATTACK": (0, 0, 1, 1, 0, 0, 0),
        },
        "timeout": 2100,
    },
}


class VizDoomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}
    def __init__(self, scenario="basic", image_size=84, frame_skip=4, render_mode=None,
                 max_episode_steps=None, reward_scale=0.01, render_tic_delay=0.75,
                 showcase_actions=False, terminal_death_penalty=0.0,
                 completion_bonus=0.0):
        super().__init__()
        if scenario not in SCENARIO_ACTIONS:
            raise ValueError(f"Unsupported scenario {scenario!r}; choose from {tuple(SCENARIO_ACTIONS)}")
        scenario_spec = SCENARIO_ACTIONS[scenario]
        self.scenario = scenario
        self.image_size, self.frame_skip = image_size, frame_skip
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps or scenario_spec["timeout"]
        self._steps = 0
        self.reward_scale = reward_scale
        self.terminal_death_penalty = terminal_death_penalty
        self.completion_bonus = completion_bonus
        self.render_tic_delay = render_tic_delay
        self.game = vzd.DoomGame()
        self.game.load_config(str(Path(vzd.scenarios_path) / f"{scenario}.cfg"))
        if scenario == "basic" and showcase_actions:
            self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_window_visible(render_mode == "human")
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.init()
        if scenario == "basic" and showcase_actions:
            # Shoot while advancing makes successful behavior visually legible.
            self._actions = [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, False, True],
                [False, False, True, True],
            ]
            self.action_names = ("MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "FORWARD_ATTACK")
        else:
            self.action_names = tuple(scenario_spec["actions"])
            self._actions = [list(action) for action in scenario_spec["actions"].values()]
        self.action_space = gym.spaces.Discrete(len(self._actions))
        self.observation_space = gym.spaces.Box(0, 255, (image_size, image_size, 1), np.uint8)

    def _observation(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        gray = cv2.cvtColor(state.screen_buffer, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return gray[..., None].astype(np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self._steps = 0
        return self._observation(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        if self.render_mode == "human":
            # Advance one engine tick at a time so movement and firing are visible.
            raw_reward = 0.0
            for _ in range(self.frame_skip):
                raw_reward += float(self.game.make_action(self._actions[action], 1))
                time.sleep(self.render_tic_delay)
                if self.game.is_episode_finished():
                    break
        else:
            raw_reward = float(self.game.make_action(self._actions[action], self.frame_skip))
        self._steps += self.frame_skip
        episode_finished = self.game.is_episode_finished()
        player_dead = bool(self.game.is_player_dead()) if episode_finished else False
        timed_out = self._steps >= self.max_episode_steps and not player_dead
        completed = bool(episode_finished and not player_dead and not timed_out)
        terminated = bool(episode_finished and not timed_out)
        truncated = bool(timed_out)
        base_reward = raw_reward * self.reward_scale
        terminal_adjustment = 0.0
        if player_dead:
            terminal_adjustment += self.terminal_death_penalty
        elif completed:
            terminal_adjustment += self.completion_bonus
        reward = base_reward + terminal_adjustment
        return self._observation(), reward, terminated, truncated, {
            "raw_reward": raw_reward,
            "base_reward": base_reward,
            "terminal_adjustment": terminal_adjustment,
            "player_dead": player_dead,
            "timed_out": timed_out,
            "completed": completed,
        }

    def render(self):
        state = self.game.get_state()
        return None if state is None else state.screen_buffer.copy()

    def close(self):
        # ViZDoom's Windows child process can block indefinitely in close().
        # Entry-point scripts use os._exit after saving/flushing as a workaround.
        if os.name != "nt":
            self.game.close()


class VizDoomBasicEnv(VizDoomEnv):
    """Backward-compatible Basic scenario wrapper."""

    def __init__(self, **kwargs):
        super().__init__(scenario="basic", **kwargs)
