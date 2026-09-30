"""Gymnasium wrapper for the visual-only ViZDoom Basic scenario."""
from pathlib import Path
import os
import time
import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd

class VizDoomBasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}
    def __init__(self, image_size=84, frame_skip=4, render_mode=None, max_episode_steps=525,
                 reward_scale=0.01):
        super().__init__()
        self.image_size, self.frame_skip = image_size, frame_skip
        self.render_mode, self.max_episode_steps, self._steps = render_mode, max_episode_steps, 0
        self.reward_scale = reward_scale
        self.game = vzd.DoomGame()
        self.game.load_config(str(Path(vzd.scenarios_path) / "basic.cfg"))
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_window_visible(render_mode == "human")
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.init()
        self._actions = [[True, False, False], [False, True, False], [False, False, True]]
        self.action_space = gym.spaces.Discrete(3)
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
        raw_reward = float(self.game.make_action(self._actions[action], self.frame_skip))
        # Synchronous ViZDoom runs as fast as possible. Pace only visible demos so
        # a human can actually follow the agent's movement and shots.
        if self.render_mode == "human":
            time.sleep(self.frame_skip / 35.0)
        reward = raw_reward * self.reward_scale
        self._steps += self.frame_skip
        terminated = self.game.is_episode_finished()
        truncated = self._steps >= self.max_episode_steps and not terminated
        return self._observation(), reward, terminated, truncated, {"raw_reward": raw_reward}

    def render(self):
        state = self.game.get_state()
        return None if state is None else state.screen_buffer.copy()

    def close(self):
        # ViZDoom's Windows child process can block indefinitely in close().
        # Entry-point scripts use os._exit after saving/flushing as a workaround.
        if os.name != "nt":
            self.game.close()
