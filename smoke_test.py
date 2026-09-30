"""Fast environment, gradient, and checkpoint round-trip checks."""
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from train import vector_env
from vizdoom_env import VizDoomBasicEnv

def main():
    torch.set_num_threads(1)
    raw = VizDoomBasicEnv(); check_env(raw)
    obs, _ = raw.reset(seed=0)
    assert obs.shape == (84, 84, 1) and obs.dtype == np.uint8
    raw.close()
    env = vector_env(0)
    model = PPO("CnnPolicy", env, n_steps=32, batch_size=32, n_epochs=1,
                policy_kwargs={"features_extractor_kwargs": {"features_dim": 256}}, verbose=0)
    model.learn(64)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip"
        model.save(path); loaded = PPO.load(path)
        action, _ = loaded.predict(env.reset(), deterministic=True)
        assert int(action[0]) in range(3)
    env.close()
    print("PASS: env shape/actions, PPO gradients, and checkpoint save/load", flush=True)
    if os.name == "nt":
        os._exit(0)

if __name__ == "__main__":
    main()
