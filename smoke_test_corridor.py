"""Short gradient/checkpoint test for the configurable Deadly Corridor baseline."""
import json
import os
import tempfile
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from vizdoom_env import VizDoomEnv


def main():
    config = json.loads(Path("configs/deadly_corridor_baseline.json").read_text(encoding="utf-8"))
    env_config = config["env"]
    raw_env = VizDoomEnv(**env_config)
    observation, _ = raw_env.reset(seed=0)
    assert observation.shape == (84, 84, 1)
    assert raw_env.action_space.n == 8
    raw_env.step(0)
    raw_env.close()

    env = VecTransposeImage(DummyVecEnv([lambda: VizDoomEnv(**env_config)]))
    model = PPO("CnnPolicy", env, n_steps=16, batch_size=8, n_epochs=1,
                policy_kwargs={"features_extractor_kwargs": {"features_dim": 256}},
                device="cpu", seed=0, verbose=0)
    model.learn(total_timesteps=16)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "corridor_smoke"
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, env=env, device="cpu")
        loaded.predict(env.reset(), deterministic=True)
    print("PASS: Deadly Corridor observation/actions, PPO gradients, and checkpoint round-trip", flush=True)
    if os.name == "nt":
        os._exit(0)


if __name__ == "__main__":
    main()
