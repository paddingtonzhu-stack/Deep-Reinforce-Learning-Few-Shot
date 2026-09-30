"""Train the visual PPO baseline and save periodic/final checkpoints."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from vizdoom_env import VizDoomEnv

def make_env(env_config, render_mode=None):
    return Monitor(VizDoomEnv(render_mode=render_mode, **env_config))

def vector_env(env_config):
    # Preserve the original vector_env(seed) helper used by smoke_test.py.
    if not isinstance(env_config, dict):
        env_config = {"scenario": "basic", "showcase_actions": False}
    return VecTransposeImage(DummyVecEnv([lambda: make_env(env_config)]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path)
    p.add_argument("--timesteps", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    env_config = config.get("env", {"scenario": "basic", "showcase_actions": False})
    ppo_config = config.get("ppo", {})
    evaluation_config = config.get("evaluation", {})
    output = args.output or Path(config.get("experiment", {}).get("output_root", "artifacts"))
    timesteps = args.timesteps or int(ppo_config.get("total_timesteps", 100_000))
    # Small CNN batches are much faster without CPU thread oversubscription.
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    for d in (output, output / "checkpoints", output / "best"):
        d.mkdir(parents=True, exist_ok=True)
    env, eval_env = vector_env(env_config), vector_env(env_config)
    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=str(output / "tensorboard"), device=args.device)
    else:
        model = PPO("CnnPolicy", env,
                    learning_rate=float(ppo_config.get("learning_rate", 2.5e-4)),
                    n_steps=int(ppo_config.get("n_steps", 512)),
                    batch_size=int(ppo_config.get("batch_size", 128)),
                    n_epochs=int(ppo_config.get("n_epochs", 4)),
                    gamma=float(ppo_config.get("gamma", 0.99)),
                    gae_lambda=float(ppo_config.get("gae_lambda", 0.95)),
                    clip_range=float(ppo_config.get("clip_range", 0.2)),
                    ent_coef=float(ppo_config.get("entropy_coef", 0.01)),
                    vf_coef=float(ppo_config.get("value_coef", 0.5)),
                    max_grad_norm=float(ppo_config.get("max_grad_norm", 0.5)),
                    policy_kwargs={"features_extractor_kwargs": {"features_dim": int(config.get("model", {}).get("features_dim", 256))}},
                    tensorboard_log=str(output / "tensorboard"), seed=args.seed,
                    verbose=1, device=args.device)
    eval_frequency = int(evaluation_config.get("frequency", 10_000))
    checkpoint = CheckpointCallback(save_freq=eval_frequency, save_path=str(output / "checkpoints"), name_prefix="ppo_vizdoom")
    evaluation = EvalCallback(eval_env, best_model_save_path=str(output / "best"),
                              log_path=str(output / "eval"), eval_freq=eval_frequency,
                              n_eval_episodes=int(evaluation_config.get("episodes", 10)), deterministic=True)
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint, evaluation],
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume),
    )
    model.save(output / "ppo_vizdoom_final")
    env.close(); eval_env.close()
    print(f"Saved final checkpoint to {output / 'ppo_vizdoom_final.zip'}", flush=True)
    if os.name == "nt":
        os._exit(0)

if __name__ == "__main__":
    main()
