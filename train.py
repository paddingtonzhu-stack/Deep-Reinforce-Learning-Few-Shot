"""Train the visual PPO baseline and save periodic/final checkpoints."""
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from vizdoom_env import VizDoomBasicEnv

def make_env(seed, render_mode=None):
    return Monitor(VizDoomBasicEnv(render_mode=render_mode))

def vector_env(seed):
    return VecTransposeImage(DummyVecEnv([lambda: make_env(seed)]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    # Small CNN batches are much faster without CPU thread oversubscription.
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    for d in (args.output, args.output / "checkpoints", args.output / "best"):
        d.mkdir(parents=True, exist_ok=True)
    env, eval_env = vector_env(args.seed), vector_env(args.seed + 10_000)
    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=str(args.output / "tensorboard"))
    else:
        model = PPO("CnnPolicy", env, learning_rate=2.5e-4, n_steps=512,
                    batch_size=128, n_epochs=4, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs={"features_extractor_kwargs": {"features_dim": 256}},
                    tensorboard_log=str(args.output / "tensorboard"), seed=args.seed,
                    verbose=1, device="auto")
    checkpoint = CheckpointCallback(save_freq=10_000, save_path=str(args.output / "checkpoints"), name_prefix="ppo_vizdoom")
    evaluation = EvalCallback(eval_env, best_model_save_path=str(args.output / "best"),
                              log_path=str(args.output / "eval"), eval_freq=10_000,
                              n_eval_episodes=10, deterministic=True)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoint, evaluation], progress_bar=True)
    model.save(args.output / "ppo_vizdoom_final")
    env.close(); eval_env.close()
    print(f"Saved final checkpoint to {args.output / 'ppo_vizdoom_final.zip'}", flush=True)
    if os.name == "nt":
        os._exit(0)

if __name__ == "__main__":
    main()
