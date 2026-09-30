"""Evaluate or visually demo a saved PPO checkpoint."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from vizdoom_env import VizDoomBasicEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("artifacts/demo/best/best_model.zip"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--render", action="store_true")
    p.add_argument("--render-delay", type=float, default=0.75,
                   help="Seconds per Doom tick in visible demos (default: 0.75)")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--showcase", action="store_true", help="Use forward-capable demo actions")
    args = p.parse_args()
    torch.set_num_threads(1)
    env = VizDoomBasicEnv(render_mode="human" if args.render else None,
                          render_tic_delay=args.render_delay,
                          showcase_actions=args.showcase)
    model = PPO.load(args.model, device="auto")
    rewards, lengths = [], []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            done = False; total = 0.0; length = 0
            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total += reward; length += 1; done = terminated or truncated
                if args.render:
                    print(f"  step={length:02d} action={env.action_names[int(action)]:>14} "
                          f"reward={reward:+.2f} total={total:+.2f}", flush=True)
            rewards.append(total); lengths.append(length)
            print(f"episode={episode + 1} reward={total:.2f} decisions={length}")
    finally:
        env.close()
    values = np.asarray(rewards)
    print(f"\nreward mean/std/median: {values.mean():.2f} / {values.std():.2f} / {np.median(values):.2f}")
    print(f"positive-reward episodes: {int((values > 0).sum())}/{len(values)}")
    print(f"mean decisions: {np.mean(lengths):.1f}", flush=True)
    if os.name == "nt":
        os._exit(0)

if __name__ == "__main__":
    main()
