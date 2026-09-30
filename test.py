"""Versioned in-distribution/OOD evaluation of the frozen PPO baseline."""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from ood_env import OODCondition, make_ood_env


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ood_eval.json"))
    parser.add_argument("--episodes", type=int, help="Override episodes per condition")
    parser.add_argument("--seed-start", type=int, help="Override first evaluation seed")
    parser.add_argument("--results-root", type=Path, help="Override versioned-results directory")
    parser.add_argument("--device", help="PyTorch device: auto, cpu, cuda, cuda:0, cuda:1, ...")
    return parser.parse_args()


def unique_run_directory(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def summarize(rows):
    rewards = [row["reward"] for row in rows]
    lengths = [row["decisions"] for row in rows]
    return {
        "episodes": len(rows),
        "mean_reward": statistics.fmean(rewards),
        "std_reward": float(np.std(rewards)),
        "median_reward": statistics.median(rewards),
        "mean_episode_decisions": statistics.fmean(lengths),
        "positive_reward_episodes": sum(row["positive_reward"] for row in rows),
    }


def main():
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    episodes = args.episodes or int(config["episodes_per_condition"])
    seed_start = args.seed_start if args.seed_start is not None else int(config["seed_start"])
    results_root = args.results_root or Path(config["results_root"])
    requested_device = args.device or config.get("device", "auto")
    run_dir = unique_run_directory(results_root)
    model_path = Path(config["model"])
    if config["scenario"] != "basic":
        raise ValueError("The current OOD suite supports the frozen Basic baseline only")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    resolved = dict(config)
    resolved.update({
        "episodes_per_condition": episodes,
        "seed_start": seed_start,
        "device": requested_device,
    })
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{requested_device}' was requested, but PyTorch cannot access CUDA. "
            "Run with --device cpu or install a CUDA-enabled PyTorch build."
        )
    torch.set_num_threads(1)
    model = PPO.load(model_path, device=requested_device)
    deterministic = bool(config.get("deterministic_policy", True))
    all_rows = []
    summaries = {}

    for condition_data in config["conditions"]:
        condition = OODCondition(**condition_data)
        env = make_ood_env(condition)
        condition_rows = []
        print(f"\ncondition={condition.name}", flush=True)
        for index in range(episodes):
            seed = seed_start + index
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            reward_sum = raw_reward_sum = 0.0
            decisions = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(int(action))
                reward_sum += float(reward)
                raw_reward_sum += float(info["raw_reward"])
                decisions += 1
            row = {
                "condition": condition.name,
                "episode": index + 1,
                "seed": seed,
                "reward": reward_sum,
                "raw_reward": raw_reward_sum,
                "decisions": decisions,
                "positive_reward": reward_sum > 0,
            }
            condition_rows.append(row)
            all_rows.append(row)
            print(f"  episode={index + 1:02d} seed={seed} reward={reward_sum:+.3f} decisions={decisions}", flush=True)
        summaries[condition.name] = summarize(condition_rows)
        print(f"  mean_reward={summaries[condition.name]['mean_reward']:+.3f}", flush=True)
        env.close()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "requested_device": requested_device,
        "model_device": str(model.device),
        "model": str(model_path),
        "deterministic_policy": deterministic,
        "seed_start": seed_start,
    }
    report = {"metadata": metadata, "config": resolved, "summary": summaries, "episodes": all_rows}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved versioned results: {run_dir}", flush=True)
    if os.name == "nt":
        os._exit(0)


if __name__ == "__main__":
    main()
