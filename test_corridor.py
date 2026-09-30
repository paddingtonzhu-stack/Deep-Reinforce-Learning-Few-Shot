"""Fixed-seed final evaluation for all Deadly Corridor training seeds."""
import argparse
import csv
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from vizdoom_env import VizDoomEnv


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


def configure_logging(run_dir: Path):
    logger = logging.getLogger("corridor_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def summarize(rows):
    rewards = [row["reward"] for row in rows]
    return {
        "episodes": len(rows),
        "mean_reward": statistics.fmean(rewards),
        "std_reward": float(np.std(rewards)),
        "median_reward": statistics.median(rewards),
        "mean_decisions": statistics.fmean(row["decisions"] for row in rows),
        "death_rate": statistics.fmean(row["player_dead"] for row in rows),
        "completion_rate": statistics.fmean(row["completed"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/deadly_corridor_baseline.json"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-seeds", help="Comma-separated subset, e.g. 0,1,2")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/deadly_corridor_baseline"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    evaluation = config["evaluation"]
    training_seeds = (
        [int(value) for value in args.training_seeds.split(",")]
        if args.training_seeds else list(experiment["seeds"])
    )
    episodes = args.episodes or int(evaluation.get("test_episodes", 100))
    test_seed_start = args.seed_start if args.seed_start is not None else int(evaluation.get("test_seed_start", 10000))
    output_root = args.output_root or Path(experiment["output_root"])
    run_dir = unique_run_directory(args.results_root)
    logger = configure_logging(run_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    torch.set_num_threads(1)
    rows = []
    summaries = {}
    checkpoint_paths = {}

    for training_seed in training_seeds:
        seed_dir = output_root / f"seed_{training_seed}"
        checkpoint = (
            seed_dir / "best" / "best_model.zip"
            if args.checkpoint == "best"
            else seed_dir / "ppo_vizdoom_final.zip"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for training seed {training_seed}: {checkpoint}")
        checkpoint_paths[str(training_seed)] = str(checkpoint)
        model = PPO.load(checkpoint, device=args.device)
        env = VizDoomEnv(**config["env"])
        seed_rows = []
        logger.info("Training seed %d: checkpoint=%s device=%s", training_seed, checkpoint, model.device)

        for episode_index in range(episodes):
            test_seed = test_seed_start + episode_index
            observation, _ = env.reset(seed=test_seed)
            terminated = truncated = False
            reward_sum = raw_reward_sum = 0.0
            decisions = 0
            final_info = {"player_dead": False, "completed": False}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(int(action))
                reward_sum += float(reward)
                raw_reward_sum += float(final_info["raw_reward"])
                decisions += 1
            row = {
                "training_seed": training_seed,
                "test_seed": test_seed,
                "episode": episode_index + 1,
                "reward": reward_sum,
                "raw_reward": raw_reward_sum,
                "decisions": decisions,
                "player_dead": bool(final_info["player_dead"]),
                "completed": bool(final_info["completed"]),
            }
            rows.append(row)
            seed_rows.append(row)
            logger.info(
                "train_seed=%d episode=%03d test_seed=%d reward=%+.3f decisions=%d dead=%s completed=%s",
                training_seed, episode_index + 1, test_seed, reward_sum, decisions,
                row["player_dead"], row["completed"],
            )
        summaries[str(training_seed)] = summarize(seed_rows)
        logger.info("Training seed %d summary: %s", training_seed, summaries[str(training_seed)])
        env.close()

    aggregate = summarize(rows)
    resolved = {
        "training_seeds": training_seeds,
        "test_seed_start": test_seed_start,
        "test_seed_end": test_seed_start + episodes - 1,
        "episodes_per_model": episodes,
        "deterministic": True,
        "checkpoint_kind": args.checkpoint,
        "device_requested": args.device,
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    report = {
        "metadata": metadata,
        "evaluation": resolved,
        "checkpoints": checkpoint_paths,
        "per_training_seed": summaries,
        "aggregate": aggregate,
        "episodes": rows,
    }
    (run_dir / "config.json").write_text(json.dumps({"source": config, "resolved_evaluation": resolved}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Aggregate summary: %s", aggregate)
    logger.info("Saved versioned evaluation: %s", run_dir)
    logging.shutdown()


if __name__ == "__main__":
    main()
