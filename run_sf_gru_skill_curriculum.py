"""Train a matched two-stage Deadly Corridor skill curriculum."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from prepare_sf_finetune import prepare_finetune


def run(command: list[str], gpu: str) -> None:
    print("+", " ".join(command), flush=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(command, check=True, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir", type=Path, default=Path("artifacts/sample_factory_gru_skill_curriculum_2m")
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--easy-steps", type=int, default=1_000_000)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]
    if args.easy_steps >= args.total_steps:
        raise ValueError("easy-steps must be smaller than total-steps")
    for seed in seeds:
        easy_name = f"sf_corridor_gru_skill1_seed_{seed}"
        curriculum_name = f"sf_corridor_gru_curriculum_seed_{seed}"
        run(
            [
                sys.executable,
                "sf_train_corridor.py",
                "--method=gru_skill_curriculum",
                f"--experiment={easy_name}",
                f"--train_dir={args.train_dir}",
                f"--seed={seed}",
                "--num_policies=1",
                "--corridor-skill=1",
                f"--train_for_env_steps={args.easy_steps}",
                f"--learning_rate={args.learning_rate}",
            ],
            args.gpu,
        )
        prepare_finetune(
            args.train_dir / easy_name,
            args.train_dir / curriculum_name,
            args.train_dir,
            args.learning_rate,
            args.total_steps,
            config_overrides={"env": "doom_deadly_corridor", "corridor_skill": 5},
        )
        run(
            [
                sys.executable,
                "sf_train_corridor.py",
                "--method=gru_long",
                f"--experiment={curriculum_name}",
                f"--train_dir={args.train_dir}",
                f"--seed={seed}",
                "--num_policies=1",
                "--env=doom_deadly_corridor",
                f"--train_for_env_steps={args.total_steps}",
                f"--learning_rate={args.learning_rate}",
            ],
            args.gpu,
        )
    print("All matched GRU skill-curriculum runs completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
