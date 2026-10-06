"""Run matched low-learning-rate continuation for completed GRU-64 seeds."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from prepare_sf_finetune import prepare_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_recurrence_10m_confirm"),
    )
    parser.add_argument(
        "--train-dir", type=Path, default=Path("artifacts/sample_factory_gru_lr_finetune_12m")
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--target-env-steps", type=int, default=12_000_000)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]
    if len(seeds) < 1:
        raise ValueError("at least one seed is required")
    for seed in seeds:
        source_name = f"sf_corridor_gru_r64_seed_{seed}"
        destination_name = f"sf_corridor_gru_lrft_seed_{seed}"
        prepare_finetune(
            args.source_train_dir / source_name,
            args.train_dir / destination_name,
            args.train_dir,
            args.learning_rate,
            args.target_env_steps,
        )
        command = [
            sys.executable,
            "sf_train_corridor.py",
            "--method=gru_long",
            f"--experiment={destination_name}",
            f"--train_dir={args.train_dir}",
            f"--seed={seed}",
            "--num_policies=1",
            f"--train_for_env_steps={args.target_env_steps}",
            f"--learning_rate={args.learning_rate}",
        ]
        print("+", " ".join(command), flush=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        subprocess.run(command, check=True, env=environment)
    print("All matched low-LR GRU continuations completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
