"""Clone a Sample Factory experiment for non-destructive low-LR fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def prepare_finetune(
    source_experiment: Path,
    destination_experiment: Path,
    destination_train_dir: Path,
    learning_rate: float,
    target_env_steps: int,
) -> Path:
    if destination_experiment.exists():
        raise FileExistsError(f"refusing to overwrite {destination_experiment}")
    checkpoints = sorted((source_experiment / "checkpoint_p0").glob("checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"no regular checkpoint in {source_experiment / 'checkpoint_p0'}")
    source_checkpoint = checkpoints[-1]
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("env_steps", 0) >= target_env_steps:
        raise ValueError(
            f"target {target_env_steps} must exceed source env_steps {checkpoint.get('env_steps')}"
        )
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict) or not optimizer.get("param_groups"):
        raise ValueError("checkpoint has no optimizer parameter groups")
    for group in optimizer["param_groups"]:
        group["lr"] = learning_rate
        if "initial_lr" in group:
            group["initial_lr"] = learning_rate
    checkpoint["curr_lr"] = learning_rate

    config = json.loads((source_experiment / "config.json").read_text(encoding="utf-8"))
    config["experiment"] = destination_experiment.name
    config["train_dir"] = str(destination_train_dir)
    config["learning_rate"] = learning_rate
    config["train_for_env_steps"] = target_env_steps

    checkpoint_dir = destination_experiment / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    (destination_experiment / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination_checkpoint = checkpoint_dir / source_checkpoint.name
    torch.save(checkpoint, destination_checkpoint)
    provenance = {
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
        "source_env_steps": checkpoint["env_steps"],
        "target_env_steps": target_env_steps,
        "fine_tune_learning_rate": learning_rate,
    }
    (destination_experiment / "finetune_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {destination_experiment} from {source_checkpoint}", flush=True)
    return destination_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--destination-experiment", type=Path, required=True)
    parser.add_argument("--destination-train-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--target-env-steps", type=int, required=True)
    args = parser.parse_args()
    prepare_finetune(
        args.source_experiment,
        args.destination_experiment,
        args.destination_train_dir,
        args.learning_rate,
        args.target_env_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
