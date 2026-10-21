"""Zero-shot and low-budget adaptation screen on held-out Corridor spawns."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from prepare_sf_finetune import prepare_finetune
from sf_corridor_spawn_variant import ENV_NAME, build_variant


DEFAULT_SOURCE = Path("artifacts/deadly-corridor-upstream/checkpoint_p0/best_000002847_11661312_reward_21.699.pth")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, default=Path("artifacts/deadly-corridor-upstream"))
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-dir", type=Path, default=Path("artifacts/sample_factory_corridor_spawn_fewshot_32k"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/sample_factory_corridor_spawn_fewshot_32k"))
    parser.add_argument("--adaptation-frames", type=int, default=32768)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--eval-seed-start", type=int, default=450000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def run(command, gpu):
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, env=env)


def latest_report(root: Path):
    reports = sorted(root.glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(f"no report under {root}")
    return json.loads(reports[-1].read_text(encoding="utf-8")), reports[-1]


def main():
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    _, variant = build_variant()
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    source_steps = int(checkpoint["env_steps"])
    target_steps = source_steps + args.adaptation_frames
    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "held-out enemy-spawn rearrangement; upstream zero-shot then matched full-network APPO adaptation",
        "environment": ENV_NAME,
        "source_checkpoint": str(args.source_checkpoint),
        "source_sha256": hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest(),
        "source_env_steps": source_steps,
        "requested_adaptation_frames": args.adaptation_frames,
        "target_env_steps": target_steps,
        "learning_rate": args.learning_rate,
        "training_seeds": seeds,
        "evaluation_seed_range": [args.eval_seed_start, args.eval_seed_start + args.episodes - 1],
        "variant_manifest": variant,
    }
    (args.results_dir / "screen_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    baseline_root = args.results_dir / "upstream_zero_shot"
    run([
        sys.executable, "sf_evaluate_corridor_variant.py",
        "--experiment=deadly-corridor-upstream", "--train-dir=artifacts", "--checkpoint=best",
        f"--episodes={args.episodes}", f"--seed-start={args.eval_seed_start}", "--device=gpu",
        f"--results-root={baseline_root}",
    ], args.gpu)
    baseline, baseline_path = latest_report(baseline_root)

    candidates = []
    for seed in seeds:
        experiment = f"upstream_spawn_holdout_fullft_{args.adaptation_frames}_seed_{seed}"
        destination = args.train_dir / experiment
        prepare_finetune(
            args.source_experiment, destination, args.train_dir, args.learning_rate, target_steps,
            config_overrides={"env": ENV_NAME, "seed": seed, "num_policies": 1},
            source_checkpoint=args.source_checkpoint,
        )
        run([
            sys.executable, "sf_train_corridor_variant.py", "--method=gru",
            f"--experiment={experiment}", f"--train_dir={args.train_dir}", f"--seed={seed}",
            "--num_policies=1", f"--train_for_env_steps={target_steps}", f"--learning_rate={args.learning_rate}",
        ], args.gpu)
        eval_root = args.results_dir / f"seed_{seed}"
        run([
            sys.executable, "sf_evaluate_corridor_variant.py",
            f"--experiment={experiment}", f"--train-dir={args.train_dir}", "--checkpoint=latest",
            f"--episodes={args.episodes}", f"--seed-start={args.eval_seed_start}", "--device=gpu",
            f"--results-root={eval_root}",
        ], args.gpu)
        report, report_path = latest_report(eval_root)
        realized_steps = int(torch.load(Path(report["checkpoint"]), map_location="cpu", weights_only=False)["env_steps"])
        candidates.append({
            "seed": seed,
            "report_path": str(report_path),
            "realized_adaptation_frames": realized_steps - source_steps,
            "report": report,
        })

    summary = {
        "manifest": manifest,
        "baseline_report_path": str(baseline_path),
        "baseline": baseline,
        "candidates": candidates,
        "completion_rates": {
            "upstream_zero_shot": baseline["metrics"]["completion_rate"],
            "adapted_by_seed": {str(item["seed"]): item["report"]["metrics"]["completion_rate"] for item in candidates},
        },
    }
    rates = list(summary["completion_rates"]["adapted_by_seed"].values())
    summary["completion_rates"]["adapted_mean"] = sum(rates) / len(rates)
    summary["completion_rates"]["adapted_worst"] = min(rates)
    (args.results_dir / "spawn_fewshot_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["completion_rates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
