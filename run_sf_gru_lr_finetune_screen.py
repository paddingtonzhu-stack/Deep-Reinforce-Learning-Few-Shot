"""Evaluate low-LR GRU continuations against exact parents and upstream."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_sf_gru_swa_screen import completion_percent, evaluate, latest_episodes, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetune-train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_lr_finetune_12m"),
    )
    parser.add_argument(
        "--source-train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_recurrence_10m_confirm"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/sample_factory_gru_lr_finetune_12m_screen"),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=90000)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]
    if len(seeds) < 3:
        raise ValueError("the controlled screen requires at least three training seeds")
    args.results_root.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        for seed in seeds:
            evaluate(
                sys.executable,
                f"sf_corridor_gru_lrft_seed_{seed}",
                args.finetune_train_dir,
                "latest",
                args.episodes,
                args.seed_start,
                args.device,
                args.results_root / f"finetune_seed{seed}",
            )
            evaluate(
                sys.executable,
                f"sf_corridor_gru_r64_seed_{seed}",
                args.source_train_dir,
                "latest",
                args.episodes,
                args.seed_start,
                args.device,
                args.results_root / f"parent_seed{seed}",
            )
        evaluate(
            sys.executable,
            "deadly-corridor-upstream",
            Path("artifacts"),
            "best",
            args.episodes,
            args.seed_start,
            args.device,
            args.results_root / "upstream",
        )

    groups: list[tuple[str, Path]] = []
    for seed in seeds:
        groups.append(
            (f"finetune_seed{seed}", latest_episodes(args.results_root / f"finetune_seed{seed}"))
        )
        groups.append((f"parent_seed{seed}", latest_episodes(args.results_root / f"parent_seed{seed}")))
    groups.append(("upstream", latest_episodes(args.results_root / "upstream")))
    paired_command = [sys.executable, "compare_corridor_failures.py"]
    for name, path in groups:
        paired_command.extend(["--group", f"{name}={path}"])
    paired_path = args.results_root / "paired_comparison.txt"
    print("+", " ".join(paired_command), flush=True)
    with paired_path.open("w", encoding="utf-8") as output:
        subprocess.run(paired_command, check=True, stdout=output, text=True)

    finetune = [
        completion_percent(latest_episodes(args.results_root / f"finetune_seed{s}")) for s in seeds
    ]
    parent = [
        completion_percent(latest_episodes(args.results_root / f"parent_seed{s}")) for s in seeds
    ]
    upstream = completion_percent(latest_episodes(args.results_root / "upstream"))
    wins = sum(candidate > baseline for candidate, baseline in zip(finetune, parent, strict=True))
    report = {
        "protocol": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "training_seeds": seeds,
            "candidate_checkpoint": "latest after 12M total steps and 2M at learning rate 3e-5",
            "baseline_checkpoint": "exact parent latest checkpoint at 10M",
        },
        "finetune_completion": finetune,
        "parent_completion": parent,
        "upstream_completion": upstream,
        "finetune_mean": sum(finetune) / len(finetune),
        "parent_mean": sum(parent) / len(parent),
        "finetune_worst": min(finetune),
        "parent_worst": min(parent),
        "paired_seed_wins": wins,
    }
    report["advance"] = bool(
        wins >= 2
        and report["finetune_mean"] > report["parent_mean"]
        and report["finetune_worst"] >= report["parent_worst"]
        and report["finetune_mean"] >= upstream - 5.0
    )
    (args.results_root / "finetune_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    run(
        [
            sys.executable,
            "sf_research_harness.py",
            "catalog",
            "--artifacts-root=artifacts",
            "--results-root=results",
            "--output=results/research_catalog.json",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
