"""Evaluate GRU skill-curriculum seeds against matched 2M parents and upstream."""

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
        "--curriculum-train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_skill_curriculum_2m"),
    )
    parser.add_argument(
        "--baseline-train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_recurrence_2m"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/sample_factory_gru_skill_curriculum_2m_screen"),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100000)
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
                f"sf_corridor_gru_curriculum_seed_{seed}",
                args.curriculum_train_dir,
                "latest",
                args.episodes,
                args.seed_start,
                args.device,
                args.results_root / f"curriculum_seed{seed}",
            )
            evaluate(
                sys.executable,
                f"sf_corridor_gru_r64_seed_{seed}",
                args.baseline_train_dir,
                "latest",
                args.episodes,
                args.seed_start,
                args.device,
                args.results_root / f"baseline_seed{seed}",
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
            (f"curriculum_seed{seed}", latest_episodes(args.results_root / f"curriculum_seed{seed}"))
        )
        groups.append(
            (f"baseline_seed{seed}", latest_episodes(args.results_root / f"baseline_seed{seed}"))
        )
    groups.append(("upstream", latest_episodes(args.results_root / "upstream")))
    paired_command = [sys.executable, "compare_corridor_failures.py"]
    for name, path in groups:
        paired_command.extend(["--group", f"{name}={path}"])
    with (args.results_root / "paired_comparison.txt").open("w", encoding="utf-8") as output:
        subprocess.run(paired_command, check=True, stdout=output, text=True)

    curriculum = [
        completion_percent(latest_episodes(args.results_root / f"curriculum_seed{s}"))
        for s in seeds
    ]
    baseline = [
        completion_percent(latest_episodes(args.results_root / f"baseline_seed{s}")) for s in seeds
    ]
    upstream = completion_percent(latest_episodes(args.results_root / "upstream"))
    wins = sum(candidate > parent for candidate, parent in zip(curriculum, baseline, strict=True))
    report = {
        "protocol": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "training_seeds": seeds,
            "candidate": "1M Doom skill 1 then 1M standard skill 5",
            "baseline": "2M standard skill 5 GRU-64",
        },
        "curriculum_completion": curriculum,
        "baseline_completion": baseline,
        "upstream_completion": upstream,
        "curriculum_mean": sum(curriculum) / len(curriculum),
        "baseline_mean": sum(baseline) / len(baseline),
        "curriculum_worst": min(curriculum),
        "baseline_worst": min(baseline),
        "paired_seed_wins": wins,
    }
    report["advance"] = bool(
        wins >= 2
        and report["curriculum_mean"] > report["baseline_mean"]
        and report["curriculum_worst"] >= report["baseline_worst"]
        and report["curriculum_mean"] >= upstream - 5.0
    )
    (args.results_root / "curriculum_report.json").write_text(
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
