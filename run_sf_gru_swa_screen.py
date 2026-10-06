"""Build and evaluate late-checkpoint averaged GRU-64 policies.

The screen is deliberately paired by training seed: each averaged policy is
compared with the newest checkpoint from the exact same completed GRU-64 run.
No training seed or checkpoint is selected using evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from average_sf_checkpoints import average_checkpoints


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def latest_episodes(root: Path) -> Path:
    matches = sorted(root.glob("*/episodes.csv"))
    if not matches:
        raise FileNotFoundError(f"no episodes.csv below {root}")
    return matches[-1]


def completion_percent(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no episodes")
    if "completed" in rows[0]:
        completed = sum(row["completed"].strip().lower() in {"true", "1"} for row in rows)
    elif "outcome" in rows[0]:
        completed = sum(row["outcome"] == "completed" for row in rows)
    else:
        raise ValueError(f"{path} has neither a completed nor outcome column")
    return 100.0 * completed / len(rows)


def evaluate(
    python: str,
    experiment: str,
    train_dir: Path,
    checkpoint: str,
    episodes: int,
    seed_start: int,
    device: str,
    results_root: Path,
) -> None:
    run(
        [
            python,
            "sf_evaluate_corridor.py",
            f"--experiment={experiment}",
            f"--train-dir={train_dir}",
            "--policy-index=0",
            f"--checkpoint={checkpoint}",
            f"--episodes={episodes}",
            f"--seed-start={seed_start}",
            f"--device={device}",
            f"--results-root={results_root}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_recurrence_10m_confirm"),
    )
    parser.add_argument(
        "--output-train-dir", type=Path, default=Path("artifacts/sample_factory_gru_swa_10m")
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/sample_factory_gru_swa_10m_screen")
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=80000)
    parser.add_argument("--device", default="gpu")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Reuse seven completed evaluations and only regenerate paired/report/catalog outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]
    if len(seeds) < 3:
        raise ValueError("the controlled screen requires at least three training seeds")
    args.results_root.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        args.output_train_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            source_name = f"sf_corridor_gru_r64_seed_{seed}"
            output_name = f"sf_corridor_gru_swa_seed_{seed}"
            source_dir = args.source_train_dir / source_name
            output_dir = args.output_train_dir / output_name
            regular = sorted((source_dir / "checkpoint_p0").glob("checkpoint_*.pth"))
            if len(regular) != 2:
                raise ValueError(
                    f"expected exactly two late checkpoints for {source_name}, found {len(regular)}"
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
            config["experiment"] = output_name
            config["train_dir"] = str(args.output_train_dir)
            (output_dir / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            output_checkpoint = output_dir / "checkpoint_p0" / regular[-1].name
            average_checkpoints(regular, output_checkpoint)

            evaluate(
                sys.executable,
                output_name,
                args.output_train_dir,
                "latest",
                args.episodes,
                args.seed_start,
                args.device,
                args.results_root / f"swa_seed{seed}",
            )
            evaluate(
                sys.executable,
                source_name,
                args.source_train_dir,
                "latest",
                args.episodes,
                args.seed_start,
                args.device,
                args.results_root / f"latest_seed{seed}",
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
        groups.append((f"swa_seed{seed}", latest_episodes(args.results_root / f"swa_seed{seed}")))
        groups.append(
            (f"latest_seed{seed}", latest_episodes(args.results_root / f"latest_seed{seed}"))
        )
    groups.append(("upstream", latest_episodes(args.results_root / "upstream")))
    paired_command = [sys.executable, "compare_corridor_failures.py"]
    for name, path in groups:
        paired_command.extend(["--group", f"{name}={path}"])
    paired_path = args.results_root / "paired_comparison.txt"
    print("+", " ".join(paired_command), flush=True)
    with paired_path.open("w", encoding="utf-8") as output:
        subprocess.run(paired_command, check=True, stdout=output, text=True)

    swa = [completion_percent(latest_episodes(args.results_root / f"swa_seed{s}")) for s in seeds]
    latest = [
        completion_percent(latest_episodes(args.results_root / f"latest_seed{s}")) for s in seeds
    ]
    upstream = completion_percent(latest_episodes(args.results_root / "upstream"))
    wins = sum(candidate > baseline for candidate, baseline in zip(swa, latest, strict=True))
    report = {
        "protocol": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "training_seeds": seeds,
            "checkpoint_rule": "mean of exactly two regular late checkpoints per seed",
        },
        "swa_completion": swa,
        "latest_completion": latest,
        "upstream_completion": upstream,
        "swa_mean": sum(swa) / len(swa),
        "latest_mean": sum(latest) / len(latest),
        "swa_worst": min(swa),
        "latest_worst": min(latest),
        "paired_seed_wins": wins,
    }
    report["advance"] = bool(
        wins >= 2
        and report["swa_mean"] > report["latest_mean"]
        and report["swa_worst"] >= report["latest_worst"]
        and report["swa_mean"] >= upstream - 5.0
    )
    (args.results_root / "swa_report.json").write_text(
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
