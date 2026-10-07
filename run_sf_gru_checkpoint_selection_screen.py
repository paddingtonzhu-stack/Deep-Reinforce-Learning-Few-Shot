"""Select GRU-64 checkpoints on development seeds, then test them unseen.

Sample Factory's ``best`` checkpoint is selected by average episode reward,
while this project's primary outcome is corridor completion.  This experiment
changes only the checkpoint-selection rule.  For each training seed it selects
among all preserved checkpoints on one fixed development range, then compares
the selected checkpoint with the original reward-best checkpoint on a disjoint
holdout range.  Holdout results never influence selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from run_sf_gru_swa_screen import completion_percent, latest_episodes, run


def checkpoint_candidates(train_dir: Path, experiment: str) -> list[Path]:
    root = train_dir / experiment / "checkpoint_p0"
    candidates = sorted({*root.glob("best_*.pth"), *root.glob("checkpoint_*.pth")})
    if not candidates:
        raise FileNotFoundError(f"no checkpoints in {root}")
    return candidates


def checkpoint_id(path: Path) -> str:
    return path.stem.replace(".", "_")


def validate_checkpoint_inventory(paths: list[Path], experiment: str) -> None:
    best = [path for path in paths if path.name.startswith("best_")]
    regular = [path for path in paths if path.name.startswith("checkpoint_")]
    if len(best) != 1 or len(regular) != 2:
        raise ValueError(
            f"{experiment} must have exactly one reward-best and two late checkpoints; "
            f"found {len(best)} best and {len(regular)} late"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_report(root: Path) -> Path:
    matches = sorted(root.glob("*/report.json"))
    if not matches:
        raise FileNotFoundError(f"no report.json below {root}")
    return matches[-1]


def read_metrics(root: Path) -> dict:
    return json.loads(latest_report(root).read_text(encoding="utf-8"))["metrics"]


def evaluation_is_complete(root: Path, episodes: int, seed_start: int) -> bool:
    try:
        report = json.loads(latest_report(root).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    return report.get("episodes") == episodes and report.get("seed_start") == seed_start


def evaluate(
    experiment: str,
    train_dir: Path,
    checkpoint: str,
    episodes: int,
    seed_start: int,
    device: str,
    results_root: Path,
    checkpoint_path: Path | None = None,
) -> None:
    if evaluation_is_complete(results_root, episodes, seed_start):
        print(f"Reusing complete evaluation: {results_root}", flush=True)
        return
    command = [
        sys.executable,
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
    if checkpoint_path is not None:
        command.append(f"--checkpoint-path={checkpoint_path}")
    run(command)


def select_checkpoint(entries: list[dict]) -> dict:
    if not entries:
        raise ValueError("cannot select from an empty checkpoint list")
    # The filename tie-break makes selection deterministic without consulting
    # holdout outcomes. Higher completion and then higher mean reward win.
    return sorted(
        entries,
        key=lambda item: (
            -item["completion_rate"],
            -item["mean_reward"],
            Path(item["checkpoint"]).name,
        ),
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_recurrence_10m_confirm"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/sample_factory_gru_checkpoint_selection_screen"),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--selection-seed-start", type=int, default=150000)
    parser.add_argument("--holdout-seed-start", type=int, default=160000)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) < 3:
        raise ValueError("the controlled screen requires at least three training seeds")
    selection_range = set(range(args.selection_seed_start, args.selection_seed_start + args.episodes))
    holdout_range = set(range(args.holdout_seed_start, args.holdout_seed_start + args.episodes))
    if selection_range & holdout_range:
        raise ValueError("selection and holdout episode seeds must be disjoint")

    args.results_root.mkdir(parents=True, exist_ok=True)
    selections: dict[str, dict] = {}
    for seed in seeds:
        experiment = f"sf_corridor_gru_r64_seed_{seed}"
        entries = []
        candidates = checkpoint_candidates(args.train_dir, experiment)
        validate_checkpoint_inventory(candidates, experiment)
        for checkpoint_path in candidates:
            result_root = (
                args.results_root / "selection" / f"seed_{seed}" / checkpoint_id(checkpoint_path)
            )
            evaluate(
                experiment,
                args.train_dir,
                "best",
                args.episodes,
                args.selection_seed_start,
                args.device,
                result_root,
                checkpoint_path,
            )
            metrics = read_metrics(result_root)
            entries.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "sha256": file_sha256(checkpoint_path),
                    "completion_rate": 100.0 * metrics["completion_rate"],
                    "mean_reward": metrics["mean_reward"],
                    "evaluation": str(latest_report(result_root)),
                }
            )
        selections[str(seed)] = {
            "candidates": entries,
            "selected": select_checkpoint(entries),
        }

    manifest = {
        "selection_rule": "max completion, then mean reward, then lexicographically smallest filename",
        "selection_episodes": args.episodes,
        "selection_seed_start": args.selection_seed_start,
        "holdout_seed_start": args.holdout_seed_start,
        "training_seeds": seeds,
        "selections": selections,
    }
    manifest_path = args.results_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for seed in seeds:
        experiment = f"sf_corridor_gru_r64_seed_{seed}"
        selected = Path(selections[str(seed)]["selected"]["checkpoint"])
        evaluate(
            experiment,
            args.train_dir,
            "best",
            args.episodes,
            args.holdout_seed_start,
            args.device,
            args.results_root / "holdout" / f"selected_seed_{seed}",
            selected,
        )
        evaluate(
            experiment,
            args.train_dir,
            "best",
            args.episodes,
            args.holdout_seed_start,
            args.device,
            args.results_root / "holdout" / f"reward_best_seed_{seed}",
        )

    evaluate(
        "deadly-corridor-upstream",
        Path("artifacts"),
        "best",
        args.episodes,
        args.holdout_seed_start,
        args.device,
        args.results_root / "holdout" / "upstream",
    )

    groups: list[tuple[str, Path]] = []
    selected_completion = []
    baseline_completion = []
    for seed in seeds:
        selected_csv = latest_episodes(args.results_root / "holdout" / f"selected_seed_{seed}")
        baseline_csv = latest_episodes(args.results_root / "holdout" / f"reward_best_seed_{seed}")
        groups.extend(((f"selected_seed{seed}", selected_csv), (f"reward_best_seed{seed}", baseline_csv)))
        selected_completion.append(completion_percent(selected_csv))
        baseline_completion.append(completion_percent(baseline_csv))
    upstream_csv = latest_episodes(args.results_root / "holdout" / "upstream")
    groups.append(("upstream", upstream_csv))

    compare = [sys.executable, "compare_corridor_failures.py"]
    for name, path in groups:
        compare.extend(["--group", f"{name}={path}"])
    with (args.results_root / "paired_comparison.txt").open("w", encoding="utf-8") as output:
        subprocess.run(compare, check=True, stdout=output, text=True)

    upstream = completion_percent(upstream_csv)
    wins = sum(a > b for a, b in zip(selected_completion, baseline_completion, strict=True))
    report = {
        "protocol": {
            "candidate": "completion-selected checkpoint from each fixed 10M GRU-64 run",
            "baseline": "reward-best checkpoint from the same run",
            "selection_episodes": args.episodes,
            "selection_seed_start": args.selection_seed_start,
            "holdout_episodes": args.episodes,
            "holdout_seed_start": args.holdout_seed_start,
            "training_seeds": seeds,
        },
        "selected_completion": selected_completion,
        "reward_best_completion": baseline_completion,
        "upstream_completion": upstream,
        "selected_mean": sum(selected_completion) / len(selected_completion),
        "reward_best_mean": sum(baseline_completion) / len(baseline_completion),
        "selected_worst": min(selected_completion),
        "reward_best_worst": min(baseline_completion),
        "paired_seed_wins": wins,
        "selection_manifest": str(manifest_path),
    }
    report["advance"] = bool(
        wins >= 2
        and report["selected_mean"] > report["reward_best_mean"]
        and report["selected_worst"] >= report["reward_best_worst"]
        and report["selected_mean"] >= upstream - 5.0
    )
    report_path = args.results_root / "checkpoint_selection_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
