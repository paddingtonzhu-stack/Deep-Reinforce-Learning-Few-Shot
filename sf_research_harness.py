"""Reproducible train/evaluate/report harness for GRU-method research.

The harness deliberately keeps orchestration separate from model code. It
launches the existing training/evaluation entry points and summarizes their
versioned JSON reports using matched seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


RECURRENCES = (32, 64)
SEEDS = (0, 1, 2)


def csv_ints(value):
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def experiment_name(recurrence, seed):
    return f"sf_corridor_gru_r{recurrence}_seed_{seed}"


def train(args):
    run(
        [
            sys.executable,
            "run_sf_gru_recurrence_study.py",
            f"--seeds={','.join(map(str, args.seeds))}",
            f"--recurrences={','.join(map(str, args.recurrences))}",
            f"--gpu={args.gpu}",
            f"--steps={args.steps}",
            f"--train-dir={args.train_dir}",
        ]
    )


def evaluate(args):
    for recurrence in args.recurrences:
        for seed in args.seeds:
            experiment = experiment_name(recurrence, seed)
            result_root = args.results_root / f"r{recurrence}_seed_{seed}_{args.checkpoint}"
            run(
                [
                    sys.executable,
                    "sf_evaluate_corridor.py",
                    f"--experiment={experiment}",
                    f"--train-dir={args.train_dir}",
                    "--policy-index=0",
                    f"--checkpoint={args.checkpoint}",
                    f"--episodes={args.episodes}",
                    f"--seed-start={args.seed_start}",
                    f"--device={args.device}",
                    f"--results-root={result_root}",
                ]
            )


def latest_reports(results_root):
    latest = {}
    for path in results_root.rglob("report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        experiment = report.get("experiment")
        if not experiment or "sf_corridor_gru_r" not in experiment:
            continue
        previous = latest.get(experiment)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            latest[experiment] = (path, report)
    return latest


def summarize(args):
    reports = latest_reports(args.results_root)
    missing = [
        experiment_name(recurrence, seed)
        for recurrence in args.recurrences
        for seed in args.seeds
        if experiment_name(recurrence, seed) not in reports
    ]
    if missing:
        raise SystemExit("Missing evaluation reports: " + ", ".join(missing))

    grouped = defaultdict(dict)
    rows = []
    for recurrence in args.recurrences:
        for seed in args.seeds:
            experiment = experiment_name(recurrence, seed)
            path, report = reports[experiment]
            metrics = report["metrics"]
            rate = float(metrics["completion_rate"])
            grouped[recurrence][seed] = rate
            row = {
                "recurrence": recurrence,
                "seed": seed,
                "completion_rate": rate,
                "death_rate": float(metrics["death_rate"]),
                "episodes": int(report["episodes"]),
                "report": str(path),
            }
            rows.append(row)
            print(
                f"GRU-{recurrence} seed {seed}: completion={100 * rate:.1f}% "
                f"death={100 * row['death_rate']:.1f}% episodes={row['episodes']}"
            )

    aggregates = {}
    for recurrence, values_by_seed in grouped.items():
        values = list(values_by_seed.values())
        aggregates[recurrence] = {
            "mean_completion_rate": statistics.fmean(values),
            "worst_seed_completion_rate": min(values),
            "best_seed_completion_rate": max(values),
        }
        aggregate = aggregates[recurrence]
        print(
            f"GRU-{recurrence} aggregate: mean={100 * aggregate['mean_completion_rate']:.1f}% "
            f"worst={100 * aggregate['worst_seed_completion_rate']:.1f}% "
            f"best={100 * aggregate['best_seed_completion_rate']:.1f}%"
        )

    decision = None
    if 32 in grouped and 64 in grouped:
        common_seeds = sorted(set(grouped[32]) & set(grouped[64]))
        wins = sum(grouped[64][seed] > grouped[32][seed] for seed in common_seeds)
        ties = sum(grouped[64][seed] == grouped[32][seed] for seed in common_seeds)
        long_mean = aggregates[64]["mean_completion_rate"]
        base_mean = aggregates[32]["mean_completion_rate"]
        long_worst = aggregates[64]["worst_seed_completion_rate"]
        base_worst = aggregates[32]["worst_seed_completion_rate"]
        advance = wins >= 2 and long_mean > base_mean and long_worst >= base_worst
        decision = {
            "gru64_seed_wins": wins,
            "ties": ties,
            "mean_delta": long_mean - base_mean,
            "worst_seed_delta": long_worst - base_worst,
            "advance_gru64_to_full_budget": advance,
        }
        print(
            f"GRU-64 vs GRU-32: seed_wins={wins}/{len(common_seeds)}, ties={ties}, "
            f"mean_delta={100 * decision['mean_delta']:+.1f} points, "
            f"worst_delta={100 * decision['worst_seed_delta']:+.1f} points"
        )
        print("DECISION:", "ADVANCE GRU-64" if advance else "DO NOT ADVANCE GRU-64")

    output = {
        "protocol": {
            "recurrences": list(args.recurrences),
            "training_seeds": list(args.seeds),
            "matched_evaluation_required": True,
            "upstream_target_completion_rate": args.upstream_target,
        },
        "runs": rows,
        "aggregates": {str(key): value for key, value in aggregates.items()},
        "screening_decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved harness report to {args.output}")


def status(args):
    launcher = args.train_dir / "gru_recurrence_runs" / "launcher.log"
    if launcher.exists():
        lines = launcher.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-args.lines :]))
    else:
        print(f"No launcher log at {launcher}")
    for recurrence in args.recurrences:
        for seed in args.seeds:
            root = args.train_dir / experiment_name(recurrence, seed) / "checkpoint_p0"
            latest = sorted(root.glob("checkpoint_*.pth"))
            best = sorted(root.glob("best_*.pth"))
            print(
                f"GRU-{recurrence} seed {seed}: "
                f"latest={latest[-1].name if latest else '-'} "
                f"best={best[-1].name if best else '-'}"
            )


def relative_or_absolute(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def catalog(args):
    """Index all immutable experiment evidence without copying large checkpoints."""
    workspace = Path.cwd()
    evaluations = []
    for report_path in args.results_root.rglob("report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable report {report_path}: {error}")
            continue
        metrics = report.get("metrics", {})
        episodes_path = report_path.with_name("episodes.csv")
        config_path = report_path.with_name("config.json")
        evaluations.append(
            {
                "experiment": report.get("experiment"),
                "policy_index": report.get("policy_index"),
                "checkpoint_kind": report.get("checkpoint_kind"),
                "checkpoint": report.get("checkpoint"),
                "episodes": report.get("episodes"),
                "seed_start": report.get("seed_start"),
                "completion_rate": metrics.get("completion_rate"),
                "death_rate": metrics.get("death_rate"),
                "timeout_rate": metrics.get("timeout_rate"),
                "mean_reward": metrics.get("mean_reward"),
                "architecture": report.get("architecture"),
                "report_path": relative_or_absolute(report_path, workspace),
                "episodes_path": relative_or_absolute(episodes_path, workspace)
                if episodes_path.exists()
                else None,
                "config_path": relative_or_absolute(config_path, workspace)
                if config_path.exists()
                else None,
                "modified_ns": report_path.stat().st_mtime_ns,
            }
        )

    training_configs = []
    for config_path in args.artifacts_root.rglob("config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        training_configs.append(
            {
                "experiment": config.get("experiment"),
                "memory": config.get("memory"),
                "seed": config.get("seed"),
                "recurrence": config.get("recurrence"),
                "train_for_env_steps": config.get("train_for_env_steps"),
                "num_policies": config.get("num_policies"),
                "path": relative_or_absolute(config_path, workspace),
            }
        )

    checkpoints = []
    for path in args.artifacts_root.rglob("*.pth"):
        stat = path.stat()
        checkpoints.append(
            {
                "path": relative_or_absolute(path, workspace),
                "name": path.name,
                "policy_dir": path.parent.name,
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        )

    logs = []
    for root in (args.artifacts_root, args.results_root):
        for pattern in ("*.log",):
            for path in root.rglob(pattern):
                stat = path.stat()
                logs.append(
                    {
                        "path": relative_or_absolute(path, workspace),
                        "size_bytes": stat.st_size,
                        "modified_ns": stat.st_mtime_ns,
                    }
                )

    payload = {
        "schema_version": 1,
        "workspace": str(workspace.resolve()),
        "artifacts_root": relative_or_absolute(args.artifacts_root, workspace),
        "results_root": relative_or_absolute(args.results_root, workspace),
        "counts": {
            "evaluations": len(evaluations),
            "training_configs": len(training_configs),
            "checkpoints": len(checkpoints),
            "logs": len(logs),
        },
        "evaluations": sorted(
            evaluations,
            key=lambda row: (str(row["experiment"]), int(row["modified_ns"])),
        ),
        "training_configs": sorted(training_configs, key=lambda row: str(row["path"])),
        "checkpoints": sorted(checkpoints, key=lambda row: str(row["path"])),
        "logs": sorted(logs, key=lambda row: str(row["path"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    fields = [
        "experiment",
        "policy_index",
        "checkpoint_kind",
        "episodes",
        "seed_start",
        "completion_rate",
        "death_rate",
        "timeout_rate",
        "mean_reward",
        "report_path",
        "episodes_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["evaluations"])
    print(
        f"Cataloged {len(evaluations)} evaluations, {len(training_configs)} training configs, "
        f"{len(checkpoints)} checkpoints, and {len(logs)} logs"
    )
    print(f"Saved {args.output} and {csv_path}")


def add_shared(parser):
    parser.add_argument("--seeds", type=csv_ints, default=SEEDS)
    parser.add_argument("--recurrences", type=csv_ints, default=RECURRENCES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="launch controlled GRU recurrence training")
    add_shared(train_parser)
    train_parser.add_argument("--gpu", type=int, default=0)
    train_parser.add_argument("--steps", type=int, default=2_000_000)
    train_parser.add_argument("--train-dir", type=Path, required=True)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("evaluate", help="evaluate every trained recurrence/seed")
    add_shared(eval_parser)
    eval_parser.add_argument("--train-dir", type=Path, required=True)
    eval_parser.add_argument("--results-root", type=Path, required=True)
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--seed-start", type=int, default=30_000)
    eval_parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    eval_parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    eval_parser.set_defaults(func=evaluate)

    summary_parser = subparsers.add_parser("summarize", help="report reliability and screening decision")
    add_shared(summary_parser)
    summary_parser.add_argument("--results-root", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.add_argument("--upstream-target", type=float, default=0.844)
    summary_parser.set_defaults(func=summarize)

    status_parser = subparsers.add_parser("status", help="show launcher tail and checkpoints")
    add_shared(status_parser)
    status_parser.add_argument("--train-dir", type=Path, required=True)
    status_parser.add_argument("--lines", type=int, default=20)
    status_parser.set_defaults(func=status)

    catalog_parser = subparsers.add_parser("catalog", help="index all experiment evidence")
    catalog_parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    catalog_parser.add_argument("--results-root", type=Path, default=Path("results"))
    catalog_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/research_catalog.json"),
    )
    catalog_parser.set_defaults(func=catalog)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
