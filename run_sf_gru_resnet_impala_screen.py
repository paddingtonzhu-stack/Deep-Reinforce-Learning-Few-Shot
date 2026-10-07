"""Evaluate ResNet-IMPALA GRU-64 seeds against matched visual baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_sf_gru_swa_screen import completion_percent, evaluate, latest_episodes, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-train-dir", type=Path,
                        default=Path("artifacts/sample_factory_gru_resnet_impala_2m"))
    parser.add_argument("--baseline-train-dir", type=Path,
                        default=Path("artifacts/sample_factory_gru_recurrence_2m"))
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/sample_factory_gru_resnet_impala_2m_screen"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=180000)
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
            evaluate(sys.executable, f"sf_corridor_gru_resnet_impala_seed_{seed}",
                     args.candidate_train_dir, "latest", args.episodes, args.seed_start,
                     args.device, args.results_root / f"candidate_seed{seed}")
            evaluate(sys.executable, f"sf_corridor_gru_r64_seed_{seed}",
                     args.baseline_train_dir, "latest", args.episodes, args.seed_start,
                     args.device, args.results_root / f"baseline_seed{seed}")
        evaluate(sys.executable, "deadly-corridor-upstream", Path("artifacts"), "best",
                 args.episodes, args.seed_start, args.device, args.results_root / "upstream")

    groups = []
    for seed in seeds:
        groups.extend(((f"candidate_seed{seed}", latest_episodes(args.results_root / f"candidate_seed{seed}")),
                       (f"baseline_seed{seed}", latest_episodes(args.results_root / f"baseline_seed{seed}"))))
    groups.append(("upstream", latest_episodes(args.results_root / "upstream")))
    command = [sys.executable, "compare_corridor_failures.py"]
    for name, path in groups:
        command.extend(["--group", f"{name}={path}"])
    with (args.results_root / "paired_comparison.txt").open("w", encoding="utf-8") as output:
        subprocess.run(command, check=True, stdout=output, text=True)

    candidate = [completion_percent(latest_episodes(args.results_root / f"candidate_seed{s}")) for s in seeds]
    baseline = [completion_percent(latest_episodes(args.results_root / f"baseline_seed{s}")) for s in seeds]
    upstream = completion_percent(latest_episodes(args.results_root / "upstream"))
    wins = sum(new > old for new, old in zip(candidate, baseline, strict=True))
    report = {
        "protocol": {"episodes": args.episodes, "seed_start": args.seed_start,
                     "training_seeds": seeds,
                     "candidate": "2M genuine ResNet-IMPALA encoder plus standard GRU-64",
                     "baseline": "2M convnet_simple plus standard GRU-64"},
        "candidate_completion": candidate, "baseline_completion": baseline,
        "upstream_completion": upstream,
        "candidate_mean": sum(candidate) / len(candidate),
        "baseline_mean": sum(baseline) / len(baseline),
        "candidate_worst": min(candidate), "baseline_worst": min(baseline),
        "paired_seed_wins": wins,
    }
    report["advance"] = bool(wins >= 2 and report["candidate_mean"] > report["baseline_mean"]
                             and report["candidate_worst"] >= report["baseline_worst"]
                             and report["candidate_mean"] >= upstream - 5.0)
    (args.results_root / "resnet_impala_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    run([sys.executable, "sf_research_harness.py", "catalog", "--artifacts-root=artifacts",
         "--results-root=results", "--output=results/research_catalog.json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
