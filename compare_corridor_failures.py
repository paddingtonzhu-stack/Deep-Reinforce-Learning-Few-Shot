"""Compare completion/death seeds across corridor evaluation CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_group(value: str):
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from error
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(path)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        required=True,
        metavar="LABEL=EPISODES_CSV",
    )
    args = parser.parse_args()

    outcomes = {}
    for label, path in args.group:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        seeds = {int(row["seed"]) for row in rows}
        completed = {int(row["seed"]) for row in rows if truthy(row["completed"])}
        deaths = {int(row["seed"]) for row in rows if truthy(row["player_dead"])}
        outcomes[label] = deaths
        print(
            f"{label}: episodes={len(rows)}, completion={len(completed)}, "
            f"death={len(deaths)}, unique_seeds={len(seeds)}"
        )

    labels = list(outcomes)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            intersection = outcomes[left] & outcomes[right]
            union = outcomes[left] | outcomes[right]
            jaccard = len(intersection) / len(union) if union else 1.0
            print(
                f"{left} vs {right}: shared_deaths={len(intersection)}, "
                f"death_union={len(union)}, jaccard={jaccard:.3f}"
            )
            if intersection:
                print("  shared death seeds:", " ".join(map(str, sorted(intersection))))


if __name__ == "__main__":
    main()
