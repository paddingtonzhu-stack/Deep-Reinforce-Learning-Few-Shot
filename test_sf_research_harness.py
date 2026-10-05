"""Tests for deterministic research-harness reporting and stage gates."""
import json
from types import SimpleNamespace

from sf_research_harness import experiment_name, latest_reports, summarize


def write_report(root, recurrence, seed, rate):
    path = root / f"r{recurrence}_seed_{seed}" / "stamp" / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "experiment": experiment_name(recurrence, seed),
                "episodes": 100,
                "metrics": {"completion_rate": rate, "death_rate": 1.0 - rate},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_latest_reports_indexes_experiments(tmp_path):
    path = write_report(tmp_path, 32, 0, 0.5)
    reports = latest_reports(tmp_path)
    assert reports[experiment_name(32, 0)][0] == path


def test_summary_advances_only_reliable_candidate(tmp_path):
    for seed, rate in enumerate((0.40, 0.50, 0.60)):
        write_report(tmp_path, 32, seed, rate)
    for seed, rate in enumerate((0.50, 0.60, 0.60)):
        write_report(tmp_path, 64, seed, rate)
    output = tmp_path / "harness_report.json"
    summarize(
        SimpleNamespace(
            results_root=tmp_path,
            recurrences=(32, 64),
            seeds=(0, 1, 2),
            upstream_target=0.844,
            output=output,
        )
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    decision = report["screening_decision"]
    assert decision["gru64_seed_wins"] == 2
    assert decision["advance_gru64_to_full_budget"] is True
