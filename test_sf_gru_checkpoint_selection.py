import json
from pathlib import Path

from run_sf_gru_checkpoint_selection_screen import (
    checkpoint_candidates,
    evaluation_is_complete,
    select_checkpoint,
    validate_checkpoint_inventory,
)


def test_checkpoint_candidates_include_best_and_regular(tmp_path):
    root = tmp_path / "exp" / "checkpoint_p0"
    root.mkdir(parents=True)
    for name in ("best_1_10_reward_2.0.pth", "checkpoint_2_20.pth", "notes.txt"):
        (root / name).write_text("x", encoding="utf-8")
    assert [path.name for path in checkpoint_candidates(tmp_path, "exp")] == [
        "best_1_10_reward_2.0.pth",
        "checkpoint_2_20.pth",
    ]


def test_selection_uses_completion_then_reward_then_filename():
    entries = [
        {"checkpoint": "z.pth", "completion_rate": 80.0, "mean_reward": 19.0},
        {"checkpoint": "b.pth", "completion_rate": 82.0, "mean_reward": 18.0},
        {"checkpoint": "a.pth", "completion_rate": 82.0, "mean_reward": 18.0},
    ]
    assert select_checkpoint(entries)["checkpoint"] == "a.pth"


def test_inventory_requires_same_three_checkpoint_opportunities():
    valid = [
        Path("best_1_10_reward_2.0.pth"),
        Path("checkpoint_2_20.pth"),
        Path("checkpoint_3_30.pth"),
    ]
    validate_checkpoint_inventory(valid, "exp")

    try:
        validate_checkpoint_inventory(valid[:-1], "exp")
    except ValueError as error:
        assert "exactly one reward-best and two late" in str(error)
    else:
        raise AssertionError("an unequal checkpoint inventory must be rejected")


def test_completed_evaluation_must_match_protocol(tmp_path):
    stamp = tmp_path / "stamp"
    stamp.mkdir()
    (stamp / "report.json").write_text(
        json.dumps({"episodes": 100, "seed_start": 150000}), encoding="utf-8"
    )
    assert evaluation_is_complete(tmp_path, 100, 150000)
    assert not evaluation_is_complete(tmp_path, 500, 150000)
    assert not evaluation_is_complete(tmp_path, 100, 160000)
