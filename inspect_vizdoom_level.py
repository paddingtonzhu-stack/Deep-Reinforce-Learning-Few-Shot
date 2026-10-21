"""Open and inspect the ViZDoom levels used by this project.

Examples:
    python inspect_vizdoom_level.py --list
    python inspect_vizdoom_level.py --level deadly_corridor --mode spectator
    python inspect_vizdoom_level.py --level battle1 --mode random --episodes 2
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Level:
    config: str
    wad: str
    objective: str
    geometry: str
    project_role: str


LEVELS = {
    "deadly_corridor": Level(
        config="deadly_corridor.cfg",
        wad="deadly_corridor.wad",
        objective="Reach the armor at the far end of a defended corridor.",
        geometry="A narrow, mostly forward route with enemies blocking progression.",
        project_role="Source task used to train the Hugging Face CNN+GRU policy.",
    ),
    "battle1": Level(
        config="battle.cfg",
        wad="battle.wad",
        objective="Fight monsters in an arena; reward comes from combat and survival rather than reaching a corridor exit.",
        geometry="A more open combat arena with threats and movement in several directions.",
        project_role="Few-shot target level: update policy components here.",
    ),
    "battle2": Level(
        config="battle2_continuous_turning.cfg",
        wad="battle2.wad",
        objective="Fight and survive in a second arena with continuous turning controls.",
        geometry="A distinct Battle layout used as a held-out transfer level.",
        project_role="Zero-shot test level: evaluate without further updates.",
    ),
}


def scenario_directory() -> Path:
    # A source checkout is enough for human play; Sample Factory does not need
    # to be installed into the small Windows player environment.
    here = Path(__file__).resolve().parent
    source_candidates = (
        here / "sample-factory-2.1.1-src" / "sf_examples" / "vizdoom" / "doom" / "scenarios",
        here.parent / "sample-factory-2.1.1-src" / "sf_examples" / "vizdoom" / "doom" / "scenarios",
    )
    for candidate in source_candidates:
        if candidate.is_dir():
            return candidate
    try:
        import sf_examples.vizdoom.doom.doom_utils as doom_utils
    except ImportError as error:
        raise SystemExit(
            "Could not find the Sample Factory ViZDoom scenario files. Keep the "
            "sample-factory-2.1.1-src folder beside this project, or install Sample Factory."
        ) from error
    return Path(doom_utils.__file__).resolve().parent / "scenarios"


def portable_config(level: Level) -> tuple[Path, tempfile.TemporaryDirectory]:
    """Copy the config and make its WAD path absolute for any working directory."""
    scenario_dir = scenario_directory()
    source_path = scenario_dir / level.config
    wad_path = scenario_dir / level.wad
    if not source_path.is_file() or not wad_path.is_file():
        raise FileNotFoundError(f"missing scenario files: {source_path} or {wad_path}")
    source = source_path.read_text(encoding="utf-8")
    temporary = tempfile.TemporaryDirectory(prefix="vizdoom_level_")
    local_wad = Path(temporary.name) / level.wad
    shutil.copy2(wad_path, local_wad)
    source, replacements = re.subn(
        r"doom_scenario_path\s*=\s*\S+\.wad",
        f"doom_scenario_path = {level.wad}",
        source,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(f"could not resolve doom_scenario_path in {source_path}")
    output = Path(temporary.name) / level.config
    output.write_text(source, encoding="utf-8")
    return output, temporary


def print_catalog() -> None:
    print(json.dumps({name: asdict(level) for name, level in LEVELS.items()}, indent=2))


def run(args) -> int:
    try:
        import vizdoom as vzd
    except ImportError as error:
        raise SystemExit(
            "ViZDoom is not installed in this Python environment. Activate .sf first."
        ) from error

    level = LEVELS[args.level]
    config, temporary = portable_config(level)
    game = vzd.DoomGame()
    game.load_config(str(config))
    game.set_window_visible(not args.headless)
    game.set_sound_enabled(not args.mute)
    game.set_seed(args.seed)
    if args.mode == "spectator":
        game.set_mode(vzd.Mode.SPECTATOR)
    else:
        game.set_mode(vzd.Mode.PLAYER)
    game.init()
    try:
        buttons = [str(button).split(".")[-1] for button in game.get_available_buttons()]
        print(f"level={args.level}")
        print(f"objective={level.objective}")
        print(f"geometry={level.geometry}")
        print(f"available_buttons={buttons}")
        print(f"episode_timeout={game.get_episode_timeout()}")
        if args.mode == "spectator":
            print("Controls: click the Doom window, then use its normal keyboard/mouse controls. ESC exits.")

        rng = random.Random(args.seed)
        for episode in range(args.episodes):
            game.new_episode()
            steps = 0
            while not game.is_episode_finished():
                if args.mode == "spectator":
                    game.advance_action()
                else:
                    action = [0] * len(buttons)
                    if args.mode == "random":
                        action = [rng.randint(0, 1) for _ in buttons]
                    game.make_action(action, args.frame_skip)
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    break
                if args.delay > 0:
                    time.sleep(args.delay)
            print(
                f"episode={episode + 1} reward={game.get_total_reward():.3f} "
                f"steps={steps} finished={game.is_episode_finished()}"
            )
    finally:
        game.close()
        temporary.cleanup()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Describe all three levels and exit.")
    parser.add_argument("--level", choices=LEVELS, default="deadly_corridor")
    parser.add_argument("--mode", choices=("spectator", "random", "noop"), default="spectator")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0, help="Zero uses the scenario timeout.")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional seconds between actions.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mute", action="store_true")
    args = parser.parse_args()
    if args.list:
        print_catalog()
        return 0
    if args.episodes < 1 or args.frame_skip < 1 or args.max_steps < 0:
        parser.error("episodes/frame-skip must be positive and max-steps cannot be negative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
