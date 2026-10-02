"""Fixed-seed evaluation for Sample Factory Deadly Corridor checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

import gymnasium as gym
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
from sf_examples.vizdoom.train_vizdoom import parse_vizdoom_cfg, register_vizdoom_components

from sf_transformer_core import GRUAttentionMemoryCore, register_transformer_core


EVAL_ENV = "doom_deadly_corridor_outcomes"


class OutcomeInfoWrapper(gym.Wrapper):
    """Capture Doom's death flag before Sample Factory automatically resets."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            dead = bool(self.env.unwrapped.game.is_player_dead())
            info["player_dead"] = dead
            info["timed_out"] = bool(truncated)
            info["completed"] = bool(terminated and not dead)
        return obs, reward, terminated, truncated, info


def register_components():
    register_vizdoom_components()
    register_transformer_core()
    spec = next(spec for spec in DOOM_ENVS if spec.name == "doom_deadly_corridor")

    def make_eval_env(_env_name, cfg, env_config, render_mode=None, **kwargs):
        env = make_doom_env_from_spec(spec, _env_name, cfg, env_config, render_mode, **kwargs)
        return OutcomeInfoWrapper(env)

    register_env(EVAL_ENV, make_eval_env)


def make_logger(output_dir: Path):
    logger = logging.getLogger("sf_corridor_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def checkpoint_path(cfg, policy_id: int, kind: str) -> Path:
    prefix = "best" if kind == "best" else "checkpoint"
    paths = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{prefix}_*")
    paths = [path for path in paths if Path(path).suffix == ".pth"]
    if not paths:
        raise FileNotFoundError(f"No {kind} checkpoint for policy {policy_id} in {Learner.checkpoint_dir(cfg, policy_id)}")
    return Path(paths[-1])


def build_env(cfg, seed: int):
    cfg.env = EVAL_ENV
    cfg.num_envs = 1
    cfg.env_frameskip = 4
    cfg.eval_env_frameskip = 4
    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    obs, _ = env.reset(seed=seed)
    return env, obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="deadly-corridor-upstream")
    parser.add_argument("--train-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--policy-index", type=int, default=0)
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--results-root", type=Path, default=Path("results/sample_factory_corridor"))
    parser.add_argument("--disable-gru-attention", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.results_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = make_logger(output_dir)

    register_components()
    sf_argv = [
        "--algo=APPO",
        "--env=doom_deadly_corridor",
        f"--experiment={args.experiment}",
        f"--train_dir={args.train_dir}",
        f"--device={args.device}",
        f"--policy_index={args.policy_index}",
        f"--load_checkpoint_kind={args.checkpoint}",
        "--no_render",
        "--eval_deterministic=True",
    ]
    cfg = load_from_checkpoint(parse_vizdoom_cfg(argv=sf_argv, evaluation=True))
    device = torch.device("cpu" if args.device == "cpu" else "cuda")

    # Build one environment to obtain the exact spaces used by the saved policy.
    probe_env, _ = build_env(cfg, args.seed_start)
    env_info = extract_env_info(probe_env, cfg)
    actor_critic = create_actor_critic(cfg, probe_env.observation_space, probe_env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    model_path = checkpoint_path(cfg, args.policy_index, args.checkpoint)
    logger.info("Loading %s policy %d from %s", args.checkpoint, args.policy_index, model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    actor_critic.load_state_dict(checkpoint["model"])
    if args.disable_gru_attention:
        cores = [
            module
            for module in actor_critic.modules()
            if isinstance(module, GRUAttentionMemoryCore)
        ]
        if not cores:
            raise ValueError(
                "--disable-gru-attention requires a GRU-attention checkpoint"
            )
        with torch.no_grad():
            for core in cores:
                core.attention_gate.fill_(-100.0)
        logger.info("Disabled GRU-attention correction in %d temporal core(s)", len(cores))
    probe_env.close()

    rows = []
    for episode_index in range(args.episodes):
        test_seed = args.seed_start + episode_index
        # A fresh process-level Doom instance is required for each seed: changing
        # VizDoom's seed after initialization does not affect the next episode.
        env, obs = build_env(cfg, test_seed)
        rnn_states = torch.zeros((env.num_agents, get_rnn_size(cfg)), dtype=torch.float32, device=device)
        reward_sum = 0.0
        decisions = 0
        terminal_info = {}
        with torch.no_grad():
            while True:
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                outputs = actor_critic(normalized_obs, rnn_states)
                actions = argmax_actions(actor_critic.action_distribution())
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)
                obs, rewards, terminated, truncated, infos = env.step(actions)
                rnn_states = outputs["new_rnn_states"]
                reward_sum += float(rewards[0].item())
                decisions += 1
                if bool(make_dones(terminated, truncated)[0].item()):
                    terminal_info = infos[0]
                    break
        env.close()
        row = {
            "episode": episode_index,
            "seed": test_seed,
            "reward": reward_sum,
            "decisions": decisions,
            "game_frames": decisions * int(cfg.env_frameskip),
            "completed": bool(terminal_info.get("completed", False)),
            "player_dead": bool(terminal_info.get("player_dead", False)),
            "timed_out": bool(terminal_info.get("timed_out", False)),
        }
        rows.append(row)
        logger.info(
            "Episode %d/%d seed=%d reward=%.3f frames=%d outcome=%s",
            episode_index + 1,
            args.episodes,
            test_seed,
            reward_sum,
            row["game_frames"],
            "completed" if row["completed"] else "death" if row["player_dead"] else "timeout",
        )

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment": args.experiment,
        "checkpoint": str(model_path),
        "checkpoint_kind": args.checkpoint,
        "policy_index": args.policy_index,
        "device": args.device,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "architecture": {
            "memory": getattr(cfg, "memory", "gru" if cfg.use_rnn else "cnn"),
            "use_rnn": bool(cfg.use_rnn),
            "recurrence": int(cfg.recurrence),
            "rnn_type": getattr(cfg, "rnn_type", None),
            "rnn_size": int(getattr(cfg, "rnn_size", 0)),
            "resolution": [int(cfg.res_w), int(cfg.res_h)],
            "frameskip": int(cfg.env_frameskip),
            "transformer_context": getattr(cfg, "transformer_context", None),
            "transformer_dim": getattr(cfg, "transformer_dim", None),
            "transformer_layers": getattr(cfg, "transformer_layers", None),
            "transformer_heads": getattr(cfg, "transformer_heads", None),
            "gtrxl_identity_bias": getattr(cfg, "gtrxl_identity_bias", None),
            "gru_attention_hidden_size": getattr(cfg, "gru_attention_hidden_size", None),
            "gru_attention_dim": getattr(cfg, "gru_attention_dim", None),
            "gru_attention_gate_init": getattr(cfg, "gru_attention_gate_init", None),
        },
        "metrics": {
            "mean_reward": statistics.fmean(row["reward"] for row in rows),
            "median_reward": statistics.median(row["reward"] for row in rows),
            "completion_count": sum(row["completed"] for row in rows),
            "completion_rate": statistics.fmean(row["completed"] for row in rows),
            "death_count": sum(row["player_dead"] for row in rows),
            "death_rate": statistics.fmean(row["player_dead"] for row in rows),
            "timeout_count": sum(row["timed_out"] for row in rows),
            "timeout_rate": statistics.fmean(row["timed_out"] for row in rows),
            "mean_game_frames": statistics.fmean(row["game_frames"] for row in rows),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    logger.info("Saved versioned results to %s", output_dir)
    logger.info("Completion %.1f%% | death %.1f%% | timeout %.1f%% | mean reward %.3f",
                100 * report["metrics"]["completion_rate"],
                100 * report["metrics"]["death_rate"],
                100 * report["metrics"]["timeout_rate"],
                report["metrics"]["mean_reward"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
