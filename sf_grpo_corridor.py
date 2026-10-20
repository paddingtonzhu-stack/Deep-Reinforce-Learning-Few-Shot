"""Grouped episodic policy optimization for ViZDoom Deadly Corridor.

This is a control-domain translation of GRPO, not the language-model trainer:
multiple stochastic trajectories start from the same Doom seed, complete
episode returns are standardized within that group, and a clipped policy
objective is optimized without a learned value advantage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import sample_actions_log_probs
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sf_examples.vizdoom.train_vizdoom import parse_vizdoom_cfg

from sf_evaluate_corridor import build_env, checkpoint_path, register_components


@dataclass
class Trajectory:
    observations: list[dict[str, torch.Tensor]]
    actions: list[torch.Tensor]
    old_log_probs: list[torch.Tensor]
    reference_log_probs: list[torch.Tensor]
    reward: float
    game_frames: int
    completed: bool


def group_advantages(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize rewards within one prompt/initial-state group."""
    if returns.ndim != 1 or returns.numel() < 2:
        raise ValueError("GRPO requires at least two returns in a one-dimensional group")
    centered = returns - returns.mean()
    scale = returns.std(unbiased=False)
    if float(scale) < eps:
        return torch.zeros_like(returns)
    return centered / (scale + eps)


def clipped_grpo_objective(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    reference_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_ratio: float,
    beta: float,
) -> torch.Tensor:
    """Per-action clipped GRPO objective with the DeepSeek-style KL estimator."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_term = torch.minimum(ratio * advantage, clipped * advantage)
    log_ref_over_policy = reference_log_prob - new_log_prob
    kl = torch.exp(log_ref_over_policy) - log_ref_over_policy - 1.0
    return policy_term - beta * kl


def _cpu_obs(normalized_obs) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in normalized_obs.items()}


def collect_trajectory(actor, reference, cfg, env_seed: int, device: torch.device) -> Trajectory:
    env, obs = build_env(cfg, env_seed)
    env_info = extract_env_info(env, cfg)
    rnn = torch.zeros((env.num_agents, get_rnn_size(cfg)), device=device)
    ref_rnn = torch.zeros_like(rnn)
    observations, actions, old_log_probs, ref_log_probs = [], [], [], []
    reward_sum, decisions, terminal_info = 0.0, 0, {}
    with torch.no_grad():
        while True:
            normalized = prepare_and_normalize_obs(actor, obs)
            out = actor(normalized, rnn)
            distribution = actor.action_distribution()
            action, old_log_prob = sample_actions_log_probs(distribution)

            ref_out = reference(normalized, ref_rnn)
            ref_log_prob = reference.action_distribution().log_prob(action)

            observations.append(_cpu_obs(normalized))
            actions.append(action.detach().cpu().clone())
            old_log_probs.append(old_log_prob.detach().cpu().clone())
            ref_log_probs.append(ref_log_prob.detach().cpu().clone())

            env_action = action
            if env_action.ndim == 1:
                env_action = unsqueeze_tensor(env_action, dim=-1)
            obs, rewards, terminated, truncated, infos = env.step(
                preprocess_actions(env_info, env_action)
            )
            rnn, ref_rnn = out["new_rnn_states"], ref_out["new_rnn_states"]
            reward_sum += float(rewards[0].item())
            decisions += 1
            if bool(make_dones(terminated, truncated)[0].item()):
                terminal_info = infos[0]
                break
    env.close()
    return Trajectory(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        reference_log_probs=ref_log_probs,
        reward=reward_sum,
        game_frames=decisions * int(cfg.env_frameskip),
        completed=bool(terminal_info.get("completed", False)),
    )


def replay_objective(actor, cfg, trajectory: Trajectory, advantage: torch.Tensor, args, device):
    rnn = torch.zeros((1, get_rnn_size(cfg)), device=device)
    terms = []
    for step, (obs, action, old_lp, ref_lp) in enumerate(
        zip(
            trajectory.observations,
            trajectory.actions,
            trajectory.old_log_probs,
            trajectory.reference_log_probs,
        )
    ):
        normalized = {key: value.to(device) for key, value in obs.items()}
        out = actor(normalized, rnn)
        new_lp = actor.action_distribution().log_prob(action.to(device))
        terms.append(
            clipped_grpo_objective(
                new_lp,
                old_lp.to(device),
                ref_lp.to(device),
                advantage,
                args.clip_ratio,
                args.beta,
            ).mean()
        )
        rnn = out["new_rnn_states"]
        if (step + 1) % args.recurrence == 0:
            rnn = rnn.detach()
    return torch.stack(terms).mean()


def load_source(args):
    argv = [
        "--algo=APPO",
        "--env=doom_deadly_corridor",
        f"--experiment={args.source_experiment}",
        f"--train_dir={args.source_train_dir}",
        f"--device={args.device}",
        "--policy_index=0",
        "--load_checkpoint_kind=best",
        "--no_render",
    ]
    cfg = load_from_checkpoint(parse_vizdoom_cfg(argv=argv, evaluation=True))
    return cfg


def save_run(actor, optimizer, cfg, args, frames, groups, history, source_path):
    experiment_dir = args.train_dir / args.experiment
    checkpoint_dir = experiment_dir / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_config = args.source_train_dir / args.source_experiment / "config.json"
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config.update(
        experiment=args.experiment,
        train_dir=str(args.train_dir),
        grpo_group_size=args.group_size,
        grpo_beta=args.beta,
        grpo_clip_ratio=args.clip_ratio,
        grpo_recurrence=args.recurrence,
        grpo_source_checkpoint=str(source_path),
    )
    (experiment_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint = {
        "model": actor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_steps": frames,
        "grpo_groups": groups,
    }
    output = checkpoint_dir / f"checkpoint_{groups:09d}_{frames}.pth"
    torch.save(checkpoint, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "method": "group_relative_episodic_ppo",
        "experiment": args.experiment,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "checkpoint": str(output),
        "checkpoint_sha256": digest,
        "requested_game_frames": args.train_frames,
        "realized_game_frames": frames,
        "groups": groups,
        "group_size": args.group_size,
        "clip_ratio": args.clip_ratio,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "recurrence": args.recurrence,
        "history": history,
    }
    (experiment_dir / "grpo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--source-experiment", default="deadly-corridor-upstream")
    parser.add_argument("--train-dir", type=Path, default=Path("artifacts/sample_factory_grpo_clean_2m"))
    parser.add_argument("--experiment", default="sf_corridor_grpo_seed_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frames", type=int, default=2_000_000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--recurrence", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=4.0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.group_size < 2:
        parser.error("--group-size must be at least 2")

    torch.manual_seed(args.seed)
    # Newer experiment branches parameterize evaluator registration by visual
    # shift, while older branches expose a zero-argument helper.
    if len(inspect.signature(register_components).parameters) == 0:
        register_components()
    else:
        register_components(False)
    cfg = load_source(args)
    device = torch.device("cuda" if args.device == "gpu" else "cpu")
    probe, _ = build_env(cfg, 0)
    actor = create_actor_critic(cfg, probe.observation_space, probe.action_space)
    source_path = checkpoint_path(cfg, 0, "best")
    source = torch.load(source_path, map_location=device, weights_only=False)
    actor.load_state_dict(source["model"])
    actor.model_to_device(device)
    reference = copy.deepcopy(actor).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    # GRPO does not train the value head; the shared visual/recurrent policy remains trainable.
    for parameter in actor.critic_linear.parameters():
        parameter.requires_grad_(False)
    actor.train()
    probe.close()

    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        eps=1e-6,
    )
    frames, groups, history = 0, 0, []
    while frames < args.train_frames:
        env_seed = args.seed * 10_000_000 + groups
        trajectories = [
            collect_trajectory(actor, reference, cfg, env_seed, device)
            for _ in range(args.group_size)
        ]
        returns = torch.tensor([trajectory.reward for trajectory in trajectories], device=device)
        advantages = group_advantages(returns)
        optimizer.zero_grad(set_to_none=True)
        objectives = [
            replay_objective(actor, cfg, trajectory, advantage, args, device)
            for trajectory, advantage in zip(trajectories, advantages)
        ]
        loss = -torch.stack(objectives).mean()
        if torch.isfinite(loss) and float(advantages.abs().sum()) > 0.0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            optimizer.step()
        group_frames = sum(trajectory.game_frames for trajectory in trajectories)
        frames += group_frames
        groups += 1
        record = {
            "group": groups,
            "env_seed": env_seed,
            "game_frames": group_frames,
            "cumulative_game_frames": frames,
            "returns": [trajectory.reward for trajectory in trajectories],
            "advantages": advantages.detach().cpu().tolist(),
            "completions": sum(trajectory.completed for trajectory in trajectories),
            "loss": float(loss.detach().cpu()),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    output = save_run(actor, optimizer, cfg, args, frames, groups, history, source_path)
    print(f"saved_checkpoint={output}", flush=True)


if __name__ == "__main__":
    main()
