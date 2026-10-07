"""Train matched Sample Factory APPO corridor ablations."""
import argparse
import os
import sys

if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

from sf_corridor_objective import (
    COMPLETION_ENV,
    SHAPED_COMPLETION_ENV,
    SKILL_CURRICULUM_ENV,
    register_corridor_completion_env,
)
from sf_methods import TRAINABLE_METHODS, format_method_catalog, method_for_training, recurrent_options
from sf_temporal_cores import register_temporal_core


MATCHED_DEFAULTS = {
    "algo": "APPO",
    "env": "doom_deadly_corridor",
    "device": "gpu",
    "num_policies": "2",
    "async_rl": "True",
    "num_batches_to_accumulate": "2",
    "worker_num_splits": "2",
    "policy_workers_per_policy": "1",
    "max_policy_lag": "1000",
    "num_workers": "8",
    "num_envs_per_worker": "4",
    "batch_size": "1024",
    "num_batches_per_epoch": "1",
    "num_epochs": "1",
    "rollout": "32",
    "gamma": "0.99",
    "reward_scale": "1.0",
    "reward_clip": "1000.0",
    "normalize_returns": "True",
    "exploration_loss_coeff": "0.001",
    "value_loss_coeff": "0.5",
    "exploration_loss": "symmetric_kl",
    "gae_lambda": "0.95",
    "ppo_clip_ratio": "0.1",
    "ppo_clip_value": "0.2",
    "optimizer": "adam",
    "adam_eps": "1e-6",
    "max_grad_norm": "4.0",
    "learning_rate": "0.0001",
    "normalize_input": "True",
    "encoder_conv_architecture": "convnet_simple",
    "encoder_conv_mlp_layers": "512",
    "nonlinearity": "elu",
    "actor_critic_share_weights": "True",
    "env_frameskip": "4",
    "env_framestack": "1",
    "pixel_format": "CHW",
    "res_w": "128",
    "res_h": "72",
    "train_for_env_steps": "10000000",
    "with_wandb": "False",
    "log_to_file": "True",
}


def has_option(argv, name):
    prefix = f"--{name}"
    return any(arg == prefix or arg.startswith(prefix + "=") for arg in argv)


def parse_corridor_cfg(argv):
    parser, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser)
    parser.add_argument(
        "--memory",
        choices=("cnn", "gru", "gru_residual_ln", "gru_orthogonal", "transformer", "gtrxl", "gru_attention"),
        default="cnn",
    )
    parser.add_argument("--transformer_context", type=int, default=32)
    parser.add_argument("--transformer_dim", type=int, default=256)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_ff_dim", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.0)
    parser.add_argument("--gtrxl_identity_bias", type=float, default=2.0)
    parser.add_argument("--gru_attention_hidden_size", type=int, default=512)
    parser.add_argument("--gru_attention_dim", type=int, default=128)
    parser.add_argument("--gru_attention_gate_init", type=float, default=-2.0)
    parser.add_argument("--completion_bonus", type=float, default=0.0)
    parser.add_argument("--corridor_skill", type=int, default=5)
    doom_override_defaults(parser)
    return parse_full_cfg(parser, argv)


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--memory",
        choices=("cnn", "gru", "gru_residual_ln", "gru_orthogonal", "transformer", "gtrxl", "gru_attention"),
        default="cnn",
    )
    custom.add_argument(
        "--method",
        choices=TRAINABLE_METHODS,
        help="Canonical experiment method (preferred over the legacy --memory flag).",
    )
    custom.add_argument("--list-methods", action="store_true")
    custom.add_argument("--transformer-context", type=int, default=32)
    custom.add_argument("--transformer-dim", type=int)
    custom.add_argument("--transformer-layers", type=int, default=2)
    custom.add_argument("--transformer-heads", type=int, default=4)
    custom.add_argument("--transformer-ff-dim", type=int, default=512)
    custom.add_argument("--transformer-dropout", type=float, default=0.0)
    custom.add_argument("--gtrxl-identity-bias", type=float, default=2.0)
    custom.add_argument("--gru-attention-hidden-size", type=int, default=512)
    custom.add_argument("--gru-attention-dim", type=int, default=128)
    custom.add_argument("--gru-attention-gate-init", type=float, default=-2.0)
    custom.add_argument("--completion-bonus", type=float, default=0.0)
    custom.add_argument("--corridor-skill", type=int, default=5)
    custom.add_argument(
        "--completion-objective",
        action="store_true",
        help="Use binary corridor completion as Sample Factory true_objective (for PBT).",
    )
    custom.add_argument("--check-config", action="store_true")
    known, remaining = custom.parse_known_args()
    if known.list_methods:
        print(format_method_catalog())
        return 0
    selected_method = method_for_training(known.method) if known.method else None
    memory = selected_method.memory if selected_method else known.memory
    if known.method and any(
        arg == "--memory" or arg.startswith("--memory=") for arg in sys.argv[1:]
    ):
        custom.error("use --method or --memory, not both")
    transformer_dim = known.transformer_dim
    if transformer_dim is None:
        # Gated residuals add parameters. Width 176 keeps the default GTrXL
        # core (~1.545M) close to the 512-unit GRU core (~1.576M), while the
        # original vanilla Transformer retains its historical width of 256.
        transformer_dim = 176 if memory == "gtrxl" else 256
    argv = list(remaining)
    if selected_method and selected_method.recurrence is not None:
        if not has_option(argv, "rollout"):
            argv.append(f"--rollout={selected_method.recurrence}")
        if not has_option(argv, "recurrence"):
            argv.append(f"--recurrence={selected_method.recurrence}")
    # Isolate actor/learner staleness without changing the GRU architecture,
    # optimizer, rollout length, or environment.
    if selected_method and selected_method.name == "gru_lag1" and not has_option(argv, "max_policy_lag"):
        argv.append("--max_policy_lag=1")
    # Test genuinely on-policy collection as a separate hypothesis from the
    # rejected lag=1 experiment, which discarded stale asynchronous batches.
    if selected_method and selected_method.name == "gru_sync" and not has_option(argv, "async_rl"):
        argv.append("--async_rl=False")
    if selected_method and selected_method.name == "gru_sync" and not has_option(
        argv, "num_batches_per_epoch"
    ):
        # 8 workers * 4 envs * rollout 64 = 2048 samples. Two 1024-sample
        # batches consume each synchronous collection exactly once.
        argv.append("--num_batches_per_epoch=2")
    # Test whether stronger action exploration prevents the observed early
    # reward~4 collapse across training seeds, without changing architecture.
    if selected_method and selected_method.name == "gru_explore" and not has_option(
        argv, "exploration_loss_coeff"
    ):
        argv.append("--exploration_loss_coeff=0.003")
    completion_bonus = known.completion_bonus
    if (
        selected_method
        and selected_method.name == "gru_completion_bonus"
        and not any(
            arg == "--completion-bonus" or arg.startswith("--completion-bonus=")
            for arg in sys.argv[1:]
        )
    ):
        completion_bonus = 10.0
    if known.completion_objective and not has_option(argv, "env"):
        argv.append(f"--env={COMPLETION_ENV}")
    elif completion_bonus != 0.0 and not has_option(argv, "env"):
        argv.append(f"--env={SHAPED_COMPLETION_ENV}")
    elif selected_method and selected_method.name == "gru_skill_curriculum" and not has_option(argv, "env"):
        argv.append(f"--env={SKILL_CURRICULUM_ENV}")
    for name, value in MATCHED_DEFAULTS.items():
        if not has_option(argv, name):
            argv.append(f"--{name}={value}")

    custom_cfg = {
        "memory": memory,
        "transformer_context": known.transformer_context,
        "transformer_dim": transformer_dim,
        "transformer_layers": known.transformer_layers,
        "transformer_heads": known.transformer_heads,
        "transformer_ff_dim": known.transformer_ff_dim,
        "transformer_dropout": known.transformer_dropout,
        "gtrxl_identity_bias": known.gtrxl_identity_bias,
        "gru_attention_hidden_size": known.gru_attention_hidden_size,
        "gru_attention_dim": known.gru_attention_dim,
        "gru_attention_gate_init": known.gru_attention_gate_init,
        "completion_bonus": completion_bonus,
        "corridor_skill": known.corridor_skill,
    }
    for name, value in custom_cfg.items():
        if not has_option(argv, name):
            argv.append(f"--{name}={value}")

    memory_options = recurrent_options(
        memory,
        known.transformer_context,
        transformer_dim,
        known.gru_attention_hidden_size,
        known.gru_attention_dim,
    )
    for name, value in memory_options.items():
        if not has_option(argv, name):
            argv.append(f"--{name}={value}")

    register_vizdoom_components()
    register_corridor_completion_env()
    register_temporal_core()
    cfg = parse_corridor_cfg(argv)
    print(
        f"Matched Sample Factory run: method={known.method or memory}, memory={cfg.memory}, "
        f"use_rnn={cfg.use_rnn}, "
        f"recurrence={cfg.recurrence}, workers={cfg.num_workers}, "
        f"envs_per_worker={cfg.num_envs_per_worker}, async_rl={cfg.async_rl}, "
        f"batches_per_epoch={cfg.num_batches_per_epoch}, "
        f"exploration_coeff={cfg.exploration_loss_coeff}, max_policy_lag={cfg.max_policy_lag}, "
        f"budget={cfg.train_for_env_steps}",
        flush=True,
    )
    if known.check_config:
        return 0
    return run_rl(cfg)


if __name__ == "__main__":
    sys.exit(main())
