"""Train matched Sample Factory APPO corridor ablations."""
import argparse
import os
import sys

if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from sample_factory.train import run_rl
from sf_examples.vizdoom.train_vizdoom import parse_vizdoom_cfg, register_vizdoom_components


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


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--memory", choices=("cnn", "gru"), default="cnn")
    custom.add_argument("--check-config", action="store_true")
    known, remaining = custom.parse_known_args()
    argv = list(remaining)
    for name, value in MATCHED_DEFAULTS.items():
        if not has_option(argv, name):
            argv.append(f"--{name}={value}")

    memory_options = {
        "cnn": {"use_rnn": "False", "recurrence": "1"},
        "gru": {"use_rnn": "True", "recurrence": "32", "rnn_size": "512", "rnn_type": "gru"},
    }[known.memory]
    for name, value in memory_options.items():
        if not has_option(argv, name):
            argv.append(f"--{name}={value}")

    register_vizdoom_components()
    cfg = parse_vizdoom_cfg(argv=argv)
    print(
        f"Matched Sample Factory run: memory={known.memory}, use_rnn={cfg.use_rnn}, "
        f"recurrence={cfg.recurrence}, workers={cfg.num_workers}, "
        f"envs_per_worker={cfg.num_envs_per_worker}, budget={cfg.train_for_env_steps}",
        flush=True,
    )
    if known.check_config:
        return 0
    return run_rl(cfg)


if __name__ == "__main__":
    sys.exit(main())
