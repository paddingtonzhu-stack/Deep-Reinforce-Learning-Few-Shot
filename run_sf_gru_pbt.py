"""Run one completion-aware GRU Population-Based Training population."""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


def describe(code):
    if code >= 0:
        return f"exit code {code}"
    try:
        return f"signal {-code} ({signal.Signals(-code).name})"
    except ValueError:
        return f"signal {-code}"


def validate_gpus(gpus, logger):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Sample Factory environment")
    count = torch.cuda.device_count()
    logger.info("CUDA preflight: PyTorch sees %d GPU(s)", count)
    for gpu in gpus:
        if gpu < 0 or gpu >= count:
            raise ValueError(f"Requested GPU {gpu}, but valid physical indices are 0..{count - 1}")
        device = torch.device(f"cuda:{gpu}")
        result = torch.randn((512, 512), device=device) @ torch.randn((512, 512), device=device)
        torch.cuda.synchronize(device)
        if not torch.isfinite(result).all().item():
            raise RuntimeError(f"CUDA numerical preflight failed on GPU {gpu}")
        logger.info("CUDA preflight passed on physical GPU %d: %s", gpu, torch.cuda.get_device_name(gpu))
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU indices")
    parser.add_argument("--train-dir", type=Path, default=Path("artifacts/sample_factory_gru_pbt"))
    parser.add_argument("--experiment", default="sf_corridor_gru_pbt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000_000, help="Environment steps per policy")
    parser.add_argument("--num-policies", type=int, default=4)
    parser.add_argument("--pbt-start", type=int, default=2_000_000)
    parser.add_argument("--pbt-period", type=int, default=2_000_000)
    parser.add_argument("--replace-fraction", type=float, default=0.25)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU")
    if args.num_policies < 2:
        parser.error("PBT requires --num-policies of at least 2")
    if args.steps <= 0 or args.pbt_start < 0 or args.pbt_period <= 0:
        parser.error("step counts must be positive (pbt-start may be zero)")
    if not 0.0 < args.replace_fraction < 1.0:
        parser.error("--replace-fraction must be between 0 and 1")

    args.train_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.train_dir / "pbt_train.log"
    logger = logging.getLogger("sf_gru_pbt")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(args.train_dir / "launcher.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    try:
        validate_gpus(gpus, logger)
    except Exception:
        logger.exception("CUDA preflight failed; training was not started")
        raise SystemExit(1)

    command = [
        sys.executable,
        "sf_train_corridor.py",
        "--memory=gru",
        "--completion-objective",
        f"--experiment={args.experiment}",
        f"--train_dir={args.train_dir}",
        f"--seed={args.seed}",
        f"--train_for_env_steps={args.steps}",
        f"--num_policies={args.num_policies}",
        "--with_pbt=True",
        "--pbt_target_objective=true_objective",
        f"--pbt_start_mutation={args.pbt_start}",
        f"--pbt_period_env_steps={args.pbt_period}",
        f"--pbt_replace_fraction={args.replace_fraction}",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    logger.info(
        "Starting completion-aware GRU PBT: policies=%d, GPUs=%s, steps/policy=%d",
        args.num_policies,
        env["CUDA_VISIBLE_DEVICES"],
        args.steps,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        logger.info("Started PID %d; training log=%s", process.pid, log_path)
        started = time.monotonic()
        while True:
            try:
                code = process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                size = log_path.stat().st_size if log_path.exists() else 0
                logger.info(
                    "Running PID %d, elapsed %.1f min, log %.1f KiB",
                    process.pid,
                    (time.monotonic() - started) / 60,
                    size / 1024,
                )
            except KeyboardInterrupt:
                logger.warning("Launcher interrupted; forwarding SIGINT to training PID %d", process.pid)
                process.send_signal(signal.SIGINT)
                code = process.wait()
                break

    logger.info("PBT training finished with %s", describe(code))
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        logger.error("Failure log tail:\n%s", "\n".join(tail))
        raise SystemExit(code if code > 0 else 1)


if __name__ == "__main__":
    main()
