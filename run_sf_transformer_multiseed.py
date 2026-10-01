"""Schedule three matched Transformer-memory Sample Factory runs across GPUs."""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


def logger_for(root):
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sf_transformer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(root / "launcher.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


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
    for value in gpus:
        gpu = int(value)
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
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--train-dir", type=Path, default=Path("artifacts/sample_factory_transformer"))
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    gpus = [value.strip() for value in args.gpus.split(",")]
    group_root = args.train_dir / "sf_corridor_transformer_runs"
    logger = logger_for(group_root)
    try:
        validate_gpus(gpus, logger)
    except Exception:
        logger.exception("CUDA preflight failed; no training processes were started")
        raise SystemExit(1)

    pending, active, failures = list(seeds), {}, []
    last_heartbeat = 0.0
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            seed = pending.pop(0)
            experiment = f"sf_corridor_transformer_seed_{seed}"
            log_path = group_root / f"seed_{seed}.log"
            handle = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                "sf_train_corridor.py",
                "--memory=transformer",
                f"--transformer-context={args.context}",
                f"--transformer-dim={args.model_dim}",
                f"--transformer-layers={args.layers}",
                f"--transformer-heads={args.heads}",
                f"--experiment={experiment}",
                f"--train_dir={args.train_dir}",
                f"--seed={seed}",
                f"--train_for_env_steps={args.steps}",
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            active[gpu] = (process, seed, handle, time.monotonic(), log_path)
            logger.info("Started seed %d on physical GPU %s with PID %d; log=%s", seed, gpu, process.pid, log_path)

        time.sleep(1)
        now = time.monotonic()
        if active and now - last_heartbeat >= 60:
            for gpu, (process, seed, _, started, path) in active.items():
                size = path.stat().st_size if path.exists() else 0
                logger.info(
                    "Running seed %d on GPU %s: PID %d, elapsed %.1f min, log %.1f KiB",
                    seed, gpu, process.pid, (now - started) / 60, size / 1024,
                )
            last_heartbeat = now

        for gpu, (process, seed, handle, _, path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[gpu]
            logger.info("Finished seed %d on GPU %s with %s", seed, gpu, describe(code))
            if code != 0:
                failures.append(seed)
                tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
                logger.error("Failure log tail for seed %d:\n%s", seed, "\n".join(tail))

    if failures:
        logger.error("Failed seeds: %s", failures)
        raise SystemExit(1)
    logger.info("All matched Transformer-memory Sample Factory runs completed successfully")


if __name__ == "__main__":
    main()
