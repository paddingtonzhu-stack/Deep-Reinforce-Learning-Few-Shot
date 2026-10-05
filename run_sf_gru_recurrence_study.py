"""Run a controlled GRU-32 versus GRU-64 Sample Factory study sequentially."""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_int_list(value):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def describe(code):
    if code >= 0:
        return f"exit code {code}"
    try:
        return f"signal {-code} ({signal.Signals(-code).name})"
    except ValueError:
        return f"signal {-code}"


def make_logger(root):
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sf_gru_recurrence_study")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(root / "launcher.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_gpu(gpu, logger):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Sample Factory environment")
    count = torch.cuda.device_count()
    logger.info("CUDA preflight: PyTorch sees %d GPU(s)", count)
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
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2"))
    parser.add_argument("--recurrences", type=parse_int_list, default=parse_int_list("32,64"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("artifacts/sample_factory_gru_recurrence_2m"),
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if any(value <= 0 for value in args.recurrences):
        parser.error("all recurrence values must be positive")

    group_root = args.train_dir / "gru_recurrence_runs"
    logger = make_logger(group_root)
    try:
        validate_gpu(args.gpu, logger)
    except Exception:
        logger.exception("CUDA preflight failed; no training was started")
        raise SystemExit(1)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    failures = []
    for recurrence in args.recurrences:
        for seed in args.seeds:
            experiment = f"sf_corridor_gru_r{recurrence}_seed_{seed}"
            log_path = group_root / f"r{recurrence}_seed_{seed}.log"
            method = {32: "gru", 64: "gru_long"}.get(recurrence, "gru")
            command = [
                sys.executable,
                "sf_train_corridor.py",
                f"--method={method}",
                f"--experiment={experiment}",
                f"--train_dir={args.train_dir}",
                f"--seed={seed}",
                f"--train_for_env_steps={args.steps}",
                "--num_policies=1",
                f"--rollout={recurrence}",
                f"--recurrence={recurrence}",
            ]
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n===== launcher start {datetime.now(timezone.utc).isoformat()} "
                    f"recurrence={recurrence} seed={seed} gpu={args.gpu} =====\n"
                )
                handle.flush()
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
                logger.info(
                    "Started GRU-%d seed %d on physical GPU %d with PID %d; log=%s",
                    recurrence,
                    seed,
                    args.gpu,
                    process.pid,
                    log_path,
                )
                started = time.monotonic()
                while True:
                    try:
                        code = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        size = log_path.stat().st_size if log_path.exists() else 0
                        logger.info(
                            "Running GRU-%d seed %d: PID %d, elapsed %.1f min, log %.1f KiB",
                            recurrence,
                            seed,
                            process.pid,
                            (time.monotonic() - started) / 60,
                            size / 1024,
                        )
                    except KeyboardInterrupt:
                        logger.warning("Interrupted; forwarding SIGINT to PID %d", process.pid)
                        process.send_signal(signal.SIGINT)
                        code = process.wait()
                        break

            logger.info("Finished GRU-%d seed %d with %s", recurrence, seed, describe(code))
            if code != 0:
                failures.append((recurrence, seed))
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                logger.error("Failure log tail:\n%s", "\n".join(tail))
                break
        if failures:
            break

    if failures:
        logger.error("Study stopped after failed runs: %s", failures)
        raise SystemExit(1)
    logger.info("All controlled GRU recurrence runs completed successfully")


if __name__ == "__main__":
    main()
