"""Run configured training seeds across one or more GPUs, one process per GPU."""
import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path


def configure_logging(output_root: Path):
    logger = logging.getLogger("multiseed")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_root / "launcher.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def exit_description(code: int):
    if code >= 0:
        return f"exit code {code}"
    try:
        return f"signal {-code} ({signal.Signals(-code).name})"
    except ValueError:
        return f"signal {-code}"


def log_tail(logger, path: Path, lines=60):
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        logger.error("Last %d lines of %s:\n%s", min(lines, len(content)), path, "\n".join(content[-lines:]))
    except OSError as exc:
        logger.error("Could not read failed training log %s: %s", path, exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/deadly_corridor_baseline_v2.json"))
    parser.add_argument("--timesteps", type=int, help="Optional short-run override")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    seeds = list(experiment["seeds"])
    devices = list(experiment["devices"])
    output_root = Path(experiment["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_root)
    pending = list(seeds)
    active = {}
    failures = []
    last_heartbeat = 0.0

    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            seed = pending.pop(0)
            output = output_root / f"seed_{seed}"
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "train.py", "--config", str(args.config), "--seed", str(seed),
                       "--device", device, "--output", str(output)]
            if args.timesteps:
                command.extend(["--timesteps", str(args.timesteps)])
            log_handle = (output / "training.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
            active[device] = (process, seed, log_handle, time.monotonic(), output / "training.log")
            logger.info("Started seed %d on %s with PID %d; log=%s", seed, device, process.pid, output / "training.log")

        time.sleep(1)
        now = time.monotonic()
        if now - last_heartbeat >= 60 and active:
            for device, (process, seed, _, started, log_path) in active.items():
                size = log_path.stat().st_size if log_path.exists() else 0
                logger.info("Running seed %d on %s: PID %d, elapsed %.1f min, log %.1f KiB",
                            seed, device, process.pid, (now - started) / 60, size / 1024)
            last_heartbeat = now

        for device, (process, seed, log_handle, _, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            del active[device]
            logger.info("Finished seed %d on %s with %s", seed, device, exit_description(code))
            if code != 0:
                failures.append(seed)
                log_tail(logger, log_path)

    if failures:
        logger.error("Training failed for seeds: %s. Successful seeds were preserved.", failures)
        raise SystemExit(1)
    logger.info("All configured training seeds completed successfully")
    logging.shutdown()


if __name__ == "__main__":
    main()
