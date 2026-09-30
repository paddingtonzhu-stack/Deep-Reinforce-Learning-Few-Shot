"""Run configured training seeds across one or more GPUs, one process per GPU."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/deadly_corridor_baseline.json"))
    parser.add_argument("--timesteps", type=int, help="Optional short-run override")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    seeds = list(experiment["seeds"])
    devices = list(experiment["devices"])
    output_root = Path(experiment["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    pending = list(seeds)
    active = {}
    failures = []

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
            active[device] = (process, seed, log_handle)
            print(f"Started seed {seed} on {device}; log={output / 'training.log'}", flush=True)

        time.sleep(1)
        for device, (process, seed, log_handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            del active[device]
            print(f"Finished seed {seed} on {device} with exit code {code}", flush=True)
            if code != 0:
                failures.append(seed)

    if failures:
        raise SystemExit(f"Training failed for seeds: {failures}")


if __name__ == "__main__":
    main()
