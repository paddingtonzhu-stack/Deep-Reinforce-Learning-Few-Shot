"""Validate every configured CUDA device before starting long experiments."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/cuda_check.json"))
    args = parser.parse_args()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch environment")
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        try:
            left = torch.randn((1024, 1024), device=device)
            right = torch.randn((1024, 1024), device=device)
            result = left @ right
            torch.cuda.synchronize(device)
            entry = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "status": "pass",
                "result_mean": float(result.mean().cpu()),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 2**20, 2),
            }
        except Exception as exc:
            entry = {"index": index, "name": torch.cuda.get_device_name(index), "status": "fail", "error": str(exc)}
        report["devices"].append(entry)
        print(f"cuda:{index} {entry['name']}: {entry['status']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any(device["status"] != "pass" for device in report["devices"]):
        raise SystemExit(1)
    print(f"Saved CUDA report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
