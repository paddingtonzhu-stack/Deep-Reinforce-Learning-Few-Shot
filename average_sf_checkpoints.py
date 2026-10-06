"""Average compatible Sample Factory model checkpoints for evaluation.

This implements a small, explicit form of stochastic weight averaging (SWA).
Only floating-point learned model tensors are averaged. Running observation and
return-normalization statistics, non-floating buffers, optimizer state, and
training metadata are copied from the final (newest) input checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


NORMALIZER_PREFIXES = ("obs_normalizer.", "returns_normalizer.")


def _load(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError(f"{path} is not a Sample Factory model checkpoint")
    return checkpoint


def average_checkpoints(inputs: Sequence[Path], output: Path) -> None:
    if len(inputs) < 2:
        raise ValueError("at least two input checkpoints are required")

    checkpoints = [_load(path) for path in inputs]
    states = [checkpoint["model"] for checkpoint in checkpoints]
    reference_keys = tuple(states[-1].keys())
    for path, state in zip(inputs, states, strict=True):
        if tuple(state.keys()) != reference_keys:
            raise ValueError(f"model keys in {path} do not match the newest checkpoint")

    averaged = states[-1].copy()
    averaged_names: list[str] = []
    for name in reference_keys:
        tensors = [state[name] for state in states]
        latest = tensors[-1]
        for path, tensor in zip(inputs, tensors, strict=True):
            if tensor.shape != latest.shape or tensor.dtype != latest.dtype:
                raise ValueError(f"tensor {name!r} in {path} is incompatible")

        if latest.is_floating_point() and not name.startswith(NORMALIZER_PREFIXES):
            # Accumulate in float64 to avoid order-dependent float32 rounding.
            mean = torch.stack([tensor.to(torch.float64) for tensor in tensors]).mean(dim=0)
            averaged[name] = mean.to(latest.dtype)
            averaged_names.append(name)
        else:
            averaged[name] = latest.clone()

    result = checkpoints[-1].copy()
    result["model"] = averaged
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(
        f"Saved {output} from {len(inputs)} checkpoints; "
        f"averaged {len(averaged_names)} floating model tensors and retained newest normalizers"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    average_checkpoints(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
