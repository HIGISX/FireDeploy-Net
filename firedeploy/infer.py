"""Deterministic deployment generation and independent service evaluation."""

import argparse
import time
from pathlib import Path

import torch

from .data import Data
from .model import FireDeployNet
from .rollout import Runner
from .runtime import evaluate, setup, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trusted FireDeploy-Net checkpoint containing policy state.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed deterministic warm-up deployments, recorded in output.",
    )
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 0:
        parser.error("Invalid threads or warm-up count.")
    output = Path(args.output)
    if output.exists():
        parser.error("Output file exists; choose a new path to retain the previous result.")
    start = time.perf_counter()
    setup(20261107, args.threads)
    data = Data(args.data)
    model = FireDeployNet().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["policy"], strict=True)
    runner = Runner(data, args.device)
    initialized = time.perf_counter()
    for _ in range(args.warmup):
        evaluate(model, runner)
    warmed = time.perf_counter()
    result = evaluate(model, runner)
    result.update(
        checkpoint_update=checkpoint.get("update"),
        initialization_seconds=initialized - start,
        warmup_count=args.warmup,
        warmup_seconds=warmed - initialized,
        total_seconds=time.perf_counter() - start,
        device=args.device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"F={result['F']:.8f}, distance={result['D_km']:.3f} km, feasible={result['feasible']}")


if __name__ == "__main__":
    main()
