"""REINFORCE fine-tuning with a leave-one-out baseline and exact replay."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import Data
from .model import FireDeployNet
from .rollout import Runner
from .runtime import (
    evaluate,
    restore_rng,
    rng_state,
    save_checkpoint,
    setup,
    synchronize,
    write_json,
)


def policy_update(model, optimizer, runner, batch_size, microbatch):
    """Sample complete deployments; replay fixed actions for memory-bounded gradients.

    The leave-one-out baseline uses the complete effective batch, not each chunk.
    Neither rewards nor baselines are differentiated.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values, layouts, logps, distances = [], [], [], []
    with torch.no_grad():
        for offset in range(0, batch_size, microbatch):
            n = min(microbatch, batch_size - offset)
            v, a, d, lp, *_ = runner.rollout(model, n, True)
            values.append(v)
            layouts.append(a)
            logps.append(lp.cpu().numpy())
            distances.append(d)
    values = np.concatenate(values)
    layouts = np.concatenate(layouts)
    original_logps = np.concatenate(logps)
    baseline = (values.sum() - values) / (batch_size - 1)
    advantages = values - baseline
    total_loss = 0.0
    for offset in range(0, batch_size, microbatch):
        end = min(offset + microbatch, batch_size)
        forced = layouts[offset:end].reshape(end - offset, 3 * runner.d.K)
        v, a, d, lp, *_ = runner.rollout(model, end - offset, False, forced)
        if not np.allclose(v, values[offset:end], atol=1e-10, rtol=0):
            raise RuntimeError("Replay changed deployment rewards.")
        if not np.allclose(
            lp.detach().cpu().numpy(), original_logps[offset:end], atol=0.01, rtol=0
        ):
            raise RuntimeError("Replay changed policy log probabilities.")
        adv = torch.as_tensor(advantages[offset:end], dtype=lp.dtype, device=runner.device)
        loss = -(adv * lp).sum() / batch_size
        loss.backward()
        total_loss += float(loss.detach())
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not math.isfinite(norm) or not math.isfinite(total_loss):
        raise FloatingPointError("Non-finite policy gradient; update not applied.")
    optimizer.step()
    return {
        "F_mean": float(values.mean()),
        "F_best": float(values.max()),
        "D_mean_km": float(np.concatenate(distances).mean()),
        "loss": total_loss,
        "gradient_norm": norm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to instance.json.")
    parser.add_argument(
        "--output",
        required=True,
        help="A new output directory (or the original directory when resuming).",
    )
    parser.add_argument("--config", required=True, help="Training configuration JSON.")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--pretrained", help="Trusted single-period policy checkpoint containing policy state."
    )
    initialization.add_argument(
        "--resume", help="latest.pt checkpoint from this public implementation."
    )
    initialization.add_argument(
        "--from-scratch",
        action="store_true",
        help="For software checks; differs from paper initialization.",
    )
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:0.")
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("epochs", "updates_per_epoch", "batch_size", "evaluate_every", "seed"):
        if not isinstance(config[key], int) or config[key] < 1:
            parser.error(f"{key} must be a positive integer.")
    if (
        config["batch_size"] < 2
        or args.microbatch < 1
        or args.threads < 1
        or not math.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
    ):
        parser.error("Invalid batch, thread or learning-rate setting.")
    setup(config["seed"], args.threads)
    data = Data(args.data)
    binding = {
        "config": config,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "device": args.device,
        "microbatch": args.microbatch,
        "threads": args.threads,
    }
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; choose a new directory or explicitly resume.")
    output.mkdir(parents=True, exist_ok=True)
    runner = Runner(data, args.device)
    model = FireDeployNet(pretrained=args.pretrained, seed=config["seed"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    update, elapsed, best, history = 0, 0.0, None, []
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload["binding"] != binding:
            parser.error("Resume configuration, data manifest or execution settings differ.")
        model.load_state_dict(payload["policy"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        update, elapsed, best, history = (
            payload[k] for k in ("update", "elapsed_seconds", "best", "history")
        )
        restore_rng(payload["rng"])
    else:
        initial = evaluate(model, runner)
        best = {"update": 0, "metrics": initial}
        save_checkpoint(
            output / "best.pt", {"policy": model.state_dict(), "update": 0, "metrics": initial}
        )
    write_json(output / "run.json", binding)
    total = config["epochs"] * config["updates_per_epoch"]
    start = time.perf_counter()
    first_update = update + 1
    for update in range(first_update, total + 1):
        synchronize(args.device)
        tick = time.perf_counter()
        record = policy_update(model, optimizer, runner, config["batch_size"], args.microbatch)
        synchronize(args.device)
        record.update(
            update=update,
            epoch=update / config["updates_per_epoch"],
            seconds=time.perf_counter() - tick,
        )
        history.append(record)
        if update % config["evaluate_every"] == 0 or update == total:
            current = evaluate(model, runner)
            record["deterministic"] = current
            if current["F"] > best["metrics"]["F"] + 1e-12:
                best = {"update": update, "metrics": current}
                save_checkpoint(
                    output / "best.pt",
                    {"policy": model.state_dict(), "update": update, "metrics": current},
                )
            print(
                json.dumps({"update": update, "F": current["F"], "best_F": best["metrics"]["F"]}),
                flush=True,
            )
        save_checkpoint(
            output / "latest.pt",
            {
                "policy": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "elapsed_seconds": elapsed + time.perf_counter() - start,
                "best": best,
                "history": history,
                "binding": binding,
                "rng": rng_state(),
            },
        )
    final = evaluate(model, runner)
    save_checkpoint(
        output / "final.pt", {"policy": model.state_dict(), "update": update, "metrics": final}
    )
    write_json(output / "history.json", history)
    write_json(
        output / "summary.json",
        {
            "final": final,
            "best": best,
            "updates": update,
            "training_layouts": update * config["batch_size"],
            "elapsed_seconds": elapsed + time.perf_counter() - start,
        },
    )


if __name__ == "__main__":
    main()
