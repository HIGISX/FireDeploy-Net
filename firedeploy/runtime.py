"""Runtime configuration, checkpoints and independently verified evaluation."""

import json
import os
import random
import time
from pathlib import Path

import numba
import numpy as np
import torch


def setup(seed, threads=4):
    torch.set_num_threads(threads)
    numba.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    Path(path).write_text(
        json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


@torch.no_grad()
def evaluate(model, runner):
    """Report decoding and verification separately; exclude loading from both."""
    model.eval()
    synchronize(runner.device)
    start = time.perf_counter()
    values, layouts, distances, *_ = runner.rollout(model, 1, False)
    synchronize(runner.device)
    decoded = time.perf_counter()
    result = runner.d.evaluate(layouts[0])
    if (
        not result["feasible"]
        or abs(values[0] - result["F"]) > 1e-9
        or abs(distances[0] - result["D_km"]) > 1e-8
    ):
        raise RuntimeError("Decoded plan disagrees with independent service/matching evaluation.")
    result["decode_seconds"] = decoded - start
    result["verification_seconds"] = time.perf_counter() - decoded
    return result


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same visible CUDA devices.")
        torch.cuda.set_rng_state_all(state["cuda"])
