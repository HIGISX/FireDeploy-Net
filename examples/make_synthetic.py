"""Create a tiny artificial instance for software checks, not scientific results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from firedeploy.data import PERIODS, utility


def make_instance(root, budget=3.0):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Synthetic output directory must be empty.")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261107)
    n, m = 12, 20
    xy = rng.uniform(0, 2000, size=(n, 2))
    weights = rng.uniform(0.1, 1, size=(3, m))
    weights /= weights.sum(axis=1, keepdims=True)
    area = np.full(m, 1 / m)
    baseline = rng.uniform(8, 22, size=(3, m))
    # Directed distances; exact zero diagonal.
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1) / 1000
    distance *= rng.uniform(1, 1.3, size=(n, n))
    np.save(root / "distance_km.npy", distance)
    np.savez(
        root / "baseline.npz",
        candidate_id=np.arange(n),
        grid_id=np.arange(m),
        weights=weights,
        area_weights=area,
        baseline_response_min=baseline,
    )
    pd.DataFrame({"candidate_id": np.arange(n), "x_m": xy[:, 0], "y_m": xy[:, 1]}).to_csv(
        root / "candidates.csv", index=False
    )
    U0, Q0, Uall, Qall = [], [], [], []
    responses, gains = {}, {}
    for t, period in enumerate(PERIODS):
        response = rng.uniform(3, 24, size=(n, m))
        np.save(root / f"{period}_response.npy", response)
        responses[period] = f"{period}_response.npy"
        gain_u = np.maximum(utility(response) - utility(baseline[t]), 0) * weights[t]
        gain_q = (
            np.maximum(np.maximum(baseline[t] - 5, 0) ** 2 - np.maximum(response - 5, 0) ** 2, 0)
            * area
        )
        gains[period] = {}
        for key, gain in [("U", gain_u), ("Q", gain_q)]:
            name = f"{period}_{key}_gain.npz"
            save_npz(root / name, csr_matrix(gain))
            gains[period][key] = name
        all_response = np.minimum(baseline[t], response.min(0))
        U0.append(float(weights[t] @ utility(baseline[t])))
        Q0.append(float(area @ np.maximum(baseline[t] - 5, 0) ** 2))
        Uall.append(float(weights[t] @ utility(all_response)))
        Qall.append(float(area @ np.maximum(all_response - 5, 0) ** 2))
    norm = {
        "U0": float(np.mean(U0)),
        "Q0": float(np.mean(Q0)),
        "delta_U": float(np.mean(Uall) - np.mean(U0)),
        "delta_Q": float(np.mean(Q0) - np.mean(Qall)),
    }
    (root / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    manifest = {
        "units_per_period": 2,
        "budget_km": budget,
        "baseline": "baseline.npz",
        "candidates": "candidates.csv",
        "normalization": "normalization.json",
        "distance_km": "distance_km.npy",
        "responses": responses,
        "gains": gains,
    }
    (root / "instance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root / "instance.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic")
    args = parser.parse_args()
    print(make_instance(args.output))
