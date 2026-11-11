"""Input loading, service metrics and exact directed repositioning matching."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import load_npz

PERIODS = ("morning", "midday", "evening")


def utility(response):
    response = np.asarray(response, dtype=np.float64)
    u = np.zeros_like(response)
    u[response <= 5] = 1
    mask = (response > 5) & (response < 20)
    u[mask] = (np.exp(-0.5 * ((response[mask] - 5) / 15) ** 2) - np.exp(-0.5)) / (1 - np.exp(-0.5))
    return u


class Data:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        cfg = json.loads(self.manifest.read_text(encoding="utf-8"))
        root = self.manifest.parent
        path = lambda value: root / value
        self.K = int(cfg["units_per_period"])
        self.budget = float(cfg["budget_km"])
        if self.K < 1 or not np.isfinite(self.budget) or self.budget < 0:
            raise ValueError("K must be positive and the budget finite and nonnegative.")
        self.base = dict(np.load(path(cfg["baseline"]), allow_pickle=False))
        self.norm = json.loads(path(cfg["normalization"]).read_text(encoding="utf-8"))
        self.ids = self.base["candidate_id"]
        self.grid_ids = self.base["grid_id"]
        self.n, self.m = len(self.ids), len(self.grid_ids)
        if self.K > self.n:
            raise ValueError("K exceeds the number of candidates.")
        for key, shape in [
            ("weights", (3, self.m)),
            ("area_weights", (self.m,)),
            ("baseline_response_min", (3, self.m)),
        ]:
            a = self.base[key]
            if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError("Invalid baseline array: " + key)
        if not np.allclose(self.base["weights"].sum(1), 1, atol=1e-12, rtol=0) or not np.isclose(
            self.base["area_weights"].sum(), 1
        ):
            raise ValueError("Demand and area weights must be normalized.")
        if (
            any(not np.isfinite(self.norm[k]) for k in ("U0", "Q0", "delta_U", "delta_Q"))
            or min(self.norm["delta_U"], self.norm["delta_Q"]) <= 0
        ):
            raise ValueError("Invalid normalization constants.")
        self.dist = np.load(path(cfg["distance_km"]), mmap_mode="r")
        if (
            self.dist.shape != (self.n, self.n)
            or np.isnan(self.dist).any()
            or (self.dist < 0).any()
            or not np.all(self.dist.diagonal() == 0)
        ):
            raise ValueError("Invalid directed distance matrix (kilometres).")
        self.od = [np.load(path(cfg["responses"][p]), mmap_mode="r") for p in PERIODS]
        if any(a.shape != (self.n, self.m) for a in self.od):
            raise ValueError("Response matrices must have shape (candidates, grids).")
        self.mats, self.csc = [], []
        self.solo = np.zeros((3, 2, self.n), dtype=np.float64)
        for t, p in enumerate(PERIODS):
            rr, cc = [], []
            for q, kind in enumerate(("U", "Q")):
                mat = load_npz(path(cfg["gains"][p][kind])).astype(np.float64).tocsr()
                if (
                    mat.shape != (self.n, self.m)
                    or not np.isfinite(mat.data).all()
                    or (mat.data < 0).any()
                ):
                    raise ValueError("Invalid service-gain matrix.")
                mat = mat / self.norm["delta_" + kind] / 3
                mat.sum_duplicates()
                mat.sort_indices()
                rr.append(mat)
                cc.append(mat.tocsc())
                self.solo[t, q] = np.asarray(mat.sum(axis=1)).ravel()
            self.mats.append(rr)
            self.csc.append(cc)
        self.scales = np.maximum(self.solo.max(axis=2), 1e-12)
        candidates = pd.read_csv(path(cfg["candidates"]), float_precision="round_trip")
        if not np.array_equal(candidates.candidate_id.to_numpy(), self.ids):
            raise ValueError("Candidate ordering does not match baseline arrays.")
        xy = candidates[["x_m", "y_m"]].to_numpy(np.float64)
        if not np.isfinite(xy).all():
            raise ValueError("Coordinates must be finite projected coordinates in metres.")
        self.xy = (xy - xy.mean(0)) / np.maximum(xy.std(0), 1.0)

    def close(self):
        """Release read-only memory maps, including Windows file handles."""
        for array in [self.dist, *self.od]:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def match(self, a, b):
        a = np.asarray(a, int)
        b = np.asarray(b, int)
        cost = self.dist[np.ix_(a, b)]
        rr, cc = linear_sum_assignment(cost)
        value = float(cost[rr, cc].sum())
        if not np.isfinite(value):
            raise ValueError("Unreachable perfect matching")
        return value, {int(b[c]): int(a[r]) for r, c in zip(rr, cc)}

    def metrics_response(self, response):
        rows = []
        for t, p in enumerate(PERIODS):
            R = response[t]
            late = np.maximum(R - 5.0, 0.0)
            u = np.zeros_like(R)
            u[R <= 5] = 1.0
            m = (R > 5) & (R < 20)
            u[m] = (np.exp(-0.5 * ((R[m] - 5) / 15) ** 2) - np.exp(-0.5)) / (1 - np.exp(-0.5))
            w = self.base["weights"][t]
            a = self.base["area_weights"]
            rows.append(
                {
                    "period": p,
                    "U": float(w @ u),
                    "Cw": float(w @ (R <= 5 + 1e-9)),
                    "EA_min": float(a @ late),
                    "Q_min2": float(a @ (late**2)),
                }
            )
        U = np.mean([v["U"] for v in rows])
        Q = np.mean([v["Q_min2"] for v in rows])
        gu = (U - self.norm["U0"]) / self.norm["delta_U"]
        gq = (self.norm["Q0"] - Q) / self.norm["delta_Q"]
        return {
            "U": U,
            "Q_min2": Q,
            "Cw": np.mean([v["Cw"] for v in rows]),
            "EA_min": np.mean([v["EA_min"] for v in rows]),
            "G_U": gu,
            "G_Q": gq,
            "F": 0.5 * (gu + gq),
            "periods": rows,
        }

    def evaluate(self, layout):
        layout = np.asarray(layout, int)
        assert layout.shape == (3, self.K) and all(len(set(row)) == self.K for row in layout)
        response = self.base["baseline_response_min"].astype(np.float64).copy()
        for t, row in enumerate(layout):
            for j in row:
                np.minimum(response[t], self.od[t][j], out=response[t])
        result = self.metrics_response(response)
        ds = []
        pairs = []
        for t in range(2):
            d, m = self.match(layout[t], layout[t + 1])
            ds.append(d)
            pairs.append(
                [
                    {"origin": j, "destination": k, "distance_km": float(self.dist[j, k])}
                    for k, j in m.items()
                ]
            )
        result.update(
            D_km=sum(ds),
            transition_km=ds,
            feasible=sum(ds) <= self.budget + 1e-8,
            selected_rows=layout.tolist(),
            selected_ids=self.ids[layout].tolist(),
            matching=pairs,
        )
        return result
