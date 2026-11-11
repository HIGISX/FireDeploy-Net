"""Batched FP64 service updates with conservative budget-feasibility masks.

The mask certifies a feasible completion; it need not enumerate all feasible actions.
Exact directed minimum-cost matching is recomputed after a completion changes.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def update_marginals(j, rp, ri, rv, cp, ci, cv, best, remaining):
    """Exact facility-location marginal updates through CSC reverse incidence."""
    gain = 0.0
    for k in range(rp[j], rp[j + 1]):
        i = ri[k]
        old = best[i]
        new = rv[k]
        if new <= old:
            continue
        gain += new - old
        best[i] = new
        for z in range(cp[i], cp[i + 1]):
            v = cv[z]
            if v > old:
                remaining[ci[z]] -= min(v, new) - old
    return gain


@njit(cache=True, parallel=True)
def batch_update(actions, rp, ri, rv, cp, ci, cv, best, remaining):
    result = np.zeros(len(actions), np.float64)
    for b in prange(len(actions)):
        result[b] = update_marginals(actions[b], rp, ri, rv, cp, ci, cv, best[b], remaining[b])
    return result


class BatchEnvironment:
    def __init__(self, d, batch):
        self.d = d
        self.batch = batch
        self.bi = np.arange(batch)
        self.reset()

    def reset(self):
        d = self.d
        b = self.batch
        self.k = 0
        self.rows = np.full((b, 3, d.K), -1, np.int64)
        self.flags = np.zeros((b, 3, d.n), bool)
        self.best = np.zeros((b, 3, 2, d.m), np.float64)
        self.remaining = np.broadcast_to(d.solo, (b, *d.solo.shape)).copy()
        self.improvement = np.zeros((b, 3, 2), np.float64)
        self.completion = np.zeros((b, d.n), bool)
        self.completion_rows = np.full((b, d.K), -1, np.int64)
        self.mapping = np.full((b, d.n), -1, np.int64)
        self.current = np.zeros(b)
        self.prefix = np.zeros(b)
        self.distances = np.zeros((b, 2))
        self.match_calls = 0
        self._observed = False
        return self

    def observe(self):
        if self._observed:
            return self.legal, self.cost, self.finite, self.context
        d = self.d
        t = self.k // d.K
        stage = self.k % d.K
        self.legal = ~self.flags[:, t].copy()
        self.cost = np.zeros((self.batch, d.n))
        self.finite = np.ones_like(self.legal)
        if t:
            available = self.completion & ~self.flags[:, t]
            assert available.any(1).all()
            self.replaced = available.argmax(1)
            self.source = self.mapping[self.bi, self.replaced]
            raw = d.dist[self.source] - d.dist[self.source, self.replaced, None]
            self.finite = np.isfinite(raw)
            certificate = self.finite & (
                self.prefix[:, None] + self.current[:, None] + raw <= d.budget + 1e-8
            )
            self.legal &= self.completion | certificate
            # For already-completing sites the accepted action leaves the layout unchanged.
            self.cost = np.where(self.completion, 0.0, np.where(self.finite, raw, 0.0)) / max(
                d.budget, 1e-12
            )
        self.context = np.zeros((self.batch, 8), np.float32)
        self.context[:, t] = 1.0
        self.context[:, 3] = (d.K - stage) / d.K
        self.context[:, 4] = self.prefix / max(d.budget, 1e-12)
        self.context[:, 5] = self.current / max(d.budget, 1e-12)
        self.context[:, 6] = (d.budget - self.prefix - self.current) / max(d.budget, 1e-12)
        self.context[:, 7] = (2 - t) / 2
        assert self.legal.any(1).all()
        self._observed = True
        return self.legal, self.cost, self.finite, self.context

    def features(self):
        legal, cost, finite, context = self.observe()
        t = self.k // self.d.K
        # Three cancels equal-period aggregation in the current gain matrices.
        components = self.remaining[:, t].transpose(0, 2, 1) * 3
        assert components.min() > -1e-9
        prev = self.flags[:, t - 1] if t else np.zeros_like(legal)
        extra = np.stack(
            (prev, self.completion if t else np.zeros_like(legal), np.clip(cost, -4, 4), finite), -1
        ).astype(np.float32)
        return components, legal, extra, context

    def step(self, actions):
        d = self.d
        actions = np.asarray(actions, np.int64)
        t = self.k // d.K
        stage = self.k % d.K
        self.observe()
        assert self.legal[self.bi, actions].all()
        if t:
            outside = ~self.completion[self.bi, actions]
            for b in np.flatnonzero(outside):
                old = self.replaced[b]
                j = actions[b]
                source = self.source[b]
                witness = self.prefix[b] + self.current[b] - d.dist[source, old] + d.dist[source, j]
                self.completion[b, old] = False
                self.completion[b, j] = True
                self.completion_rows[b, np.flatnonzero(self.completion_rows[b] == old)[0]] = j
                value, mapping = d.match(self.rows[b, t - 1], self.completion_rows[b])
                assert self.prefix[b] + value <= witness + 1e-8
                self.current[b] = value
                for destination, origin in mapping.items():
                    self.mapping[b, destination] = origin
                self.match_calls += 1
            assert (self.prefix + self.current <= d.budget + 1e-8).all()
        self.rows[:, t, stage] = actions
        self.flags[self.bi, t, actions] = True
        for q in range(2):
            m = d.mats[t][q]
            c = d.csc[t][q]
            self.improvement[:, t, q] += batch_update(
                actions,
                m.indptr,
                m.indices,
                m.data,
                c.indptr,
                c.indices,
                c.data,
                self.best[:, t, q],
                self.remaining[:, t, q],
            )
        self.k += 1
        self._observed = False
        if stage == d.K - 1:
            if t:
                assert np.array_equal(self.completion, self.flags[:, t])
                self.distances[:, t - 1] = self.current
                self.prefix += self.current
            if t < 2:
                self.completion = self.flags[:, t].copy()
                self.mapping[:] = -1
                self.completion_rows = self.rows[:, t].copy()
                for b in range(self.batch):
                    self.mapping[b, self.rows[b, t]] = self.rows[b, t]
                self.current[:] = 0.0

    def finish(self):
        assert self.k == 3 * self.d.K and (self.flags.sum(2) == self.d.K).all()
        assert (self.prefix <= self.d.budget + 1e-8).all()
        return 0.5 * self.improvement.sum((1, 2)), self.rows.copy(), self.prefix.copy()
