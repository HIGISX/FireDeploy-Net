"""Sampling and deterministic decoding on CPU or one CUDA device."""

import numpy as np
import torch

from .environment import BatchEnvironment


class Runner:
    def __init__(self, data, device="cpu"):
        self.device = torch.device(device)
        self.d = data
        self.xy = torch.as_tensor(data.xy, dtype=torch.float32, device=self.device)
        self.future = torch.as_tensor(
            (data.solo / data.scales[:, :, None]).reshape(6, data.n).T,
            dtype=torch.float32,
            device=self.device,
        )
        self.scales = np.maximum((0.5 * data.solo[:, 0] + 0.5 * data.solo[:, 1]).max(1) * 3, 1e-12)
        self.envs = {}

    def rollout(self, model, batch, sample, forced=None):
        if batch not in self.envs:
            self.envs[batch] = BatchEnvironment(self.d, batch)
        env = self.envs[batch].reset()
        cache = model.begin(self.xy, batch)
        lp = []
        entropy = []
        previous = hidden = None
        previous_pool = self.xy.new_zeros(batch, 128)
        current_sum = self.xy.new_zeros(batch, 128)
        for k in range(3 * self.d.K):
            t = k // self.d.K
            stage = k % self.d.K
            if stage == 0:
                previous = hidden = None
                if t:
                    previous_pool = current_sum / self.d.K
                current_sum = self.xy.new_zeros(batch, 128)
            legal, cost, finite, context = env.observe()
            components = env.remaining[:, t].transpose(0, 2, 1) * 3
            features = torch.as_tensor(components, dtype=torch.float32, device=self.device)
            gain = torch.as_tensor(
                components.mean(-1) / float(self.scales[t]), dtype=torch.float32, device=self.device
            )
            blocked = ~torch.as_tensor(legal, device=self.device)
            prev = torch.as_tensor(
                env.flags[:, t - 1].copy() if t else np.zeros_like(legal), device=self.device
            )
            inside = torch.as_tensor(
                env.completion.copy() if t else np.zeros_like(legal), device=self.device
            )
            cost_t = torch.as_tensor(np.clip(cost, -4, 4), dtype=torch.float32, device=self.device)
            finite_t = torch.as_tensor(finite, device=self.device)
            ctx = torch.as_tensor(context, device=self.device)
            force = (
                torch.as_tensor(forced[:, k], device=self.device) if forced is not None else None
            )
            selected, h, act, en = model.action(
                cache,
                previous,
                hidden,
                gain,
                (self.d.K - stage) / self.d.K,
                blocked,
                features,
                (prev, inside, cost_t, finite_t),
                ctx,
                previous_pool,
                current_sum / max(stage, 1),
                self.future,
                sample,
                force,
            )
            # Snapshot all environment arrays passed to the network by CUDA copies.
            actions = act.detach().cpu().numpy()
            env.step(actions)
            lp.append(selected)
            entropy.append(en)
            hidden = h
            previous = act
            if model.adapted:
                current_sum = current_sum + cache["h"][act]
        values, rows, distance = env.finish()
        return (
            values,
            rows,
            distance,
            torch.stack(lp, 1).sum(1),
            torch.stack(entropy).mean(),
            env.distances.copy(),
        )
