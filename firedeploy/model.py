"""FireDeploy-Net architecture and masked autoregressive action selection."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .encoder import GraphAttentionEncoder


def shared_mlp(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 11001)
        return nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )


class MLPPolicy(nn.Module):
    def __init__(self, seed=17):
        super().__init__()
        self.local = shared_mlp(seed)

    def begin(self, xy, batch):
        first = self.local[0]
        return {"xy": xy, "local_xy": F.linear(xy, first.weight[:, :2], first.bias), "batch": batch}

    def local_score(self, cache, gain, remaining):
        # Algebraically the same MLP([x,y,gain,remaining]); reuse static first-layer work.
        dynamic = torch.stack([gain, torch.full_like(gain, remaining)], -1)
        h = cache["local_xy"][None, :, :] + F.linear(dynamic, self.local[0].weight[:, 2:], None)
        h = self.local[1](h)
        h = self.local[2](h)
        h = self.local[3](h)
        return self.local[4](h).squeeze(-1)

    def decision(self, cache, previous, hidden, gain, remaining, blocked):
        logits = 10 * torch.tanh(self.local_score(cache, gain, remaining))
        return logits.masked_fill(blocked.clone(), -torch.inf), None


class SpatialPolicy(MLPPolicy):
    def __init__(self, seed=17):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 22001)
            self.xy_embedding = nn.Linear(2, 128)
            self.encoder = GraphAttentionEncoder(8, 128, 3)
            self.gru = nn.GRU(128, 128, 1, batch_first=True)
            self.first = nn.Parameter(torch.empty(128).uniform_(-1, 1))
            self.context = nn.Linear(128, 128, bias=False)
            self.keys = nn.Linear(128, 384, bias=False)
            self.output = nn.Linear(128, 128, bias=False)
            nn.init.zeros_(self.output.weight)

    def begin(self, xy, batch):
        cache = super().begin(xy, batch)
        h, _ = self.encoder(self.xy_embedding(xy)[None, :, :])
        h = h[0]
        k, v, l = self.keys(h).chunk(3, -1)
        heads = lambda a: a.reshape(-1, 8, 16).transpose(0, 1).contiguous()
        cache.update(
            h=h,
            k=heads(k)[None, :, :, :],
            v=heads(v)[None, :, :, :],
            l=l,
            hidden=self.context(h.mean(0))[None, None, :].expand(1, batch, -1).contiguous(),
        )
        return cache

    def decision(self, cache, previous, hidden, gain, remaining, blocked):
        batch = gain.shape[0]
        x = (
            self.first[None, None, :].expand(batch, 1, -1)
            if previous is None
            else cache["h"][previous][:, None, :]
        )
        query, hidden = self.gru(x, cache["hidden"] if hidden is None else hidden)
        q = query.reshape(batch, 8, 16).permute(1, 0, 2)[None, :, :, :]
        # Queries are independent trajectories sharing the same static keys/values.
        h = F.scaled_dot_product_attention(
            q,
            cache["k"],
            cache["v"],
            attn_mask=(~blocked)[None, None, :, :],
            dropout_p=0.0,
            scale=1 / math.sqrt(16),
        )[0]
        glimpse = self.output(h.permute(1, 0, 2).reshape(batch, 128))
        pointer = glimpse @ cache["l"].T / math.sqrt(128)
        raw = self.local_score(cache, gain, remaining) + pointer
        return (10 * torch.tanh(raw)).masked_fill(blocked.clone(), -torch.inf), hidden


class ServicePolicy(SpatialPolicy):
    def __init__(self, seed=17):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 44001)
            self.service_d = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            )
            self.service_b = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            )
            self.service_state = nn.Linear(128, 16, bias=False)
            self.service_space = nn.Linear(128, 16, bias=False)
            # 16 GRU-state projection + remaining budget + two pooled 16D gains.
            self.service_gate = nn.Sequential(nn.Linear(49, 32), nn.ReLU(), nn.Linear(32, 16))
            nn.init.zeros_(self.service_gate[-1].weight)
            nn.init.zeros_(self.service_gate[-1].bias)
            self.service_output = nn.Linear(16, 1, bias=False)
            nn.init.zeros_(self.service_output.weight)

    def begin(self, xy, batch):
        cache = super().begin(xy, batch)
        cache["service_space"] = self.service_space(cache["h"])
        return cache


class FireDeployNet(ServicePolicy):
    """Spatial policy with gated service fusion and a period-repositioning branch."""

    def __init__(self, pretrained=None, seed=17):
        super().__init__(seed=17)
        if pretrained is not None:
            payload = torch.load(pretrained, map_location="cpu", weights_only=False)
            self.load_state_dict(payload["policy"], strict=True)
        self.adapted = True
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 5000000)
            self.adapt_candidate = nn.Linear(10, 16)
            self.adapt_context = nn.Linear(8, 16)
            self.adapt_previous_pool = nn.Linear(128, 16, bias=False)
            self.adapt_current_pool = nn.Linear(128, 16, bias=False)
            self.adapt_output = nn.Linear(16, 1, bias=False)
            nn.init.zeros_(self.adapt_output.weight)
        self.activation_checkpointing = True

    def raw(
        self,
        cache,
        previous,
        hidden,
        gain,
        remaining,
        blocked,
        features,
        extra,
        context,
        previous_pool,
        current_pool,
        future,
    ):
        batch = len(gain)
        x = (
            self.first[None, None, :].expand(batch, 1, -1)
            if previous is None
            else cache["h"][previous][:, None, :]
        )
        query, hidden = self.gru(x, cache["hidden"] if hidden is None else hidden)
        q = query.reshape(batch, 8, 16).permute(1, 0, 2)[None]
        att = F.scaled_dot_product_attention(
            q,
            cache["k"],
            cache["v"],
            attn_mask=(~blocked)[None, None],
            dropout_p=0.0,
            scale=1 / math.sqrt(16),
        )[0]
        glimpse = self.output(att.permute(1, 0, 2).reshape(batch, 128))
        pointer = glimpse @ cache["l"].T / math.sqrt(128)
        local = self.local_score(cache, gain, remaining)
        # Explicit arguments: checkpoint replay cannot depend on mutable service_inputs.
        zd = self.service_d(features[:, :, :1])
        zb = self.service_b(features[:, :, 1:])
        allowed = (~blocked).to(features.dtype)[:, :, None]
        count = allowed.sum(1).clamp_min(1.0)
        hs = self.service_state(hidden[0])
        budget = hs.new_full((batch, 1), remaining)
        alpha = torch.sigmoid(
            self.service_gate(
                torch.cat(
                    (hs, budget, (zd * allowed).sum(1) / count, (zb * allowed).sum(1) / count), -1
                )
            )
        )
        service = self.service_output(
            (alpha[:, None] * zd + (1 - alpha[:, None]) * zb)
            * torch.tanh(cache["service_space"][None] + hs[:, None])
        ).squeeze(-1)
        raw = local + pointer + service
        if self.adapted:
            prev, inside, cost, finite = extra
            # Static future service features are cached and do not use future layouts.
            dynamic = torch.stack((prev, inside, cost, finite), -1).to(features.dtype)
            a = F.linear(future, self.adapt_candidate.weight[:, :6], self.adapt_candidate.bias)[
                None
            ]
            a = a + F.linear(dynamic, self.adapt_candidate.weight[:, 6:], None)
            c = (
                self.adapt_context(context)
                + self.adapt_previous_pool(previous_pool)
                + self.adapt_current_pool(current_pool)
            )
            raw = raw + self.adapt_output(torch.tanh(a + c[:, None])).squeeze(-1)
        return (10 * torch.tanh(raw)).masked_fill(blocked, -torch.inf), hidden

    def action(
        self,
        cache,
        previous,
        hidden,
        gain,
        remaining,
        blocked,
        features,
        extra,
        context,
        previous_pool,
        current_pool,
        future,
        sample,
        forced=None,
    ):
        def fn(
            hidden,
            gain,
            blocked,
            features,
            prev,
            inside,
            cost,
            finite,
            context,
            previous_pool,
            current_pool,
        ):
            logits, h = self.raw(
                cache,
                previous,
                hidden,
                gain,
                remaining,
                blocked,
                features,
                (prev, inside, cost, finite),
                context,
                previous_pool,
                current_pool,
                future,
            )
            with torch.no_grad():
                act = (
                    forced
                    if forced is not None
                    else (
                        (logits - torch.empty_like(logits).exponential_().log_()).argmax(-1)
                        if sample
                        else logits.argmax(-1)
                    )
                )
            lp = torch.log_softmax(logits, -1)
            selected = lp.gather(1, act[:, None])[:, 0]
            with torch.no_grad():
                entropy = torch.where(torch.isfinite(lp), -lp * lp.exp(), 0.0).sum(-1)
            return selected, h, act, entropy

        args = (hidden, gain, blocked, features, *extra, context, previous_pool, current_pool)
        if torch.is_grad_enabled() and self.activation_checkpointing:
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)
        return fn(*args)
