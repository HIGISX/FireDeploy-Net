"""Graph attention encoder with checkpoint-compatible parameter names.

Adapted from attention-learn-to-route, copyright (c) 2018 Wouter Kool.
See licenses/attention-learn-to-route-MIT.txt.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super().__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """Original Wq/Wk/Wv/Wout projections; replace only softmax(QK')V.

        No dropout, pruning, sparse approximation, or changed normalization. The
        original mask marks *blocked* entries; PyTorch's Boolean mask marks allowed
        entries. Encoder self-attention ordinarily receives no mask.
        """
        if h is None:
            h = q
        batch, n, dim = h.shape
        nq = q.shape[1]
        if q.shape[0] != batch or q.shape[2] != dim or dim != self.input_dim:
            raise ValueError("Wrong attention input dimensions")
        qflat, hflat = q.reshape(-1, dim), h.reshape(-1, dim)
        query = torch.matmul(qflat, self.W_query).view(self.n_heads, batch, nq, self.key_dim)
        key = torch.matmul(hflat, self.W_key).view(self.n_heads, batch, n, self.key_dim)
        value = torch.matmul(hflat, self.W_val).view(self.n_heads, batch, n, self.val_dim)
        allowed = None if mask is None else ~mask.to(torch.bool).reshape(batch, 1, nq, n)
        query, key, value = (t.permute(1, 0, 2, 3) for t in (query, key, value))
        heads = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
            scale=self.norm_factor,
        )
        return torch.mm(
            heads.permute(0, 2, 1, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch, nq, self.embed_dim)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads=8, embed_dim=128, n_layers=3):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    SkipConnection(MultiHeadAttention(n_heads, embed_dim, embed_dim)),
                    nn.LayerNorm(embed_dim),
                    SkipConnection(
                        nn.Sequential(
                            nn.Linear(embed_dim, 512), nn.ReLU(), nn.Linear(512, embed_dim)
                        )
                    ),
                    nn.LayerNorm(embed_dim),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        h = self.layers(x)
        return h, h.mean(dim=1)
