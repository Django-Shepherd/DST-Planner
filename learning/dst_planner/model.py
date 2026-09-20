"""Shared linear-attention graph encoder, history actor, twin Q, value diffusion."""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    hidden: int = 128
    graph_layers: int = 3
    history_layers: int = 2
    heads: int = 4
    horizon: int = 5
    diffusion_steps: int = 10
    diffusion_samples: int = 5
    trust_alpha: float = 1.0


class GraphLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.local = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, h, valid, edges):
        b, n, d = h.shape
        q, k, v = self.qkv(h).chunk(3, -1)
        q = F.elu(q) + 1
        k = (F.elu(k) + 1) * valid[..., None]
        v = v * valid[..., None]
        kv = torch.einsum("bnd,bne->bde", k, v)
        global_h = (
            torch.einsum("bnd,bde->bne", q, kv)
            / torch.einsum("bnd,bd->bn", q, k.sum(1)).clamp_min(1e-6)[..., None]
        )
        flat = h.reshape(b * n, d)
        local = torch.zeros_like(flat)
        degree = torch.zeros(b * n, device=h.device, dtype=h.dtype)
        if edges.numel():
            src, dst = edges
            local.index_add_(0, dst, flat[src])
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        local = (local / degree.clamp_min(1)[:, None]).reshape(b, n, d)
        h = self.norm(h + self.out(global_h) + self.local(local))
        return self.norm2(h + self.ff(h)) * valid[..., None]


class GraphEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input = nn.Linear(7, c.hidden)
        self.layers = nn.ModuleList(
            [GraphLayer(c.hidden) for _ in range(c.graph_layers)]
        )

    def forward(self, batch):
        h = self.input(batch["nodes"]) * batch["valid"][..., None]
        for layer in self.layers:
            h = layer(h, batch["valid"], batch["edges"])
        return h


class SequenceActor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        d = c.hidden
        self.spatial = nn.MultiheadAttention(d, c.heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d, c.heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.history = nn.TransformerEncoder(
            layer, c.history_layers, enable_nested_tensor=False
        )
        self.position = nn.Embedding(c.horizon, d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Linear(2 * d, d)
        self.key = nn.Linear(d, d)

    def context(self, h, b):
        current = h[torch.arange(len(h), device=h.device), b["current"]]
        global_h = self.spatial(
            current[:, None], h, h, key_padding_mask=~b["valid"], need_weights=False
        )[0][:, 0]
        candidates = h.gather(1, b["candidates"][..., None].expand(-1, -1, h.shape[-1]))
        return current, global_h, candidates

    def logits(self, context, b, prefix):
        current, global_h, candidates = context
        batch_size, v, d = candidates.shape
        allowed = b["candidate_valid"].clone()
        if prefix.shape[1]:
            safe = prefix.clamp_min(0)
            hist = candidates.gather(1, safe[..., None].expand(-1, -1, d))
            hist = (
                hist
                + self.position(torch.arange(prefix.shape[1], device=hist.device))[None]
            )
            valid_hist = prefix >= 0
            # Finished rows retain at least one unmasked element to avoid all-masked attention NaNs.
            padding = ~valid_hist
            padding = padding.clone()
            padding[:, 0] = False
            encoded = self.history(hist, src_key_padding_mask=padding)
            pooled = (encoded * valid_hist[..., None]).sum(1) / valid_hist.sum(
                1
            ).clamp_min(1)[:, None]
            state = self.fuse(torch.cat([global_h, pooled], -1))
            for k in range(prefix.shape[1]):
                allowed.scatter_(
                    1,
                    safe[:, k, None],
                    allowed.gather(1, safe[:, k, None]) & ~valid_hist[:, k, None],
                )
            last = safe[:, -1] + 1
            allowed &= b["reachable"][
                torch.arange(batch_size, device=last.device), last, 1:
            ]
        else:
            state = self.fuse(torch.cat([global_h, torch.zeros_like(global_h)], -1))
            allowed &= b["reachable"][:, 0, 1:]
        q = self.query(torch.cat([state, current], -1))
        logits = torch.einsum("bd,bvd->bv", q, self.key(candidates)) / math.sqrt(d)
        return logits.masked_fill(~allowed, -1e9), allowed

    def rollout(self, h, b, sample=False):
        context = self.context(h, b)
        batch_size = len(h)
        prefix = torch.empty(batch_size, 0, dtype=torch.long, device=h.device)
        logs = []
        masks = []
        alive = b["candidate_valid"].any(1)
        for _ in range(self.config.horizon):
            logits, allowed = self.logits(context, b, prefix)
            valid = alive & allowed.any(1)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample() if sample else logits.argmax(1)
            logs.append(dist.log_prob(a) * valid)
            masks.append(valid)
            prefix = torch.cat([prefix, torch.where(valid, a, -1)[:, None]], 1)
            alive = valid
        return prefix, torch.stack(logs, 1), torch.stack(masks, 1)

    def bc(self, h, b, labels):
        context = self.context(h, b)
        prefix = labels[:, :0]
        loss = h.sum() * 0.0
        count = 0
        for k in range(labels.shape[1]):
            logits, allowed = self.logits(context, b, prefix)
            target = labels[:, k]
            valid = target >= 0
            if valid.any():
                if not allowed[valid, target[valid]].all():
                    raise ValueError("Expert sequence violates action mask")
                loss = loss + F.cross_entropy(
                    logits[valid], target[valid], reduction="sum"
                )
                count += int(valid.sum())
            prefix = labels[:, : k + 1]
        return loss / max(count, 1)


class TwinQ(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )

    def forward(self, condition):
        return self.q1(condition).squeeze(-1), self.q2(condition).squeeze(-1)


class ValueDiffusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.steps = c.diffusion_steps
        d = c.hidden
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.net = nn.Sequential(
            nn.Linear(3 * d + 33, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        # Cosine schedule works for a genuinely short (10-step) diffusion chain.
        x = torch.linspace(0, self.steps, self.steps + 1)
        abar = torch.cos(((x / self.steps + 0.008) / 1.008) * math.pi / 2) ** 2
        abar = abar / abar[0]
        beta = (1 - abar[1:] / abar[:-1]).clamp(0.0001, 0.999)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1 - beta)
        self.register_buffer("abar", torch.cumprod(1 - beta, 0))

    def forward(self, condition, noisy, t):
        te = self.time((t.float() / max(1, self.steps - 1))[..., None])
        return self.net(torch.cat([condition, noisy[..., None], te], -1)).squeeze(-1)

    def loss(self, condition, target):
        t = torch.randint(self.steps, target.shape, device=target.device)
        noise = torch.randn_like(target)
        abar = self.abar[t]
        noisy = abar.sqrt() * target + (1 - abar).sqrt() * noise
        return F.mse_loss(self(condition, noisy, t), noise)

    @torch.no_grad()
    def samples(self, condition, count):
        shape = condition.shape[:-1]
        c = condition.unsqueeze(0).expand(count, *condition.shape)
        q = torch.randn((count, *shape), device=condition.device)
        for step in reversed(range(self.steps)):
            t = torch.full_like(q, step, dtype=torch.long)
            eps = self(c, q, t)
            a = self.alpha[step]
            ab = self.abar[step]
            prev = self.abar[step - 1] if step else torch.ones_like(ab)
            mean = (q - self.beta[step] / (1 - ab).sqrt() * eps) / a.sqrt()
            variance = self.beta[step] * (1 - prev) / (1 - ab)
            q = mean + variance.sqrt() * torch.randn_like(q) if step else mean
        return q


class DST(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = GraphEncoder(config)
        self.actor = SequenceActor(config)
        self.critic = TwinQ(config.hidden)
        self.diffusion = ValueDiffusion(config)

    def conditions(self, h, b):
        d = h.shape[-1]
        cur = h[torch.arange(len(h), device=h.device), b["current"]]
        glob = (h * b["valid"][..., None]).sum(1) / b["valid"].sum(1).clamp_min(1)[
            :, None
        ]
        cand = h.gather(1, b["candidates"][..., None].expand(-1, -1, d))
        return torch.cat(
            [cur[:, None].expand_as(cand), glob[:, None].expand_as(cand), cand], -1
        )

    @torch.no_grad()
    def trust(self, condition, valid):
        q = self.diffusion.samples(condition, self.config.diffusion_samples)
        std = q.std(0, unbiased=False)
        mean = (std * valid).sum(-1, keepdim=True) / valid.sum(
            -1, keepdim=True
        ).clamp_min(1)
        return torch.exp(-self.config.trust_alpha * std / mean.clamp_min(1e-8)) * valid
