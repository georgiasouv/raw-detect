"""
Learnable RAW preprocessing front-end for task-driven object detection.

Maps a packed RAW frame [B, 4, h, w] in [0, 1] to a detector-ready 3-channel
image [B, 3, H, W] in [0, 1], optimized end-to-end against a FROZEN off-the-shelf
detector. Components mirror the method spec:

    global tone op  ->  conv  ->  FiLM(gamma, beta from RAW statistics)
                    ->  gated adaptive-compute stages  ->  conv -> sigmoid
                    ->  (optional) pixel-shuffle upsample back to full resolution

Per-sensor LoRA can be attached after shared training for cheap specialization.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
from mmdet.registry import MODELS


# ---------------------------------------------------------------------------
# 1. Statistics encoder  -- a fixed (non-learnable) feature of the input.
#    Computed under no_grad: we never backprop into s(x).
# ---------------------------------------------------------------------------
class StatsEncoder(nn.Module):
    """s(x) = [ K-bin histogram | m percentiles | per-channel mean | per-channel std ]."""

    def __init__(self, num_bins: int = 16,
                 percentiles: Tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("percentiles", torch.tensor(percentiles))

    def out_dim(self, num_channels: int) -> int:
        return self.num_bins + len(self.percentiles) + 2 * num_channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.reshape(B, C, -1)                                   # [B, C, N]

        # Histogram of mean intensity -- captures exposure / distribution shape.
        intensity = flat.mean(dim=1)                                 # [B, N]
        edges = torch.linspace(0.0, 1.0, self.num_bins + 1, device=x.device)
        idx = torch.bucketize(intensity, edges[1:-1])                # [B, N] in [0, K-1]
        hist = torch.zeros(B, self.num_bins, device=x.device)
        hist.scatter_add_(1, idx, torch.ones_like(intensity))
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)

        # Percentiles -- capture dynamic range location and clipping.
        qs = torch.quantile(intensity, self.percentiles.to(x.device), dim=1)  # [m, B]
        qs = qs.transpose(0, 1)                                       # [B, m]

        # Per-channel mean / std -- capture colour cast and per-channel noise.
        ch_mean = flat.mean(dim=2)                                   # [B, C]
        ch_std = flat.std(dim=2)                                     # [B, C]

        return torch.cat([hist, qs, ch_mean, ch_std], dim=1)         # [B, d_s]


# ---------------------------------------------------------------------------
# 2. FiLM conditioning  -- statistics -> per-channel (gamma, beta).
#    Identity-initialized: gamma = 1, beta = 0 at start.
# ---------------------------------------------------------------------------
class FiLMGenerator(nn.Module):
    def __init__(self, in_dim: int, num_channels: int, hidden: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * num_channels),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias[:num_channels] = 1.0          # gamma -> 1
            last.bias[num_channels:] = 0.0          # beta  -> 0

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.net(s).chunk(2, dim=1)   # each [B, C]
        return gamma, beta


def film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """feat: [B, C, H, W];  gamma, beta: [B, C]."""
    return gamma[:, :, None, None] * feat + beta[:, :, None, None]


# ---------------------------------------------------------------------------
# 3. Per-sensor low-rank adaptation for a conv layer.
#    base(x) + (alpha / r) * B(A(x)).  B is zero-initialized -> starts as no-op.
# ---------------------------------------------------------------------------
class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int = 4, alpha: float = 4.0):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Conv2d(base.in_channels, r, kernel_size=1, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, kernel_size=base.kernel_size,
                           stride=base.stride, padding=base.padding, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / r)
        nn.init.zeros_(self.B.weight)               # delta W = 0 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.B(self.A(x))


# ---------------------------------------------------------------------------
# 4. Global tone operator  -- per-channel learnable power curve, init = identity.
# ---------------------------------------------------------------------------
class GlobalTone(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.log_g = nn.Parameter(torch.zeros(num_channels))   # g = exp(0) = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.log_g.exp()[None, :, None, None]
        return x.clamp(self.eps, 1.0).pow(g)


# ---------------------------------------------------------------------------
# 5. Adaptive-compute gated stage  -- soft skip with a straight-through gate.
#    Forward uses a hard {0,1} decision; backward uses the soft sigmoid gradient.
# ---------------------------------------------------------------------------
class GatedStage(nn.Module):
    def __init__(self, channels: int, tau: float = 1.0, cost: float = 1.0):
        super().__init__()
        self.tau = tau
        self.cost = cost                            # relative FLOP cost c_t
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // 2), nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 3.0)  # start open: pi ~ 1 (full pipeline)

    def forward(self, h: torch.Tensor, gumbel: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        logit = self.gate(h)                                    # [B, 1]
        pi = torch.sigmoid(logit / self.tau).mean()             # clean penalty target

        s_logit = logit
        if gumbel and self.training:
            u = torch.rand_like(logit).clamp(1e-6, 1.0 - 1e-6)
            s_logit = logit + (u.log() - (1.0 - u).log())       # Gumbel noise
        z_soft = torch.sigmoid(s_logit / self.tau)
        z_hard = (z_soft > 0.5).float()
        z = (z_soft + (z_hard - z_soft).detach())[:, :, None, None]   # STE

        out = z * self.refine(h) + (1.0 - z) * h
        return out, pi


# ---------------------------------------------------------------------------
# 6. The full front-end.
# ---------------------------------------------------------------------------
@MODELS.register_module()
class RAWFrontEnd(nn.Module):
    def __init__(self, in_ch: int = 4, feat: int = 16, num_stages: int = 2,
                 num_bins: int = 16,
                 percentiles: Tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99),
                 upsample: bool = True):
        super().__init__()
        self.stats = StatsEncoder(num_bins, percentiles)
        d_s = self.stats.out_dim(in_ch)
        self.tone = GlobalTone(in_ch)
        self.proj = nn.Conv2d(in_ch, feat, 1)
        self.film_gen = FiLMGenerator(d_s, feat)
        self.stages = nn.ModuleList([GatedStage(feat) for _ in range(num_stages)])
        out_ch = 12 if upsample else 3              # 12 = 3 * (2 * 2) for pixel-shuffle
        self.head = nn.Conv2d(feat, out_ch, 3, padding=1)
        self.ps = nn.PixelShuffle(2) if upsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s = self.stats(x)                                       # [B, d_s], no grad inside
        h = self.proj(self.tone(x))                             # [B, feat, h, w]
        gamma, beta = self.film_gen(s)
        h = film(h, gamma, beta)
        pis: List[torch.Tensor] = []
        for stage in self.stages:
            h, pi = stage(h)
            pis.append(pi)
        out = torch.sigmoid(self.ps(self.head(h)))              # [B, 3, H, W] in [0, 1]
        return out, pis


def add_lora(frontend: RAWFrontEnd, r: int = 4, alpha: float = 4.0) -> RAWFrontEnd:
    """Freeze the shared front-end and attach trainable low-rank residuals.

    After this call, only the LoRA A/B convs carry gradients.
    """
    for p in frontend.parameters():
        p.requires_grad_(False)
    frontend.proj = LoRAConv2d(frontend.proj, r, alpha)
    frontend.head = LoRAConv2d(frontend.head, r, alpha)
    return frontend
