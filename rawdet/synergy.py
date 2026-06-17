"""
Synergy experiment: couple the adaptive-compute gate to the sensor statistics,
and measure whether compute spend rises with sensor unfamiliarity.

Two pieces:
  1. CoupledGatedStage -- the gate now sees the SAME statistics s(x) that drive
     cross-sensor FiLM adaptation, in addition to the local feature map.
  2. A measurement harness -- Mahalanobis distance of s(x) from the training
     distribution (the "unfamiliarity" axis) + per-image compute depth +
     a rank correlation between them.

The bottom block is a SYNTHETIC methodology check: it verifies the harness
reports a strong correlation when the effect is present (a coupled toy gate)
and ~zero when it is absent (an uncoupled toy gate). The real result requires
training on the five datasets; this only proves the measurement is sound.
"""
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. The coupled gate.
# ---------------------------------------------------------------------------
class CoupledGatedStage(nn.Module):
    """Soft-skip stage whose gate is conditioned on pooled features AND the
    sensor statistics s(x). Set stats_dim=0 to get the UNCOUPLED control."""

    def __init__(self, channels, stats_dim=0, stats_proj=8, tau=1.0, cost=1.0):
        super().__init__()
        self.tau, self.cost = tau, cost
        self.couple = stats_dim > 0
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        if self.couple:
            self.stats_proj = nn.Linear(stats_dim, stats_proj)
        gate_in = channels + (stats_proj if self.couple else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, max(gate_in // 2, 2)), nn.ReLU(inplace=True),
            nn.Linear(max(gate_in // 2, 2), 1),
        )
        nn.init.constant_(self.gate[-1].bias, 3.0)        # start open

    def forward(self, h, s=None, gumbel=True):
        feat = self.pool(h)                                # [B, C]
        if self.couple:
            feat = torch.cat([feat, self.stats_proj(s)], dim=1)
        logit = self.gate(feat)                            # [B, 1]
        pi = torch.sigmoid(logit / self.tau).mean()        # penalty target
        s_logit = logit
        if gumbel and self.training:
            u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
            s_logit = logit + (u.log() - (1 - u).log())
        z_soft = torch.sigmoid(s_logit / self.tau)
        z_hard = (z_soft > 0.5).float()
        z = (z_soft + (z_hard - z_soft).detach())[:, :, None, None]
        out = z * self.refine(h) + (1 - z) * h
        return out, pi, z_hard.squeeze(1)                  # hard decision -> depth


# ---------------------------------------------------------------------------
# 2. Measurement harness.
# ---------------------------------------------------------------------------
def fit_training_gaussian(stats_matrix: np.ndarray):
    """Fit mean and inverse covariance of s(x) over the TRAINING set."""
    mean = stats_matrix.mean(axis=0)
    cov = np.cov(stats_matrix, rowvar=False) + 1e-6 * np.eye(stats_matrix.shape[1])
    return mean, np.linalg.inv(cov)


def mahalanobis(s: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray):
    """Distance of each row of s from the training distribution (unfamiliarity)."""
    d = s - mean
    return np.sqrt(np.einsum("ni,ij,nj->n", d, cov_inv, d))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, no scipy dependency."""
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


# ---------------------------------------------------------------------------
# Verifications.
# ---------------------------------------------------------------------------
def _check_module():
    torch.manual_seed(0)
    h = torch.rand(4, 16, 8, 8)
    s = torch.rand(4, 29)

    coupled = CoupledGatedStage(16, stats_dim=29).train()
    out, pi, depth = coupled(h, s)
    (out.mean() + pi).backward()
    assert tuple(out.shape) == (4, 16, 8, 8)
    assert tuple(depth.shape) == (4,)
    assert coupled.stats_proj.weight.grad is not None       # gate learns from s(x)

    uncoupled = CoupledGatedStage(16, stats_dim=0).train()   # control: no s(x)
    out2, _, _ = uncoupled(h)                                # s not needed
    assert tuple(out2.shape) == (4, 16, 8, 8)
    print("[PASS] coupled gate consumes s(x) and is trainable; uncoupled control runs")


def _check_measurement():
    """Synthetic: does the harness detect the effect when present, null when absent?"""
    rng = np.random.default_rng(0)
    n, d_stats, T = 600, 6, 4
    S = rng.standard_normal((n, d_stats))
    mean, cov_inv = fit_training_gaussian(S)
    dist = mahalanobis(S, mean, cov_inv)
    p = (dist - dist.min()) / (dist.max() - dist.min())      # normalized distance

    depth_coupled = rng.binomial(T, 0.15 + 0.75 * p)         # more stages when far
    depth_uncoupled = rng.binomial(T, 0.5, size=n)           # independent of distance

    rc = spearman(depth_coupled.astype(float), dist)
    ru = spearman(depth_uncoupled.astype(float), dist)
    print(f"[INFO] coupled   Spearman(depth, unfamiliarity) = {rc:+.3f}  (want strong +)")
    print(f"[INFO] uncoupled Spearman(depth, unfamiliarity) = {ru:+.3f}  (want ~0)")
    assert rc > 0.35 and abs(ru) < 0.15 and (rc - abs(ru)) > 0.3   # clear separation
    print("[PASS] harness detects the coupling when present and reports null when absent")


if __name__ == "__main__":
    _check_module()
    _check_measurement()
    print("ALL SYNERGY CHECKS PASSED")
