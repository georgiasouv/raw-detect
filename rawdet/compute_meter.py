"""
Input-dependent compute measurement for the fair accuracy-vs-compute Pareto.

Our front-end's cost varies per image because the gates skip stages, so a
standard FLOPs counter (which assumes a fixed graph) cannot represent it. This
meter measures EXECUTED compute: run the gates hard at inference and average the
FLOPs of the stages that actually ran. Fixed-compute competitors get a constant
value; our adaptive method gets its true mean -- so the Pareto compares like
with like instead of crediting us a budget we don't always spend.
"""
import torch


def stage_flops(channels, h, w, kernel=3):
    """MACs of one refinement stage = two conv(channels->channels, k x k)."""
    per_conv = channels * channels * kernel * kernel * h * w
    return 2 * per_conv                                   # the stage has two convs


@torch.no_grad()
def mean_executed_flops(per_stage_decisions, per_stage_flops):
    """per_stage_decisions: [N, T] hard 0/1 (1 = stage ran for that image).
    per_stage_flops: [T]. Returns mean executed FLOPs per image."""
    d = torch.as_tensor(per_stage_decisions, dtype=torch.float64)   # [N, T]
    f = torch.as_tensor(per_stage_flops, dtype=torch.float64)       # [T]
    return float((d * f).sum(dim=1).mean())


@torch.no_grad()
def trace_gates(frontend, loader, device="cpu"):
    """Collect per-image hard gate decisions [N, T] over a loader. Assumes the
    front-end exposes gate_trace(x) -> [B, T] of hard 0/1 (a small method that
    stacks each CoupledGatedStage's z_hard); adapt to your forward if different."""
    frontend.eval().to(device)
    rows = [frontend.gate_trace(x.to(device)).cpu() for x in loader]
    return torch.cat(rows, 0)


if __name__ == "__main__":
    f1 = stage_flops(16, 64, 64)
    assert f1 == 2 * 16 * 16 * 9 * 64 * 64

    # two stages; image A runs both, image B runs only the first -> mean 1.5x one stage
    m = mean_executed_flops([[1, 1], [1, 0]], [f1, f1])
    assert abs(m - 1.5 * f1) < 1.0, m

    # an always-on (fixed-compute) model would report 2x for every image
    fixed = mean_executed_flops([[1, 1], [1, 1]], [f1, f1])
    assert abs(fixed - 2.0 * f1) < 1.0

    print(f"one stage      : {f1:.3e} MACs")
    print(f"adaptive mean  : {m:.3e}  (1.5x one stage)")
    print(f"fixed-compute  : {fixed:.3e}  (2x one stage)")
    print("COMPUTE METER OK")
