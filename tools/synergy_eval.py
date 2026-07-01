"""Synergy measurement: does compute spend track sensor unfamiliarity?

THE contribution figure. For a trained model:
  1. Fit a Gaussian to s(x) over the TRAINING set (the "familiar" distribution).
  2. For each VAL image, measure
        unfamiliarity = Mahalanobis(s(x), train_gaussian)
        compute       = sum_t cost_t * gate_t(x)        (from frontend.gate_trace)
  3. Report Spearman(compute, unfamiliarity) + scatter plot.

Run on the COUPLED model (gate sees s(x)) and on an UNCOUPLED control
(gate blind to s(x)); the coupled curve should rise with unfamiliarity, the
uncoupled one should be flat. That contrast is the result.

    python tools/synergy_eval.py CONFIG CHECKPOINT \
        --out-dir work_dirs/synergy/Syn02 --max-train 1000 --max-val 851
"""
import argparse, os
import numpy as np
import torch

from rawdet.synergy import fit_training_gaussian, mahalanobis, spearman


# ---------------------------------------------------------------------------
# Analysis core (pure: takes a frontend + iterables of preprocessed inputs).
# Factored out so it can be unit-tested without mmdet.
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_stats(frontend, inputs_iter, device, max_items):
    """Run frontend.stats over inputs; return [N, d_s] matrix of s(x)."""
    rows, n = [], 0
    for batch_inputs in inputs_iter:
        batch_inputs = batch_inputs.to(device)
        s = frontend.stats(batch_inputs)             # [B, d_s], no grad
        rows.append(s.cpu().numpy())
        n += s.shape[0]
        if n >= max_items:
            break
    return np.concatenate(rows, axis=0)[:max_items]


@torch.no_grad()
def collect_stats_and_compute(frontend, inputs_iter, device, max_items):
    """Return (S [N,d_s], compute [N]) over inputs, compute from gate_trace."""
    costs = torch.tensor([float(st.cost) for st in frontend.stages], device=device)
    S, C, n = [], [], 0
    for batch_inputs in inputs_iter:
        batch_inputs = batch_inputs.to(device)
        s = frontend.stats(batch_inputs)             # [B, d_s]
        z = frontend.gate_trace(batch_inputs)        # [B, num_stages] in {0,1}
        comp = (z * costs[None, :]).sum(dim=1)       # [B] weighted depth
        S.append(s.cpu().numpy()); C.append(comp.cpu().numpy()); n += s.shape[0]
        if n >= max_items:
            break
    return (np.concatenate(S, 0)[:max_items],
            np.concatenate(C, 0)[:max_items])


def run_analysis(frontend, train_iter, val_iter, device,
                 max_train=1000, max_val=100000):
    """The measurement: fit on train s(x), correlate val compute vs unfamiliarity."""
    S_train = collect_stats(frontend, train_iter, device, max_train)
    mean, cov_inv = fit_training_gaussian(S_train)

    S_val, compute = collect_stats_and_compute(frontend, val_iter, device, max_val)
    unfamiliarity = mahalanobis(S_val, mean, cov_inv)
    rho = spearman(compute, unfamiliarity)
    return dict(rho=rho, compute=compute, unfamiliarity=unfamiliarity,
                mean=mean, cov_inv=cov_inv, n_train=len(S_train), n_val=len(S_val))


def save_outputs(res, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{tag}_synergy.npz"),
             compute=res['compute'], unfamiliarity=res['unfamiliarity'],
             rho=res['rho'])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.scatter(res['unfamiliarity'], res['compute'], s=8, alpha=0.4)
        plt.xlabel('sensor unfamiliarity  (Mahalanobis of s(x))')
        plt.ylabel('compute depth  (gated stages)')
        plt.title(f"{tag}:  Spearman = {res['rho']:+.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_synergy.png"), dpi=130)
        plt.close()
    except Exception as e:
        print(f"(plot skipped: {e})")


# ---------------------------------------------------------------------------
# mmdet plumbing: build model + loaders from the config, load checkpoint.
# ---------------------------------------------------------------------------
def _inputs_iter(dataloader, model):
    """Yield preprocessed batch_inputs [B,4,h,w] from an mmdet dataloader."""
    for data in dataloader:
        data = model.data_preprocessor(data, training=False)
        yield data['inputs']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', default='work_dirs/synergy')
    ap.add_argument('--tag', default='run')
    ap.add_argument('--max-train', type=int, default=1000)
    ap.add_argument('--max-val', type=int, default=100000)
    args = ap.parse_args()

    import rawdet  # noqa: registers custom modules
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.out_dir
    # strip W&B during analysis -- local only
    cfg.visualizer = dict(type='DetLocalVisualizer',
                          vis_backends=[dict(type='LocalVisBackend')],
                          name='visualizer')
    cfg.custom_hooks = []
    # 1-CPU node: 2 workers can deadlock -> force main-process loading
    for dl in ('train_dataloader', 'val_dataloader'):
        if dl in cfg:
            cfg[dl]['num_workers'] = 0
            cfg[dl]['persistent_workers'] = False                            # no early-stop etc. for eval

    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    device = next(model.parameters()).device; model.eval().to(device)
    frontend = model.frontend

    # device set above from model params
    res = run_analysis(
        frontend,
        _inputs_iter(runner.train_dataloader, model),
        _inputs_iter(runner.val_dataloader, model),
        device, max_train=args.max_train, max_val=args.max_val)

    print(f"\n=== {args.tag} ===")
    print(f"train images (Gaussian fit): {res['n_train']}")
    print(f"val images                 : {res['n_val']}")
    print(f"Spearman(compute, unfamiliarity) = {res['rho']:+.4f}")
    save_outputs(res, args.out_dir, args.tag)
    print(f"saved scatter + npz to {args.out_dir}")


if __name__ == '__main__':
    main()